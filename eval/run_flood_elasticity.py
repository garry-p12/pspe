"""Why a calibrated margin is hardest to fit exactly where the levee matters.

The precondition `b + z_delta*sigma < f*s` failed on flood with a negligible bias,
so the whole margin was `z_delta*sigma`. A 25% log-spread in discharge produced an
83% spread in realised depth -- a 3.3x amplification. This script measures the
mechanism instead of asserting it.

Over-topping is a threshold. To first order the amplification is the elasticity

    E = d ln D / d ln Q

of exposure-weighted depth with respect to discharge, because for a small
log-spread sigma_lnQ the depth spread is sigma_lnD ~ E * sigma_lnQ. The
prediction is that E peaks where the flood is just topping the berm's weakest gap
-- which is precisely the regime where a levee has leverage, since a structure can
only matter when the water is near its crest.

If that holds, the tension is structural rather than a quirk of one scenario:
    leverage and margin-feasibility are in conflict in threshold-governed hazards.

All discharges are evaluated in a single batched solve, so the comparison shares
one timestep sequence and cannot drift between arms.

Run:
    python3 eval/run_flood_elasticity.py --out runs/flood_elasticity
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from pspe.simulate.flood import (
    FloodConfig,
    FloodSolver,
    floodplain_terrain,
    levee_basis,
)

SITES = (0.42, 0.50, 0.60, 0.68, 0.78, 0.86)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--dx", type=float, default=480.0)
    ap.add_argument("--slope", type=float, default=1.5e-4)
    ap.add_argument("--manning", type=float, default=0.03)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--inflows", type=float, nargs="+",
                    default=[4e-3, 6e-3, 8e-3, 1.0e-2, 1.2e-2, 1.4e-2, 1.8e-2, 2.4e-2])
    ap.add_argument("--rel-step", type=float, default=0.10,
                    help="central-difference step in ln Q")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--f-fraction", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/flood_elasticity")
    args = ap.parse_args()

    device = torch.device(args.device)
    z0, exposure = floodplain_terrain(
        grid=args.grid, dx=args.dx, slope=args.slope, channel_depth=5.0,
        channel_width=5.0, berm_height=3.0,
        gaps=((0.42, 0.55), (0.60, 0.80), (0.78, 0.40)),
        town_centre=(0.62, 0.20), town_radius=0.060, town_drop=1.2, device=device,
    )
    masks = levee_basis(args.grid, dx=args.dx, sites=SITES).to(device)
    k = masks.shape[0]
    cfg = FloodConfig(dx=args.dx, open_edges=("south",), manning=args.manning,
                      bed_slope=args.slope)
    rows = torch.arange(args.grid, device=device)
    berm_col = z0[:, 25:45].argmax(dim=1) + 25

    # Three discharges per operating point (Q, Q e^-h, Q e^+h) x two plans
    # (do nothing, spend the budget) -> one batched solve.
    # Per operating point: do-nothing at three discharges (for the elasticity),
    # plus every candidate allocation at the central discharge (for the span).
    # The span must be the BEST achievable: two of the six sites make flooding
    # worse (§5.6), so an even spread understates it and biases sigma/s upward.
    h_step = args.rel_step
    alloc = [torch.full((k,), args.budget / k, device=device)]      # even spread
    for j in range(k):
        c = torch.zeros(k, device=device)
        c[j] = min(args.budget, 3.0)
        alloc.append(c)
    per_q = 3 + len(alloc)
    qs, plan_rows = [], []
    for q in args.inflows:
        for mult in (math.exp(-h_step), 1.0, math.exp(h_step)):     # idx 0,1,2
            qs.append(q * mult)
            plan_rows.append(torch.zeros(k, device=device))
        for a_vec in alloc:                                          # idx 3..
            qs.append(q)
            plan_rows.append(a_vec)
    inflow = torch.tensor(qs, device=device)
    plans = torch.stack(plan_rows)
    b = len(qs)
    print(f"one batched solve of {b} configurations", flush=True)

    z = z0 + (plans[:, :, None, None] * masks[None]).sum(1)
    solver = FloodSolver(z, cfg)
    h = torch.zeros(b, args.grid, args.grid, device=device)
    qx, qy = solver.zeros_flux(b)
    peak = torch.zeros_like(h)
    t, dur = 0.0, args.days * 86400.0
    while t < dur:
        dt = min(solver.adaptive_dt(h), dur - t)
        frac = t / dur
        shape = min(frac / 0.33, max(0.0, (1.0 - frac) / 0.67))
        h, qx, qy = solver.step(h, qx, qy, dt, z=z)
        h[:, 0:3, 46:51] = h[:, 0:3, 46:51] + dt * (inflow[:, None, None] * shape)
        peak = torch.maximum(peak, h)
        t += dt
    depth = (peak * exposure).flatten(1).sum(1)
    crest_wet = (peak[:, rows, berm_col] > 0.05).sum(dim=1)

    z_delta = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - args.delta)))
    out_rows = []
    print(f"\n{'Q':>9} {'D_none':>8} {'D_best':>8} {'span s':>8} {'elasticity':>11} "
          f"{'berm rows wet':>13} {'sig/s @0.25':>12} {'need':>7} {'verdict':>8}", flush=True)
    for i, q in enumerate(args.inflows):
        base = i * per_q
        d_lo = float(depth[base + 0]); d_mid = float(depth[base + 1]); d_hi = float(depth[base + 2])
        cands = [float(depth[base + 3 + j]) for j in range(len(alloc))]
        best = min(cands)
        best_j = int(min(range(len(cands)), key=lambda j: cands[j]))
        span = max(d_mid - best, 1e-9)
        # Central difference in logs; guard the dry regime where ln D is undefined.
        if min(d_lo, d_hi, d_mid) > 1e-6:
            elast = (math.log(d_hi) - math.log(d_lo)) / (2.0 * h_step)
        else:
            elast = float("nan")
        # sigma/s implied at a 0.25 log-spread, the operating point measured earlier.
        sig_over_s = (elast * 0.25 * d_mid / span) if elast == elast else float("nan")
        need = args.f_fraction / z_delta
        verdict = ("PASS" if sig_over_s < need else "FAIL") if sig_over_s == sig_over_s else "dry"
        out_rows.append({
            "inflow": q, "depth_do_nothing": d_mid, "depth_best": best, "span": span,
            "elasticity": elast, "berm_rows_wet": int(crest_wet[base + 1]),
            "best_allocation_index": best_j,
            "sigma_over_s_at_0.25": sig_over_s, "required_sigma_over_s": need,
            "verdict": verdict,
        })
        print(f"{q:9.1e} {d_mid:8.4f} {best:8.4f} {span:8.4f} {elast:11.2f} "
              f"{int(crest_wet[base+1]):10d}/96 {sig_over_s:12.3f} {need:7.3f} {verdict:>8}",
              flush=True)

    finite = [r for r in out_rows if r["elasticity"] == r["elasticity"]]
    peak_row = max(finite, key=lambda r: r["elasticity"]) if finite else None
    summary = {
        "rows": out_rows,
        "peak_elasticity": peak_row,
        "z_delta": z_delta, "f_fraction": args.f_fraction,
        "claim": (
            "Amplification E = dlnD/dlnQ is largest near the over-topping "
            "threshold, which is also where a levee has leverage. Leverage and "
            "margin feasibility are therefore in tension in threshold-governed "
            "hazards."
        ),
        "config": vars(args),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "elasticity.json").write_text(json.dumps(summary, indent=2))
    if peak_row:
        print(f"\npeak amplification {peak_row['elasticity']:.2f}x at Q={peak_row['inflow']:.1e} "
              f"({peak_row['berm_rows_wet']}/96 berm rows wet)", flush=True)
    print(f"wrote {out/'elasticity.json'}", flush=True)


if __name__ == "__main__":
    main()
