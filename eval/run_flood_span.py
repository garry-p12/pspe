"""Phase 0 gate for the flood testbed: can actuation move the objective at all?

`swe` was retired because levee-free, topography-free actuation ranked planners
inside a ~7% achievable band (defect 12). Before building any planning or margin
result on flood, this script measures the same quantity on the flood testbed.

It also asserts the scenario is well posed before solving anything, because three
separate faults in earlier versions produced physically plausible floods on which
no intervention could work:

  * the hydrograph was injected across the berm, delivering water straight onto
    the protected floodplain;
  * the settlement's depression overlapped the berm and the channel, so it had
    carved its own gap and was scored on channel water;
  * the forcing submerged the structure entirely, so levee height was irrelevant.

Run:
    python3 eval/run_flood_span.py --out runs/flood_span
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pspe.simulate.flood import (
    FloodConfig,
    FloodSolver,
    floodplain_terrain,
    levee_basis,
)

SITES = (0.42, 0.50, 0.60, 0.68, 0.78, 0.86)


def build(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, FloodConfig, tuple[int, int]]:
    z, exposure = floodplain_terrain(
        grid=args.grid,
        dx=args.dx,
        slope=args.slope,
        channel_depth=args.channel_depth,
        channel_width=5.0,
        berm_height=args.berm_height,
        gaps=((0.42, 0.55), (0.60, 0.80), (0.78, 0.40)),
        town_centre=(0.62, 0.20),
        town_radius=0.060,
        town_drop=1.2,
    )
    masks = levee_basis(args.grid, dx=args.dx, sites=SITES)
    cfg = FloodConfig(
        dx=args.dx, open_edges=("south",), manning=args.manning, bed_slope=args.slope
    )

    # Geometry must be separated, or the study is meaningless. Measure it.
    town = exposure > exposure.max() * 0.3
    cells = town.nonzero()
    berm_col = z[:, 25:45].argmax(dim=1) + 25
    town_max_col = int(cells[:, 1].max())
    berm_min_col = int(berm_col.min())
    if town_max_col >= berm_min_col:
        raise SystemExit(
            f"geometry overlap: town reaches col {town_max_col}, berm starts at "
            f"col {berm_min_col}. The settlement would carve its own gap."
        )

    # Inject inside the channel only, at the measured meander position.
    chan0 = int(round(0.5 * (args.grid - 1)))
    patch = (chan0 - 2, chan0 + 3)
    if patch[0] <= berm_min_col:
        raise SystemExit("inflow patch straddles the berm")
    return z, exposure, masks, cfg, patch


def damage(z0, exposure, masks, cfg, patch, plan, args) -> tuple[float, float]:
    """Time-averaged exposure-weighted depth, and peak mean depth over the town."""
    z = z0 + (torch.as_tensor(plan, dtype=torch.float32)[:, None, None] * masks).sum(0)
    solver = FloodSolver(z, cfg)
    h = torch.zeros(1, args.grid, args.grid)
    qx, qy = solver.zeros_flux(1)
    town = exposure > exposure.max() * 0.3
    t, dur = 0.0, args.days * 86400.0
    total, peak = 0.0, torch.zeros(1, args.grid, args.grid)
    lo, hi = patch
    while t < dur:
        dt = min(solver.adaptive_dt(h), dur - t)
        frac = t / dur
        shape = min(frac / 0.33, max(0.0, (1.0 - frac) / 0.67))
        h, qx, qy = solver.step(h, qx, qy, dt, z=z)
        h[..., 0:3, lo:hi] = h[..., 0:3, lo:hi] + dt * args.inflow * shape
        peak = torch.maximum(peak, h)
        total += dt * float((h[0] * exposure).sum())
        t += dt
    return total / dur, float(peak[0][town].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--dx", type=float, default=480.0)
    ap.add_argument("--slope", type=float, default=1.5e-4)
    ap.add_argument("--channel-depth", type=float, default=5.0)
    ap.add_argument("--berm-height", type=float, default=3.0)
    ap.add_argument("--manning", type=float, default=0.03)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--inflow", type=float, default=1.4e-2)
    ap.add_argument("--max-height", type=float, default=3.0)
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="runs/flood_span")
    args = ap.parse_args()

    z0, exposure, masks, cfg, patch = build(args)
    k = masks.shape[0]
    run = lambda plan: damage(z0, exposure, masks, cfg, patch, plan, args)

    base, base_town = run([0.0] * k)
    print(f"do-nothing: damage {base:.5f}  town mean peak {base_town:.3f} m", flush=True)

    singles = []
    for i in range(k):
        plan = [0.0] * k
        plan[i] = args.max_height
        d, tm = run(plan)
        singles.append({"site": i, "row": SITES[i], "damage": d, "town": tm,
                        "reduction_pct": 100 * (base - d) / base})
        print(f"  site {i} (row {SITES[i]:.2f}) damage {d:.5f} "
              f"reduction {100*(base-d)/base:+7.2f}% town {tm:.3f} m", flush=True)

    d_all, _ = run([args.max_height] * k)

    plan, spent, cur, trace = [0.0] * k, 0.0, base, []
    while spent + args.step <= args.budget + 1e-9:
        best = None
        for i in range(k):
            if plan[i] + args.step > args.max_height:
                continue
            cand = list(plan)
            cand[i] += args.step
            d, _ = run(cand)
            if best is None or d < best[0]:
                best = (d, i)
        if best is None or best[0] >= cur - 1e-7:
            break
        cur, i = best
        plan[i] += args.step
        spent += args.step
        trace.append({"spent": spent, "site": i, "damage": cur,
                      "reduction_pct": 100 * (base - cur) / base})
        print(f"  +{args.step} at site {i} -> {cur:.5f} "
              f"({100*(base-cur)/base:+7.2f}%) spent {spent:.1f}", flush=True)

    span = 100 * (base - cur) / base
    result = {
        "do_nothing": base, "do_nothing_town_m": base_town,
        "best_budgeted": cur, "best_plan": plan,
        "unconstrained": d_all,
        "span_pct_budgeted": span,
        "span_pct_unconstrained": 100 * (base - d_all) / base,
        "singles": singles, "greedy": trace,
        "reference": {"swe_retired_at_pct": 7.0, "flood_attempt1_pct": 1.63},
        "config": vars(args),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "span.json").write_text(json.dumps(result, indent=2))
    print(f"\nachievable span {span:.2f}% budgeted, "
          f"{100*(base-d_all)/base:.2f}% unconstrained -> {out/'span.json'}", flush=True)


if __name__ == "__main__":
    main()
