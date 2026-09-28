"""Is the real-terrain span low because levees don't work here, or because I
picked the wrong settlement?

§5.6a measured a 6.18% span at the block that floods *deepest* in the reference.
That criterion plausibly selects a sump -- an area that fills regardless of where
water comes from -- which would bias the answer toward "uncontrollable" by
construction. That is a confound in a headline negative result, so it gets tested.

The screen is cheap because the depth field does not depend on the settlement,
only the exposure weighting does. Two solves -- one with rainfall, one without --
decompose the water budget at EVERY candidate site simultaneously:

    already-present   from h0, present before the planner can act
    direct rain       falls inside any ring; no levee reaches it
    routed in         arrives across the perimeter; the only controllable part

A levee can only act on the routed-in share. Sites are then ranked by
controllable depth, and the best is given the same levee test as §5.6a. If the
span stays low there too, the finding is about the hazard and not about my
choice of target.

Run:
    python eval/run_flood_sites.py --root <extracted> --out runs/flood_sites
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pspe.simulate.flood import FloodConfig, FloodSolver
from pspe.simulate.real.floodcast_scenario import build_scenario


def peak_depth(scn, cfg, rain_scale, plan=None, frac=1.0):
    z = scn.z[None].clone()
    if plan is not None:
        z = z + (torch.as_tensor(plan, dtype=torch.float32,
                                 device=scn.z.device)[None, :, None, None] * scn.masks[None]).sum(1)
    solver = FloodSolver(z, cfg)
    h = scn.h0[None].clone()
    qx, qy = solver.zeros_flux(1)
    peak = h.clone()
    t, dur = 0.0, scn.duration * frac
    while t < dur:
        dt = min(solver.adaptive_dt(h), dur - t)
        if dt <= 0:
            break
        h, qx, qy = solver.step(h, qx, qy, dt, rain=scn.rain * rain_scale, z=z)
        peak = torch.maximum(peak, h)
        t += dt
    return peak[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--event", default="australia2022")
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--start-hours", type=float, default=48.0)
    ap.add_argument("--end-hours", type=float, default=120.0)
    ap.add_argument("--settle-radius", type=int, default=10)
    ap.add_argument("--n-sites", type=int, default=6)
    ap.add_argument("--manning", type=float, default=0.035)
    ap.add_argument("--max-height", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/flood_sites")
    args = ap.parse_args()

    device = torch.device(args.device)
    scn = build_scenario(args.root, args.event, coarsen=args.coarsen,
                         start_hours=args.start_hours, end_hours=args.end_hours,
                         settle_radius=args.settle_radius, n_sites=args.n_sites,
                         device=device)
    rows = scn.z.mean(dim=1)
    slope = max(abs(float((rows[0] - rows[-1]) / (scn.z.shape[0] * scn.dx))), 1e-5)
    cfg = FloodConfig(dx=scn.dx, open_edges=(), manning=args.manning,
                      bed_slope=slope, dt_max=300.0)
    print(json.dumps(scn.meta, indent=2), flush=True)

    print("\n=== two solves: with and without rainfall ===", flush=True)
    p_full = peak_depth(scn, cfg, 1.0)
    print("  full rainfall done", flush=True)
    p_norain = peak_depth(scn, cfg, 0.0)
    print("  no rainfall done", flush=True)

    direct = scn.rain * scn.duration            # falls inside any ring
    h0 = scn.h0
    H, W = scn.z.shape
    R = args.settle_radius
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")

    cands = []
    for r in range(R + 2, H - R - 2, args.stride):
        for c in range(R + 2, W - R - 2, args.stride):
            d2 = ((yy - r) ** 2 + (xx - c) ** 2).float() / float(R) ** 2
            w = torch.exp(-d2)
            w = w / w.sum()
            total = float((p_full * w).sum())
            if total < 0.05:
                continue
            pre = float((h0 * w).sum())
            routed = float((p_norain * w).sum()) - pre
            rain_part = total - float((p_norain * w).sum())
            controllable = max(routed, 0.0) + max(rain_part - direct, 0.0)
            cands.append({
                "row": r, "col": c, "total": total, "pre": pre,
                "routed": routed, "rain_driven": rain_part,
                "direct_rain_cap": float(direct),
                "controllable": controllable,
                "controllable_frac": controllable / total if total > 0 else 0.0,
                "elev": float(scn.z[r, c]),
            })

    cands.sort(key=lambda d: -d["controllable"])
    print(f"\n=== {len(cands)} candidate settlements screened ===", flush=True)
    print(f"{'row':>5}{'col':>5}{'elev':>7}{'total':>8}{'pre':>8}{'routed':>8}"
          f"{'rain':>8}{'ctrl':>8}{'ctrl%':>7}", flush=True)
    for d in cands[: args.top]:
        print(f"{d['row']:5d}{d['col']:5d}{d['elev']:7.1f}{d['total']:8.3f}"
              f"{d['pre']:8.3f}{d['routed']:8.3f}{d['rain_driven']:8.3f}"
              f"{d['controllable']:8.3f}{100*d['controllable_frac']:6.0f}%", flush=True)

    # Nearest candidate by distance, not the first within a tolerance box: a
    # +/-stride window spans several candidates and `next` over a SORTED list
    # returns the best-ranked of them, which reported the wrong site's rank.
    used = min(cands, key=lambda d: (d["row"] - scn.settlement[0]) ** 2
               + (d["col"] - scn.settlement[1]) ** 2) if cands else None
    if used:
        rank = cands.index(used) + 1
        dist = ((used["row"] - scn.settlement[0]) ** 2
                + (used["col"] - scn.settlement[1]) ** 2) ** 0.5
        print(f"\ndefault site {scn.settlement}: nearest candidate "
              f"({used['row']},{used['col']}), {dist:.1f} cells away, ranks "
              f"{rank}/{len(cands)} by controllable depth "
              f"({100*used['controllable_frac']:.0f}% controllable)", flush=True)

    # Levee test at the most controllable site, same protocol as 5.6a.
    best = cands[0]
    print(f"\n=== levee test at the MOST controllable site ({best['row']},{best['col']}) ===",
          flush=True)
    scn2 = build_scenario(args.root, args.event, coarsen=args.coarsen,
                          start_hours=args.start_hours, end_hours=args.end_hours,
                          settle_radius=args.settle_radius, n_sites=args.n_sites,
                          device=device)
    d2 = ((yy - best["row"]) ** 2 + (xx - best["col"]) ** 2).float() / float(R) ** 2
    bowl = torch.exp(-d2)
    scn2.exposure = bowl / bowl.sum()
    ring = R + 2
    masks, elevs = [], []
    for s in range(args.n_sites):
        a0, a1 = 2 * np.pi * s / args.n_sites, 2 * np.pi * (s + 1) / args.n_sites
        be, brc = np.inf, None
        for a in np.linspace(a0, a1, 24):
            rr = int(round(best["row"] + ring * np.sin(a)))
            cc = int(round(best["col"] + ring * np.cos(a)))
            if 0 <= rr < H and 0 <= cc < W and float(scn2.z[rr, cc]) < be:
                be, brc = float(scn2.z[rr, cc]), (rr, cc)
        if brc is None:
            continue
        g = torch.exp(-(((yy - brc[0]) ** 2 + (xx - brc[1]) ** 2).float() / 9.0))
        masks.append(g); elevs.append(be)
    scn2.masks = torch.stack(masks)
    k = scn2.masks.shape[0]

    base = float((peak_depth(scn2, cfg, 1.0) * scn2.exposure).sum())
    print(f"  do-nothing {base:.5f} m", flush=True)
    singles = []
    for i in range(k):
        plan = [0.0] * k; plan[i] = args.max_height
        d = float((peak_depth(scn2, cfg, 1.0, plan) * scn2.exposure).sum())
        singles.append({"site": i, "elev": elevs[i], "depth": d,
                        "reduction_pct": 100 * (base - d) / base})
        print(f"  site {i} (elev {elevs[i]:5.1f} m) {d:.5f}  "
              f"{100*(base-d)/base:+7.2f}%", flush=True)
    d_all = float((peak_depth(scn2, cfg, 1.0, [args.max_height] * k) * scn2.exposure).sum())
    best_single = max(s["reduction_pct"] for s in singles)
    print(f"\n  all sites at full height: {d_all:.5f}  "
          f"{100*(base-d_all)/base:+7.2f}%", flush=True)
    print(f"\n=== best single levee {best_single:+.2f}%, all-sites "
          f"{100*(base-d_all)/base:+.2f}% ===", flush=True)
    print("(5.6a's site gave +5.55% single; swe was retired at ~7%; "
          "synthetic fluvial valley gave ~100%)", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sites.json").write_text(json.dumps({
        "screened": len(cands), "top": cands[: args.top],
        "site_used_in_5_6a": used,
        "levee_test": {"site": [best["row"], best["col"]], "do_nothing": base,
                       "singles": singles, "all_sites": d_all,
                       "best_single_pct": best_single,
                       "all_sites_pct": 100 * (base - d_all) / base},
        "scenario": scn.meta, "config": vars(args),
    }, indent=2))
    print(f"wrote {out/'sites.json'}", flush=True)


if __name__ == "__main__":
    main()
