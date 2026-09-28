"""Precompute every mitigation option so the app never runs a solver.

An operational tool has to answer "what if we build this?" in milliseconds. A
shallow-water solve takes minutes. So the options are enumerated here, once, and
the app queries the results.

Every scenario shares ONE batched solve: the timestep loop is the expensive
part and it is identical across options, only the bed differs. Twelve options
therefore cost roughly one solve, not twelve.

Outputs, per scenario:
    peak depth raster    what the app draws
    exposure             which cells flood, for counting buildings
    a cost estimate      indicative only, and labelled as such

Run on Vista:
    python eval/build_scenario_library.py --root <extracted> --out runs/scenarios
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from pspe.simulate.flood import FloodConfig, FloodSolver
from pspe.simulate.real.floodcast_scenario import build_scenario

# Indicative unit cost for an earthen levee, AUD per metre of crest per metre of
# height. Order-of-magnitude only: real costs depend on foundation, land
# acquisition and access. The app labels every figure as indicative.
COST_PER_M_PER_M = 2600.0


def enumerate_options(k: int, max_h: float, seed: int = 0):
    """Do-nothing, each site alone, and a CALIBRATION SET of combinations.

    The combinations are not decoration. Single-site effects are exact by
    construction, so everything the surrogate has to guess about lives in the
    interaction between measures, and the spread of that guess is what the
    conformal margin is a quantile of. Split conformal needs
    `n >= 1/delta - 1` calibration points before any finite order statistic
    carries the stated rate: **9 for delta = 0.1, 19 for delta = 0.05**. The
    original five hand-picked combinations plus "all" gave six, which is not
    enough to state 90% and `pspe.plan.margins.conformal_quantile` correctly
    refuses rather than returning a number it cannot justify.

    Heights are varied, not all at max. The calibration sample must be
    exchangeable with what will actually be predicted, and a planner spending a
    budget proposes partial heights far more often than full ones -- calibrating
    only on full-height combinations would measure the wrong distribution.
    """
    opts = [{"id": "base", "name": "Do nothing", "heights": [0.0] * k}]
    for i in range(k):
        h = [0.0] * k
        h[i] = max_h
        opts.append({"id": f"s{i}", "name": f"Levee at site {i}", "heights": h})

    rng = random.Random(seed)
    combos: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for combo in pairs:                                   # every pair, full height
        combos.append((combo, tuple(max_h for _ in combo)))
    triples = [(i, j, l) for i in range(k) for j in range(i + 1, k)
               for l in range(j + 1, k)]
    for combo in rng.sample(triples, min(8, len(triples))):
        combos.append((combo, tuple(max_h for _ in combo)))
    for combo in rng.sample(pairs, min(6, len(pairs))):   # partial heights
        combos.append((combo, tuple(rng.choice([0.5, 1.0, 1.5, 2.0, 2.5])
                                    for _ in combo)))
    combos.append((tuple(range(k)), tuple(max_h for _ in range(k))))

    seen = set()
    for combo, heights in combos:
        h = [0.0] * k
        for i, hi in zip(combo, heights):
            h[i] = hi
        key = tuple(h)
        if key in seen:
            continue
        seen.add(key)
        tag = "".join(f"{i}@{hi:g}" for i, hi in zip(combo, heights))
        opts.append({
            "id": "c" + tag.replace(".", "p").replace("@", ""),
            "name": "Levees at sites " + ", ".join(
                f"{i} ({hi:g} m)" for i, hi in zip(combo, heights)),
            "heights": h,
        })
    return opts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--event", default="australia2022")
    ap.add_argument("--coarsen", type=int, default=2, help="2 -> 60 m")
    ap.add_argument("--start-hours", type=float, default=48.0)
    ap.add_argument("--end-hours", type=float, default=120.0)
    ap.add_argument("--settle-rc", type=int, nargs=2, default=[104, 104],
                    help="settlement centre in the COARSENED grid")
    ap.add_argument("--settle-radius", type=int, default=20)
    ap.add_argument("--n-sites", type=int, default=6)
    ap.add_argument("--max-height", type=float, default=3.0)
    ap.add_argument("--site-width-m", type=float, default=900.0,
                    help="levee footprint in METRES (not cells: see defect 20). "
                         "900 m gives a structure of roughly 2 km, which is the "
                         "scale of a council scheme rather than a farm bund.")
    ap.add_argument("--manning", type=float, default=0.035)
    ap.add_argument("--rain-scales", type=float, nargs="+", default=[1.0],
                    help="hydrograph multipliers, e.g. 0.8 1.0 1.25 for a "
                         "smaller / recorded / larger event")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/scenarios")
    args = ap.parse_args()

    device = torch.device(args.device)
    scn = build_scenario(args.root, args.event, coarsen=args.coarsen,
                         start_hours=args.start_hours, end_hours=args.end_hours,
                         settle_radius=args.settle_radius, n_sites=args.n_sites,
                         settle_rc=tuple(args.settle_rc),
                         site_width_m=args.site_width_m, device=device)
    k = scn.k
    print(json.dumps(scn.meta, indent=2), flush=True)
    print(f"sites at elevations {[round(e,1) for e in scn.site_elevations]}", flush=True)

    rows = scn.z.mean(dim=1)
    slope = max(abs(float((rows[0] - rows[-1]) / (scn.z.shape[0] * scn.dx))), 1e-5)
    cfg = FloodConfig(dx=scn.dx, open_edges=(), manning=args.manning,
                      bed_slope=slope, dt_max=300.0)

    opts = enumerate_options(k, args.max_height)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"scenario": scn.meta, "sites": [], "options": [],
             "rain_scales": args.rain_scales, "dx_m": scn.dx,
             "grid": list(scn.z.shape),
             "cost_note": ("Indicative only. Unit rate "
                           f"A${COST_PER_M_PER_M:,.0f} per metre of crest per "
                           "metre of height; real cost depends on foundation, "
                           "land acquisition and access.")}

    # Crest length per site, from the mask footprint, for the cost estimate.
    for i, e in enumerate(scn.site_elevations):
        cells = float((scn.masks[i] > 0.3).sum())
        length_m = float(np.sqrt(cells)) * scn.dx
        index["sites"].append({"id": i, "elev_m": e, "crest_length_m": length_m})

    for rs in args.rain_scales:
        plans = torch.tensor([o["heights"] for o in opts], dtype=torch.float32,
                             device=device)
        b = plans.shape[0]
        z = scn.z[None] + (plans[:, :, None, None] * scn.masks[None]).sum(1)
        solver = FloodSolver(z, cfg)
        h = scn.h0[None].expand(b, *scn.z.shape).clone()
        qx, qy = solver.zeros_flux(b)
        peak = h.clone()
        t, dur = 0.0, scn.duration
        t0 = time.time()
        step = 0
        while t < dur:
            dt = min(solver.adaptive_dt(h), dur - t)
            if dt <= 0:
                break
            h, qx, qy = solver.step(h, qx, qy, dt, rain=scn.rain * rs, z=z)
            peak = torch.maximum(peak, h)
            t += dt
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"  rain x{rs}: {100*t/dur:5.1f}%  {el:.0f}s elapsed, "
                      f"~{el*(dur-t)/max(t,1e-9):.0f}s left", flush=True)

        tag = f"r{rs:g}".replace(".", "p")
        np.save(out / f"peak_{tag}.npy", peak.cpu().numpy().astype(np.float32))
        base_exposed = None
        for j, o in enumerate(opts):
            d = peak[j]
            exp_depth = float((d * scn.exposure).sum())
            wet = int((d > 0.10).sum())
            if j == 0:
                base_exposed = wet
            cost = sum(
                index["sites"][i]["crest_length_m"] * hgt * COST_PER_M_PER_M
                for i, hgt in enumerate(o["heights"]) if hgt > 0
            )
            index["options"].append({
                "rain_scale": rs, "id": o["id"], "name": o["name"],
                "heights": o["heights"], "cost_aud": cost,
                "exposure_depth_m": exp_depth,
                "wet_cells_10cm": wet,
                "wet_change_vs_base": wet - (base_exposed or 0),
                "raster_index": j,
            })
            print(f"  [{o['id']:>6}] exposure {exp_depth:.5f} m  wet {wet}  "
                  f"cost A${cost/1e6:.2f}M", flush=True)

        # Checkpoint after each event scale. Writing the index only at the end
        # means a timeout leaves saved rasters that nothing can interpret.
        (out / "index.json").write_text(json.dumps(index, indent=2))
        print(f"  checkpointed index.json after rain x{rs}", flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {out/'index.json'} and peak rasters", flush=True)


if __name__ == "__main__":
    main()
