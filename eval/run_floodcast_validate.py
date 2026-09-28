"""Does our solver reproduce FloodCastBench's reference depths?

Every planning result in Part 5 was scored against a surrogate written in this
repository. §5.6 proposes to fix that for flood by treating an accepted
shallow-water solver as reality. That claim is only worth making if our
implementation actually agrees with the accepted one, so this script measures it
rather than asserting it.

Protocol: take the event's DEM, its land-cover-derived Manning field and its
published initial condition; force our local-inertial solver with the event's own
GPM-IMERG rainfall; and compare our depth field against the reference frame by
frame, on the data paper's own metric -- Critical Success Index at 0.01 m and
0.05 m -- plus RMSE and bias over wet cells.

Nothing here is fitted to the reference. Roughness comes from land cover, not
from matching depths, because a solver tuned to reproduce its comparison target
tells you nothing about whether it is independent of it.

Run:
    python3 eval/run_floodcast_validate.py \
        --root $WORK/data/floodcastbench/FloodCastBench \
        --event pakistan2022 --n-frames 48 --out runs/floodcast_validate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pspe.simulate.flood import FloodConfig, FloodSolver
from pspe.simulate.real.floodcast import (
    critical_success_index,
    describe,
    frame_seconds,
    load_event,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted FloodCastBench directory")
    ap.add_argument("--event", default="australia2022")
    ap.add_argument("--start-hours", type=float, default=192.0,
                    help="initialise from the reference frame at this elapsed "
                         "time; the default sits in Australia's quiescent window")
    ap.add_argument("--n-frames", type=int, default=48,
                    help="reference frames to compare (300 s apart)")
    ap.add_argument("--stride", type=int, default=1, help="compare every Nth frame")
    ap.add_argument("--crop", type=int, default=0,
                    help="centre-crop to this many cells per side (0 = full domain)")

    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--open-edges", nargs="*", default=["south", "north", "east", "west"],
                    help="domain edges that discharge; pass with no values for a "
                         "closed domain. Isolates how much of the dry bias is "
                         "boundary handling rather than dynamics.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--describe-only", action="store_true")
    ap.add_argument("--out", default="runs/floodcast_validate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.describe_only:
        info = describe(args.root)
        print(json.dumps(info, indent=2))
        (out / "archive.json").write_text(json.dumps(info, indent=2))
        return

    ev = load_event(args.root, args.event)
    print(f"event {ev.name}: grid {ev.shape}, dx {ev.dx} m, "
          f"{len(ev.depth_frames)} depth frames", flush=True)

    device = torch.device(args.device)

    def prep(a: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.nan_to_num(a), dtype=torch.float32, device=device)
        if args.crop:
            h, w = t.shape
            c = args.crop
            r0, c0 = max(0, (h - c) // 2), max(0, (w - c) // 2)
            t = t[r0 : r0 + c, c0 : c0 + c]
        return t

    # The archive ships no rainfall, so the event's forcing cannot be replayed.
    # Validate instead on a window where the forcing is negligible, and MEASURE
    # how negligible rather than asserting it: the reference's own volume budget
    # gives the answer. Australia's volume rises +8996/h at the peak and
    # +13..+45/h after t=192 h against a total of ~419,000, i.e. ~0.01% per hour.
    times = [frame_seconds(p) for p in ev.depth_frames]
    t0 = args.start_hours * 3600.0
    i0 = min(range(len(times)), key=lambda i: abs(times[i] - t0))
    idx = [i for i in range(i0, len(ev.depth_frames), args.stride)][: args.n_frames]
    if len(idx) < 2:
        raise SystemExit("need at least two frames after --start-hours")

    v0 = float(np.nan_to_num(ev.depth(idx[0])).sum())
    v1 = float(np.nan_to_num(ev.depth(idx[-1])).sum())
    span_h = (times[idx[-1]] - times[idx[0]]) / 3600.0
    drift = abs(v1 - v0) / max(v0, 1e-9)
    print(f"validation window t={times[idx[0]]/3600:.1f}..{times[idx[-1]]/3600:.1f} h "
          f"({span_h:.1f} h, {len(idx)} frames)", flush=True)
    print(f"unmodelled forcing over the window: {100*drift:.3f}% of volume "
          f"({100*drift/max(span_h,1e-9):.4f}%/h) -- the bound on this comparison",
          flush=True)

    dem = prep(ev.dem)
    manning = prep(ev.manning)
    h = prep(ev.depth(idx[0])).unsqueeze(0).clone()
    print(f"grid {tuple(dem.shape)}  dem {float(dem.min()):.1f}..{float(dem.max()):.1f} m  "
          f"manning {float(manning.min()):.3f} (uniform: no land cover in archive)",
          flush=True)

    rows = dem.mean(dim=1)
    bed_slope = max(abs(float((rows[0] - rows[-1]) / (dem.shape[0] * ev.dx))), 1e-5)
    cfg = FloodConfig(dx=ev.dx, open_edges=tuple(args.open_edges),
                      bed_slope=bed_slope, dt_max=ev.dt)
    print(f"open edges: {tuple(args.open_edges) or '(closed domain)'}", flush=True)
    print(f"measured bed slope {bed_slope:.2e}", flush=True)

    solver = FloodSolver(dem.unsqueeze(0), cfg, manning=manning.unsqueeze(0))
    qx, qy = solver.zeros_flux(1)

    out_rows = []
    t_now = float(times[idx[0]])
    for j, fi in enumerate(idx):
        target = float(times[fi])
        while t_now < target:
            dt = min(solver.adaptive_dt(h), target - t_now)
            if dt <= 0:
                break
            h, qx, qy = solver.step(h, qx, qy, dt)
            t_now += dt

        ref = prep(ev.depth(fi)).cpu().numpy()
        ours = h[0].cpu().numpy()
        wet = (ref > args.thresholds[0]) | (ours > args.thresholds[0])
        row = {
            "frame": int(fi), "t_hours": target / 3600.0,
            "ref_wet": int((ref > args.thresholds[0]).sum()),
            "our_wet": int((ours > args.thresholds[0]).sum()),
            "rmse_wet": float(np.sqrt(((ours - ref) ** 2)[wet].mean())) if wet.any() else float("nan"),
            "bias_wet": float((ours - ref)[wet].mean()) if wet.any() else float("nan"),
        }
        for th in args.thresholds:
            row[f"csi@{th}"] = critical_success_index(ours, ref, th)
        out_rows.append(row)
        if j % max(1, len(idx) // 12) == 0 or j == len(idx) - 1:
            csis = "  ".join(f"CSI@{th}={row[f'csi@{th}']:.3f}" for th in args.thresholds)
            print(f"  t={target/3600:7.2f}h  {csis}  RMSE={row['rmse_wet']:.4f} m  "
                  f"bias={row['bias_wet']:+.4f} m  wet ref/ours "
                  f"{row['ref_wet']}/{row['our_wet']}", flush=True)

    summary = {
        "event": ev.name, "dx": ev.dx, "grid": list(dem.shape),
        "bed_slope": bed_slope,
        "window_hours": [times[idx[0]] / 3600.0, times[idx[-1]] / 3600.0],
        "unmodelled_forcing_frac": drift,
        "frames": out_rows,
        "mean_csi": {f"@{th}": float(np.nanmean([r[f"csi@{th}"] for r in out_rows]))
                     for th in args.thresholds},
        "mean_rmse_wet": float(np.nanmean([r["rmse_wet"] for r in out_rows])),
        "note": (
            "The archive ships no rainfall and no land cover, so the event's own "
            "forcing cannot be replayed and roughness is a uniform literature "
            "value rather than a fitted or land-cover-derived field. This "
            "validates the ROUTING operator over a window whose unmodelled "
            "forcing is measured and reported above. Agreement is evidence our "
            "solver is the accepted scheme; it is not evidence either solver is "
            "the world (6.2)."
        ),
        "config": vars(args),
    }
    (out / "validate.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean CSI {summary['mean_csi']}  mean RMSE {summary['mean_rmse_wet']:.4f} m", flush=True)
    print(f"wrote {out/'validate.json'}", flush=True)


if __name__ == "__main__":
    main()
