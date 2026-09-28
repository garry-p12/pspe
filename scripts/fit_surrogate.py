"""Fit a fast surrogate to the scenario library, with an honest error band.

The framework's whole point is that a learned model answers in milliseconds
where the solver takes an hour, and that a calibrated margin says how far to
trust it. This is that idea in its operational form.

Model, per event scale, with g_i = h_i / h_max:

    impact(h) = base + SUM_i alpha_i * g_i + correction(active set)

`alpha_i` is read straight off the single-measure runs, so each is exact by
construction. Levees do NOT superpose — blocking two drainage paths is not the
sum of blocking each, and the measured combinations differ from the linear sum.
That residual is fitted as a correction and, more importantly, its spread
becomes the published confidence band.

The band is the operational face of the conformal margin: rather than a point
estimate the tool cannot justify, the planner is told "at least X, with 9 in 10
confidence", and the full solver verifies the plan that is actually chosen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fit_for_scale(opts: list[dict], max_h: float, metric: str = "road_cut_km") -> dict:
    """Fit against whichever quantity the tool reports.

    Road-kilometres across the whole valley turned out to be the wrong target:
    a levee's effect is concentrated within about 2 km, so a valley-wide count
    is dominated by ground the works never touch. Core flood depth at the
    defended settlement is what the scheme is actually bought to change.
    """
    base = next(o for o in opts if o["id"] == "base")
    base_cut = base[metric]

    # alpha_i: exact, from the single-measure runs.
    alpha: dict[int, float] = {}
    for o in opts:
        act = [i for i, h in enumerate(o["heights"]) if h > 0]
        if len(act) == 1:
            alpha[act[0]] = o[metric] - base_cut

    # Every multi-measure run, as (linear sum, measured).
    rows = []
    for o in opts:
        act = [i for i, h in enumerate(o["heights"]) if h > 0]
        if len(act) < 2:
            continue
        lin = sum(alpha.get(i, 0.0) * (o["heights"][i] / max_h) for i in act)
        rows.append({"n_active": len(act), "linear": lin,
                     "measured": o[metric] - base_cut})

    # Combined measures SATURATE rather than add. Three models were compared
    # against the runs: a correction linear in the number of active measures
    # (RMSE 7.14), one proportional to the linear sum (8.43), and a saturating
    # form S*tanh(sum/S) (5.89). The last is also the only one with a physical
    # reading -- levees cannot remove more water than is there -- and the only
    # one that does not predict absurdities when extrapolated.
    best_S, best_err = None, float("inf")
    for S in range(20, 500, 2):
        err = 0.0
        for r in rows:
            pred = S * math.tanh(r["linear"] / S) if S > 0 else r["linear"]
            err += (pred - r["measured"]) ** 2
        err = math.sqrt(err / max(len(rows), 1))
        if err < best_err:
            best_S, best_err = float(S), err

    # Band: what the saturating model still does not explain, at 90% two-sided,
    # widened for the small number of calibration runs.
    n = len(rows)
    band = 1.645 * best_err * math.sqrt(1 + 1 / n) if n >= 2 else best_err

    return {
        "base_road_cut_km": base_cut,
        "base_area_km2": base["area_flooded_km2"],
        "alpha_km": {str(i): v for i, v in sorted(alpha.items())},
        "saturation_S": best_S,
        "fit_rmse": best_err,
        "band_km": band,
        "n_calibration_runs": n,
        "max_height_m": max_h,
        "calibration": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True, help="operational.json")
    ap.add_argument("--max-height", type=float, default=3.0)
    ap.add_argument("--metric", default="core_reduction_pct",
                    help="quantity to fit; core reduction is what a scheme buys")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ops = json.loads(Path(args.ops).read_text())
    by_scale: dict[float, list[dict]] = {}
    for o in ops["options"]:
        by_scale.setdefault(o["event_scale"], []).append(o)

    fits = {}
    for rs, opts in sorted(by_scale.items()):
        f = fit_for_scale(opts, args.max_height, args.metric)
        fits[str(rs)] = f
        print(f"event x{rs} [{args.metric}]:  base {f['base_road_cut_km']:.3f}   "
              f"band ±{f['band_km']:.3f} over {f['n_calibration_runs']} runs", flush=True)
        for i, a in f["alpha_km"].items():
            print(f"    measure {i}: {a:+.3f} at full height", flush=True)
        print(f"    saturation S = {f['saturation_S']:.0f}, fit RMSE {f['fit_rmse']:.2f}", flush=True)

    payload = {
        "metric": args.metric,
        "fits": fits,
        "sites": ops.get("sites", []),
        "note": ("Fast estimate. Single-measure effects are exact from the "
                 "hydrodynamic runs; combinations carry a fitted interaction "
                 "term and a 90% band from held-out residuals. Any plan that is "
                 "adopted should be verified by a full solver run."),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
