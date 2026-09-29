#!/usr/bin/env python
"""Re-report the headline numbers as IQM with bootstrap intervals.

    python eval/run_bootstrap_report.py

Section 9.11 cites [Agarwal et al., 2021] against exactly the reporting this
project used for most of its life: a point estimate, sometimes a standard
deviation, over a handful of seeds. Section 8.4 item 9 is the remedy. The
machinery already exists in eval/metrics.py -- `iqm`, `bootstrap_ci`,
`seed_report` -- so what was missing is applying it to the claims that appear
in the abstract.

Resampling is over SEEDS, because that is the unit of independence: two
episodes from one training run share a policy. With five seeds the intervals
are wide. That is the honest width, not a defect of the estimator, and a claim
whose interval straddles zero is reported as straddling zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import bootstrap_ci, iqm, markdown_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The arms behind the section 5.4 table. Same tree layout on both families.
TREES = {
    "dar": "runs/constraint_fix_conf",
    "rdf": "runs/constraint_fix_conf_rdf",
}
ARMS = ["baseline", "margin", "probe", "probe+margin"]

# The section 5.4 table reports `eval/violating_eval_fraction` -- the fraction
# of periodic evaluations during the run whose episode cost exceeded the limit
# -- NOT the final evaluation's `violation_rate`. They are different numbers
# (rdf baseline: 14.5% against 17.5%) and only the first reproduces the paper,
# so it is the primary here and the other is carried alongside.
PRIMARY = "eval/violating_eval_fraction"
SECONDARY = "violation_rate"


def per_seed(tree: Path, arm: str, field: str) -> list[float]:
    out = []
    for seed_dir in sorted(tree.glob("seed_*")):
        f = seed_dir / arm / "summary.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if field in d and d[field] is not None:
            out.append(float(d[field]))
    return out


def summarise(values: list[float], digits: int, pct: bool) -> dict:
    if not values:
        return {"n": 0, "iqm": None, "ci_lo": None, "ci_hi": None, "mean": None}
    scale = 100.0 if pct else 1.0
    v = [x * scale for x in values]
    lo, hi = bootstrap_ci(v) if len(v) > 1 else (float("nan"), float("nan"))
    return {
        "n": len(v),
        "iqm": round(iqm(v), digits),
        "ci_lo": round(lo, digits),
        "ci_hi": round(hi, digits),
        "mean": round(float(np.mean(v)), digits),
        "sd": round(float(np.std(v, ddof=1)), digits) if len(v) > 1 else None,
    }


def paired_delta(a: list[float], b: list[float], digits: int, pct: bool) -> dict:
    """Seed-paired difference b - a, bootstrapped on the paired differences.

    Pairing matters: seed 3 is hard for every arm, so the unpaired interval is
    dominated by across-seed spread that cancels within a seed.
    """
    n = min(len(a), len(b))
    if n < 2:
        return {"n": n, "iqm": None, "ci_lo": None, "ci_hi": None, "crosses_zero": None}
    scale = 100.0 if pct else 1.0
    d = [(b[i] - a[i]) * scale for i in range(n)]
    lo, hi = bootstrap_ci(d)
    return {
        "n": n,
        "iqm": round(iqm(d), digits),
        "ci_lo": round(lo, digits),
        "ci_hi": round(hi, digits),
        "wins": f"{sum(1 for x in d if x < 0)}/{n}",
        "losses": f"{sum(1 for x in d if x > 0)}/{n}",
        "crosses_zero": int(lo <= 0.0 <= hi),
    }


# ---------------------------------------------------------------------------
# The horizon ablation (section 5.1). This is the claim section 9.6 calls
# "novel as evidence" -- null at one day where the theory requires it, growing
# with horizon -- so it is the one that most needs an interval rather than a
# paired t over five seeds.
HORIZONS = {1: "runs/ndws_horiz_h1", 2: "runs/ndws_horiz_h2",
            3: "runs/ndws_horiz_h3", 5: "runs/ndws_horiz_h5"}
FIELD = "reduction vs none %"


def horizon_pairs(tree: Path, a: str = "greedy", b: str = "pspe per-instance"
                  ) -> tuple[list[float], list[float]]:
    """Per-seed (greedy, pspe) burn reduction. Same seed, same fire, same budget."""
    ga, pb = [], []
    for seed_dir in sorted(tree.glob("seed_*")):
        f = seed_dir / "results.json"
        if not f.exists():
            continue
        rows = {r["policy"]: r[FIELD] for r in json.loads(f.read_text())
                if FIELD in r}
        if a in rows and b in rows:
            ga.append(float(rows[a]))
            pb.append(float(rows[b]))
    return ga, pb


def horizon_report() -> tuple[list[dict], dict]:
    rows, raw = [], {}
    for h, rel in HORIZONS.items():
        tree = ROOT / rel
        if not tree.exists():
            continue
        g, v = horizon_pairs(tree)
        if len(g) < 2:
            continue
        raw[f"h{h}"] = {"greedy": g, "pspe": v}
        d = [v[i] - g[i] for i in range(len(g))]
        lo, hi = bootstrap_ci(d)
        rows.append({
            "horizon (days)": h,
            "greedy (IQM)": round(iqm(g), 1),
            "PSPE (IQM)": round(iqm(v), 1),
            "gap (IQM)": round(iqm(d), 1),
            "gap mean": round(float(np.mean(d)), 1),
            "95% CI": f"[{lo:.1f}, {hi:.1f}]",
            "seeds": len(d),
            "seeds PSPE wins": f"{sum(1 for x in d if x > 0)}/{len(d)}",
            "CI excludes 0": "**yes**" if (lo > 0 or hi < 0) else "no",
        })
    return rows, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bootstrap_report")
    ap.add_argument("--trees", nargs="*", default=None,
                    help="name=path pairs to analyse instead of the defaults, "
                         "e.g. rdf20=runs/constraint_fix_conf_rdf_20")
    ap.add_argument("--skip-horizon", action="store_true",
                    help="the horizon ablation is a separate experiment; skip it "
                         "when only re-analysing a constraint-fix tree")
    args = ap.parse_args()

    trees = TREES
    if args.trees:
        trees = {}
        for spec in args.trees:
            name, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--trees wants name=path, got {spec!r}")
            trees[name] = path
        # A named tree is the whole analysis; the horizon ablation lives in a
        # different run directory and would otherwise be reported beside it as
        # though it came from the same experiment.
        args.skip_horizon = True

    rows, deltas, raw = [], [], {}
    for testbed, rel in trees.items():
        tree = ROOT / rel
        if not tree.exists():
            print(f"skip {testbed}: {rel} missing", file=sys.stderr)
            continue
        for arm in ARMS:
            vals = per_seed(tree, arm, PRIMARY)
            fin = per_seed(tree, arm, SECONDARY)
            ret = per_seed(tree, arm, "return")
            raw[f"{testbed}/{arm}"] = {PRIMARY: vals, SECONDARY: fin, "return": ret}
            if not vals:
                continue
            s = summarise(vals, 1, pct=True)
            r = summarise(ret, 3, pct=False)
            rows.append({
                "testbed": testbed, "arm": arm, "seeds": s["n"],
                "violating % (mean)": s["mean"],
                "violating % (IQM)": s["iqm"],
                "95% CI": f"[{s['ci_lo']}, {s['ci_hi']}]",
                "final-eval %": summarise(fin, 1, pct=True)["mean"],
                "return (IQM)": r["iqm"],
                "return 95% CI": f"[{r['ci_lo']}, {r['ci_hi']}]",
            })
        base = per_seed(tree, "baseline", PRIMARY)
        for arm in ARMS[1:]:
            arm_v = per_seed(tree, arm, PRIMARY)
            d = paired_delta(base, arm_v, 1, pct=True)
            if d["iqm"] is None:
                continue
            deltas.append({
                "testbed": testbed, "comparison": f"{arm} - baseline",
                "seeds": d["n"], "Δ violating pts (IQM)": d["iqm"],
                "95% CI": f"[{d['ci_lo']}, {d['ci_hi']}]",
                "seeds improved": d["wins"],
                "seeds worse": d["losses"],
                "CI crosses 0": "yes" if d["crosses_zero"] else "**no**",
            })

    hrows, hraw = ([], {}) if args.skip_horizon else horizon_report()
    if hraw:
        raw["horizon"] = hraw

    print("## Violation rate and return, IQM with 95% bootstrap CI over seeds\n")
    print(markdown_table(rows))
    print("\n## Seed-paired differences against the baseline arm\n")
    print(markdown_table(deltas))

    if hrows:
        print("\n## Horizon ablation: PSPE - greedy burn reduction, seed-paired\n")
        print(markdown_table(hrows))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"rows": rows, "deltas": deltas, "horizon": hrows,
         "raw_per_seed": raw}, indent=2))
    (out / "results.md").write_text(
        "## Violation rate and return, IQM with 95% bootstrap CI over seeds\n\n"
        + markdown_table(rows)
        + "\n\n## Seed-paired differences against the baseline arm\n\n"
        + markdown_table(deltas)
        + ("\n\n## Horizon ablation: PSPE - greedy burn reduction, seed-paired\n\n"
           + markdown_table(hrows) if hrows else "") + "\n")
    print(f"\nwrote {out}/results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
