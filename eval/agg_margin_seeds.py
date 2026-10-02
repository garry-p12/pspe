"""Pool a margin comparison across seeds, with the feasibility split Rule 45 requires.

A coverage rate may only be pooled over configurations in which a plan exists.
Where the margin consumes the whole limit there is no admissible action, so the
realised violation measures the task's feasibility and not the method's
calibration -- and averaging the two moved swe's headline by four points
(defect 39). This script therefore always reports three pools and never just the
mean.

    python3 eval/agg_margin_seeds.py runs/constraint_fix_swe20
    python3 eval/agg_margin_seeds.py runs/constraint_fix_swe20 --evals 11
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

ARMS = ["baseline", "margin", "probe", "probe+margin"]


def load(root: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(f"{root}/seed_*"),
                    key=lambda x: int(x.rsplit("_", 1)[-1])):
        f = Path(d) / "results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows = r if isinstance(r, list) else r.get("rows", r.get("arms", []))
        got = {x.get("arm"): x for x in rows if x.get("arm") in ARMS}
        if len(got) < 2:
            continue
        got["_seed"] = int(d.rsplit("_", 1)[-1])
        out.append(got)
    return out


def ci(p: float, n: int) -> tuple[float, float]:
    se = math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0
    return p - 1.96 * se, p + 1.96 * se


def infer_evals(seeds: list[dict]) -> int:
    """Recover evaluations per seed from the granularity of the fractions.

    Violation fractions are k/n_ev, so the smallest positive value seen is
    1/n_ev. Inferring it beats hard-coding a number that changes with
    --probe-every.
    """
    vals = {float(s[a]["violating_evals"]) for s in seeds for a in ARMS
            if a in s and float(s[a]["violating_evals"]) > 0}
    if not vals:
        return 1
    return int(round(1.0 / min(vals)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding seed_* subdirectories")
    ap.add_argument("--evals", type=int, default=None,
                    help="evaluations per seed; inferred from granularity if omitted")
    ap.add_argument("--arm", default="probe+margin", help="arm whose coverage is pooled")
    ap.add_argument("--vs", default="probe", help="arm to pair against")
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()

    seeds = load(args.root)
    if not seeds:
        raise SystemExit(f"no seed_*/results.json under {args.root}")
    n_ev = args.evals or infer_evals(seeds)
    print(f"{len(seeds)} seeds, {n_ev} evaluations each "
          f"({'given' if args.evals else 'inferred from fraction granularity'})")

    print("\n| arm | violating mean | per-seed spread | return mean |")
    print("|---|---|---|---|")
    for a in ARMS:
        v = [float(s[a]["violating_evals"]) for s in seeds if a in s]
        r = [float(s[a]["return"]) for s in seeds if a in s]
        if v:
            print(f"| {a} | {st.mean(v):.4f} | {min(v):.3f}-{max(v):.3f} | {st.mean(r):.3f} |")

    # Rule 45: pool only where a plan exists.
    def feasible(s: dict) -> bool:
        row = s.get(args.arm)
        if row is None:
            return False
        lim = float(row.get("cost_limit", 0.0))
        eff = float(row.get("effective_limit", lim))
        return not (lim > 1e-9 and eff <= 1e-9)

    pools = {"all seeds": seeds,
             "feasible (a plan exists)": [s for s in seeds if feasible(s)],
             "infeasible (margin = limit)": [s for s in seeds if not feasible(s)]}
    print(f"\ncoverage of `{args.arm}` against delta = {args.delta}")
    print("| pool | seeds | violations | rate | 95% CI |")
    print("|---|---|---|---|---|")
    for name, rows in pools.items():
        if not rows:
            print(f"| {name} | 0 | — | — | — |")
            continue
        v = sum(round(float(s[args.arm]["violating_evals"]) * n_ev) for s in rows)
        t = n_ev * len(rows)
        p = v / t
        lo, hi = ci(p, t)
        mark = ""
        if name.startswith("feasible") or name.startswith("all"):
            mark = ("  contains delta" if lo <= args.delta <= hi
                    else "  EXCLUDES delta")
        print(f"| {name} | {len(rows)} | {v}/{t} | {100*p:.2f}% | "
              f"[{100*lo:.2f}, {100*hi:.2f}]{mark} |")

    both = [s for s in seeds if args.arm in s and args.vs in s]
    if len(both) > 1:
        d = [float(s[args.arm]["violating_evals"]) - float(s[args.vs]["violating_evals"])
             for s in both]
        se = st.stdev(d) / math.sqrt(len(d)) if st.stdev(d) > 0 else 0.0
        print(f"\npaired {args.arm} - {args.vs} over {len(d)} seeds: "
              f"{st.mean(d):+.4f}"
              + (f", 95% CI [{st.mean(d)-1.96*se:+.4f}, {st.mean(d)+1.96*se:+.4f}]"
                 if se > 0 else "  (no variance across seeds)"))


if __name__ == "__main__":
    main()
