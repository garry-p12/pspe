#!/usr/bin/env python
"""Per-scenario sign, worst case and 25% tail, learned against the incumbent.

The paired mean test in `scripts/paired_gym.py` answers one question: is the
average difference over the twelve shared scenarios distinguishable from zero.
It is the right question and it is not the only one, because a policy can be
worse on average and better where being worse is expensive.

`FloodEvacuate-v0` is why this script exists. Its learned policy improves the
mean by +996 and does not reach significance, and the reason is not the sample
size: it wins 3 of 12 scenarios and loses 7, and the mean is positive only
because `jit_action` returns -7453.704 on the three it wins while the learner
returns -2198.112. That is a tail-risk result wearing a mean-return
comparison, and reading it as an underpowered mean win -- which this project
did, for an hour -- gets both the finding and the next experiment wrong.

So: three columns the paired test does not report, computed from the same
`paired.json` arrays so the two can never disagree about which episodes they
describe.

  * win/loss/tie, pooled over seeds -- whether the sign is consistent at all.
  * worst case, learned minus incumbent, meaned over seeds. Positive is better.
  * CVaR at 25%, the mean of the worst quarter of scenarios, same convention.

A positive mean with an inconsistent sign is a different claim from a positive
mean with a consistent one, and only this table separates them.

  scripts/gym_tail.py                       # table to stdout
  scripts/gym_tail.py --json runs/gym/tail.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

#: Tail fraction for the CVaR column. A quarter of twelve is three scenarios,
#: which is also how many `FloodEvacuate-v0`'s whole effect lives in.
TAIL_FRACTION = 0.25

TOL = 1e-9


def tail_row(incumbent: list, per_seed: list) -> dict:
    """Sign counts and tail gaps for one environment.

    `per_seed` is one return array per training seed, all scored on the same
    scenarios in the same order as `incumbent` -- which is what `paired.json`
    guarantees and the reason this reads that file rather than re-rolling.
    """
    better = worse = same = 0
    means, worst, cvar = [], [], []
    k = max(1, int(len(incumbent) * TAIL_FRACTION))
    inc_worst = min(incumbent)
    inc_cvar = st.mean(sorted(incumbent)[:k])
    for returns in per_seed:
        diffs = [a - b for a, b in zip(returns, incumbent)]
        better += sum(d > TOL for d in diffs)
        worse += sum(d < -TOL for d in diffs)
        same += sum(abs(d) <= TOL for d in diffs)
        means.append(st.mean(diffs))
        worst.append(min(returns) - inc_worst)
        cvar.append(st.mean(sorted(returns)[:k]) - inc_cvar)
    return {
        "seeds": len(per_seed),
        "scenarios": len(incumbent),
        "better": better, "worse": worse, "same": same,
        "mean_gap": st.mean(means),
        "worst_gap": st.mean(worst),
        "cvar_gap": st.mean(cvar),
        "tail_k": k,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paired", default="runs/gym/paired.json")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    path = Path(args.paired)
    if not path.exists():
        print(f"{path} not found -- run scripts/paired_gym.py first",
              file=sys.stderr)
        return 2
    paired = json.loads(path.read_text())

    rows: dict[str, dict] = {}
    skipped: list[str] = []
    for env, rec in paired.items():
        incumbent = rec.get("incumbent_returns")
        per_seed = [s["returns"] for s in rec.get("seeds", {}).values()
                    if s.get("returns") and not s.get("stale")]
        if not incumbent or not per_seed:
            # A stale or unrolled row has no episodes to compare. Naming it is
            # the honest result; dropping it silently would make this table
            # read as though the catalog were smaller than it is.
            skipped.append(env)
            continue
        rows[env] = tail_row(incumbent, per_seed)

    order = sorted(rows, key=lambda e: -rows[e]["worst_gap"])
    print(f"{'environment':22s} {'win/loss/tie':>14s} {'mean':>12s} "
          f"{'worst':>12s} {'CVaR25':>12s}")
    print("-" * 76)
    for env in order:
        r = rows[env]
        sg = lambda v: ("+" if v > 0 else "") + f"{v:.4g}"  # noqa: E731
        print(f"{env:22s} "
              f"{r['better']:4d}/{r['worse']:3d}/{r['same']:3d}   "
              f"{sg(r['mean_gap']):>12s} {sg(r['worst_gap']):>12s} "
              f"{sg(r['cvar_gap']):>12s}")
    print("-" * 76)
    wins = sum(r["worst_gap"] > TOL for r in rows.values())
    none = sum(r["better"] == 0 for r in rows.values())
    print(f"{len(rows)} environments; {wins} improve their worst case; "
          f"{none} win on no scenario at all")
    if skipped:
        print(f"not compared (no paired episodes): {', '.join(sorted(skipped))}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
