#!/usr/bin/env python
"""How many eval scenarios can actually tell two policies apart.

`of: 12` counts seeds. It does not count information, and on three of these
fourteen environments the difference is most of the number:

* `FloodBuyout-v0`'s incumbent returns **0.0 on eleven of twelve** eval
  scenarios. There is no harm left to prevent on those eleven, so nothing a
  policy does there can show up in the score.
* `FloodEvacuate-v0` has six scenarios with no flood at all.
* `FloodCrossing-v0`'s returns take three distinct values.

A scenario DISCRIMINATES when the span between the worst and the best policy
on it is non-trivial: if `nothing` and the best feasible rung give the same
number, that scenario is a constant and contributes nothing but a smaller
standard error. This measures the span per scenario over the whole ladder, so
the count can be reported beside every result instead of `of: 12` implying
twelve independent observations.

The span is taken over policies that are FEASIBLE on that scenario. A policy
that buys a better outcome by overspending does not establish that the
scenario can be improved within budget, which is the only question a
constrained benchmark asks.

    scripts/gym_discrimination.py                    # every environment
    scripts/gym_discrimination.py --env FloodBuyout-v0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_gym import (FEASIBLE_TOL, INCUMBENT, LADDER_POLICIES,   # noqa: E402
                       _policy, roll)

#: A scenario counts as discriminating when the feasible policies on it span
#: more than this fraction of the whole environment's feasible span. Relative
#: because returns here run from 0.16 to 3.7e3; 1% is low enough to admit a
#: real but small difference and high enough to reject floating-point noise.
SPAN_FRAC = 0.01


def main() -> int:
    import gymnasium as gym
    from pspe.envs import register_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="runs/gym/discrimination.json")
    a = ap.parse_args()

    register_all()
    env_ids = a.env or list(LADDER_POLICIES)
    seeds = list(range(a.seeds))
    out = {}
    for env_id in env_ids:
        env = gym.make(env_id, split=a.split)
        inner = env.unwrapped
        limit = float(inner.task.cost_limit)
        slack = abs(limit) * FEASIBLE_TOL
        rungs = dict(LADDER_POLICIES[env_id])
        rungs.setdefault("incumbent", INCUMBENT[env_id])
        # per scenario: the returns of the policies that were feasible on it
        feas_by_seed = {s: [] for s in seeds}
        for _name, spec in rungs.items():
            pol = _policy(env, inner, spec)
            for s in seeds:
                r, c, _lim, _st = roll(env, pol, s)
                if c <= limit + slack:
                    feas_by_seed[s].append(float(r))
        env.close()
        spans = {s: (max(v) - min(v) if len(v) > 1 else 0.0)
                 for s, v in feas_by_seed.items()}
        overall = max(spans.values()) if spans else 0.0
        thresh = overall * SPAN_FRAC
        disc = [s for s in seeds if spans[s] > thresh and spans[s] > 0.0]
        out[env_id] = {
            "discriminating": len(disc), "of": len(seeds),
            "seeds": disc, "span_per_seed": {str(k): v for k, v in spans.items()},
            "largest_span": overall, "threshold": thresh,
            "n_rungs": len(rungs),
        }
        print(f"{env_id:<22} discriminating {len(disc):>2}/{len(seeds)}  "
              f"largest span {overall:>10.4g}  rungs {len(rungs)}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
