#!/usr/bin/env python
"""Paired learned-vs-incumbent comparison on the catalog gym.

`scripts/score_gym.py` reports a mean and a standard deviation per rung, and
an unpaired comparison of two such rungs is far weaker than the design
allows: every rung is scored on the SAME twelve evaluation scenarios, so the
scenario is a blocking factor and the right test is paired. On
`FloodOperateFine-v0` the per-episode spread is about 21 while the
learned-minus-incumbent gap is about 0.5 -- unpaired that gap is a tenth of a
standard error and unreportable either way, which says nothing about whether
it is real.

So this re-rolls the saved checkpoints and the incumbent over the same seeds,
keeps the PER-EPISODE arrays `score()` throws away, and reports the paired
difference with a t statistic on the twelve differences. Costs come back too,
because a return comparison between a feasible and an infeasible row is not a
comparison.

    scripts/paired_gym.py                      # every environment with runs
    scripts/paired_gym.py --env CMDPToy-v0
    scripts/paired_gym.py --out runs/gym/paired.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_gym import FEASIBLE_TOL, INCUMBENT, roll              # noqa: E402

#: `score_gym.py` takes `--seeds 12` and rolls `range(12)`; the catalog's
#: recorded numbers and every learned row in `runs/gym` were measured on that
#: set. Pairing is only valid against the same set, so this is the default
#: here too and the flag exists to change both halves at once or not at all.
DEFAULT_SEEDS = 12


def episodes(env, policy, seeds):
    """Per-episode returns and costs -- the arrays `score()` reduces away."""
    rets, costs = [], []
    for s in seeds:
        r, c, _limit, _steps = roll(env, policy, s)
        rets.append(float(r))
        costs.append(float(c))
    return rets, costs


def paired_t(a, b):
    """Paired t on a - b. Returns (mean difference, t, n).

    t is None when the differences are identical to floating point, which is
    not a degenerate edge case here: `FloodDefend-v0` returns the same number
    for every policy that spends little, so the zero-variance case is a real
    answer and must not raise.
    """
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return m, None, n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var <= 0.0:
        return m, None, n
    return m, m / math.sqrt(var / n), n


def main() -> int:
    import gymnasium as gym
    import torch

    from pspe.envs import register_all
    from pspe.plan.gym_ppo import load_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/gym")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--out", default="runs/gym/paired.json")
    args = ap.parse_args()

    register_all()
    eval_seeds = tuple(range(args.seeds))
    root = Path(args.root)
    env_ids = args.env or sorted(
        {p.parent.parent.name for p in root.glob("*/seed*/policy.pt")})

    out: "dict[str, dict]" = {}
    for env_id in env_ids:
        env = gym.make(env_id, split=args.split)
        inner = env.unwrapped
        limit = float(inner.task.cost_limit)
        slack = abs(limit) * FEASIBLE_TOL
        inc_name = INCUMBENT[env_id]
        inc_r, inc_c = episodes(env, getattr(inner, inc_name), eval_seeds)
        rec = {"limit": limit, "incumbent": inc_name,
               "incumbent_returns": inc_r, "incumbent_costs": inc_c,
               "incumbent_feasible": sum(c <= limit + slack for c in inc_c),
               "seeds": {}}
        for ckpt in sorted(root.glob(f"{env_id}/seed*/policy.pt")):
            sd = ckpt.parent.name.replace("seed", "")
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            pol = load_policy(state, env.action_space.shape)
            r, c = episodes(env, pol, eval_seeds)
            m, t, n = paired_t(r, inc_r)
            rec["seeds"][sd] = {
                "returns": r, "costs": c,
                "feasible": sum(x <= limit + slack for x in c), "of": len(c),
                "paired_diff": m, "t": t, "n": n}
            flag = "" if rec["seeds"][sd]["feasible"] == len(c) else "  INFEASIBLE"
            ts = "   n/a" if t is None else f"{t:+6.2f}"
            print(f"{env_id:<22} s{sd}  diff {m:+10.4f}  t {ts}"
                  f"  feas {rec['seeds'][sd]['feasible']}/{len(c)}{flag}",
                  flush=True)
        env.close()
        out[env_id] = rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
