"""The trigger family's whole frontier, not three points on it.

Why this exists: every "best feasible baseline" in this project's evacuation
write-up was the best trigger *at a budget of 32*, which the trigger clears
with slack, while the learned policies were compared against it while
spending half as much disruption. That is not a comparison. On a constrained
problem the baseline is a CURVE -- return as a function of what you are
willing to spend -- and a learned policy beats it only if it sits above that
curve at its OWN cost.

So: sweep the family finely, and report the trigger's return at the cost each
learned arm actually used.

The curve turns out to be a step function with three Pareto points, which is
itself the finding. Eleven consecutive thresholds return the same number to
one decimal while their cost varies by 1.6x. One threshold on one global
signal selects which SLOT the order fires at, and a 48-step episode has only
a handful of slots where that changes anything -- so a one-degree-of-freedom
controller on an eight-zone problem expresses three behaviours.

    python3 -m baselines.run_evac_frontier
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import numpy as np
import torch

from baselines.evac import table as T
from pspe.envs.evacuate import EvacuateTask, FloodEvacuateEnv

#: The learned arms to place against the curve, as (name, return, cost).
#: Cost must be `episode_cost` -- the final policy's -- and not
#: `eval/cost_mean_over_run`, which averages over training and describes a
#: different policy from the one `return` describes.
LEARNED_DEFAULT: list[tuple[str, float, float]] = []


def score_fixed(env, th: float, seeds=(0, 1, 2, 3), episodes: int = 64):
    rs, cs = [], []
    for sd in seeds:
        s = env.reset(episodes, generator=torch.Generator().manual_seed(sd))
        R = torch.zeros(episodes)
        C = torch.zeros(episodes)
        while True:
            s, r, c, d = env.step(env.trigger_action(s, th))
            R += r
            C += c
            if d:
                break
        rs.append(float(R.mean()))
        cs.append(float(C.mean()))
    return st.fmean(rs), st.fmean(cs), st.pstdev(rs)


def pareto(curve: list[tuple[float, float, float]]):
    """The thresholds nothing else beats on both axes."""
    best: list[tuple[float, float, float]] = []
    for th, r, c in sorted(curve, key=lambda row: row[2]):
        if not best or r > best[-1][1]:
            best.append((th, r, c))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=32.0,
                    help="only sets the env's cost_limit; the sweep reports "
                         "cost rather than filtering on it")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--learned", type=str, default="",
                    help="JSON [[name, return, episode_cost], ...] to place "
                         "against the curve")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    tab = T.load()
    env = FloodEvacuateEnv(tab, EvacuateTask(cost_limit=args.budget))
    print(f"egress table: {tab.members} members x {tab.zones} zones x "
          f"{tab.slots} slots, lead {T.LEAD * T.INTERVAL:.0f} min\n")

    print(f"{'thresh':>8} {'return':>10} {'cost':>8} {'sd':>8}")
    curve = []
    for th in np.arange(0.30, 0.96, 0.025):
        r, c, sd = score_fixed(env, float(th), episodes=args.episodes)
        curve.append((float(th), r, c))
        print(f"{th:>8.3f} {r:>10.1f} {c:>8.2f} {sd:>8.1f}")

    print("\nPareto points (nothing is both cheaper and better):")
    for th, r, c in pareto(curve):
        print(f"  trigger {th:.3f}  {r:>10.1f} at cost {c:>6.2f}")

    learned = json.loads(args.learned) if args.learned else LEARNED_DEFAULT
    if learned:
        print("\nbest trigger affordable at each learned arm's own cost:")
    for name, lr, lc in learned:
        afford = [(r, th, c) for th, r, c in curve if c <= lc]
        if not afford:
            print(f"  {name:<18} cost {lc:5.1f}: no trigger is that cheap")
            continue
        br, bth, bc = max(afford)
        print(f"  {name:<18} at cost {lc:5.1f}: trigger {bth:.3f} scores "
              f"{br:>9.1f} (cost {bc:5.2f})  ->  learned {lr:>9.1f}, "
              f"delta {lr - br:+9.1f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"curve": curve, "pareto": pareto(curve)}, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
