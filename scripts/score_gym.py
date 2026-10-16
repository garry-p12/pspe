"""Score every registered environment against its incumbent, and against the
numbers `pspe/envs/catalog.py` claims.

Two jobs. Without `--check` it measures and prints; with `--check` it compares
what it measured against the `scores` recorded in the catalog and **exits
nonzero on drift**, so the one file that claims to be authoritative about this
gym cannot quietly stop being true.

The thing it reports that a reward column alone does not: `feasible` is the
number of episodes whose SUMMED cost cleared the limit. Per episode, never a
mean over episodes -- `safe_rl.py` winds its multiplier on
`episode_cost - cost_limit`, so the episode sum is the constrained quantity,
and a mean over a bimodal cost distribution describes no episode that exists.

    python3 scripts/score_gym.py                 # measure and print
    python3 scripts/score_gym.py --check         # fail if the catalog drifted
    python3 scripts/score_gym.py --env Firebreak-v0 --seeds 32
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The incumbent each environment exposes. A learned policy that does not beat
#: the named method here has not shown anything.
INCUMBENT = {
    "FloodOperate-v0": "baseline_action",
    "FloodOperateFine-v0": "baseline_action",
    "FloodEvacuate-v0": "trigger_action",
    "Firebreak-v0": "uniform_action",
}

#: How far a measured number may move before `--check` calls it drift. Returns
#: are O(10) to O(1000) across these environments, so the return tolerance is
#: relative; costs are compared on an absolute scale because the quantity that
#: matters about a cost is its distance from the limit.
RET_TOL = 0.02      # 2% of the recorded magnitude
COST_TOL = 0.01     # absolute, in the cost's own unit


def roll(env, policy, seed: int, cap: int = 256):
    """One episode. Returns (summed reward, summed cost, limit, steps).

    Both seeds, deliberately. `reset(seed=)` seeds the environment's own
    stream; `action_space.seed()` seeds the one `action_space.sample()` draws
    from, and Gymnasium keeps them separate on purpose. Seeding only the first
    leaves the random arm unreproducible -- which is how the catalog's random
    numbers drifted by up to 8% between two runs of this script.
    """
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    total_r = total_c = 0.0
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total_r += reward
        total_c += info["cost"]
        steps += 1
        if terminated or truncated or steps >= cap:
            break
    return total_r, total_c, info["cost_limit"], steps


def score(env, policy, seeds, limit) -> "dict[str, float]":
    import numpy as np

    rolls = [roll(env, policy, s) for s in seeds]
    ret = np.array([r[0] for r in rolls])
    cost = np.array([r[1] for r in rolls])
    return {"return": float(ret.mean()), "return_sd": float(ret.std()),
            "cost": float(cost.mean()), "cost_sd": float(cost.std()),
            "feasible": int((cost <= limit).sum()), "of": len(seeds),
            "steps": rolls[0][3]}


def measure(env_ids, seeds) -> "dict[str, dict]":
    import gymnasium as gym

    from pspe.envs import register_all

    register_all()
    out: "dict[str, dict]" = {}
    for env_id in env_ids:
        name = INCUMBENT[env_id]
        env = gym.make(env_id)
        inner = env.unwrapped
        fn = getattr(inner, name)
        # `uniform_action` needs no observation: the budget is the whole input.
        incumbent = ((lambda _obs, fn=fn: fn()) if name == "uniform_action"
                     else (lambda obs, fn=fn: fn(obs)))
        limit = float(inner.task.cost_limit)
        out[env_id] = {
            "incumbent": name, "limit": limit,
            "random": score(env, lambda _o: env.action_space.sample(), seeds, limit),
            "incumbent_scores": score(env, incumbent, seeds, limit),
        }
        env.close()
    return out


def report(results) -> None:
    for env_id, row in results.items():
        r, i = row["random"], row["incumbent_scores"]
        print(f"\n{env_id}   {r['steps']} steps, limit {row['limit']}")
        for label, s in (("random", r), (row["incumbent"], i)):
            print(f"  {label:<16} return {s['return']:+10.2f} +- {s['return_sd']:7.2f}"
                  f"   cost {s['cost']:8.3f}   feasible {s['feasible']:3d}/{s['of']}")
        if i["return"] < r["return"] and i["feasible"] > r["feasible"]:
            print("  -> reward alone ranks RANDOM first; it is infeasible")


def check(results) -> int:
    """Compare against the catalog. Nonzero exit on drift."""
    from pspe.envs import BY_ID

    bad = 0
    for env_id, row in results.items():
        recorded = BY_ID[env_id].scores
        if not recorded:
            print(f"FAIL {env_id}: catalog records no scores")
            bad += 1
            continue
        for key, measured in (("random", row["random"]),
                              ("incumbent", row["incumbent_scores"])):
            want = recorded[key]
            dr = abs(measured["return"] - want["return"])
            if dr > RET_TOL * max(abs(want["return"]), 1e-9):
                print(f"FAIL {env_id} {key}: return {measured['return']:.2f} vs "
                      f"catalog {want['return']:.2f}")
                bad += 1
            if abs(measured["cost"] - want["cost"]) > COST_TOL:
                print(f"FAIL {env_id} {key}: cost {measured['cost']:.3f} vs "
                      f"catalog {want['cost']:.3f}")
                bad += 1
            if measured["feasible"] != want["feasible"]:
                print(f"FAIL {env_id} {key}: feasible {measured['feasible']}/"
                      f"{measured['of']} vs catalog {want['feasible']}/{want['of']}")
                bad += 1
    print(f"\n{'catalog matches the environments' if not bad else f'{bad} mismatch(es)'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", action="append", default=[],
                    help="environment id; repeatable. Default: all four.")
    ap.add_argument("--seeds", type=int, default=12,
                    help="episodes per policy, seeded 0..n-1. The catalog's "
                         "numbers are 12; --check only means anything at 12.")
    ap.add_argument("--check", action="store_true",
                    help="compare against the catalog and exit nonzero on drift")
    ap.add_argument("--json", type=str, default="",
                    help="also write the measurements here")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    env_ids = args.env or list(INCUMBENT)
    unknown = [e for e in env_ids if e not in INCUMBENT]
    if unknown:
        print(f"unknown environment(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.check and args.seeds != 12:
        print(f"--check compares against numbers measured on 12 seeds; "
              f"--seeds {args.seeds} would compare different quantities.",
              file=sys.stderr)
        return 2

    results = measure(env_ids, list(range(args.seeds)))
    report(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json}")
    return check(results) if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
