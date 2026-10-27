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

#: The strongest FEASIBLE policy known on each environment -- the bar a
#: learned policy has to clear to have shown anything. These are not the
#: baselines each environment originally named; three of those were weak
#: enough that a reward-only table ranked a random policy above them, which
#: reads as a result and is an artefact. `pspe/envs/catalog.py` carries the
#: full ladder, the displaced baselines included.
#:
#: `FloodEvacuate-v0`'s entry is doing nothing, because on that environment
#: nothing known is both feasible and better than doing nothing.
INCUMBENT = {
    "FloodOperate-v0": "budget_edge_action",
    "FloodOperateFine-v0": "budget_edge_action",
    "FloodEvacuate-v0": "never_action",
    "Firebreak-v0": "greedy_action",
    "FloodDefend-v0": "uniform_action",
    'CMDPToy-v0': 'optimal_action_vector',
    'PDEControl-dar-v0': 'proportional_action',
    'PDEControl-swe-v0': 'proportional_action',
    'PDEControl-rdf-v0': 'proportional_action',
    'FireDispatch-v0': 'biggest_fire_action',
    # Was `even_action`, which its own ladder row beat: `lowest_first_action`
    # returns -0.519 to even's -0.572 and does it on 8.7 of the 17.8 houses.
    # An incumbent that something else in the same table dominates sets the
    # bar below the strongest feasible policy known, which is the one thing
    # this constant exists to avoid.
    'FloodBuyout-v0': 'lowest_first_action',
    'FloodCrossing-v0': 'budgeted_react_action',
    'FloodGreenInfra-v0': 'most_paved_action',
    'HeatRetrofit-v0': 'exposure_first_action',
}

#: Every policy in the catalog's ladder, keyed the same way, so `--ladder
#: --check` re-measures the whole table rather than taking it on trust. A
#: value is either the name of a method on the environment, a float meaning
#: "hold this on every action dimension", an explicit per-dimension action, or
#: a (method, kwargs) pair.
LADDER_POLICIES = {
    "FloodOperate-v0": {
        "nothing": -1.0,
        "gate_only": [1.0, -1.0],
        "deadband": "baseline_action",
        "budget_edge": "budget_edge_action",
        "sealed_aware": "sealed_aware_action",
        "flat_out": 1.0,
    },
    # `trigger_*` is kept at the thresholds the whole-pool sweep chose, and
    # the low ones are new. The three original rungs CANNOT FIRE on the eval
    # split: its signal never exceeds 0.4781, so 0.55, 0.70 and 0.90 all
    # score exactly `never` at a cost of exactly zero, and four of the six
    # rungs in this table were the same policy wearing different names. The
    # thresholds were tuned on the whole pool before the split existed and
    # were never re-tuned after it; `trigger_33`/`_40`/`_45` bracket the range
    # the eval scenarios actually reach.
    #
    # `jit` is the rung the family was missing. Every `trigger_*` reads one
    # GLOBAL number and orders every zone at once, which is all the old
    # four-channel observation supported. With per-zone water and egress now
    # in channels 4 and 5, `jit_action` orders zone by zone as each one wets,
    # and on the train split it returns -1863 feasible 12/12 against `never`'s
    # -4206 -- a 2343 improvement that no member of the threshold family can
    # reach at any setting.
    "FloodEvacuate-v0": {
        "never": "never_action",
        "trigger_90": ("trigger_action", {"threshold": 0.90}),
        "trigger_70": ("trigger_action", {"threshold": 0.70}),
        "trigger_55": ("trigger_action", {"threshold": 0.55}),
        "trigger_45": ("trigger_action", {"threshold": 0.45}),
        "trigger_40": ("trigger_action", {"threshold": 0.40}),
        "trigger_33": ("trigger_action", {"threshold": 0.33}),
        "jit": ("jit_action", {"water": 0.02}),
        "jit_early": ("jit_action", {"water": 0.01}),
        "always": 1.0,
    },
    "FloodDefend-v0": {
        "nothing": -1.0,
        "uniform": "uniform_action",
        "lowest_first": "lowest_first_action",
        "flat_out": 1.0,
    },
    "Firebreak-v0": {
        "nothing": -1.0,
        "uniform": "uniform_action",
        "greedy": "greedy_action",
        "flat_out": 1.0,
    },
}
# The two flood presets share a ladder: same policies, same action layout.
LADDER_POLICIES["FloodOperateFine-v0"] = LADDER_POLICIES["FloodOperate-v0"]

#: The families added later.
LADDER_POLICIES.update({
    'CMDPToy-v0': {'nothing': 'zero_action', 'optimal': 'optimal_action_vector'},
    'PDEControl-dar-v0': {'nothing': 'zero_action', 'proportional': 'proportional_action'},
    'PDEControl-swe-v0': {'nothing': 'zero_action', 'proportional': 'proportional_action'},
    'PDEControl-rdf-v0': {'nothing': 'zero_action', 'proportional': 'proportional_action'},
    'FireDispatch-v0': {'nothing': -1.0, 'even': 'even_action', 'biggest': 'biggest_fire_action', 'most_people': 'most_people_action'},
    'FloodBuyout-v0': {'nothing': -1.0, 'even': 'even_action', 'lowest_first': 'lowest_first_action', 'flat_out': 1.0},
    "HeatRetrofit-v0": {"nothing": -1.0, "even": "even_action",
                        "hottest": "hottest_action",
                        "people": "most_people_action",
                        "both": "exposure_first_action", "flat_out": 1.0},
    "FloodGreenInfra-v0": {"nothing": -1.0, "even": "even_action",
                           "uphill": "highest_first_action",
                           "paved": "most_paved_action", "flat_out": 1.0},
    "FloodCrossing-v0": {"never": "never_close_action",
                         "budgeted": "budgeted_react_action",
                         "ration": "ration_action",
                         "react": "react_action",
                         "always": "always_close_action"},
})


def _policy(env, inner, spec):
    """Turn a ladder entry into a callable taking an observation."""
    import numpy as np

    if isinstance(spec, str):
        return getattr(inner, spec)
    if isinstance(spec, tuple):
        name, kwargs = spec
        fn = getattr(inner, name)
        return lambda obs, fn=fn, kwargs=kwargs: fn(obs, **kwargs)
    if isinstance(spec, (int, float)):
        const = np.full(env.action_space.shape, float(spec), dtype=np.float32)
    else:
        const = np.asarray(spec, dtype=np.float32)
        if const.shape != env.action_space.shape:
            # A flood ladder entry is [gate, pump]; an environment with more
            # than one pump needs the pump setting repeated, not broadcast by
            # accident into the wrong slot.
            const = np.concatenate(
                [const[:1], np.full(env.action_space.shape[0] - 1, const[-1],
                                    dtype=np.float32)])
    return lambda _obs, c=const: c


#: How far a measured number may move before `--check` calls it drift. Returns
#: are O(10) to O(1000) across these environments, so the return tolerance is
#: relative; costs are compared on an absolute scale because the quantity that
#: matters about a cost is its distance from the limit.
RET_TOL = 0.02       # 2% of the recorded magnitude
#: Costs are compared RELATIVE TO THE BUDGET, because that is the only scale
#: on which a spend means anything. An absolute tolerance cannot serve both a
#: 2.0 pump-hour limit and a $261,755 one: 0.01 is a fifth of a per cent of
#: the first and a rounding artefact of the second, and it failed every
#: FloodDefend row on differences of eighteen cents.
COST_TOL_FRAC = 0.002
COST_TOL_FLOOR = 1.0e-4


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


#: A violation of one part per million is not a violation.
#:
#: Episode cost is a sum of float32 steps and the limit is one number, so a
#: policy that spends the budget EXACTLY lands a few parts in 1e8 either side
#: of it. `CMDPToy-v0` makes this unavoidable rather than incidental: its
#: optimum sits on the constraint boundary by definition, so nudging the
#: action inside would stop it being the optimum. The tolerance belongs in the
#: comparison.
FEASIBLE_TOL = 1.0e-6


def score(env, policy, seeds, limit) -> "dict[str, float]":
    import numpy as np

    rolls = [roll(env, policy, s) for s in seeds]
    ret = np.array([r[0] for r in rolls])
    cost = np.array([r[1] for r in rolls])
    slack = abs(limit) * FEASIBLE_TOL
    return {"return": float(ret.mean()), "return_sd": float(ret.std()),
            "cost": float(cost.mean()), "cost_sd": float(cost.std()),
            "feasible": int((cost <= limit + slack).sum()), "of": len(seeds),
            "steps": rolls[0][3]}


def learned_rungs(env, env_id: str, seeds, limit, root: str) -> "dict[str, dict]":
    """Score every checkpoint `scripts/train_gym_policy.py` left under `root`.

    One rung per training seed rather than one averaged rung, because a mean
    over three training seeds hides the thing worth knowing about a learned
    policy on a 26-window pool: whether all three runs landed in the same
    place. The rungs are scored by the same `score` as every hand-written rung
    above them, on the same twelve evaluation seeds.
    """
    import torch

    from pspe.plan.gym_ppo import load_policy

    out: "dict[str, dict]" = {}
    for ckpt in sorted(Path(root).glob(f"{env_id}/seed*/policy.pt")):
        seed = ckpt.parent.name.replace("seed", "")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        policy = load_policy(state, env.action_space.shape)
        out[f"learned(s{seed})"] = score(env, policy, seeds, limit)
    return out


def measure(env_ids, seeds, ladder: bool = False,
            split: str = "eval",
            learned_root: str = "") -> "dict[str, dict]":
    import numpy as np
    import gymnasium as gym

    from pspe.envs import register_all

    register_all()
    out: "dict[str, dict]" = {}
    for env_id in env_ids:
        name = INCUMBENT[env_id]
        env = gym.make(env_id, split=split)
        inner = env.unwrapped
        # Every baseline takes an observation and returns an action, whether
        # or not it reads the observation, so no special-casing here.
        incumbent = getattr(inner, name)
        limit = float(inner.task.cost_limit)
        row = {
            "incumbent": name, "limit": limit,
            "random": score(env, lambda _o: env.action_space.sample(), seeds, limit),
            "incumbent_scores": score(env, incumbent, seeds, limit),
        }
        if ladder:
            rungs = {"random": row["random"]}
            for key, spec in LADDER_POLICIES.get(env_id, {}).items():
                rungs[key] = score(env, _policy(env, inner, spec), seeds, limit)
            row["ladder"] = rungs
        if learned_root:
            row["learned"] = learned_rungs(env, env_id, seeds, limit,
                                           learned_root)
        out[env_id] = row
        env.close()
    return out


def report(results) -> None:
    for env_id, row in results.items():
        r, i = row["random"], row["incumbent_scores"]
        print(f"\n{env_id}   {r['steps']} steps, limit {row['limit']}")
        for label, s in (("random", r), (row["incumbent"], i)):
            print(f"  {label:<16} return {s['return']:+10.2f} +- {s['return_sd']:7.2f}"
                  f"   cost {s['cost']:8.3f}   feasible {s['feasible']:3d}/{s['of']}")
        # The two arms above are also ladder rungs; print each number once.
        already = {("random", round(r["return"], 2)),
                   (row["incumbent"], round(i["return"], 2))}
        rungs = {**row.get("ladder", {}), **row.get("learned", {})}
        for label, s in sorted(rungs.items(),
                               key=lambda kv: kv[1]["return"]):
            if label == "random" or round(s["return"], 2) == round(i["return"], 2) \
                    and round(s["cost"], 4) == round(i["cost"], 4):
                continue
            flag = "" if s["feasible"] == s["of"] else "   INFEASIBLE"
            print(f"  {label:<16} return {s['return']:+10.2f} +- {s['return_sd']:7.2f}"
                  f"   cost {s['cost']:8.3f}   feasible {s['feasible']:3d}/{s['of']}{flag}")
        if i["return"] < r["return"] and i["feasible"] > r["feasible"]:
            print("  -> reward alone ranks RANDOM first; it is infeasible")


def check(results) -> int:
    """Compare against the catalog. Nonzero exit on drift."""
    from pspe.envs import BY_ID

    bad = 0
    for env_id, row in results.items():
        # Costs mean something only against their own budget.
        tol = max(COST_TOL_FRAC * row['limit'], COST_TOL_FLOOR)
        spec = BY_ID[env_id]
        if abs(spec.cost_limit - row["limit"]) > max(
                COST_TOL_FRAC * row["limit"], COST_TOL_FLOOR):
            print(f"FAIL {env_id}: catalog records a budget of "
                  f"{spec.cost_limit} but the environment's is {row['limit']}")
            bad += 1
        recorded = spec.scores
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
            if abs(measured["cost"] - want["cost"]) > tol:
                print(f"FAIL {env_id} {key}: cost {measured['cost']:.3f} vs "
                      f"catalog {want['cost']:.3f}")
                bad += 1
            if measured["feasible"] != want["feasible"]:
                print(f"FAIL {env_id} {key}: feasible {measured['feasible']}/"
                      f"{measured['of']} vs catalog {want['feasible']}/{want['of']}")
                bad += 1

        measured_ladder = row.get("ladder")
        if measured_ladder is None:
            continue
        recorded_rungs = {r["key"]: r for r in BY_ID[env_id].ladder}
        # Every rung the catalog prints has to be one this script can produce,
        # and the other way round. A ladder row with no policy behind it is a
        # number nobody can reproduce.
        for key in sorted(set(recorded_rungs) ^ set(measured_ladder)):
            side = "catalog only" if key in recorded_rungs else "measured only"
            print(f"FAIL {env_id} ladder rung {key!r}: {side}")
            bad += 1
        for key in sorted(set(recorded_rungs) & set(measured_ladder)):
            want, got = recorded_rungs[key], measured_ladder[key]
            if abs(got["return"] - want["return"]) > RET_TOL * max(abs(want["return"]), 1e-9):
                print(f"FAIL {env_id} ladder {key}: return {got['return']:.2f} "
                      f"vs catalog {want['return']:.2f}")
                bad += 1
            if abs(got["cost"] - want["cost"]) > tol:
                print(f"FAIL {env_id} ladder {key}: cost {got['cost']:.4f} "
                      f"vs catalog {want['cost']:.4f}")
                bad += 1
            if got["feasible"] != want["feasible"]:
                print(f"FAIL {env_id} ladder {key}: feasible {got['feasible']}/"
                      f"{got['of']} vs catalog {want['feasible']}/{want['of']}")
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
    ap.add_argument("--split", default="eval", choices=("train", "eval", "all"),
                    help="which half of each scenario pool to measure on. "
                         "`eval` by default, because a number worth reporting "
                         "is one from scenarios nothing was fitted to, and "
                         "that is what the catalog records.")
    ap.add_argument("--check", action="store_true",
                    help="compare against the catalog and exit nonzero on drift")
    ap.add_argument("--ladder", action="store_true",
                    help="also score every policy in the catalog's ladder. "
                         "With --check this verifies the whole table, which is "
                         "where three of this gym's four findings came from.")
    ap.add_argument("--learned", type=str, default="",
                    help="also score the trained checkpoints under this "
                         "directory (see scripts/train_gym_policy.py), one "
                         "rung per training seed. Cannot be combined with "
                         "--check: a learned row is not in the catalog, and "
                         "silently exempting it from the drift check would "
                         "make the check mean less than it does.")
    ap.add_argument("--json", type=str, default="",
                    help="also write the measurements here")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    env_ids = args.env or list(INCUMBENT)
    unknown = [e for e in env_ids if e not in INCUMBENT]
    if unknown:
        print(f"unknown environment(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.check and args.split != "eval":
        print("--check compares against numbers measured on the eval split; "
              f"--split {args.split} would compare different quantities.",
              file=sys.stderr)
        return 2
    if args.check and args.learned:
        print("--learned scores policies the catalog does not record; run it "
              "without --check.", file=sys.stderr)
        return 2
    if args.check and args.seeds != 12:
        print(f"--check compares against numbers measured on 12 seeds; "
              f"--seeds {args.seeds} would compare different quantities.",
              file=sys.stderr)
        return 2

    results = measure(env_ids, list(range(args.seeds)),
                      ladder=args.ladder, split=args.split,
                      learned_root=args.learned)
    report(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json}")
    return check(results) if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
