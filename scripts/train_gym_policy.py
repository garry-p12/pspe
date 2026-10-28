"""Train a constrained policy on one catalog environment, then score it the way
the catalog scores everything else.

`scripts/score_gym.py` measures a ladder of hand-written policies and calls the
strongest feasible one the *incumbent* -- "the bar a learned policy has to clear
to have shown anything". Nothing in this repository had ever actually put a
learned policy against that bar on these fourteen environments, so the sentence
was a promise rather than a measurement. This closes it.

## The protocol, which is the whole point

**Fit on `train`, report on `eval`.** Training calls `gym.make(env_id,
split="train")`; the reported number comes from `gym.make(env_id,
split="eval")`. `pspe/envs/splits.py` fixes that partition under a constant
seed, so an eval number here is a number about scenarios the policy was never
fitted to. Two pools are too thin to partition and say so in `SPLIT_NOTES`;
their rows inherit that caveat and this script repeats it.

**Scored by `score_gym.score`, not by a second evaluator.** The learned row
goes through the same `roll` -- same twelve seeds, same episode-sum cost, same
feasibility tolerance -- as `nothing`, `uniform` and the incumbent. A separate
eval loop is a separate definition of the quantity.

**The deterministic policy is what gets scored.** A stochastic policy's cost is
a distribution around whatever the multiplier settled on, and the thing a city
would deploy is the mean action. Scoring the sampler would report a policy
nobody would run.

**Checkpoint selection never sees `eval`.** Every `--eval-every` iterations the
deterministic policy is scored on eight *training* episodes; the checkpoint
kept is the feasible one with the highest training return, or, if no probe was
ever feasible, the one with the lowest training cost -- flagged as such in the
output. Selecting on the eval split would make the eval number a training
number wearing a different name.

**One hyperparameter setting for all fourteen.** See `GymPPOConfig`. A
per-environment sweep would turn "a learned policy clears the bar" into a
statement about the sweep.

    python3 scripts/train_gym_policy.py --env CMDPToy-v0 --seed 0
    python3 scripts/train_gym_policy.py --report          # aggregate what exists
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

#: Environment steps of experience per seed. Equal across the catalog, because
#: sample budget is the currency a constrained learner is judged in, with one
#: declared exception: `FloodGreenInfra-v0` runs its catchment hydrology at
#: 0.79 s per step -- 20k steps is four and a half hours per seed on this
#: machine. It gets 6k, and the table says so rather than hiding an unequal
#: budget behind a single column header.
DEFAULT_STEPS = 20_000
STEP_BUDGET = {"FloodGreenInfra-v0": 6_000}

#: The eval protocol the catalog already uses. Not a flag: a learned row
#: measured on a different number of seeds is not comparable to the rungs above
#: and below it.
EVAL_SEEDS = tuple(range(12))

#: Relative tolerance when comparing a run's saved `cost_limit` to the live
#: one. A saved limit is a float32 round-trip of the live float64, so an exact
#: comparison reports a changed budget where none changed: `HeatRetrofit-v0`
#: stores 1480464.5 against a live 1480464.375, which is 8.4e-8 apart and the
#: same budget. The same 1e-6 relative tolerance feasibility uses.
LIMIT_TOL = 1e-6

#: Training-split probe episodes for checkpoint selection. Deliberately few --
#: it is a selection signal, not a reported number.
PROBE_SEEDS = tuple(range(90_000, 90_008))


def out_dir(env_id: str, seed: int, root: str) -> Path:
    return Path(root) / env_id / f"seed{seed}"


def train_one(env_id: str, seed: int, steps: int, root: str,
              eval_every: int, quiet: bool = False) -> dict:
    import numpy as np
    import gymnasium as gym
    import torch

    from pspe.envs import register_all
    from pspe.plan.gym_ppo import GymPPOConfig, GymPPOLagrangian, load_policy
    from score_gym import INCUMBENT, score

    torch.set_num_threads(1)
    register_all()

    env = gym.make(env_id, split="train")
    cfg = GymPPOConfig(total_steps=steps, seed=seed)
    agent = GymPPOLagrangian(env, cfg)
    limit = agent.cost_limit
    action_shape = env.action_space.shape

    best = {"score": None, "state": None, "probe": None, "iteration": 0,
            "feasible_probe": False}
    probes: "list[dict]" = []

    def probe(row, ag) -> None:
        """Score the deterministic policy on TRAINING episodes, for selection."""
        s = score(env, ag.deterministic, list(PROBE_SEEDS), limit)
        feasible = s["feasible"] == s["of"]
        probes.append({"iteration": row["iteration"], "steps": row["steps"],
                       "feasible": feasible, **s})
        # Lexicographic: feasibility first, then return. A policy that beats
        # the incumbent by spending more than the budget has not beaten it.
        key = (1 if feasible else 0, s["return"] if feasible else -s["cost"])
        if best["score"] is None or key > best["score"]:
            best.update(score=key, state=ag.state(), probe=s,
                        iteration=row["iteration"], feasible_probe=feasible)

    t0 = time.time()
    history: "list[dict]" = []

    def on_iteration(row, ag):
        history.append(row)
        if row["iteration"] % eval_every == 0:
            probe(row, ag)
        if not quiet:
            print(f"  it {row['iteration']:3d}  steps {row['steps']:6d}  "
                  f"R {row['train_return']:+11.3f}  C {row['train_cost']:11.4g}"
                  f"/{limit:<11.4g} feas {row['train_feasible_frac']:.2f}  "
                  f"lam {row['multiplier']:7.3f}  kl {row['kl']:.4f}",
                  flush=True)

    agent.fit(on_iteration=on_iteration)
    if best["state"] is None:                 # fewer iterations than eval_every
        probe(history[-1], agent)
    train_seconds = time.time() - t0
    env.close()

    # ---- the reported number: eval split, twelve seeds, same `score` ------ #
    eval_env = gym.make(env_id, split="eval")
    policy = load_policy(best["state"], action_shape)
    learned = score(eval_env, policy, list(EVAL_SEEDS),
                    float(eval_env.unwrapped.task.cost_limit))
    # The untrained network through the same code path. Not a baseline anyone
    # should care about on its own -- it is the control that says how much of
    # the learned row is the architecture's inductive bias and how much is the
    # twenty thousand steps.
    init_cfg = GymPPOConfig(total_steps=1, seed=seed)
    init_agent = GymPPOLagrangian(gym.make(env_id, split="train"), init_cfg)
    untrained = score(eval_env, load_policy(init_agent.state(), action_shape),
                      list(EVAL_SEEDS), float(eval_env.unwrapped.task.cost_limit))
    eval_env.close()

    d = out_dir(env_id, seed, root)
    d.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save(best["state"], d / "policy.pt")
    result = {
        "env_id": env_id, "seed": seed, "steps_budget": steps,
        "steps_done": agent.steps_done, "cost_limit": limit,
        "incumbent": INCUMBENT[env_id],
        "selected_iteration": best["iteration"],
        "selection_feasible_on_train": best["feasible_probe"],
        "train_probe": best["probe"], "eval": learned,
        "eval_untrained": untrained,
        "train_seconds": round(train_seconds, 1),
        "iterations": len(history),
        "algorithm": "ppo_lagrangian",
        "cfg": {k: v for k, v in cfg.__dict__.items()},
    }
    (d / "result.json").write_text(json.dumps(result, indent=1))
    (d / "history.json").write_text(json.dumps(
        {"history": history, "probes": probes}, indent=1))
    return result


def stale_against_live(env_id: str, results: "list[dict]",
                       seeds: "list[Path]") -> "dict | None":
    """Why these runs no longer describe the environment of that name.

    A finished run records the budget it was trained against and leaves a
    checkpoint whose normaliser records the observation space it saw. When
    either has since moved, the saved eval return is a number about a
    different problem, and putting it in a column headed by the current
    incumbent compares two environments and calls the difference a result.
    Both of the fourteen that moved on 2026-10-27 are caught here:
    `FloodBuyout-v0` by its budget (17.803 houses on the 33-window pool
    against 8.3125 on the eight wet ones) and `FloodEvacuate-v0` by its
    observation space (4 channels before the forecast was exposed, 6 after).
    """
    import gymnasium as gym
    import torch

    from pspe.envs import register_all

    register_all()
    env = gym.make(env_id, split="eval")
    try:
        live_limit = float(env.unwrapped.task.cost_limit)
        live_obs = tuple(env.observation_space.shape)
    finally:
        env.close()

    why: "dict[str, dict]" = {}
    saved_limit = float(results[0]["cost_limit"])
    if abs(saved_limit - live_limit) > LIMIT_TOL * max(abs(live_limit), 1e-9):
        why["limit"] = {"saved": saved_limit, "live": live_limit}
    for p in seeds:
        ckpt = p.parent / "policy.pt"
        if not ckpt.exists():
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        shape = (state.get("obs_norm") or {}).get("shape")
        if shape is None:
            # Nothing recorded the observation space, so this check cannot
            # answer either way. Silence is the honest result: the limit
            # comparison above still stands on its own.
            break
        if tuple(shape) != live_obs:
            why["obs"] = {"saved": list(shape), "live": list(live_obs)}
        break
    return why or None


def report(root: str, env_ids: "list[str]") -> int:
    """Aggregate every seed that finished into one table."""
    import numpy as np

    from pspe.envs import BY_ID
    from score_gym import INCUMBENT

    rows = []
    for env_id in env_ids:
        seeds = sorted(Path(root).glob(f"{env_id}/seed*/result.json"))
        results = [json.loads(p.read_text()) for p in seeds]
        if not results:
            continue
        spec = BY_ID[env_id]
        rec = spec.scores or {}
        ret = np.array([r["eval"]["return"] for r in results])
        cost = np.array([r["eval"]["cost"] for r in results])
        feas = np.array([r["eval"]["feasible"] for r in results])
        of = results[0]["eval"]["of"]
        inc = rec.get("incumbent", {})
        rnd = rec.get("random", {})
        unt = np.array([r["eval_untrained"]["return"] for r in results])
        stale = stale_against_live(env_id, results, seeds)
        rows.append({
            "env_id": env_id, "n_seeds": len(results),
            "stale": stale,
            "limit": results[0]["cost_limit"],
            "steps": results[0]["steps_budget"],
            "learned_return": float(ret.mean()), "learned_return_sd": float(ret.std()),
            "learned_cost": float(cost.mean()),
            "learned_feasible_seeds": int((feas == of).sum()),
            "learned_feasible_episodes": int(feas.sum()), "episodes": of * len(results),
            "untrained_return": float(unt.mean()),
            "incumbent": INCUMBENT[env_id],
            "incumbent_return": inc.get("return"), "incumbent_cost": inc.get("cost"),
            "incumbent_feasible": inc.get("feasible"),
            "random_return": rnd.get("return"),
            "selection_feasible": sum(
                1 for r in results if r["selection_feasible_on_train"]),
            "seconds": float(np.sum([r["train_seconds"] for r in results])),
        })

    if not rows:
        print(f"no finished runs under {root}", file=sys.stderr)
        return 1

    print(f"\n{'environment':22s} {'learned return':>22s} {'cost/limit':>21s} "
          f"{'feas':>8s} {'incumbent':>13s} {'verdict':>10s}")
    print("-" * 104)
    beats = clears = stale_rows = 0
    for r in rows:
        feasible = r["learned_feasible_seeds"] == r["n_seeds"]
        better = (r["incumbent_return"] is not None
                  and r["learned_return"] > r["incumbent_return"])
        # `STALE` is not a rank. The saved return is a true number about the
        # environment as it was when the run finished, and the incumbent
        # column is the environment as it is now, so there is no comparison
        # to make and the row is excluded from both counters below.
        verdict = ("STALE" if r["stale"] else
                   "beats" if feasible and better else
                   "ties" if feasible and r["incumbent_return"] is not None
                   and abs(r["learned_return"] - r["incumbent_return"])
                   <= 0.02 * max(abs(r["incumbent_return"]), 1e-9) else
                   "below" if feasible else "INFEASIBLE")
        if r["stale"]:
            stale_rows += 1
        elif feasible:
            clears += 1
            if better:
                beats += 1
        print(f"{r['env_id']:22s} {r['learned_return']:+12.3f} "
              f"+-{r['learned_return_sd']:8.3f} "
              f"{r['learned_cost']:10.4g}/{r['limit']:<10.4g} "
              f"{r['learned_feasible_episodes']:3d}/{r['episodes']:<4d} "
              f"{r['incumbent_return']:+12.3f} {verdict:>10s}")
    print("-" * 104)
    print(f"{len(rows)} environments measured; {clears} feasible on every seed; "
          f"{beats} of those beat the incumbent")
    if stale_rows:
        print(f"{stale_rows} of the {len(rows)} are STALE -- the environment "
              "changed under the saved run, so no verdict is reported for "
              "them. Retrain to fill those rows:")
        for r in rows:
            if not r["stale"]:
                continue
            for what, v in r["stale"].items():
                print(f"  {r['env_id']:22s} {what}: trained on {v['saved']}, "
                      f"environment is now {v['live']}")
    Path(root, "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {Path(root, 'summary.json')}")
    return 0


def main() -> int:
    from score_gym import INCUMBENT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", action="append", default=[],
                    help="environment id; repeatable. Default: all fourteen.")
    ap.add_argument("--seed", type=int, action="append", default=[],
                    help="training seed; repeatable. Default: 0 1 2.")
    ap.add_argument("--steps", type=int, default=0,
                    help=f"environment steps per seed. Default {DEFAULT_STEPS}, "
                         f"overridden per environment in STEP_BUDGET.")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="iterations between training-split selection probes")
    ap.add_argument("--root", default="runs/gym",
                    help="where checkpoints and results go")
    ap.add_argument("--report", action="store_true",
                    help="aggregate finished runs instead of training")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    env_ids = args.env or list(INCUMBENT)
    unknown = [e for e in env_ids if e not in INCUMBENT]
    if unknown:
        print(f"unknown environment(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.report:
        return report(args.root, env_ids)

    seeds = args.seed or [0, 1, 2]
    for env_id in env_ids:
        steps = args.steps or STEP_BUDGET.get(env_id, DEFAULT_STEPS)
        for seed in seeds:
            print(f"\n=== {env_id}  seed {seed}  {steps} steps ===", flush=True)
            r = train_one(env_id, seed, steps, args.root, args.eval_every,
                          quiet=args.quiet)
            e = r["eval"]
            print(f"  -> eval return {e['return']:+.3f} +- {e['return_sd']:.3f}  "
                  f"cost {e['cost']:.4g}/{r['cost_limit']:.4g}  "
                  f"feasible {e['feasible']}/{e['of']}  "
                  f"({r['train_seconds']:.0f}s, selected iteration "
                  f"{r['selected_iteration']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
