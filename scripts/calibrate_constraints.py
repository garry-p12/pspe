#!/usr/bin/env python
"""Calibrate each testbed's `cost_limit` so the constraint actually binds.

A constrained comparison is meaningless when the limit is slack: the dual stays
at zero, every algorithm runs unconstrained, and all of them tie. That is
exactly what the first Phase 2 table showed — cost ~0.16 against a limit of
2.0, violation rate 0 for all five learners.

The limit has to sit between two reference points:

* `cost_zero`   - the cost of doing nothing (a = 0). The floor: no policy that
                  declines to act can be charged less than this.
* `cost_greedy` - the cost incurred by a *reward-greedy* policy, trained with
                  the dual controller disabled (kp = ki = kd = 0, so lambda is
                  pinned at 0).

If `cost_greedy <= cost_zero` there is no tension between reward and cost in
the task at all, and no choice of limit will produce a meaningful constrained
problem — the script says so rather than emitting a number.

Otherwise the proposed limit is

    limit = cost_zero + fraction * (cost_greedy - cost_zero)

so a constrained learner must give up a real share of the reward-greedy
behaviour. `fraction=0.35` is aggressive enough that the dual has to work
without making the task infeasible.

    python scripts/calibrate_constraints.py
    python scripts/calibrate_constraints.py --testbeds dar --iterations 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


@torch.no_grad()
def zero_action_cost(env, episodes: int, horizon: int, seed: int) -> tuple[float, float]:
    """Cost and return of the do-nothing policy — the floor for any limit."""
    state = env.reset(episodes, torch.Generator().manual_seed(seed))
    total_cost = torch.zeros(episodes, device=state.device)
    total_reward = torch.zeros(episodes, device=state.device)
    for _ in range(horizon):
        action = torch.zeros(episodes, env.action_dim, device=state.device)
        state, reward, cost, _ = env.step(action)
        total_cost += cost
        total_reward += reward
    return float(total_cost.mean()), float(total_reward.mean())


def greedy_cost(testbed: str, grid: int, horizon: int, iterations: int,
                device, seed: int, log_root: Path) -> dict[str, float]:
    """Train a reward-greedy planner (dual disabled) and report what it costs."""
    env_kwargs = dict(testbed=testbed, grid=grid, horizon=horizon, n_actuators=9,
                      device=device, batched=True)
    train_env = make_env(dynamics="truth", **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)

    trainer = HybridPlannerTrainer(
        train_env, policy,
        # kp = ki = kd = 0 pins lambda at 0: pure reward maximisation.
        cfg=PlannerConfig(iterations=iterations, horizon=horizon, kp=0.0, ki=0.0,
                          kd=0.0, eval_every=10**9, seed=seed),
        eval_env=eval_env,
        logger=RunLogger(log_root / f"{testbed}_greedy", use_tensorboard=False),
        device=device,
    )
    trainer.train()
    metrics = trainer.evaluate(episodes=32)
    assert trainer.dual.multiplier == 0.0, "dual was not disabled"
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbeds", nargs="*", default=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--fraction", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/calibration/limits.json")
    args = parser.parse_args()

    device = get_device(args.device)
    log_root = project_path("runs/calibration")
    log_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for testbed in args.testbeds:
        seed_everything(args.seed)
        env = make_env(testbed=testbed, dynamics="truth", grid=args.grid,
                       horizon=args.horizon, n_actuators=9, device=device, batched=True)
        current = env.task.cost_limit

        cost_zero, return_zero = zero_action_cost(
            env, args.episodes, args.horizon, args.seed
        )
        greedy = greedy_cost(testbed, args.grid, args.horizon, args.iterations,
                             device, args.seed, log_root)
        cost_greedy, return_greedy = greedy["episode_cost"], greedy["return"]

        spread = cost_greedy - cost_zero
        binds = spread > 1e-3
        proposed = (
            round(cost_zero + args.fraction * spread, 3) if binds else None
        )

        results[testbed] = {
            "current_limit": current,
            "cost_zero_action": cost_zero,
            "return_zero_action": return_zero,
            "cost_reward_greedy": cost_greedy,
            "return_reward_greedy": return_greedy,
            "spread": spread,
            "constraint_can_bind": binds,
            "proposed_limit": proposed,
            "current_limit_slack_multiple": (
                current / cost_greedy if cost_greedy > 1e-9 else float("inf")
            ),
        }

        print(f"\n=== {testbed} ===")
        print(f"  do-nothing      cost {cost_zero:7.3f}   return {return_zero:+8.3f}")
        print(f"  reward-greedy   cost {cost_greedy:7.3f}   return {return_greedy:+8.3f}")
        print(f"  spread                {spread:7.3f}")
        print(f"  current limit   {current:7.3f}  "
              f"({results[testbed]['current_limit_slack_multiple']:.1f}x the greedy cost)")
        if binds:
            print(f"  -> proposed limit {proposed:.3f}  "
                  f"(gives up {args.fraction:.0%} of the greedy cost budget)")
        else:
            print("  -> NO TENSION: the reward-greedy policy is already at or below "
                  "the do-nothing cost. Reward and cost are aligned in this task; "
                  "tighten TaskSpec.u_max or TaskSpec.budget instead.")

    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))

    print("\n\nSuggested TASK_SPECS edits (pspe/envs/task.py):")
    for testbed, r in results.items():
        if r["proposed_limit"] is not None:
            print(f'  "{testbed}": cost_limit={r["proposed_limit"]}   '
                  f'# was {r["current_limit"]}')
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
