#!/usr/bin/env python
"""Validate the in-tree safe-RL four against a standard benchmark.

    python baselines/validate_safety_gym.py --steps 1000000

**This script has not been run.** `safety-gymnasium` does not install in the
development environment (its `pygame` dependency fails to build a wheel on
macOS/arm64), so the numbers it would produce do not exist yet. Run it on Linux
or Colab before reporting any Phase 2 comparison, and record the output in
`docs/proposal_deltas.md`.

What it checks, and why it is not redundant with the unit tests:

* `tests/test_safe_rl_correctness.py` verifies each *component* against a closed
  form (CG against an explicit inverse, the Fisher-vector product against the
  analytic Fisher, the PID terms against hand arithmetic) and each *algorithm*
  against a one-step CMDP whose optimum is `min(p, d)`. That catches wrong
  update rules — it already caught two.
* It cannot catch failures that only appear at scale: poor exploration over long
  horizons, value-function collapse, or trust-region settings that are fine on a
  scalar action and wrong on a 12-dimensional one.

SafetyPointGoal1-v0 is the standard reference. Published behaviour at ~1e6
steps, from the Safety-Gymnasium and OmniSafe benchmark suites: episode return
rises into the ~15-25 band and episode cost falls toward the cost limit (25 by
convention). The assertions below encode the *trend*, not exact values, since
those vary with implementation details and seeds.

Note this needs a continuous-observation adapter: the agents in this repo take
image-like field observations, whereas SafetyPointGoal1 emits a flat vector.
`_VectorObsEnv` reshapes the vector into a 1xNxN grid so the same convolutional
policy applies unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.safe_rl import ALGORITHMS, SafeRLConfig, make_agent  # noqa: E402
from pspe.envs.task import TaskSpec  # noqa: E402
from pspe.utils import RunLogger, project_path, seed_everything  # noqa: E402

# Published reference bands at ~1e6 steps (Safety-Gymnasium / OmniSafe suites).
# Deliberately wide: this is a "did it learn and did it respect the budget"
# check, not a leaderboard reproduction.
REFERENCE = {
    "return_min": 5.0,      # a learner that never improves stays near 0
    "cost_limit": 25.0,     # the benchmark convention
    "cost_tolerance": 15.0,  # allowed steady-state overshoot above the limit
}


class _VectorObsEnv:
    """Adapt a flat-observation Gymnasium env to the batched field interface."""

    def __init__(self, env_id: str, cost_limit: float, horizon: int, batch: int) -> None:
        import safety_gymnasium  # type: ignore[import-not-found]

        self.envs = [safety_gymnasium.make(env_id) for _ in range(batch)]
        self.batch = batch
        self.horizon = horizon
        self.t = 0
        self.task = TaskSpec(cost_limit=cost_limit)
        self.basis = None
        obs_dim = self.envs[0].observation_space.shape[0]
        self.side = int(math.ceil(math.sqrt(obs_dim)))
        self.obs_dim = obs_dim
        self._action_dim = self.envs[0].action_space.shape[0]
        self.state: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (1, self.side, self.side)

    def _grid(self, flat: list) -> torch.Tensor:
        out = torch.zeros(self.batch, 1, self.side, self.side)
        for i, obs in enumerate(flat):
            padded = torch.zeros(self.side * self.side)
            padded[: self.obs_dim] = torch.as_tensor(obs, dtype=torch.float32)
            out[i, 0] = padded.view(self.side, self.side)
        return out

    def reset(self, batch: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        observations = [env.reset()[0] for env in self.envs[:batch]]
        self.t = 0
        self.state = self._grid(observations)
        return self.state

    def step(self, action: torch.Tensor):
        observations, rewards, costs = [], [], []
        for i, env in enumerate(self.envs[: action.shape[0]]):
            obs, reward, cost, terminated, truncated, _ = env.step(
                action[i].detach().cpu().numpy()
            )
            if terminated or truncated:
                obs, _ = env.reset()
            observations.append(obs)
            rewards.append(reward)
            costs.append(cost)
        self.t += 1
        self.state = self._grid(observations)
        return (
            self.state,
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(costs, dtype=torch.float32),
            self.t >= self.horizon,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SafetyPointGoal1-v0")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/validation/safety_gym.json")
    args = parser.parse_args()

    try:
        import safety_gymnasium  # noqa: F401
    except ImportError:
        print(
            "safety-gymnasium is not installed. It does not build on macOS/arm64 "
            "(pygame wheel failure); run this on Linux or Colab:\n"
            "    pip install safety-gymnasium\n"
            "Until then the safe-RL baselines are validated only against the "
            "closed-form CMDP in tests/test_safe_rl_correctness.py."
        )
        return 1

    iterations = max(1, args.steps // (args.batch * args.horizon))
    results = {}
    for name in ALGORITHMS:
        seed_everything(args.seed)
        env = _VectorObsEnv(args.env_id, REFERENCE["cost_limit"], args.horizon, args.batch)
        agent = make_agent(
            name, env,
            cfg=SafeRLConfig(iterations=iterations, batch=args.batch, horizon=args.horizon,
                             seed=args.seed, log_dir="runs/validation"),
            logger=RunLogger(project_path("runs/validation") / name),
        )
        summary = agent.train()
        learned = summary["return"] >= REFERENCE["return_min"]
        respected = (
            summary["episode_cost"]
            <= REFERENCE["cost_limit"] + REFERENCE["cost_tolerance"]
        )
        results[name] = {
            **summary,
            "learned": bool(learned),
            "respected_budget": bool(respected),
        }
        verdict = "PASS" if learned and respected else "FAIL"
        print(
            f"[{verdict}] {name}: return {summary['return']:.1f} "
            f"(>= {REFERENCE['return_min']}), cost {summary['episode_cost']:.1f} "
            f"(limit {REFERENCE['cost_limit']})"
        )

    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwritten to {out}")
    return 0 if all(r["learned"] and r["respected_budget"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
