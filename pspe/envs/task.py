"""Task reward and constraint functionals, shared by every planner and baseline.

Defined once, in batched differentiable form, and reused by:
  * the Gymnasium env (scalar, numpy-facing),
  * the differentiable rollout used by the hybrid-gradient planner,
  * the safe-RL baselines.

If reward or cost were defined twice, the "controlled comparison" in Phase 2
would silently compare different problems.

Reward (per step, batched):
    R = -w_track * mean((u - u_target)^2) - w_effort * mean(a^2)

Cost (per step, batched), the proposal's constraint functional:
    C = w_exposure * mean(relu(u_0 - u_max))          # field must stay below a cap
      + w_budget   * relu(mean|a| - budget)           # actuation budget
A trajectory is "safe" when sum_t C_t <= cost_limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class TaskSpec:
    """Reward/cost weights and constraint thresholds for one testbed."""

    target_value: float = 0.0     # drive the tracked channel toward this level
    tracked_channel: int = 0
    w_track: float = 1.0
    w_effort: float = 0.02
    u_max: float = 0.55           # exposure threshold on the tracked channel
    w_exposure: float = 1.0
    budget: float = 0.35          # mean |amplitude| allowed per step
    w_budget: float = 1.0
    cost_limit: float = 2.0       # episode cost budget d in the proposal's CMDP

    def reward(self, state: Tensor, action: Tensor) -> Tensor:
        """(B, C, H, W), (B, K) -> (B,) task reward."""
        tracked = state[:, self.tracked_channel]
        track = ((tracked - self.target_value) ** 2).mean(dim=(-2, -1))
        effort = (action**2).mean(dim=-1)
        return -(self.w_track * track + self.w_effort * effort)

    def cost(self, state: Tensor, action: Tensor) -> Tensor:
        """(B, C, H, W), (B, K) -> (B,) constraint cost, >= 0."""
        tracked = state[:, self.tracked_channel]
        exposure = torch.relu(tracked - self.u_max).mean(dim=(-2, -1))
        overspend = torch.relu(action.abs().mean(dim=-1) - self.budget)
        return self.w_exposure * exposure + self.w_budget * overspend


# Per-testbed task settings. The tracked channel differs by testbed: surface
# height for shallow water, activator for the reaction-diffusion front.
TASK_SPECS: dict[str, TaskSpec] = {
    "dar": TaskSpec(
        target_value=0.0, tracked_channel=0, u_max=0.55, budget=0.35, cost_limit=2.0
    ),
    "swe": TaskSpec(
        target_value=0.0, tracked_channel=0, u_max=0.12, budget=0.35,
        w_track=4.0, cost_limit=1.0,
    ),
    "rdf": TaskSpec(
        target_value=-1.0, tracked_channel=0, u_max=0.5, budget=0.35,
        w_track=0.5, cost_limit=2.0,
    ),
}


def make_task(testbed: str, **overrides: float) -> TaskSpec:
    if testbed not in TASK_SPECS:
        raise KeyError(f"no task spec for testbed {testbed!r}")
    base = TASK_SPECS[testbed]
    return TaskSpec(**{**base.__dict__, **overrides})
