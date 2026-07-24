"""Task reward and constraint functionals, shared by every planner and baseline.

Defined once, in batched differentiable form, and reused by:
  * the Gymnasium env (scalar, numpy-facing),
  * the differentiable rollout used by the hybrid-gradient planner,
  * the safe-RL baselines.

If reward or cost were defined twice, the "controlled comparison" in Phase 2
would silently compare different problems.

Reward (per step, batched):
    R = -w_track * mean((u - u_target)^2) - w_effort * mean(a^2)

The paper's C-POMDP (Section 3) carries a *set* of constraints {g_j}, not one.
Two are implemented here, each an independent constraint with its own limit and
its own dual:

  safety (g_1, per step):
    C = w_exposure * mean(relu(u_0 - u_max))          # field must stay below a cap
      + w_budget   * relu(mean|a| - budget)           # actuation budget

  equity (g_2, per step) — the distributional fairness functional the paper
  calls "first-class, formally enforced" (Sections 3 and 8):
    partition the domain into R x R sub-regions; let h_r be the per-region mean
    residual harm (squared deviation from target). Equity cost is the spread of
    harm across regions,
        E = variance_r(h_r),
    which is zero exactly when every sub-region is treated equally and grows
    when the policy sacrifices some regions to help others. Minimising it forbids
    "help the easy regions, abandon the rest" solutions.

A trajectory is feasible when sum_t g_j(t) <= d_j for every enabled constraint.
Equity is OFF by default (equity_enabled=False) so the head-to-head Phase 2
comparison stays a single-constraint controlled comparison; turn it on to
exercise the multi-constraint machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def region_means(field: Tensor, n_sub: int) -> Tensor:
    """(B, H, W) -> (B, n_sub*n_sub): mean over each of R x R sub-regions.

    Uses adaptive average pooling so it works at any grid resolution, which
    matters because the equity functional has to be identical at 64x64 and the
    resolution-generalisation grids.
    """
    pooled = F.adaptive_avg_pool2d(field.unsqueeze(1), output_size=n_sub)
    return pooled.flatten(1)


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
    cost_limit: float = 2.0       # safety budget d_1 in the proposal's CMDP
    # Equity constraint g_2 (off by default; see module docstring).
    equity_enabled: bool = False
    n_subregions: int = 2         # R: domain split into R x R sub-regions
    w_equity: float = 1.0
    equity_limit: float = 0.05    # d_2, calibrated like cost_limit when enabled

    def reward(self, state: Tensor, action: Tensor) -> Tensor:
        """(B, C, H, W), (B, K) -> (B,) task reward."""
        tracked = state[:, self.tracked_channel]
        track = ((tracked - self.target_value) ** 2).mean(dim=(-2, -1))
        effort = (action**2).mean(dim=-1)
        return -(self.w_track * track + self.w_effort * effort)

    def cost(self, state: Tensor, action: Tensor) -> Tensor:
        """Safety constraint g_1: (B, C, H, W), (B, K) -> (B,), >= 0."""
        tracked = state[:, self.tracked_channel]
        exposure = torch.relu(tracked - self.u_max).mean(dim=(-2, -1))
        overspend = torch.relu(action.abs().mean(dim=-1) - self.budget)
        return self.w_exposure * exposure + self.w_budget * overspend

    def equity_cost(self, state: Tensor, action: Tensor) -> Tensor:
        """Equity constraint g_2: spread of residual harm across sub-regions.

        (B, C, H, W), (B, K) -> (B,), >= 0. Zero iff every sub-region carries the
        same residual harm; positive when the policy leaves some regions behind.
        """
        tracked = state[:, self.tracked_channel]
        harm = (tracked - self.target_value) ** 2          # (B, H, W)
        per_region = region_means(harm, self.n_subregions)  # (B, R*R)
        return self.w_equity * per_region.var(dim=-1, unbiased=False)

    def costs(self, state: Tensor, action: Tensor) -> dict[str, Tensor]:
        """All enabled constraints as {name: (B,) cost}. Used by the planner."""
        out = {"safety": self.cost(state, action)}
        if self.equity_enabled:
            out["equity"] = self.equity_cost(state, action)
        return out

    def limits(self) -> dict[str, float]:
        out = {"safety": self.cost_limit}
        if self.equity_enabled:
            out["equity"] = self.equity_limit
        return out


# Per-testbed task settings. The tracked channel differs by testbed: surface
# height for shallow water, activator for the reaction-diffusion front.
#
# `cost_limit` is CALIBRATED, not guessed: `scripts/calibrate_constraints.py`
# measures the cost of doing nothing and the cost a reward-greedy policy (dual
# disabled) incurs, then places the limit at
#
#     cost_zero + 0.35 * (cost_greedy - cost_zero)
#
# so a constrained learner has to give up a real share of reward-greedy
# behaviour. A limit above `cost_greedy` leaves the dual pinned at zero and the
# whole constrained comparison degenerates into five unconstrained learners
# tying — which is exactly what the uncalibrated defaults produced.
# Measurements in runs/calibration/limits.json.
TASK_SPECS: dict[str, TaskSpec] = {
    # do-nothing 0.229, reward-greedy 2.249 -> 0.936
    "dar": TaskSpec(
        target_value=0.0, tracked_channel=0, u_max=0.55, budget=0.35, cost_limit=0.936
    ),
    # u_max and budget are also calibrated here, not just the limit. The
    # original thresholds (u_max 0.12, budget 0.35) sat entirely outside the
    # realised distributions - |h| reaches 0.113 at the 95th percentile and
    # mean|a| peaks at 0.051 - so neither cost term ever activated and the task
    # had no reward/cost tension at all to constrain.
    # after retuning: do-nothing 0.108, reward-greedy 0.347 -> 0.192
    "swe": TaskSpec(
        target_value=0.0, tracked_channel=0, u_max=0.04, budget=0.03,
        w_track=4.0, cost_limit=0.192,
    ),
    # do-nothing 0.594, reward-greedy 8.213 -> 3.26
    "rdf": TaskSpec(
        target_value=-1.0, tracked_channel=0, u_max=0.5, budget=0.35,
        w_track=0.5, cost_limit=3.26,
    ),
}


def make_task(testbed: str, **overrides: float) -> TaskSpec:
    if testbed not in TASK_SPECS:
        raise KeyError(f"no task spec for testbed {testbed!r}")
    base = TASK_SPECS[testbed]
    return TaskSpec(**{**base.__dict__, **overrides})
