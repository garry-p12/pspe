"""A constrained MDP with a closed-form optimum, for validating the safe-RL four.

The problem with reimplementing CPO, PID-Lagrangian, Sauté RL and primal-dual
NPG is that all four *run* whether or not they are correct. Comparing them only
against each other on the PDE testbeds cannot detect a wrong natural-gradient
step or a mis-signed dual update: every algorithm would be wrong together and
the table would look self-consistent.

This module fixes a task whose constrained optimum is known analytically, so
each algorithm can be checked against a number that does not come from this
repo.

    single step, scalar action a in (-1, 1)
    reward(a) = -(a - p)^2          maximised at a = p (default p = 0.8)
    cost(a)   = relu(a)             the budget is spent by acting positively
    constraint: E[cost] <= d

    Unconstrained optimum:  a* = p
    Constrained optimum:    a* = min(p, d)
    Optimal multiplier:     lambda* = |dR/da| / |dC/da| = 2 * (p - d)

The reward peak `p` is deliberately *interior* to the tanh-squashed action
range rather than at the boundary. With p = 1 the unconstrained optimum sits
exactly on the squashing limit, the policy's pre-tanh mean runs away, and the
gradient dies before any dual variable can pull it back - so every Lagrangian
method fails the canary for a reason that has nothing to do with its
constraint handling. That is a real pathology, but it is a policy-parameterisation
pathology, and this file is here to test the constrained-optimisation logic.

So with p = 0.8, d = 0.4 a correct constrained learner converges to a = 0.4
with lambda -> 0.8: it must move toward the reward optimum and then *stop
exactly at the budget*. Two failure modes are caught: an algorithm that ignores
the constraint runs to 0.8, and one whose dual update is mis-signed collapses
to a <= 0.

This is a correctness canary, not a benchmark. It says the update rules are
right; it says nothing about behaviour at scale. For that see
`baselines/validate_safety_gym.py`.
"""

from __future__ import annotations

import torch

from pspe.envs.task import TaskSpec

Tensor = torch.Tensor


class ToyConstrainedBandit:
    """Batched one-step CMDP matching the `BatchedFieldEnv` interface.

    Observations are a constant field so the policy's convolutional encoder can
    be reused unchanged; all the signal is in the reward and cost.
    """

    def __init__(
        self,
        cost_limit: float = 0.4,
        horizon: int = 1,
        grid: int = 8,  # the policy encoder strides by 2 three times; 8 is the floor
        reward_peak: float = 0.8,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.horizon = horizon
        self.grid = grid
        self.reward_peak = reward_peak
        self.t = 0
        self.state: Tensor | None = None
        self.task = TaskSpec(cost_limit=cost_limit)
        self.basis = None  # unused; the action is the actuator here

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (1, self.grid, self.grid)

    @property
    def optimal_action(self) -> float:
        """The closed-form constrained optimum, min(p, d)."""
        return min(self.reward_peak, self.task.cost_limit / self.horizon)

    @property
    def optimal_multiplier(self) -> float:
        """lambda* = |dR/da| / |dC/da| at the constrained optimum.

        Zero when the constraint is slack, since the reward optimum is then
        already feasible.
        """
        return max(0.0, 2.0 * (self.reward_peak - self.optimal_action))

    def reset(self, batch: int = 1, generator: torch.Generator | None = None) -> Tensor:
        self.state = torch.ones(batch, 1, self.grid, self.grid, device=self.device)
        self.t = 0
        return self.state

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:
        a = action[:, 0]
        reward = -((a - self.reward_peak) ** 2)
        cost = torch.relu(a)
        self.t += 1
        assert self.state is not None
        return self.state, reward, cost, self.t >= self.horizon


def action_error(policy, env: ToyConstrainedBandit) -> float:
    """|mean deterministic action - closed-form optimum|."""
    with torch.no_grad():
        state = env.reset(64)
        observation = state
        # Sauté augments the observation with the remaining budget.
        if policy.encoder.net[0].in_channels == state.shape[1] + 1:
            budget = torch.ones(state.shape[0], 1, env.grid, env.grid, device=state.device)
            observation = torch.cat([state, budget], dim=1)
        action, _ = policy.sample(observation, deterministic=True)
    return abs(float(action.mean()) - env.optimal_action)
