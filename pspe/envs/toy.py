"""The environment whose answer is known.

Every other environment in this gym is scored against the best policy anybody
has found. This one is scored against the **right** policy, because its
optimum has a closed form: a one-step constrained bandit where reward rises to
a peak at `p` and cost rises linearly, so the constrained optimum is
`min(p, d)` and the Lagrange multiplier at that point is known too.

That makes it the first id a user should call. A constrained RL algorithm that
does not land on `optimal_action` here is broken, and the failure is visible
in one line instead of inferred from a league table on problems whose answers
nobody knows. This is not a hypothetical use: the closed form in
`baselines/toy_cmdp.py` is what caught two real bugs in this repository's own
reimplemented baselines, which is recorded in the README under "Baseline
validation".

It is deliberately trivial otherwise -- a constant observation field, one
action dimension, one step -- so that anything that goes wrong here is the
algorithm and not the problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]


class CMDPToyEnv(gym.Env):
    """Single-instance Gymnasium view of the constrained bandit.

    `info["cost"]` per step and `cost_limit` for the episode, the same
    convention as everything else here. `optimal_action` and
    `optimal_multiplier` are exposed so a test can assert against them rather
    than against a number somebody wrote down.
    """

    metadata = {"render_modes": []}

    def __init__(self, cost_limit: float = 0.4, horizon: int = 1,
                 reward_peak: float = 0.8, grid: int = 8,
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from baselines.toy_cmdp import ToyConstrainedBandit     # noqa: PLC0415

        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = ToyConstrainedBandit(cost_limit=cost_limit, horizon=horizon,
                                          grid=grid, reward_peak=reward_peak,
                                          device=self.device)
        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._core.action_dim,),
                                       dtype=np.float32)
        self.episode_cost = 0.0

    # -- the answer --------------------------------------------------------- #
    @property
    def optimal_action(self) -> float:
        """The constrained optimum in ACTION units, `min(peak, limit/horizon)`."""
        return float(self._core.optimal_action)

    @property
    def optimal_multiplier(self) -> float:
        """lambda* at that optimum, for anything checking a dual variable."""
        return float(self._core.optimal_multiplier)

    @property
    def task(self):
        return self._core.task

    @property
    def core(self):
        return self._core

    def optimal_action_vector(self, obs: "np.ndarray | None" = None) -> "np.ndarray":
        """The optimum as an action this environment accepts. The bar, exactly."""
        return np.full(self.action_space.shape, self.optimal_action,
                       dtype=np.float32)

    def zero_action(self, obs: "np.ndarray | None" = None) -> "np.ndarray":
        """Do nothing: feasible by construction, and worth nothing."""
        return np.zeros(self.action_space.shape, dtype=np.float32)

    # -- gymnasium API ------------------------------------------------------ #
    def reset(self, *, seed: "int | None" = None,
              options: "dict | None" = None) -> "tuple[np.ndarray, dict]":
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        state = self._core.reset(batch=1, generator=self._generator)
        self._last_action = 0.0
        self.episode_cost = 0.0
        return state[0].detach().cpu().numpy().astype(np.float32), {"cost": 0.0}

    def step(self, action: "np.ndarray"
             ) -> "tuple[np.ndarray, float, bool, bool, dict]":
        act = torch.as_tensor(action, dtype=torch.float32,
                              device=self.device).view(1, -1).clamp(-1.0, 1.0)
        # Kept so a viewer can draw the dial that was set against the known
        # answer; there is no field in the observation to draw instead.
        self._last_action = float(act[0, 0])
        with torch.no_grad():
            state, reward, cost, done = self._core.step(act)
        self.episode_cost += float(cost[0])
        info = {"cost": float(cost[0]), "episode_cost": self.episode_cost,
                "cost_limit": self._core.task.cost_limit,
                "optimal_action": self.optimal_action}
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        # TERMINATED, not truncated -- the one place in this gym where that is
        # the right answer. A one-step bandit's episode is over by
        # construction; there is no time limit being hit and no value beyond
        # the end for a critic to bootstrap. The flood and fire environments
        # are the other way round, and Gymnasium's own checker rejects an
        # environment that truncates after one step.
        return obs, float(reward[0]), bool(done), False, info


def register_envs() -> None:
    """Make `gymnasium.make("CMDPToy-v0")` work. Idempotent."""
    from gymnasium.envs.registration import register, registry

    if "CMDPToy-v0" in registry:
        return
    register(id="CMDPToy-v0", entry_point="pspe.envs.toy:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> CMDPToyEnv:
    return CMDPToyEnv(**kw)
