"""Gymnasium wrappers around the PDE testbeds.

Two views of the same problem:

* `PDEControlEnv`   - standard single-instance Gymnasium env (numpy in/out).
  Dynamics come from either the numerical solver ("truth") or the learned
  surrogate. This is what the safe-RL baselines consume.

* `BatchedFieldEnv` - batched, all-torch, differentiable-through-the-surrogate.
  This is what the hybrid-gradient planner consumes: the pathwise branch needs
  d(reward)/d(action) through the dynamics, which a numpy Gymnasium step cannot
  provide.

Both use the same `TaskSpec` and the same actuator basis, so a reward reported
by one is directly comparable to the other.
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from ..simulate.solvers import PDETestbed, make_testbed
from .actuators import GaussianActuatorBasis
from .task import TaskSpec, make_task

Tensor = torch.Tensor
Dynamics = Literal["truth", "surrogate"]


class BatchedFieldEnv:
    """Batched differentiable environment over `B` parallel field episodes."""

    def __init__(
        self,
        testbed: PDETestbed,
        task: TaskSpec,
        basis: GaussianActuatorBasis,
        surrogate: nn.Module | None = None,
        dynamics: Dynamics = "truth",
        horizon: int = 16,
        device: torch.device | str = "cpu",
    ) -> None:
        if dynamics == "surrogate" and surrogate is None:
            raise ValueError("dynamics='surrogate' requires a surrogate model")
        self.testbed = testbed
        self.task = task
        self.basis = basis
        self.surrogate = surrogate
        self.dynamics = dynamics
        self.horizon = horizon
        self.device = torch.device(device)
        self.state: Tensor | None = None
        self.t = 0

    @property
    def action_dim(self) -> int:
        return self.basis.n_actuators

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.testbed.n_channels, self.testbed.grid, self.testbed.grid)

    def reset(self, batch: int = 1, generator: torch.Generator | None = None) -> Tensor:
        self.state = self.testbed.initial_condition(batch, generator).to(self.device)
        self.t = 0
        return self.state

    def set_state(self, state: Tensor) -> None:
        """Seed the env from an externally supplied field (e.g. Perceive output)."""
        self.state = state.to(self.device)
        self.t = 0

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:
        """action: (B, K) in [-1, 1]. Returns (next_state, reward, cost, done).

        The graph is retained: with `dynamics='surrogate'` the returned reward is
        differentiable w.r.t. `action`.
        """
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        control = self.basis.field(action)
        if self.dynamics == "truth":
            next_state = self.testbed.step(self.state, control)
        else:
            assert self.surrogate is not None
            next_state = self.surrogate(self.state, control)

        reward = self.task.reward(next_state, action)
        cost = self.task.cost(next_state, action)
        self.state = next_state
        self.t += 1
        return next_state, reward, cost, self.t >= self.horizon


class PDEControlEnv(gym.Env):
    """Single-instance Gymnasium env; cost is returned in `info["cost"]`.

    `info["cost"]` is the convention the safe-RL baselines in `baselines/`
    read, matching the Safety-Gymnasium / OmniSafe interface so an OmniSafe
    install could be dropped in unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        testbed: str = "dar",
        grid: int = 64,
        horizon: int = 16,
        n_actuators: int = 9,
        dynamics: Dynamics = "truth",
        surrogate: nn.Module | None = None,
        task: TaskSpec | None = None,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._testbed_name = testbed
        self._testbed = make_testbed(testbed, grid=grid, device=self.device)
        self._task = task or make_task(testbed)
        self._basis = GaussianActuatorBasis(n_actuators, grid, device=self.device,
                                            max_amplitude=self._task.actuator_amplitude)
        self._core = BatchedFieldEnv(
            self._testbed, self._task, self._basis, surrogate, dynamics, horizon, self.device
        )
        self._generator = torch.Generator().manual_seed(seed if seed is not None else 0)

        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actuators,), dtype=np.float32)
        self.episode_cost = 0.0

    # -- gymnasium API ------------------------------------------------------ #
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Seeds `self.np_random`, which Gymnasium's `check_env` requires and
        # which this env did not do. No behaviour change -- the episode is
        # driven by `self._generator` below -- but without it the env fails
        # conformance, and a caller reading `env.np_random` gets an unseeded
        # stream. Found while writing the flood env's sibling wrapper.
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        state = self._core.reset(batch=1, generator=self._generator)
        self.episode_cost = 0.0
        return state[0].detach().cpu().numpy().astype(np.float32), {"cost": 0.0}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device).view(1, -1)
        act = act.clamp(-1.0, 1.0)
        with torch.no_grad():
            state, reward, cost, done = self._core.step(act)
        self.episode_cost += float(cost[0])
        info = {
            "cost": float(cost[0]),
            "episode_cost": self.episode_cost,
            "cost_limit": self._task.cost_limit,
        }
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        return obs, float(reward[0]), False, bool(done), info

    # -- accessors used by the trainers ------------------------------------- #
    @property
    def task(self) -> TaskSpec:
        return self._task

    @property
    def basis(self) -> GaussianActuatorBasis:
        return self._basis

    @property
    def core(self) -> BatchedFieldEnv:
        return self._core

    # -- baselines ---------------------------------------------------------- #
    def zero_action(self, obs: "np.ndarray | None" = None) -> "np.ndarray":
        """Actuate nothing. The floor every other number is read against."""
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def proportional_action(self, obs: "np.ndarray",
                            gain: "float | None" = None) -> "np.ndarray":
        """Push each actuator against the error under it. The incumbent.

        A least-squares projection of the tracking error onto each Gaussian --
        `<b_k, e> / <b_k, b_k>` -- negated, so an actuator over ground that is
        above target pushes down. It is the controller anybody writes first on
        a tracking problem, it reads only the observation, and it has no model
        of the dynamics and no notion of the budget, which is exactly why it is
        worth beating rather than a strawman.
        """
        task = self._task
        field = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device)[task.tracked_channel]
        err = field - task.target_value
        b = self._basis.basis                                   # (K, H, W)
        num = (b * err[None]).sum(dim=(-2, -1))
        den = (b * b).sum(dim=(-2, -1)).clamp(min=1e-8)
        g = PROPORTIONAL_GAIN.get(self._testbed_name, 1.0) if gain is None else gain
        amp = -g * (num / den) / max(task.actuator_amplitude, 1e-8)
        return amp.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)


#: The largest proportional gain that stays inside each testbed's budget,
#: found by sweeping. Set by measurement, not taste: at gain 1.0 the
#: controller beats doing nothing on all three testbeds AND violates the
#: budget on all three, so shipping it unscaled would have left "do nothing"
#: as the best feasible policy on three new environments -- the weak-incumbent
#: error this project has already published once.
PROPORTIONAL_GAIN = {"dar": 0.4, "swe": 0.3, "rdf": 0.3}

#: `swe` is the exception and it is worth stating. Its cost carries a
#: state-driven exposure term, so on some episodes the budget is exceeded
#: whatever the actuators do -- gain 0.0 clears it in only 11 of 12 episodes.
#: The constraint there is not fully controllable, which is a property of the
#: testbed and not of any policy, and `task.py` already notes that swe "went a
#: whole project without anyone noticing its margin".

#: The three testbeds, with the grid each is calibrated at. `task.py` sets a
#: separate `cost_limit` and actuator amplitude per testbed; those calibrations
#: are what make the constraint bind, so they are not parameters to vary here.
PDE_TESTBEDS = ("dar", "swe", "rdf")


def register_envs() -> None:
    """Make `gymnasium.make("PDEControl-dar-v0")` and friends work.

    These are the three PDE problems the transfer matrix runs on, and the
    environment behind them has been a complete `gymnasium.Env` for months --
    it simply had no id. Registering costs nothing and takes the gym from one
    hazard family to two kinds of problem: a field to track, and a hazard to
    decide about.
    """
    from gymnasium.envs.registration import register, registry

    for name in PDE_TESTBEDS:
        env_id = f"PDEControl-{name}-v0"
        if env_id in registry:
            continue
        register(id=env_id, entry_point="pspe.envs.pde_env:_make_default",
                 kwargs={"testbed": name}, max_episode_steps=None)


def _make_default(testbed: str = "dar", grid: int = 32, horizon: int = 8,
                  **kw) -> "PDEControlEnv":
    """Entry point for `gymnasium.make`.

    Grid 32 and horizon 8 rather than the research default of 64 and 16: this
    is the id a stranger calls first, and an environment whose first step takes
    a minute is one nobody runs twice. `make_env` still builds the research
    configuration.
    """
    return PDEControlEnv(testbed=testbed, grid=grid, horizon=horizon, **kw)


def make_env(
    testbed: str = "dar",
    dynamics: Dynamics = "truth",
    surrogate: nn.Module | None = None,
    grid: int = 64,
    horizon: int = 16,
    n_actuators: int = 9,
    device: torch.device | str = "cpu",
    batched: bool = False,
    **task_overrides: float,
) -> PDEControlEnv | BatchedFieldEnv:
    """One construction point for both env views.

    `task_overrides` go to `make_task` — used to sweep actuation authority when
    calibrating a testbed, which is how swe's dead task was diagnosed.
    """
    if not batched:
        return PDEControlEnv(
            testbed=testbed, grid=grid, horizon=horizon, n_actuators=n_actuators,
            dynamics=dynamics, surrogate=surrogate, device=device,
            task=make_task(testbed, **task_overrides) if task_overrides else None,
        )
    device_t = torch.device(device)
    task = make_task(testbed, **task_overrides)
    return BatchedFieldEnv(
        testbed=make_testbed(testbed, grid=grid, device=device_t),
        task=task,
        basis=GaussianActuatorBasis(n_actuators, grid, device=device_t,
                                    max_amplitude=task.actuator_amplitude),
        surrogate=surrogate,
        dynamics=dynamics,
        horizon=horizon,
        device=device_t,
    )
