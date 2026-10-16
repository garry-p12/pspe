"""Constrained firebreak planning on observed wildfire data (NDWS).

The first planning environment in this repository whose dynamics are a model
fitted to what a real fire did, not a solver written here. State is a real
day-t patch: the fire mask plus the eleven driver channels (terrain, weather,
drought, vegetation, population). The action is a coarse grid of firebreak
intensities. Dynamics are the trained next-day U-Net, differentiable, so the
planner gets a pathwise gradient exactly as it does through the FNO.

Action model, stated so it can be argued with rather than discovered:

* a firebreak removes fuel: the (standardised) NDVI channel is lowered by
  `fuel_removal` standard deviations times the local treatment intensity, and
  the surrogate is asked what the fire does with less fuel;
* a firebreak also blocks spread into the treated cells: the predicted
  ignition probability there is scaled by `1 - block * intensity`.

The first mechanism goes through the learned model and can be as weak as the
data made it; the second is imposed physics. Both are logged separately.

Reward is minus the expected burned fraction of the patch, with burned cells
weighted 1 + `population_weight` * normalised population density. Cost is the
fraction of the patch treated, against a crew budget: the hard constraint the
dual enforces.

What cannot be measured here, and is not: the effect of an intervention on
the real fire. The data records what happened without a firebreak. The
surrogate's forecast is checked against the observed next-day mask on the same
patches (`forecast_aucpr`); the intervention's effect is scored in-model only.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from pspe.simulate.real.ndws import DRIVER_FEATURES

Tensor = torch.Tensor


@dataclass
class FireTask:
    """Constraint set. Mirrors TaskSpec's interface as far as the planner reads it."""
    cost_limit: float = 0.03        # fraction of the patch that may be treated
    equity_limit: float = 0.05
    equity_enabled: bool = False
    population_weight: float = 4.0  # burned populated cell counts 1 + w * density
    fuel_removal: float = 2.0       # NDVI drop, in standard deviations, at full intensity
    block: float = 0.9              # spread suppression in treated cells at full intensity
    reward_scale: float = 100.0     # burned fraction in percent, so returns are O(1)


class FireEnv:
    """BatchedFieldEnv-compatible: reset(batch, gen) / step(action) / obs_shape / action_dim."""

    dynamics = "surrogate"

    def __init__(self, surrogate: nn.Module, data: dict[str, np.ndarray],
                 task: FireTask | None = None, *, grid: int = 64, patches: int = 4,
                 horizon: int = 1, device: torch.device | str = "cpu") -> None:
        self.surrogate = surrogate.eval()
        for p in self.surrogate.parameters():
            p.requires_grad_(False)
        self.task = task or FireTask()
        self.grid, self.patches, self.horizon = grid, patches, horizon
        self.device = torch.device(device)

        prev = data["states"][:, 0].astype(np.float32)            # (N, 1, H, W)
        prev = np.where(prev >= 0, prev, 0.0).astype(np.float32)
        self.x = torch.as_tensor(np.concatenate([prev, data["drivers"]], axis=1))  # (N, 12, H, W)
        self.next_mask = torch.as_tensor(data["states"][:, 1, 0].astype(np.float32))
        self.next_valid = torch.as_tensor(data["valid"][:, 1, 0])
        self.n = self.x.shape[0]

        self.ndvi = 1 + DRIVER_FEATURES.index("NDVI")
        self.pop = 1 + DRIVER_FEATURES.index("population")
        self.state: Tensor | None = None
        self._idx: Tensor | None = None
        self.t = 0

    # -- interface ----------------------------------------------------------- #
    @property
    def action_dim(self) -> int:
        return self.patches * self.patches

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.x.shape[1], self.grid, self.grid)

    def reset(self, batch: int = 1, generator: torch.Generator | None = None,
              idx: Tensor | None = None) -> Tensor:
        if idx is None:
            idx = torch.randint(0, self.n, (batch,), generator=generator)
        self._idx = idx
        self.state = self.x[idx].to(self.device)
        self.t = 0
        return self.state

    def set_state(self, state: Tensor) -> None:
        self.state = state.to(self.device)
        self.t = 0

    # -- action model -------------------------------------------------------- #
    # Intensity map. Squared rather than linear so that "no treatment" is
    # reachable inside the planner's saturation wall: with |pre-tanh mean| kept
    # under 1.5, a linear map floors every patch at 4.7% intensity and a 3%
    # budget cannot be met (seed 0: lambda pinned at its cap, policy smeared
    # thin, 4% burn reduction against greedy's 29%). Squared, the floor is 0.2%.
    @staticmethod
    def intensity(action: Tensor) -> Tensor:
        return ((action.clamp(-1, 1) + 1) / 2) ** 2

    @staticmethod
    def action_for(intensity: Tensor) -> Tensor:
        return 2 * intensity.clamp(0, 1).sqrt() - 1

    def treatment(self, action: Tensor) -> Tensor:
        """(B, K) in [-1, 1] -> (B, H, W) intensity in [0, 1], patchwise constant."""
        return self.treatment_from_intensity(self.intensity(action))

    def treatment_from_intensity(self, u: Tensor) -> Tensor:
        """(B, K) intensities in [0, 1] -> (B, H, W) field. Direct path for
        optimisers that work in intensity space (sqrt has no gradient at 0)."""
        u = u.view(-1, 1, self.patches, self.patches)
        return F.interpolate(u, size=(self.grid, self.grid), mode="nearest")[:, 0]

    def population(self, state: Tensor) -> Tensor:
        """Population density scaled to [0, 1] within each patch."""
        pop = state[:, self.pop].clamp(min=0)
        peak = pop.flatten(1).max(dim=1).values.clamp(min=1e-6)[:, None, None]
        return pop / peak

    def predict(self, state: Tensor, intensity: Tensor | None = None) -> Tensor:
        """Next-day ignition probability (B, H, W) under the action model."""
        x = state
        if intensity is not None:
            x = state.clone()
            x[:, self.ndvi] = x[:, self.ndvi] - self.task.fuel_removal * intensity
        p = torch.sigmoid(self.surrogate(x)[:, 0])
        if intensity is not None:
            p = p * (1 - self.task.block * intensity)
        return p

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        b = self.treatment(action)
        p = self.predict(self.state, b)
        weight = 1 + self.task.population_weight * self.population(self.state)
        reward = -self.task.reward_scale * (p * weight).flatten(1).mean(dim=1)
        cost = b.flatten(1).mean(dim=1)                       # treated fraction
        # Autoregressive continuation: the predicted mask becomes tomorrow's
        # day-t mask, drivers held fixed. Horizon 1 is the honest default.
        next_state = self.state.clone()
        next_state[:, 0] = p
        self.state = next_state
        self.t += 1
        return next_state, reward, cost, self.t >= self.horizon

    # -- reality check ------------------------------------------------------- #
    @torch.no_grad()
    def forecast_scores(self, idx: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """No-action forecast vs the observed next-day mask, for AUC-PR."""
        p = self.predict(self.x[idx].to(self.device)).cpu()
        valid = self.next_valid[idx]
        return p[valid].numpy(), (self.next_mask[idx][valid] > 0.5).numpy().astype(np.float32)


class FirebreakEnv(gym.Env):
    """Single-instance Gymnasium view of firebreak placement.

    `FireEnv` is batched and all-torch, which suits this repository's planner
    and nothing outside it. This is the `gymnasium.make` view: one fire at a
    time, numpy in and numpy out, with the `info["cost"]` convention the
    safe-RL baselines and OmniSafe both read.

    **Reward** is minus the population-weighted burned fraction the spread
    model predicts. **Cost** is the fraction of the patch treated, and the
    budget is a crew-day limit.

    Two things a user of this environment has to know, and they are the same
    two the portal states:

    * **The dynamics are a learned model, not physics.** There is no
      rate-of-spread equation anywhere in it. A U-Net trained on ~18k historic
      fire days predicts tomorrow's mask from today's plus eleven drivers, and
      the environment steps by feeding its own prediction back in. Errors
      compound in a way a conservation law would not allow.
    * **The counterfactual is unobservable.** No fire on record had a break
      cut on our instruction, so every reward here is what the model expects
      and not what happened. Beating `uniform_action` is the meaningful
      comparison; beating no-treatment only shows the treatment model does
      something, which it does by construction.
    """

    metadata = {"render_modes": []}

    def __init__(self, surrogate: nn.Module, data: "dict[str, np.ndarray]", *,
                 budget: float = 0.03, horizon: int = 3, grid: int = 64,
                 patches: int = 4, device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = FireEnv(surrogate, data,
                             FireTask(cost_limit=budget),
                             grid=grid, patches=patches, horizon=horizon,
                             device=self.device)
        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self._core.action_dim,),
                                       dtype=np.float32)
        self.episode_cost = 0.0

    def reset(self, *, seed: "int | None" = None,
              options: "dict | None" = None) -> "tuple[np.ndarray, dict]":
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        state = self._core.reset(batch=1, generator=self._generator)
        self.episode_cost = 0.0
        return state[0].detach().cpu().numpy().astype(np.float32), {"cost": 0.0}

    def step(self, action: "np.ndarray"
             ) -> "tuple[np.ndarray, float, bool, bool, dict]":
        act = torch.as_tensor(action, dtype=torch.float32,
                              device=self.device).view(1, -1).clamp(-1.0, 1.0)
        with torch.no_grad():
            state, reward, cost, done = self._core.step(act)
        self.episode_cost += float(cost[0])
        info = {"cost": float(cost[0]),
                "episode_cost": self.episode_cost,
                "cost_limit": self._core.task.cost_limit}
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        # A crew-day horizon is a time limit, not an absorbing state.
        return obs, float(reward[0]), False, bool(done), info

    @property
    def task(self) -> FireTask:
        return self._core.task

    @property
    def core(self) -> "FireEnv":
        return self._core

    #: One part per million, taken off every budgeted intensity below.
    #:
    #: An action is converted to a treated fraction by `intensity()`, which
    #: squares, and back by `action_for`, which takes a square root. In float32
    #: that round trip is not exact, so an intensity chosen to spend the budget
    #: precisely comes back a few parts in 1e8 over it -- enough for a
    #: feasibility test with a tight tolerance to call the budget violated.
    #: Both baselines here aim just inside instead. The margin is 1e-6 of the
    #: budget, which is six orders of magnitude below the precision any of
    #: these numbers are reported at, and it makes "feasible" mean feasible
    #: rather than feasible-up-to-the-reader's-tolerance.
    _MARGIN = 1.0 - 1e-6

    def _action_for_intensity(self, u: "np.ndarray") -> "np.ndarray":
        """Intensity per patch -> a legal action, aiming just inside the budget."""
        u = np.clip(np.asarray(u, dtype=np.float32) * self._MARGIN, 0.0, 1.0)
        return FireEnv.action_for(
            torch.as_tensor(u, dtype=torch.float32).view(1, -1))[0].cpu().numpy()

    def greedy_action(self, obs: "np.ndarray") -> "np.ndarray":
        """Put the day's share of the budget on the patch that is alight most.

        The strongest feasible policy known here, and it is barely stronger,
        which is the finding. At this environment's 3% budget, 12 episodes:

            uniform over patches and days        -31.90
            whole budget on day 1                -31.91
            whole budget on day 3                -31.88
            greedy onto the 8 hottest patches    -31.59
            greedy onto the 4 hottest            -31.36
            greedy onto the 2 hottest            -31.17
            THIS (greedy onto the 1 hottest)     -31.08

        Every one of those spends exactly 0.030 and is feasible in all 12
        episodes. The spread from worst to best is 0.83, or 2.6% of the return,
        so at 3% of the patch *where you cut barely matters*. The regime where
        it does matter is a larger budget: the gap between uniform and this
        widens to 2.06 at a 6% budget and 5.16 at 24%, then collapses again at
        48% when a single patch saturates at full intensity.

        Worth stating plainly, because "our policy beat uniform placement on
        Firebreak-v0" at the default budget can mean a 2.6% window at best, and
        an experiment that reports a larger margin than that at this budget is
        more likely to have a bug than a result.
        """
        core = self._core
        g, grid = core.patches, core.grid
        cell = grid // g
        # Channel 0 is the previous day's fire mask -- the same channel the
        # surrogate reads as "what is burning now".
        fire = np.asarray(obs)[0]
        score = fire.reshape(g, cell, g, cell).sum(axis=(1, 3)).ravel()
        share = core.task.cost_limit / max(core.horizon, 1)
        u = np.zeros(core.action_dim, dtype=np.float32)
        # Before anything is alight the score is flat and argmax would always
        # pick patch 0; that is the honest behaviour -- there is nothing to be
        # greedy about -- and it is what the measured numbers above include.
        pick = int(np.argmax(score))
        # The whole episode's share for every patch, concentrated on one, so
        # the spend matches `uniform_action` exactly rather than approximately.
        u[pick] = min(share * core.action_dim, 1.0)
        return self._action_for_intensity(u)

    def uniform_action(self, obs: "np.ndarray | None" = None) -> "np.ndarray":
        """Spread the budget evenly. `obs` is accepted and ignored.

        Every baseline across these four environments takes an observation and
        returns an action, whether or not it reads the observation, so a
        harness can call them all the same way. This one is open-loop.

        `cost_limit` is the budget for the whole episode -- `safe_rl.py` winds
        its multiplier on `episode_cost - cost_limit`, a sum over steps -- so a
        uniform policy that treats `cost_limit` of the patch on *every* step
        spends `horizon` times the budget. Divide by the horizon or the
        incumbent is infeasible, and an infeasible incumbent is not a baseline,
        it is a policy that was allowed to cheat.
        """
        k = self._core.action_dim
        share = self._core.task.cost_limit / max(self._core.horizon, 1)
        return self._action_for_intensity(np.full(k, share))


def register_envs() -> None:
    """Make `gymnasium.make("Firebreak-v0")` work.

    The entry point loads the held-out NDWS records and the trained spread
    model through `service.fire`, which caches both -- so constructing this is
    cheap after the first call and raises a named error, rather than silently
    producing a useless environment, when the dataset is absent.
    """
    from gymnasium.envs.registration import register, registry

    if "Firebreak-v0" in registry:
        return
    register(id="Firebreak-v0", entry_point="pspe.envs.fire_env:_make_default",
             max_episode_steps=None)


def _make_default(budget: float = 0.03, horizon: int = 3, **kw) -> FirebreakEnv:
    """Entry point for `gymnasium.make`, on the held-out NDWS records."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from service.fire import _model, _records          # noqa: PLC0415

    return FirebreakEnv(_model(), _records(), budget=budget,
                        horizon=horizon, **kw)
