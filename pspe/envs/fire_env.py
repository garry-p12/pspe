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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
