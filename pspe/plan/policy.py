"""Field-conditioned Gaussian policy over actuator amplitudes.

`rsample` (reparameterised) feeds the pathwise branch of the hybrid estimator;
`log_prob` on the same action feeds the likelihood-ratio branch. Both branches
must see the *same* sampled action, or the two gradient estimates are not
estimating the same expectation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

Tensor = torch.Tensor

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


class FieldEncoder(nn.Module):
    """Small strided CNN: (B, C, H, W) -> (B, feature_dim)."""

    def __init__(self, in_channels: int, width: int = 32, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 4, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(width * 2 * 4, feature_dim), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GaussianFieldPolicy(nn.Module):
    """Tanh-squashed diagonal Gaussian over actuator amplitudes in [-1, 1]."""

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        width: int = 32,
        feature_dim: int = 128,
        init_log_std: float = -0.7,
    ) -> None:
        super().__init__()
        self.encoder = FieldEncoder(in_channels, width, feature_dim)
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.action_dim = action_dim

    def distribution(self, state: Tensor) -> torch.distributions.Normal:
        features = self.encoder(state)
        mean = self.mean_head(features)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, state: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Return (action, log_prob). `action` is reparameterised (pathwise-ready)."""
        dist = self.distribution(state)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        # Change-of-variables correction for the tanh squash.
        log_prob = dist.log_prob(pre_tanh).sum(-1) - (
            2 * (math.log(2.0) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))
        ).sum(-1)
        return action, log_prob

    def log_prob_of(self, state: Tensor, action: Tensor) -> Tensor:
        """Log-density of an already-taken action; used by the LR branch and PPO."""
        dist = self.distribution(state)
        clamped = action.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(clamped)
        return dist.log_prob(pre_tanh).sum(-1) - (
            2 * (math.log(2.0) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))
        ).sum(-1)

    def entropy(self, state: Tensor) -> Tensor:
        return self.distribution(state).entropy().sum(-1)


class FieldCritic(nn.Module):
    """State-value head. Two heads: reward value and cost value (CMDP)."""

    def __init__(self, in_channels: int, width: int = 32, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = FieldEncoder(in_channels, width, feature_dim)
        self.value = nn.Linear(feature_dim, 1)
        self.cost_value = nn.Linear(feature_dim, 1)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(state)
        return self.value(features).squeeze(-1), self.cost_value(features).squeeze(-1)
