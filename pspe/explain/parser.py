"""Frozen brief parser: text -> action distribution.

Frozen in the strict sense: no parameters, no training, no gradient. That is the
point. If the parser were learned jointly with the explainer, the pair could
agree on a private encoding and score itself faithful while saying nothing a
reader could act on. A fixed parser makes the faithfulness score a statement
about the text as a reader would decode it.

Parse rules:
    "actuator <i> at <x> <y> set to <a>"  -> mean[i] = a
    unmentioned actuators                 -> mean[i] = 0
    "confidence <level>"                  -> std for every component
"""

from __future__ import annotations

import re

import torch

from .brief import CONFIDENCE_LEVELS

Tensor = torch.Tensor

_ACTUATOR_RE = re.compile(
    r"actuator\s+(\d+)\s+at\s+[-+0-9.]+\s+[-+0-9.]+\s+set\s+to\s+([-+]?\d*\.?\d+)"
)
_CONFIDENCE_RE = re.compile(r"confidence\s+(\w+)")
_COST_RE = re.compile(r"predicted\s+cost\s+([-+]?\d*\.?\d+)")


class FrozenBriefParser:
    """Deterministic text -> Normal(mean, std) over actuator amplitudes."""

    def __init__(self, n_actuators: int, default_std: float = 0.4) -> None:
        self.n_actuators = n_actuators
        self.default_std = default_std

    def parse(self, brief: str) -> tuple[Tensor, Tensor, dict[str, float]]:
        mean = torch.zeros(self.n_actuators)
        mentioned = 0
        for match in _ACTUATOR_RE.finditer(brief):
            idx = int(match.group(1))
            if 0 <= idx < self.n_actuators:
                mean[idx] = float(match.group(2))
                mentioned += 1

        level = _CONFIDENCE_RE.search(brief)
        std_value = CONFIDENCE_LEVELS.get(
            level.group(1) if level else "", self.default_std
        )
        std = torch.full((self.n_actuators,), std_value)

        cost = _COST_RE.search(brief)
        extras = {
            "mentioned_actuators": float(mentioned),
            "stated_cost": float(cost.group(1)) if cost else float("nan"),
        }
        return mean, std, extras

    def distribution(self, brief: str) -> tuple[torch.distributions.Normal, dict[str, float]]:
        mean, std, extras = self.parse(brief)
        return torch.distributions.Normal(mean, std), extras

    def batch_distribution(
        self, briefs: list[str], device: torch.device | str = "cpu"
    ) -> tuple[torch.distributions.Normal, list[dict[str, float]]]:
        means, stds, extras = [], [], []
        for brief in briefs:
            mean, std, extra = self.parse(brief)
            means.append(mean)
            stds.append(std)
            extras.append(extra)
        return (
            torch.distributions.Normal(
                torch.stack(means).to(device), torch.stack(stds).to(device)
            ),
            extras,
        )
