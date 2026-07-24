"""Rollout helpers and the surrogate-fidelity metric.

`surrogate_fidelity` is the number Phase 1's acceptance check reports: relative
L2 of an H-step surrogate rollout against the numerical ground truth, plus the
per-step error curve that shows how fast error compounds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import relative_l2
from .solvers import PDETestbed

Tensor = torch.Tensor


@torch.no_grad()
def rollout_truth(
    testbed: PDETestbed, u0: Tensor, controls: Tensor | None, steps: int
) -> Tensor:
    """Numerical ground-truth trajectory under a given control sequence."""
    traj = [u0]
    state = u0
    for t in range(steps):
        state = testbed.step(state, controls[:, t] if controls is not None else None)
        traj.append(state)
    return torch.stack(traj, dim=1)


def rollout_surrogate(
    model: nn.Module,
    u0: Tensor,
    controls: Tensor | None,
    steps: int,
    detach: bool = False,
) -> Tensor:
    """Surrogate trajectory. `detach=False` keeps the graph for pathwise planning."""
    traj = [u0]
    state = u0
    for t in range(steps):
        state = model(state, controls[:, t] if controls is not None else None)
        if detach:
            state = state.detach()
        traj.append(state)
    return torch.stack(traj, dim=1)


@torch.no_grad()
def surrogate_fidelity(
    model: nn.Module,
    testbed: PDETestbed,
    u0: Tensor,
    controls: Tensor | None,
    steps: int,
) -> dict[str, float | list[float]]:
    """Relative L2 per step and at the horizon, surrogate vs numerical truth."""
    model.eval()
    truth = rollout_truth(testbed, u0, controls, steps)
    pred = rollout_surrogate(model, u0, controls, steps, detach=True)
    per_step = [float(relative_l2(pred[:, t], truth[:, t])) for t in range(1, steps + 1)]
    return {
        "rel_l2_1step": per_step[0],
        "rel_l2_final": per_step[-1],
        "rel_l2_mean": sum(per_step) / len(per_step),
        "rel_l2_per_step": per_step,
        "horizon": steps,
    }
