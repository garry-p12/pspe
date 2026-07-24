"""Surrogate training objectives.

Three terms, matching the proposal's Simulate module:

* data term        - relative L2 against the numerical ground truth;
* physics term     - the PDE residual of the surrogate's own transition,
                     differentiated through the model by autograd;
* rollout term     - multi-step consistency, which is what actually controls
                     error growth when the planner unrolls the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .solvers import PDETestbed

Tensor = torch.Tensor


def relative_l2(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Batch-mean relative L2 error, the proposal's surrogate-fidelity metric."""
    dims = tuple(range(1, pred.ndim))
    num = torch.linalg.vector_norm(pred - target, dim=dims)
    den = torch.linalg.vector_norm(target, dim=dims).clamp_min(eps)
    return (num / den).mean()


def physics_residual_loss(
    testbed: PDETestbed,
    state: Tensor,
    pred_next: Tensor,
    control: Tensor | None = None,
) -> Tensor:
    """Mean-squared PDE residual of the surrogate's predicted transition.

    `testbed.residual` builds the residual from differentiable finite-difference
    stencils, so this term backpropagates into the surrogate weights exactly
    like an autograd-derived residual would, without needing per-sample
    second-order graph construction on a 64x64 grid.
    """
    res = testbed.residual(state, pred_next, control)
    return (res**2).mean()


def rollout_consistency_loss(
    model: nn.Module,
    testbed: PDETestbed,
    traj: Tensor,
    controls: Tensor | None,
    horizon: int,
    physics_weight: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Unroll `horizon` steps from traj[:, 0] and score against the truth.

    Returns (data_loss, physics_loss, final_relative_l2). Horizons are short
    (proposal's compute budget); the graph is kept through the whole unroll,
    which is also what makes the surrogate usable for pathwise planning.
    """
    state = traj[:, 0]
    data_loss = state.new_zeros(())
    phys_loss = state.new_zeros(())
    last_rel = state.new_zeros(())

    for t in range(horizon):
        control = controls[:, t] if controls is not None else None
        pred = model(state, control)
        target = traj[:, t + 1]
        data_loss = data_loss + relative_l2(pred, target)
        if physics_weight > 0.0:
            phys_loss = phys_loss + physics_residual_loss(testbed, state, pred, control)
        last_rel = relative_l2(pred.detach(), target)
        state = pred

    return data_loss / horizon, phys_loss / max(horizon, 1), last_rel


@dataclass
class SurrogateLossWeights:
    data: float = 1.0
    physics: float = 0.1
    rollout: float = 1.0


def surrogate_loss(
    model: nn.Module,
    testbed: PDETestbed,
    traj: Tensor,
    controls: Tensor | None,
    horizon: int = 4,
    weights: SurrogateLossWeights | None = None,
    use_physics: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    """Total surrogate objective plus a dict of loggable components.

    `use_physics=False` is the proposal's physics-loss ablation switch.
    """
    weights = weights or SurrogateLossWeights()
    physics_weight = weights.physics if use_physics else 0.0

    # One-step term: the supervised signal that anchors the operator.
    one_step_pred = model(traj[:, 0], controls[:, 0] if controls is not None else None)
    one_step = relative_l2(one_step_pred, traj[:, 1])

    rollout, physics, final_rel = rollout_consistency_loss(
        model, testbed, traj, controls, horizon, physics_weight
    )

    total = weights.data * one_step + weights.rollout * rollout + physics_weight * physics
    components = {
        "loss/total": float(total.detach()),
        "loss/one_step": float(one_step.detach()),
        "loss/rollout": float(rollout.detach()),
        "loss/physics": float(physics.detach()),
        "metric/rel_l2_final": float(final_rel.detach()),
    }
    return total, components
