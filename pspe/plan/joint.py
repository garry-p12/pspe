"""Joint Simulate + Plan training — the paper's contribution #1, made concrete.

The paper says the modules train "under a single objective". Until this file,
nothing did: the surrogate was fitted to data, frozen, and handed to the
planner. Here the planner's Lagrangian loss also backpropagates into the
surrogate, so the dynamics model is shaped by what the decision needs.

The design is governed by one failure mode, which this repository has already
met once. A surrogate that receives the planner's gradient is being asked to
make the policy look good. Left to itself it will: predict lower cost and
higher reward than the true dynamics deliver, the dual relaxes, and the
constraint is violated in reality while satisfied in-model. That is the same
mechanism behind the 7.3% violation rate the probe + margin fix closed, now
with the surrogate as an active participant rather than a passive source of
error.

Three guards, each reported so a reader can see whether they held:

1. **A data anchor.** Every surrogate step also fits a minibatch of real
   transitions. The planning term is scaled by `beta`; the data term is not.
   `beta = 0` is exactly the disaggregated pipeline.
2. **Truth-side constraint enforcement.** Probe + margin stays on, so the dual
   is driven by real cost regardless of what the surrogate claims.
3. **Held-out surrogate accuracy, tracked through joint training.** If rel L2
   on data the planner never saw rises while return rises, the surrogate has
   been captured. That is reported as a number, not hidden in a loss curve.

An `anchor=False` arm exists to demonstrate the failure deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .trainer import HybridPlannerTrainer, RolloutBatch

Tensor = torch.Tensor


@dataclass
class JointConfig:
    # Planning-gradient share, as a fraction of the data gradient's norm. The
    # raw planning loss is O(1) and the data loss O(1e-4), so an unnormalised
    # weight is meaningless: at raw beta = 0.1 the planning term swamped the
    # anchor within 8 iterations and held-out error rose 10x. With this
    # scaling, beta = 0.1 means "the planner moves the surrogate one tenth as
    # hard as the data does", whatever the loss magnitudes happen to be.
    beta: float = 0.1
    lr_surrogate: float = 1e-4     # small: the surrogate is being nudged, not retrained
    data_batch: int = 32           # transitions per anchor step
    anchor: bool = True            # False -> the failure-mode arm
    grad_clip: float = 1.0
    heldout_fraction: float = 0.1  # slice of the dataset never used for anchoring
    eval_every: int = 20


class JointPlannerTrainer(HybridPlannerTrainer):
    """HybridPlannerTrainer whose surrogate also learns from the planning loss."""

    def __init__(self, *args, surrogate: nn.Module, dataset: dict[str, np.ndarray],
                 joint: JointConfig | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.joint = joint or JointConfig()
        self.surrogate = surrogate
        for p in self.surrogate.parameters():
            p.requires_grad_(True)
        self.surrogate_params = [p for p in self.surrogate.parameters() if p.requires_grad]
        self.opt_surrogate = torch.optim.Adam(self.surrogate_params, lr=self.joint.lr_surrogate)

        # Split transitions into anchor / held-out by trajectory, not by step,
        # so the held-out set is genuinely unseen dynamics rather than the next
        # frame of a trajectory the anchor already fitted.
        states = torch.as_tensor(dataset["states"], dtype=torch.float32)
        controls = torch.as_tensor(dataset["controls"], dtype=torch.float32)
        n = states.shape[0]
        n_hold = max(1, int(round(n * self.joint.heldout_fraction)))
        self._anchor = (states[n_hold:], controls[n_hold:])
        self._heldout = (states[:n_hold], controls[:n_hold])
        self._data_gen = torch.Generator().manual_seed(self.cfg.seed + 30_000)
        self._pending_plan_grads: list[Tensor | None] | None = None
        self.surrogate_rel_l2_history: list[tuple[int, float]] = []
        # A FIXED held-out batch, so before/after comparisons measure the
        # surrogate and not the draw. A frozen surrogate must score identically.
        self._heldout_batch = self._sample_transitions(self._heldout, 64)

    # -- data anchor ----------------------------------------------------------- #
    def _sample_transitions(self, split: tuple[Tensor, Tensor], batch: int) -> tuple[Tensor, Tensor, Tensor]:
        states, controls = split
        n, t = controls.shape[0], controls.shape[1]
        idx_n = torch.randint(0, n, (batch,), generator=self._data_gen)
        idx_t = torch.randint(0, t, (batch,), generator=self._data_gen)
        u = states[idx_n, idx_t].to(self.device)
        a = controls[idx_n, idx_t].to(self.device)
        u_next = states[idx_n, idx_t + 1].to(self.device)
        return u, a, u_next

    def _data_loss(self, u: Tensor, a: Tensor, u_next: Tensor) -> Tensor:
        pred = self.surrogate(u, a)
        return ((pred - u_next) ** 2).mean() / ((u_next ** 2).mean() + 1e-8)

    @torch.no_grad()
    def surrogate_rel_l2(self) -> float:
        """One-step relative L2 on a fixed held-out batch."""
        self.surrogate.eval()
        u, a, u_next = self._heldout_batch
        pred = self.surrogate(u, a)
        rel = (pred - u_next).flatten(1).norm(dim=1) / (u_next.flatten(1).norm(dim=1) + 1e-8)
        self.surrogate.train()
        return float(rel.mean())

    # -- hooks ----------------------------------------------------------------- #
    def _before_policy_step(self, batch: RolloutBatch, pathwise: Tensor,
                            likelihood: Tensor) -> dict[str, float]:
        # Gradient of the planning loss w.r.t. the *surrogate*, taken while the
        # rollout graph is still alive. Stored, applied after the policy moves.
        if self.joint.beta <= 0.0:
            self._pending_plan_grads = None
            return {}
        grads = torch.autograd.grad(
            pathwise.mean(), self.surrogate_params, retain_graph=True, allow_unused=True
        )
        self._pending_plan_grads = [g.detach() if g is not None else None for g in grads]
        # abs(): the Fourier weights are complex, and g**2 of a complex tensor
        # is not a magnitude.
        norm = float(torch.sqrt(sum((g.abs() ** 2).sum() for g in self._pending_plan_grads
                                    if g is not None)))
        return {"joint/plan_grad_norm": norm}

    def _after_policy_step(self, it: int) -> dict[str, float]:
        record: dict[str, float] = {}
        self.opt_surrogate.zero_grad(set_to_none=True)

        data_norm = None
        if self.joint.anchor:
            u, a, u_next = self._sample_transitions(self._anchor, self.joint.data_batch)
            data_loss = self._data_loss(u, a, u_next)
            data_loss.backward()
            record["joint/data_loss"] = float(data_loss.detach())
            data_norm = float(torch.sqrt(sum(
                (p.grad.abs() ** 2).sum() for p in self.surrogate_params if p.grad is not None
            )))

        if self._pending_plan_grads is not None:
            plan_norm = float(torch.sqrt(sum(
                (g.abs() ** 2).sum() for g in self._pending_plan_grads if g is not None
            )))
            # Scale the planning gradient to `beta` x the data gradient's norm.
            # Without an anchor there is no data norm to scale against, so the
            # unanchored arm applies the raw planning gradient at `beta` — its
            # job is to show the failure, not to be tuned around it.
            scale = (self.joint.beta * data_norm / max(plan_norm, 1e-12)
                     if data_norm is not None else self.joint.beta)
            record["joint/plan_to_data_ratio"] = scale * plan_norm / max(data_norm or plan_norm, 1e-12)
            for p, g in zip(self.surrogate_params, self._pending_plan_grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = scale * g
                else:
                    p.grad.add_(g, alpha=scale)
            self._pending_plan_grads = None

        if any(p.grad is not None for p in self.surrogate_params):
            torch.nn.utils.clip_grad_norm_(self.surrogate_params, self.joint.grad_clip)
            self.opt_surrogate.step()
            if getattr(self.surrogate, "lipschitz", False):
                self.surrogate.project_spectral_weights(1.0)

        if it % self.joint.eval_every == 0:
            rel = self.surrogate_rel_l2()
            self.surrogate_rel_l2_history.append((it, rel))
            record["joint/surrogate_rel_l2_heldout"] = rel
        return record

    def train(self) -> dict[str, float]:
        before = self.surrogate_rel_l2()
        summary = super().train()
        after = self.surrogate_rel_l2()
        summary.update({
            "joint/beta": self.joint.beta,
            "joint/anchor": float(self.joint.anchor),
            "joint/surrogate_rel_l2_before": before,
            "joint/surrogate_rel_l2_after": after,
            # The capture diagnostic: positive means the surrogate got worse on
            # data it never saw while serving the planner.
            "joint/surrogate_drift": after - before,
        })
        return summary
