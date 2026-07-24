"""Hybrid pathwise / likelihood-ratio gradient estimator with adaptive mixing.

The proposal's Plan module combines two estimators of the same policy gradient:

* pathwise (reparameterisation) - differentiates the objective through the
  differentiable surrogate. Low variance, but biased by surrogate error and
  prone to exploding gradients over long unrolls of stiff dynamics;
* likelihood ratio (score function) - unbiased, model-free, high variance.

The mixed estimator is  g(a) = a * g_pw + (1 - a) * g_lr.

The adaptive coefficient is the variance-minimising combination of two
correlated estimators of the same quantity:

    a* = (V_lr - Cov) / (V_pw + V_lr - 2 * Cov),   clipped to [0, 1]

with V and Cov estimated by splitting the batch into folds, computing each
estimator on each fold, and taking the trace of the empirical (co)variance over
the flattened parameter gradient. Setting `adaptive=False` pins `a` to a fixed
value - the proposal's fixed-vs-adaptive-mixing ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

Tensor = torch.Tensor
Params = Sequence[torch.nn.Parameter]
LossFn = Callable[[Tensor], Tensor]


def _flat_grad(
    loss: Tensor, params: Params, retain_graph: bool = True
) -> Tensor:
    """Flattened gradient of `loss` w.r.t. `params`; missing grads count as zero."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True, create_graph=False
    )
    flat = [
        torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
        for p, g in zip(params, grads)
    ]
    return torch.cat(flat)


def _assign_flat_grad(params: Params, flat: Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n


@dataclass
class HybridStats:
    alpha: float = 0.5
    var_pathwise: float = 0.0
    var_likelihood: float = 0.0
    covariance: float = 0.0
    grad_norm_pathwise: float = 0.0
    grad_norm_likelihood: float = 0.0
    grad_norm_mixed: float = 0.0
    history: list[float] = field(default_factory=list)


class HybridGradientEstimator:
    """Mixes two per-sample loss functions into one gradient on `params`."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        adaptive: bool = True,
        alpha_init: float = 0.5,
        alpha_ema: float = 0.9,
        n_folds: int = 4,
        estimate_every: int = 5,
        grad_clip: float | None = 10.0,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.adaptive = adaptive
        self.alpha = alpha_init
        self.alpha_ema = alpha_ema
        self.n_folds = n_folds
        self.estimate_every = estimate_every
        self.grad_clip = grad_clip
        self._step = 0
        self.stats = HybridStats(alpha=alpha_init)

    # -- variance-optimal mixing ------------------------------------------- #
    def _estimate_alpha(
        self, per_sample_pw: Tensor, per_sample_lr: Tensor
    ) -> tuple[float, float, float, float]:
        batch = per_sample_pw.shape[0]
        folds = min(self.n_folds, batch)
        if folds < 2:
            return self.alpha, 0.0, 0.0, 0.0

        chunks_pw = torch.chunk(per_sample_pw, folds)
        chunks_lr = torch.chunk(per_sample_lr, folds)
        g_pw = torch.stack([_flat_grad(c.mean(), self.params) for c in chunks_pw])
        g_lr = torch.stack([_flat_grad(c.mean(), self.params) for c in chunks_lr])

        dev_pw = g_pw - g_pw.mean(0, keepdim=True)
        dev_lr = g_lr - g_lr.mean(0, keepdim=True)
        denom = max(folds - 1, 1)
        var_pw = float((dev_pw**2).sum() / denom)
        var_lr = float((dev_lr**2).sum() / denom)
        cov = float((dev_pw * dev_lr).sum() / denom)

        spread = var_pw + var_lr - 2 * cov
        if spread <= 1e-12:
            alpha = self.alpha
        else:
            alpha = (var_lr - cov) / spread
        alpha = float(min(max(alpha, 0.0), 1.0))
        return alpha, var_pw, var_lr, cov

    # -- main entry point --------------------------------------------------- #
    def step(
        self,
        per_sample_pathwise: Tensor,
        per_sample_likelihood: Tensor,
    ) -> HybridStats:
        """Set `.grad` on `params` to the mixed estimate; return diagnostics.

        Both inputs are per-sample *losses* (minimise), shape (B,), sharing the
        same sampled actions.
        """
        self._step += 1
        if self.adaptive and self._step % self.estimate_every == 1:
            alpha, var_pw, var_lr, cov = self._estimate_alpha(
                per_sample_pathwise, per_sample_likelihood
            )
            # EMA-smoothed: the fold-wise variance estimate is itself noisy, and
            # an alpha that jumps every step destabilises the policy update.
            self.alpha = self.alpha_ema * self.alpha + (1 - self.alpha_ema) * alpha
            self.stats.var_pathwise = var_pw
            self.stats.var_likelihood = var_lr
            self.stats.covariance = cov

        g_pw = _flat_grad(per_sample_pathwise.mean(), self.params)
        g_lr = _flat_grad(per_sample_likelihood.mean(), self.params, retain_graph=False)
        mixed = self.alpha * g_pw + (1.0 - self.alpha) * g_lr

        if self.grad_clip is not None:
            norm = mixed.norm()
            if norm > self.grad_clip:
                mixed = mixed * (self.grad_clip / (norm + 1e-8))

        _assign_flat_grad(self.params, mixed)

        self.stats.alpha = self.alpha
        self.stats.grad_norm_pathwise = float(g_pw.norm())
        self.stats.grad_norm_likelihood = float(g_lr.norm())
        self.stats.grad_norm_mixed = float(mixed.norm())
        self.stats.history.append(self.alpha)
        return self.stats
