"""Perceive losses: field regression + contrastive alignment to weak text.

    L = w_reg * relative_L2(field_hat, field) + w_con * InfoNCE(image, text)

The contrastive term is what stops the decoder from collapsing to the dataset
mean: it forces the representation to carry which-scene-is-this information that
the regression head alone would not require.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..simulate.losses import relative_l2

Tensor = torch.Tensor


def info_nce(
    image_embed: Tensor, text_embed: Tensor, logit_scale: Tensor
) -> tuple[Tensor, float]:
    """Symmetric InfoNCE over an in-batch matrix of image/text pairs."""
    image_embed = F.normalize(image_embed, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * image_embed @ text_embed.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    accuracy = float((logits.argmax(dim=-1) == labels).float().mean())
    return loss, accuracy


def perceive_loss(
    field_pred: Tensor,
    field_true: Tensor,
    image_embed: Tensor,
    text_embed: Tensor,
    logit_scale: Tensor,
    w_regression: float = 1.0,
    w_contrastive: float = 0.2,
) -> tuple[Tensor, dict[str, float]]:
    regression = relative_l2(field_pred, field_true)
    mse = F.mse_loss(field_pred, field_true)
    contrastive, accuracy = info_nce(image_embed, text_embed, logit_scale)
    total = w_regression * regression + w_contrastive * contrastive
    return total, {
        "loss/total": float(total.detach()),
        "loss/regression": float(regression.detach()),
        "loss/mse": float(mse.detach()),
        "loss/contrastive": float(contrastive.detach()),
        "metric/retrieval_acc": accuracy,
    }
