"""Minimal LoRA injection for backbones that are not HuggingFace models.

For `transformers` backbones the repo uses `peft`. But Phases 3 and 4 must also
run with zero downloads (offline CI, free-tier boxes with no HF cache), and the
"backbone frozen, adapters only" acceptance check has to be exercised in that
path too - otherwise the check only ever runs when the big model is present.
This module is that path: the same low-rank reparameterisation, ~60 lines.

    W x  ->  W x + (alpha / r) * B (A x),   A, B trainable, W frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

Tensor = torch.Tensor


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable rank-`r` update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # B starts at zero so the adapted model is exactly the base model at init.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        update = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return out + self.scaling * update


def inject_lora(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_suffixes: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2", "q_proj", "v_proj"),
) -> int:
    """Replace matching `nn.Linear` children with `LoRALinear`. Returns the count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name.endswith(target_suffixes):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            replaced += 1
        else:
            replaced += inject_lora(child, r, alpha, target_suffixes)
    return replaced


def mark_only_lora_trainable(module: nn.Module) -> None:
    """Freeze everything except LoRA A/B matrices."""
    for name, param in module.named_parameters():
        param.requires_grad_("lora_a" in name or "lora_b" in name)


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in module.named_parameters() if "lora_a" in n or "lora_b" in n]
