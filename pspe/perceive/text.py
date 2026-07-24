"""Weak text supervision: caption synthesis and a frozen text encoder.

The proposal pairs imagery with *weak* text ("elevated levels in the north-east,
front advancing east"), not with dense labels. `describe_field` produces exactly
that from field statistics, and `WeakTextEncoder` embeds it.

The encoder's feature extractor is a frozen hashed bag-of-bigrams with a fixed
random projection - deterministic, download-free, and adequate for a contrastive
term whose job is alignment rather than language understanding. Pass
`hf_model=...` to swap in a real sentence encoder when one is available.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

Tensor = torch.Tensor

_DIRECTIONS = ["south-west", "south", "south-east", "west", "centre", "east",
               "north-west", "north", "north-east"]


def describe_field(field: Tensor, channel: int = 0) -> str:
    """One weak caption per field: intensity level, hot region, drift direction."""
    u = field[channel] if field.ndim == 3 else field
    grid = u.shape[-1]
    level = float(u.mean())
    peak = float(u.max())

    flat_idx = int(torch.argmax(u))
    py, px = divmod(flat_idx, grid)
    region = _DIRECTIONS[min(int(py / grid * 3), 2) * 3 + min(int(px / grid * 3), 2)]

    # Coarse drift proxy: centre of mass offset from the domain centre.
    weights = (u - u.min()).clamp_min(0)
    total = weights.sum().clamp_min(1e-6)
    lin = torch.linspace(0, 1, grid, device=u.device)
    cx = float((weights.sum(0) * lin).sum() / total)
    cy = float((weights.sum(1) * lin).sum() / total)
    drift_x = "east" if cx > 0.55 else "west" if cx < 0.45 else "steady"
    drift_y = "north" if cy > 0.55 else "south" if cy < 0.45 else "steady"

    intensity = "high" if peak > 0.7 else "moderate" if peak > 0.35 else "low"
    return (
        f"{intensity} concentration with peak in the {region}; "
        f"mean level {level:.2f}, peak {peak:.2f}; "
        f"mass drifting {drift_x} and {drift_y}"
    )


class WeakTextEncoder(nn.Module):
    """Frozen hashed features + a trainable projection into the shared space."""

    def __init__(self, feature_dim: int = 512, embed_dim: int = 128, seed: int = 0) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(feature_dim, feature_dim, generator=generator) / feature_dim**0.5
        self.register_buffer("frozen_projection", projection)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def _hash(self, texts: list[str]) -> Tensor:
        out = torch.zeros(len(texts), self.feature_dim)
        for i, text in enumerate(texts):
            tokens = text.lower().replace(";", " ").replace(",", " ").split()
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                out[i, int.from_bytes(digest, "big") % self.feature_dim] += 1.0
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, texts: list[str]) -> Tensor:
        feats = self._hash(texts).to(self.frozen_projection.device)
        feats = torch.tanh(feats @ self.frozen_projection)
        return self.head(feats)
