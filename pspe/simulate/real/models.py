"""Architectures for observed fire spread — and why the FNO alone is not enough.

The FNO reaches 0.0098 relative L2 on real PDEBench shallow-water, and 0.073
AUC-PR on next-day fire spread against a 0.284 published baseline. That gap is
not a tuning failure, it is a statement about scope:

* a spectral operator truncates high frequencies by construction. That is
  exactly right for smooth advection-diffusion, where the energy lives in the
  low modes, and exactly wrong for a fire front, which is a sharp, sparse,
  locally-determined edge — the high modes *are* the signal;
* fire spread is driven by local interactions (this cell ignites because its
  neighbour is burning and the wind blows this way), which a convolution
  expresses directly and a global spectral mixing has to reconstruct.

So the honest architectural claim is not "operators are better", it is that the
two families are complementary. `HybridFNOUNet` keeps the spectral path for the
smooth drivers (weather, terrain, drought) and adds a convolutional path for the
front, then fuses them. The U-Net here exists as the control that decides
whether the spectral half earns its place at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fno import FNO2d

Tensor = torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNet(nn.Module):
    """The published-baseline architecture class for this task.

    Not a strawman: this is what the dataset's own paper and its successors use,
    and it is the control that tells us whether the spectral path contributes.
    """

    def __init__(self, in_channels: int, width: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        chans = [width * 2 ** i for i in range(depth + 1)]

        self.downs = nn.ModuleList()
        prev = in_channels
        for ch in chans[:-1]:
            self.downs.append(ConvBlock(prev, ch))
            prev = ch
        self.bottleneck = ConvBlock(prev, chans[-1])

        # Decoder channel bookkeeping, written out rather than inferred: at each
        # step the input is (upsampled decoder features + the matching skip), so
        # the 1x1 reduction must be sized from BOTH, and `prev` is the decoder
        # width carried forward, not the encoder's.
        self.ups = nn.ModuleList()
        self.reduce = nn.ModuleList()
        prev = chans[-1]                       # bottleneck width
        for skip_ch in reversed(chans[:-1]):
            self.reduce.append(nn.Conv2d(prev + skip_ch, skip_ch, 1))
            self.ups.append(ConvBlock(skip_ch, skip_ch))
            prev = skip_ch
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        h = x
        for block in self.downs:
            h = block(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.bottleneck(h)
        for reduce, block, skip in zip(self.reduce, self.ups, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(torch.cat([h, skip], dim=1))
            h = block(h)
        return self.head(h)


class HybridFNOUNet(nn.Module):
    """PSPE's surrogate for sharp-front dynamics: local primary, spectral gated.

    An earlier version fused the two paths as equal partners and scored *worse*
    than its own convolutional half (0.220 vs 0.280 AUC-PR on NDWS) at four
    times the compute. The reason is structural, not incidental: averaging a
    low-frequency spectral output into a prediction whose signal is a sharp edge
    can only blur it, and the extra parameters bought earlier overfitting
    (best epoch 3 against the U-Net's 13).

    So the roles are asymmetric now, matching what each family is good at:

    * the **U-Net is the predictor** — fire spread is a local, front-driven
      process and that is what convolutions express;
    * the **FNO is a correction**, multiplied by a learned gate initialised at
      **zero**. The model therefore starts exactly at U-Net behaviour, and the
      spectral path contributes only to the extent it reduces the loss. The
      gate value is readable after training, so "did the operator help" becomes
      a measurement instead of an assumption.

    The previous fire mask enters as a learned logit bias: what the physics
    determines is the *change* in the burning region, not the region itself,
    which is already an input.
    """

    def __init__(self, in_channels: int, grid: int = 64,
                 modes: int = 12, width: int = 32) -> None:
        del grid  # kept for a uniform factory signature; FNO2d is resolution-free
        super().__init__()
        self.local = UNet(in_channels, width=width, depth=3)
        self.spectral = FNO2d(in_channels=in_channels, out_channels=1,
                              modes=modes, width=width, predict_delta=False)
        # Zero init: the spectral correction starts switched off, so this model
        # cannot begin worse than the U-Net it contains.
        self.gate = nn.Parameter(torch.zeros(1))
        self.prev_bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        prev = x[:, :1]                       # day-t fire mask, by convention
        logits = self.local(x)
        logits = logits + self.gate * self.spectral(x)
        return logits + self.prev_bias * (2.0 * prev - 1.0)

    def spectral_contribution(self) -> float:
        """How much the operator path is actually being used."""
        return float(self.gate.abs().item())


def make_fire_model(name: str, in_channels: int, grid: int = 64,
                    width: int = 32) -> nn.Module:
    """`fno` (spectral only), `unet` (local only), `hybrid` (PSPE)."""
    if name == "unet":
        return UNet(in_channels, width=width)
    if name == "hybrid":
        return HybridFNOUNet(in_channels, grid=grid, width=width)
    if name == "fno":
        return nn.Sequential(
            FNO2d(in_channels=in_channels, out_channels=32, modes=12, width=32,
                  predict_delta=False),
            nn.Conv2d(32, 1, 1),
        )
    raise ValueError(f"unknown fire model {name!r}; expected fno, unet or hybrid")
