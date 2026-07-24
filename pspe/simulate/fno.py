"""Reduced-order Fourier Neural Operator surrogate (plain PyTorch).

Written in-tree rather than pulled from `neuraloperator` so that Phase 1 has no
download or build dependency; `neuraloperator`'s FNO is a drop-in alternative
via `pip install -e ".[extras]"` and `make_surrogate("fno_neuralop", ...)`.

"Reduced-order" in the proposal's sense means the spectral truncation: only the
lowest `modes` Fourier coefficients per axis are kept, so the surrogate's state
lives on a low-dimensional manifold and both memory and FLOPs scale with
`modes^2` rather than `grid^2`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

Tensor = torch.Tensor


class SpectralConv2d(nn.Module):
    """Truncated-spectrum convolution: R-linear mixing of low Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        # Two blocks: positive and negative x-wavenumbers (rfft2 keeps y >= 0 only).
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    @staticmethod
    def _mul(inp: Tensor, weight: Tensor) -> Tensor:
        # (B, Cin, X, Y) x (Cin, Cout, X, Y) -> (B, Cout, X, Y)
        return torch.einsum("bixy,ioxy->boxy", inp, weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        in_dtype = x.dtype
        # The spectral path runs in float32 regardless of autocast. Under CUDA
        # mixed precision the input is float16, and rfft2 of a half tensor
        # produces ComplexHalf (complex32) — which has no baddbmm/einsum kernel
        # on CUDA ("baddbmm_cuda not implemented for ComplexHalf") and is
        # numerically poor for spectral convolutions anyway. Casting to float32
        # here keeps AMP for the rest of the network while the FFT stays stable.
        work = x.float()
        # MPS has no complex FFT kernel; fall back to CPU for the spectral block.
        needs_cpu = work.device.type == "mps"
        if needs_cpu:
            work = work.cpu()

        x_ft = torch.fft.rfft2(work, norm="ortho")
        mx = min(self.modes_x, height // 2)
        my = min(self.modes_y, width // 2 + 1)

        out_ft = torch.zeros(
            batch, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=work.device,
        )
        # weight_pos / weight_neg are cfloat (float32 complex); keep them so.
        w_pos = self.weight_pos.to(work.device)[:, :, :mx, :my]
        w_neg = self.weight_neg.to(work.device)[:, :, :mx, :my]
        out_ft[:, :, :mx, :my] = self._mul(x_ft[:, :, :mx, :my], w_pos)
        out_ft[:, :, -mx:, :my] = self._mul(x_ft[:, :, -mx:, :my], w_neg)

        out = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        out = out.to(x.device) if needs_cpu else out
        # Back to the surrounding autocast dtype so the residual add is consistent.
        return out.to(in_dtype)


class FNOBlock(nn.Module):
    """Spectral branch + pointwise branch + residual, the standard FNO layer."""

    def __init__(self, width: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)

    def forward(self, x: Tensor) -> Tensor:
        h = self.spectral(x) + self.pointwise(x)
        return x + F.gelu(self.norm(h))


class FNO2d(nn.Module):
    """Maps (state, control) at time t to the state increment at t + dt.

    Predicting the increment rather than the next state matters for the physics
    residual: the residual is (u_next - u) / dt - rhs(...), so a model that
    outputs the increment directly is learning the discrete time derivative.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        control_channels: int = 1,
        predict_delta: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.control_channels = control_channels
        self.predict_delta = predict_delta
        # Gradient checkpointing: recompute each FNO block in the backward pass
        # instead of storing its activations. Trades ~30% compute for a large
        # activation-memory saving, which is what keeps a multi-step rollout at
        # 128^2 inside a commodity GPU (paper Section 7.5). Off by default since
        # it is pure overhead at the 64^2 default resolution.
        self.use_checkpoint = use_checkpoint
        # +2 for the (x, y) coordinate grid appended to every input.
        self.lift = nn.Conv2d(in_channels + control_channels + 2, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes, modes) for _ in range(n_layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * width, out_channels, kernel_size=1),
        )

    def _grid(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        gy = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        gx = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1)

    def forward(self, state: Tensor, control: Tensor | None = None) -> Tensor:
        if control is None:
            control = torch.zeros(
                state.shape[0], self.control_channels, *state.shape[-2:],
                device=state.device, dtype=state.dtype,
            )
        x = torch.cat([state, control, self._grid(state)], dim=1)
        h = self.lift(x)
        for block in self.blocks:
            if self.use_checkpoint and h.requires_grad:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        out = self.project(h)
        return state + out if self.predict_delta else out
