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

    @torch.no_grad()
    def project_operator_norm(self, max_norm: float = 1.0) -> float:
        """Clip each mode's (Cin x Cout) mixing matrix to spectral norm <= max_norm.

        Under the orthonormal FFT this layer is block-diagonal across modes, so
        its operator norm is exactly the largest per-mode spectral norm. Clipping
        per mode is therefore the tight constraint, not a proxy. Returns the
        largest norm seen before clipping, so training logs show how binding
        the constraint is.
        """
        worst = 0.0
        for weight in (self.weight_pos, self.weight_neg):
            # (Cin, Cout, X, Y) -> (X*Y, Cin, Cout): one matrix per mode
            mats = weight.permute(2, 3, 0, 1).reshape(-1, self.in_channels, self.out_channels)
            # SVD on CPU: MPS has no complex SVD kernel and these matrices are
            # (modes^2) x width x width — small enough that the copy is free.
            norms = torch.linalg.matrix_norm(mats.detach().cpu(), ord=2).to(mats.device)
            worst = max(worst, float(norms.max()))
            scale = torch.clamp(max_norm / norms.clamp(min=1e-12), max=1.0)
            mats.mul_(scale[:, None, None].to(mats.dtype))
        return worst

    def operator_norm(self) -> float:
        """Current operator norm (max per-mode spectral norm), without clipping."""
        with torch.no_grad():
            worst = 0.0
            for weight in (self.weight_pos, self.weight_neg):
                mats = weight.permute(2, 3, 0, 1).reshape(-1, self.in_channels, self.out_channels)
                worst = max(worst, float(torch.linalg.matrix_norm(mats.detach().cpu(), ord=2).max()))
            return worst

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
    """Spectral branch + pointwise branch + residual, the standard FNO layer.

    `lipschitz=True` is the paper's Section 4.2 / Assumption 1 mode (Miyato et
    al. spectral normalisation), and it changes two things besides normalising
    the 1x1 conv:

    * GroupNorm is dropped. Normalisation layers are not Lipschitz — they
      rescale a small-magnitude input by an arbitrary factor — so no bound
      survives them. The block keeps its residual structure without it.
    * The branch is scaled by `residual_scale`. A residual block's constant is
      1 + L_branch, so even with both branches at norm <= 1 the block is
      2-Lipschitz, and n stacked blocks are 2^n. The scale is what makes a
      near-1 constant *possible*; `estimate_lipschitz` is what says whether it
      was *achieved*. The two must not be confused: the paper's L_G <= 1 is a
      claim about the trained network, and only the measurement can support it.
    """

    def __init__(self, width: int, modes_x: int, modes_y: int,
                 lipschitz: bool = False, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.lipschitz = lipschitz
        self.residual_scale = residual_scale if lipschitz else 1.0
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        pointwise = nn.Conv2d(width, width, kernel_size=1)
        if lipschitz:
            pointwise = torch.nn.utils.parametrizations.spectral_norm(pointwise)
        self.pointwise = pointwise
        self.norm = nn.Identity() if lipschitz else nn.GroupNorm(
            num_groups=min(8, width), num_channels=width
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.spectral(x) + self.pointwise(x)
        return x + self.residual_scale * F.gelu(self.norm(h))


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
        lipschitz: bool = False,
    ) -> None:
        super().__init__()
        self.lipschitz = lipschitz
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
        lift = nn.Conv2d(in_channels + control_channels + 2, width, kernel_size=1)
        proj_in = nn.Conv2d(width, 2 * width, kernel_size=1)
        proj_out = nn.Conv2d(2 * width, out_channels, kernel_size=1)
        if lipschitz:
            sn = torch.nn.utils.parametrizations.spectral_norm
            lift, proj_in, proj_out = sn(lift), sn(proj_in), sn(proj_out)
        self.lift = lift
        self.blocks = nn.ModuleList(
            FNOBlock(width, modes, modes, lipschitz=lipschitz) for _ in range(n_layers)
        )
        self.project = nn.Sequential(proj_in, nn.GELU(), proj_out)

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

    @torch.no_grad()
    def project_spectral_weights(self, max_norm: float = 1.0) -> float:
        """Clip every Fourier layer's per-mode operator norm. Call after each
        optimiser step in Lipschitz mode; the 1x1 convs are handled by their
        spectral_norm parametrisation, but complex Fourier weights are not
        covered by that utility and need the explicit projection."""
        return max(block.spectral.project_operator_norm(max_norm) for block in self.blocks)

    def estimate_lipschitz(self, state: Tensor, control: Tensor | None = None,
                           iters: int = 20) -> float:
        """Local Lipschitz constant of state -> next state, by power iteration
        on the Jacobian at `state`.

        This is the number Assumption 1 is about, measured rather than
        assumed. Power iteration on J^T J converges to the largest singular
        value of the Jacobian; averaged over sampled states it is the local
        constant the Prop. 1 recursion actually uses. Global sup is not
        computable; this is the honest proxy, and the docstring says so.
        """
        was_training = self.training
        self.eval()
        x = state.detach().clone().requires_grad_(True)
        v = torch.randn_like(x)
        v = v / v.norm()
        sigma = 0.0
        f = lambda s: self.forward(s, control)  # noqa: E731
        x0 = x.detach()
        for _ in range(iters):
            _, jv = torch.func.jvp(f, (x0,), (v,))          # J v
            _, vjp_fn = torch.func.vjp(f, x0)
            jtjv = vjp_fn(jv)[0]                            # J^T J v
            # Rayleigh quotient of J^T J at the current v -> sigma_max^2.
            with torch.no_grad():
                sigma = float(torch.sqrt((v * jtjv).sum().abs() / (v * v).sum()))
                v = jtjv / max(float(jtjv.norm()), 1e-12)
        if was_training:
            self.train()
        return sigma
