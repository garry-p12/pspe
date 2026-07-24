"""Neural-operator baselines: DeepONet and a GNOT-style attention operator.

Both are in-tree reimplementations sharing the FNO's interface
`forward(state, control) -> next_state`, so the Phase 1 training loop, the
physics residual, and the Phase 2 planner are identical across all three. That
is what makes the comparison controlled: only the operator body changes.

`make_surrogate` is the single construction point; it also exposes
`neuraloperator`'s reference FNO when that optional extra is installed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .fno import FNO2d

Tensor = torch.Tensor


def _coord_grid(x: Tensor) -> Tensor:
    batch, _, height, width = x.shape
    gy = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
    gx = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


class DeepONet2d(nn.Module):
    """Branch/trunk operator net, evaluated on the full grid.

    Branch encodes the input function (state + control) into `p` coefficients;
    trunk encodes query coordinates into `p` basis functions; the output is
    their inner product, per output channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid: int = 64,
        p: int = 64,
        width: int = 64,
        control_channels: int = 1,
        predict_delta: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.control_channels = control_channels
        self.p = p
        self.predict_delta = predict_delta
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels + control_channels, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, p * out_channels),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 + 4 * 2 * 2, width),  # coords + Fourier features
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, p),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    @staticmethod
    def _features(coords: Tensor) -> Tensor:
        feats = [coords]
        for freq in (1.0, 2.0, 4.0, 8.0):
            feats.append(torch.sin(2 * math.pi * freq * coords))
            feats.append(torch.cos(2 * math.pi * freq * coords))
        return torch.cat(feats, dim=-1)[..., : 2 + 4 * 2 * 2]

    def forward(self, state: Tensor, control: Tensor | None = None) -> Tensor:
        if control is None:
            control = torch.zeros(
                state.shape[0], self.control_channels, *state.shape[-2:],
                device=state.device, dtype=state.dtype,
            )
        batch, _, height, width = state.shape
        coeff = self.branch(torch.cat([state, control], dim=1))
        coeff = coeff.view(batch, self.out_channels, self.p)

        coords = _coord_grid(state).permute(0, 2, 3, 1).reshape(batch, height * width, 2)
        basis = self.trunk(self._features(coords))  # (B, HW, p)

        out = torch.einsum("bcp,bnp->bcn", coeff, basis) / math.sqrt(self.p)
        out = out.view(batch, self.out_channels, height, width) + self.bias.view(1, -1, 1, 1)
        return state + out if self.predict_delta else out


class GNOT2d(nn.Module):
    """GNOT-style operator transformer: patchified linear-attention blocks.

    The general operator transformer's defining pieces are heterogeneous-input
    cross-attention and a linear-attention kernel that keeps cost linear in the
    number of query points. Reproduced here at testbed scale: the state/control
    field is patchified into tokens, mixed by linear attention, and unpatchified.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid: int = 64,
        patch: int = 4,
        width: int = 96,
        n_layers: int = 4,
        n_heads: int = 4,
        control_channels: int = 1,
        predict_delta: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.control_channels = control_channels
        self.patch = patch
        self.width = width
        self.n_heads = n_heads
        self.predict_delta = predict_delta
        token_dim = (in_channels + control_channels + 2) * patch * patch
        self.embed = nn.Linear(token_dim, width)
        self.blocks = nn.ModuleList(
            _LinearAttentionBlock(width, n_heads) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_channels * patch * patch)

    def forward(self, state: Tensor, control: Tensor | None = None) -> Tensor:
        if control is None:
            control = torch.zeros(
                state.shape[0], self.control_channels, *state.shape[-2:],
                device=state.device, dtype=state.dtype,
            )
        batch, _, height, width = state.shape
        p = self.patch
        x = torch.cat([state, control, _coord_grid(state)], dim=1)
        # (B, C, H, W) -> (B, n_tokens, C*p*p)
        tokens = (
            x.unfold(2, p, p)
            .unfold(3, p, p)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch, (height // p) * (width // p), -1)
        )
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        out_tokens = self.head(self.norm(h))
        out = (
            out_tokens.view(batch, height // p, width // p, self.out_channels, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, self.out_channels, height, width)
        )
        return state + out if self.predict_delta else out


class _LinearAttentionBlock(nn.Module):
    """Softmax-free linear attention (elu+1 feature map) + MLP, pre-norm."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, n_tokens, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (batch, n_tokens, self.n_heads, self.head_dim)
        q = torch.nn.functional.elu(q.view(shape)) + 1.0
        k = torch.nn.functional.elu(k.view(shape)) + 1.0
        v = v.view(shape)
        kv = torch.einsum("bnhd,bnhe->bhde", k, v)
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, k.sum(dim=1)) + 1e-6)
        attn = torch.einsum("bnhd,bhde,bnh->bnhe", q, kv, z).reshape(batch, n_tokens, width)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


def make_surrogate(
    kind: str,
    n_channels: int,
    grid: int = 64,
    control_channels: int = 1,
    padded: bool = False,
    **kwargs: object,
) -> nn.Module:
    """Construct a surrogate by name.

    Names: "fno" (default, in-tree), "deeponet", "gnot", "fno_neuralop"
    (requires the optional `neuraloperator` extra).

    `padded=True` returns a channel-padded model that can be trained on one PDE
    family and rolled out on another (see `multifamily.py`); this is what makes
    the cross-family transfer gap measurable. `n_channels` is then ignored, the
    model always carrying the maximum arity.
    """
    kind = kind.lower()
    if padded:
        from .multifamily import make_padded_surrogate

        return make_padded_surrogate(kind, grid=grid, **kwargs)
    if kind == "fno":
        return FNO2d(
            in_channels=n_channels,
            out_channels=n_channels,
            control_channels=control_channels,
            **kwargs,  # type: ignore[arg-type]
        )
    if kind == "deeponet":
        return DeepONet2d(
            in_channels=n_channels,
            out_channels=n_channels,
            grid=grid,
            control_channels=control_channels,
            **kwargs,  # type: ignore[arg-type]
        )
    if kind == "gnot":
        return GNOT2d(
            in_channels=n_channels,
            out_channels=n_channels,
            grid=grid,
            control_channels=control_channels,
            **kwargs,  # type: ignore[arg-type]
        )
    if kind == "fno_neuralop":
        return _neuralop_fno(n_channels, control_channels, **kwargs)
    raise KeyError(f"unknown surrogate {kind!r}")


def _neuralop_fno(n_channels: int, control_channels: int, **kwargs: object) -> nn.Module:
    try:
        from neuralop.models import FNO  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "surrogate 'fno_neuralop' needs the optional extra: "
            'pip install -e ".[extras]"'
        ) from exc

    modes = int(kwargs.pop("modes", 12))  # type: ignore[arg-type]
    width = int(kwargs.pop("width", 32))  # type: ignore[arg-type]

    class _Wrapper(nn.Module):
        """Adapt neuraloperator's signature to (state, control) -> next_state."""

        def __init__(self) -> None:
            super().__init__()
            self.net = FNO(
                n_modes=(modes, modes),
                hidden_channels=width,
                in_channels=n_channels + control_channels,
                out_channels=n_channels,
            )
            self.control_channels = control_channels

        def forward(self, state: Tensor, control: Tensor | None = None) -> Tensor:
            if control is None:
                control = torch.zeros(
                    state.shape[0], self.control_channels, *state.shape[-2:],
                    device=state.device, dtype=state.dtype,
                )
            return state + self.net(torch.cat([state, control], dim=1))

    return _Wrapper()
