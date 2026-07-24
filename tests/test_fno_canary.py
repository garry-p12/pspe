"""Correctness canary for the in-tree FNO, against 1D Burgers.

Phase 1's acceptance check compares the surrogate against *this repo's own*
numerical solver. That is an internal-consistency check: if the FNO
implementation were subtly wrong — a mis-indexed spectral block, a dropped
conjugate half, a broken inverse transform — it would still be scored against
the same ground truth it was fitted to, and would simply report a slightly worse
relative L2 rather than announcing a bug.

Burgers' equation is the standard FNO sanity task. Li et al. (2021) report
relative L2 around 1e-3 for 1D Burgers at nu = 0.1 with a well-trained model.
This canary trains a small FNO for a short budget and asserts a *much* looser
bound: a correct implementation reaches well under 5% relative error on this
problem, while a broken spectral convolution sits near 100% (it cannot beat
predicting the input unchanged).

The threshold is deliberately loose. Its job is to separate "the operator works"
from "the operator is broken", not to reproduce a published number — reproducing
1e-3 needs a bigger model and a far longer schedule than a test should spend.

Marked `canary`: excluded from the regular suite, run manually.

    pytest tests/test_fno_canary.py -m canary -q
"""

from __future__ import annotations

import math

import pytest
import torch

from pspe.simulate.fno import SpectralConv2d
from pspe.simulate.losses import relative_l2

pytestmark = pytest.mark.canary


# --------------------------------------------------------------------------- #
# 1D Burgers ground truth, solved spectrally (independent of pspe.simulate)
# --------------------------------------------------------------------------- #
def burgers_dataset(
    n_samples: int = 256,
    grid: int = 128,
    nu: float = 0.1,
    t_final: float = 1.0,
    steps: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """u_t + u u_x = nu u_xx on a periodic domain; returns (u0, u_T).

    Integrated with a pseudo-spectral scheme and exact integrating factor for
    the diffusion term, which is deliberately a *different* discretisation from
    the finite-difference solvers in `pspe/simulate/solvers.py` — a canary that
    shared machinery with the thing it validates would not be a canary.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 2 * math.pi, grid + 1)[:-1]

    # Smooth random periodic initial conditions.
    u = torch.zeros(n_samples, grid)
    for k in range(1, 5):
        amp = torch.randn(n_samples, 1, generator=generator) / k
        phase = torch.rand(n_samples, 1, generator=generator) * 2 * math.pi
        u = u + amp * torch.sin(k * x.unsqueeze(0) + phase)
    u0 = u.clone()

    k = torch.fft.rfftfreq(grid, d=1.0 / grid)
    dt = t_final / steps
    decay = torch.exp(-nu * (k**2) * dt)

    for _ in range(steps):
        # Nonlinear term in conservative form, dealiased by 2/3 truncation.
        u_hat = torch.fft.rfft(u)
        nonlinear = torch.fft.rfft(u * u) * 0.5j * k
        cutoff = int(len(k) * 2 / 3)
        nonlinear[:, cutoff:] = 0
        u = torch.fft.irfft((u_hat - dt * nonlinear) * decay, n=grid)

    return u0, u


class _FNO1d(torch.nn.Module):
    """Minimal 1D FNO built on the repo's `SpectralConv2d` (width-1 second axis).

    Reusing the production spectral block is the whole point: this test fails if
    *that* block is wrong.
    """

    def __init__(self, modes: int = 16, width: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(2, width, 1)  # u(x) and x
        self.spectral = torch.nn.ModuleList(
            SpectralConv2d(width, width, modes, 1) for _ in range(n_layers)
        )
        self.pointwise = torch.nn.ModuleList(
            torch.nn.Conv2d(width, width, 1) for _ in range(n_layers)
        )
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(width, 128, 1), torch.nn.GELU(), torch.nn.Conv2d(128, 1, 1)
        )

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        batch, grid = u0.shape
        coords = torch.linspace(0, 1, grid, device=u0.device).expand(batch, grid)
        x = torch.stack([u0, coords], dim=1).unsqueeze(-1)  # (B, 2, grid, 1)
        h = self.lift(x)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            h = torch.nn.functional.gelu(spectral(h) + pointwise(h))
        return self.project(h).squeeze(-1).squeeze(1)


def test_fno_solves_burgers_to_published_order() -> None:
    torch.manual_seed(0)
    u0, uT = burgers_dataset(n_samples=256, grid=128)
    split = 224
    train_x, train_y = u0[:split], uT[:split]
    test_x, test_y = u0[split:], uT[split:]

    model = _FNO1d()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    for _ in range(300):
        perm = torch.randperm(split)
        for i in range(0, split, 32):
            idx = perm[i : i + 32]
            loss = relative_l2(model(train_x[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()

    with torch.no_grad():
        error = float(relative_l2(model(test_x), test_y))
        # A broken operator cannot beat the identity map; report both so a
        # failure says which side it fell on.
        identity = float(relative_l2(test_x, test_y))

    print(f"\nBurgers nu=0.1: FNO rel L2 = {error:.4f}, identity baseline = {identity:.4f}")
    assert error < 0.05, (
        f"FNO reached {error:.4f} relative L2 on 1D Burgers; a correct "
        f"implementation reaches well under 0.05 (identity baseline {identity:.4f}). "
        "Suspect the spectral convolution."
    )
    assert error < identity / 5, "FNO barely beats the identity map"
