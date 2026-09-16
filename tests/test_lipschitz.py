"""Assumption 1 (paper Sec. 4.2 / 5.1): a Lipschitz-bounded surrogate.

The paper asserts L_G <= 1 via spectral normalisation. These tests establish
what the code can actually guarantee (per-layer norms) and what it can only
measure (the network's Lipschitz constant), and that the measurement is right.
"""

from __future__ import annotations

import torch

from pspe.simulate.fno import FNO2d, SpectralConv2d

GRID = 16


def test_estimator_recovers_a_known_lipschitz_constant() -> None:
    """Power iteration on the Jacobian must return sigma_max for a linear map."""
    class Scale(torch.nn.Module):
        def __init__(self, k: float) -> None:
            super().__init__()
            self.k = k
            self.lipschitz = False
        def forward(self, s, c=None):
            return self.k * s
        estimate_lipschitz = FNO2d.estimate_lipschitz
        training = False
        def eval(self): return self
        def train(self, mode=True): return self

    x = torch.randn(2, 1, GRID, GRID)
    for k in (0.5, 2.0, 3.7):
        assert abs(Scale(k).estimate_lipschitz(x, iters=10) - k) < 1e-4


def test_fourier_projection_clips_per_mode_operator_norm() -> None:
    layer = SpectralConv2d(8, 8, 4, 4)
    with torch.no_grad():
        layer.weight_pos.mul_(50.0)                # blow up the norms
    assert layer.operator_norm() > 1.0
    worst_before = layer.project_operator_norm(1.0)
    assert worst_before > 1.0
    assert layer.operator_norm() <= 1.0 + 1e-5
    # Idempotent: a second projection changes nothing.
    assert abs(layer.project_operator_norm(1.0) - layer.operator_norm()) < 1e-5


def test_lipschitz_mode_swaps_groupnorm_and_normalises_convs() -> None:
    m = FNO2d(1, 1, modes=4, width=8, n_layers=2, lipschitz=True)
    assert all(isinstance(b.norm, torch.nn.Identity) for b in m.blocks)
    # spectral_norm parametrisation registers `parametrizations.weight`
    assert hasattr(m.lift, "parametrizations")
    assert all(hasattr(b.pointwise, "parametrizations") for b in m.blocks)


def test_default_fno_exceeds_one_and_lipschitz_mode_does_not_at_init() -> None:
    """The paper's assumption, checked rather than asserted.

    The default architecture is a residual stack with GroupNorm: its constant
    is not bounded by 1 and the measurement says so. Lipschitz mode brings the
    *initialised* network under 1; whether that survives training is what the
    experiment measures, not this test.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 1, GRID, GRID); c = torch.zeros(2, 1, GRID, GRID)
    plain = FNO2d(1, 1, modes=6, width=16, n_layers=4, lipschitz=False)
    lip = FNO2d(1, 1, modes=6, width=16, n_layers=4, lipschitz=True)
    lip.project_spectral_weights(1.0)
    assert plain.estimate_lipschitz(x, c, iters=30) > 1.0
    assert lip.estimate_lipschitz(x, c, iters=30) <= 1.05


def test_projection_runs_through_the_factory_and_forward_is_unchanged_in_shape() -> None:
    from pspe.simulate import make_surrogate
    m = make_surrogate("fno", 1, grid=GRID, lipschitz=True)
    x = torch.randn(2, 1, GRID, GRID)
    assert m(x).shape == x.shape
    assert m.project_spectral_weights(1.0) >= 0.0
