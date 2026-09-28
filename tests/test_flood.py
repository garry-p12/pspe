"""Properties the flood solver must hold, including the one that caught a bug.

The mass test is not decoration. The first version of the solver gained 6.4%
volume on a dam break, because clamping a cell's depth at zero after an explicit
flux had over-drained it *creates* water. A solver that manufactures the very
quantity a constraint is declared on would have invalidated every margin result
built on it.
"""

from __future__ import annotations

import torch

from pspe.simulate.flood import (
    FloodConfig,
    FloodSolver,
    floodplain_terrain,
    levee_basis,
)

GRID = 48
DX = 480.0


def _closed(z: torch.Tensor) -> FloodSolver:
    return FloodSolver(z, FloodConfig(dx=DX, open_edges=()))


def test_lake_at_rest_does_not_move() -> None:
    """A flat free surface over uneven bed must stay flat: the C-property.

    This is the test that catches a sign error in the topography term, which no
    amount of "the flood looks plausible" would reveal.
    """
    z, _ = floodplain_terrain(GRID, dx=DX)
    solver = _closed(z)
    h0 = (float(z.min()) + 3.0 - z).clamp(min=0.0).unsqueeze(0)
    assert (h0 > 0).any(), "test scenario must actually be wet"

    h, qx, qy = h0.clone(), *solver.zeros_flux(1)
    for _ in range(100):
        h, qx, qy = solver.step(h, qx, qy, dt=30.0)

    assert torch.allclose(h, h0, atol=1e-6)
    wet = h0[0] > 1e-6
    assert float((h + z)[0][wet].std()) < 1e-5


def test_closed_domain_conserves_mass() -> None:
    """No rain, no inflow, no open edge: volume is invariant."""
    z, _ = floodplain_terrain(GRID, dx=DX)
    solver = _closed(z)
    h = torch.zeros(1, GRID, GRID)
    h[0, 2:6, 20:28] = 8.0
    v0 = float(h.sum())

    qx, qy = solver.zeros_flux(1)
    for _ in range(300):
        h, qx, qy = solver.step(h, qx, qy, dt=solver.adaptive_dt(h))

    assert abs(float(h.sum()) - v0) / v0 < 1e-5


def test_flux_limiter_prevents_negative_depth() -> None:
    """The limiter, not the clamp, is what keeps depth non-negative."""
    z, _ = floodplain_terrain(GRID, dx=DX)
    solver = _closed(z)
    h = torch.zeros(1, GRID, GRID)
    h[0, 2:6, 20:28] = 12.0
    qx, qy = solver.zeros_flux(1)
    for _ in range(200):
        h, qx, qy = solver.step(h, qx, qy, dt=solver.adaptive_dt(h))
        assert solver.last_clamped < 1e-5, "clamp had to rescue a negative depth"
        assert float(h.min()) >= 0.0


def test_open_edge_drains_and_closed_edge_does_not() -> None:
    z, _ = floodplain_terrain(GRID, dx=DX)
    h0 = torch.zeros(1, GRID, GRID)
    h0[0, 2:6, 20:28] = 8.0

    drained = {}
    for edges in ((), ("south",)):
        solver = FloodSolver(z, FloodConfig(dx=DX, open_edges=edges))
        h, qx, qy = h0.clone(), *solver.zeros_flux(1)
        for _ in range(800):
            h, qx, qy = solver.step(h, qx, qy, dt=solver.adaptive_dt(h))
        drained[edges] = float(h.sum())

    assert drained[("south",)] < 0.5 * drained[()]


def test_raising_the_bed_displaces_water_without_adding_any() -> None:
    """A levee is a gate, not a source: it must not change the water budget.

    This is the property that distinguishes the flood actuator from the retired
    `swe` one, which added an equal source everywhere.
    """
    z, _ = floodplain_terrain(GRID, dx=DX)
    masks = levee_basis(GRID, dx=DX)
    z_levee = z + 3.0 * masks[2]

    h0 = torch.zeros(1, GRID, GRID)
    h0[0, 2:6, 20:28] = 6.0
    volumes = []
    for bed in (z, z_levee):
        solver = FloodSolver(bed, FloodConfig(dx=DX, open_edges=()))
        h, qx, qy = h0.clone(), *solver.zeros_flux(1)
        for _ in range(300):
            h, qx, qy = solver.step(h, qx, qy, dt=solver.adaptive_dt(h))
        volumes.append(float(h.sum()))

    v0 = float(h0.sum())
    for v in volumes:
        assert abs(v - v0) / v0 < 1e-5


def test_levee_basis_shapes_and_range() -> None:
    masks = levee_basis(GRID, dx=DX)
    assert masks.shape[1:] == (GRID, GRID)
    assert float(masks.min()) >= 0.0
    assert float(masks.max()) <= 1.0 + 1e-6
    # Sites must be distinguishable, or the allocation problem is degenerate.
    flat = masks.flatten(1)
    corr = torch.corrcoef(flat)
    off = corr - torch.eye(masks.shape[0])
    assert float(off.max()) < 0.9


def test_exposure_is_a_normalised_weighting() -> None:
    _, exposure = floodplain_terrain(GRID, dx=DX)
    assert abs(float(exposure.sum()) - 1.0) < 1e-5
    assert float(exposure.min()) >= 0.0
