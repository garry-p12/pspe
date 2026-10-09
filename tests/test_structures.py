"""Floodgates and pumps: the properties that separate a model from a prop.

Both measures were added because the planner could otherwise only offer a
levee, and a levee is a bed perturbation -- it keeps water OUT of a perimeter
and can do nothing about water already inside one. Section 5.9's pluvial result
is the case with no instrument: no levee placement helped, because a quarter of
the rain fell inside the area being defended.

The tests that matter here are the ones a cosmetic implementation would fail.
A constant subtracted from depth would pass "the pump removes water". It would
not pass mass conservation, it would not refuse to lift below its own sump, and
it would not shut when the river rises.
"""

from __future__ import annotations

import torch

from pspe.simulate.flood import FloodConfig, FloodSolver, Structures

DX = 50.0
N = 24


def _flat(h0: float = 0.0, bed: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.full((1, N, N), bed)
    h = torch.full((1, N, N), h0)
    return z, h


def _bowls(a: tuple[int, int], b: tuple[int, int], wall: float = 20.0
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two isolated pits in a high plateau, with nothing but a structure
    between them.

    A flat bed does not test a structure, it tests overland spreading. Three
    metres dropped on one cell of a 24x24 flat domain settles to 5 mm
    everywhere in a few hundred steps, so "the level fell" says nothing about
    whether the gate or the pump did it -- measured, the first version of
    these tests read 0.015 m in the source cell and called the sill broken.
    Walling each cell in means the only route out is the one under test.
    """
    z = torch.full((1, N, N), wall)
    z[0, a[0], a[1]] = 0.0
    z[0, b[0], b[1]] = 0.0
    h = torch.zeros(1, N, N)
    return z, h


def _closed(z: torch.Tensor) -> FloodSolver:
    return FloodSolver(z, FloodConfig(dx=DX, open_edges=()))


def _gate(inside: tuple[int, int], outside: tuple[int, int],
          area: float = 4.0, invert: float = 0.0,
          open_: bool = True) -> Structures:
    return Structures(
        gate_in=torch.tensor([inside]),
        gate_out=torch.tensor([outside]),
        gate_area_m2=torch.tensor([area]),
        gate_invert_m=torch.tensor([invert]),
        gate_open=torch.tensor([[open_]]),
    )


def _pump(inside: tuple[int, int], outside: tuple[int, int] | None,
          q: float = 5.0, sump: float = 0.0,
          on: float = 0.05, off: float = 0.01) -> Structures:
    return Structures(
        pump_in=torch.tensor([inside]),
        pump_out=None if outside is None else torch.tensor([outside]),
        pump_q_m3s=torch.tensor([[q]]),
        pump_sump_m=torch.tensor([sump]),
        pump_on_m=torch.tensor([on]),
        pump_off_m=torch.tensor([off]),
    )


# ---- the gate ------------------------------------------------------------

def test_gate_drains_when_the_inside_is_higher() -> None:
    """The condition a floodgate is built for: let the catchment out."""
    z, h = _bowls((5, 5), (5, 18))
    h[0, 5, 5] = 3.0                       # inside, ponded
    solver = _closed(z)
    solver.structures = _gate((5, 5), (5, 18))
    qx, qy = solver.zeros_flux(1)
    before_out = float(h[0, 5, 18])
    for _ in range(40):
        h, qx, qy = solver.step(h, qx, qy, dt=30.0)
    assert float(h[0, 5, 18]) > before_out + 1e-3, "no water reached the outlet"


def test_gate_is_one_way() -> None:
    """The condition it exists to PREVENT: the river must not come in.

    A notch in the bed cannot do this. `_interface_depth` is symmetric in the
    two cells, so a lowered crest flows whichever way the surface tilts. This
    is the test that a bed-based 'gate' fails.
    """
    z, h = _bowls((5, 5), (5, 18))
    h[0, 5, 18] = 3.0                      # OUTSIDE is the high side now
    solver = _closed(z)
    solver.structures = _gate((5, 5), (5, 18))
    qx, qy = solver.zeros_flux(1)
    inside_before = float(h[0, 5, 5])
    for _ in range(40):
        h, qx, qy = solver.step(h, qx, qy, dt=30.0)
    # Nothing may arrive at the protected cell THROUGH THE GATE. The two cells
    # are far apart on a closed flat bed, so overland spreading cannot reach it
    # in this many steps either; any arrival would be the flap leaking.
    assert float(h[0, 5, 5]) <= inside_before + 1e-6, "the gate let the river in"


def test_gate_conserves_mass() -> None:
    z, h = _bowls((5, 5), (5, 18))
    h[0, 5, 5] = 3.0
    solver = _closed(z)
    solver.structures = _gate((5, 5), (5, 18))
    qx, qy = solver.zeros_flux(1)
    v0 = float(h.sum())
    for _ in range(60):
        h, qx, qy = solver.step(h, qx, qy, dt=30.0)
        assert solver.last_clamped < 1e-5
    assert abs(float(h.sum()) - v0) / v0 < 1e-4


def test_a_shut_gate_moves_nothing() -> None:
    """`gate_open` is what lets one batched solve cover plans that build it
    and plans that do not."""
    z, h = _bowls((5, 5), (5, 18))
    h[0, 5, 5] = 3.0
    solver = _closed(z)
    solver.structures = _gate((5, 5), (5, 18), open_=False)
    qx, qy = solver.zeros_flux(1)
    for _ in range(40):
        h, qx, qy = solver.step(h, qx, qy, dt=30.0)
    assert float(h[0, 5, 18]) < 1e-6


def test_gate_will_not_drain_below_its_sill() -> None:
    z, h = _bowls((5, 5), (5, 18))
    h[0, 5, 5] = 2.0
    solver = _closed(z)
    solver.structures = _gate((5, 5), (5, 18), invert=1.5)
    qx, qy = solver.zeros_flux(1)
    for _ in range(200):
        h, qx, qy = solver.step(h, qx, qy, dt=30.0)
    # It may drain down to the sill and no further.
    assert float(h[0, 5, 5]) > 1.4


# ---- the pump ------------------------------------------------------------

def test_pump_lifts_at_its_rated_capacity() -> None:
    """Rated discharge, converted honestly: Q m3/s over a cell of dx^2."""
    z, h = _bowls((9, 9), (9, 20))
    h[0, 9, 9] = 4.0
    solver = _closed(z)
    solver.structures = _pump((9, 9), (9, 20), q=5.0)
    qx, qy = solver.zeros_flux(1)
    dt = 10.0
    before = float(h[0, 9, 9])
    h, qx, qy = solver.step(h, qx, qy, dt=dt)
    moved = (before - float(h[0, 9, 9])) * DX * DX / dt
    # Some water also spreads to the neighbours on a flat bed, so the pumped
    # share is a lower bound on what left the cell, not the whole of it.
    assert moved > 4.0, f"pumped only {moved:.2f} m3/s of a rated 5"


def test_pump_cannot_lift_what_is_not_there() -> None:
    """A 5 m3/s station on a two-centimetre puddle must not drain the cell.

    Relying on the solver's own limiter alone would be mass-safe and still
    wrong: the station would take exactly 100% of whatever the cell held and
    the answer would depend on the timestep.
    """
    z, h = _bowls((9, 9), (9, 20))
    h[0, 9, 9] = 0.02
    solver = _closed(z)
    solver.structures = _pump((9, 9), (9, 20), q=5.0, on=0.005, off=0.001)
    qx, qy = solver.zeros_flux(1)
    h, qx, qy = solver.step(h, qx, qy, dt=300.0)
    assert float(h[0, 9, 9]) >= 0.0
    assert solver.last_clamped < 1e-9


def test_pump_respects_its_sump() -> None:
    z, h = _bowls((9, 9), (9, 20))
    h[0, 9, 9] = 2.0
    solver = _closed(z)
    solver.structures = _pump((9, 9), (9, 20), q=3.0, sump=0.8, on=0.02, off=0.005)
    qx, qy = solver.zeros_flux(1)
    for _ in range(300):
        h, qx, qy = solver.step(h, qx, qy, dt=20.0)
    assert float(h[0, 9, 9]) > 0.7, "the pump lifted water below its own intake"


def test_pump_conserves_mass_when_it_discharges_in_domain() -> None:
    z, h = _bowls((9, 9), (9, 20))
    h[0, 9, 9] = 3.0
    solver = _closed(z)
    solver.structures = _pump((9, 9), (9, 20), q=4.0)
    qx, qy = solver.zeros_flux(1)
    v0 = float(h.sum())
    for _ in range(120):
        h, qx, qy = solver.step(h, qx, qy, dt=20.0)
        assert solver.last_clamped < 1e-5
    assert abs(float(h.sum()) - v0) / v0 < 1e-4


def test_pump_off_domain_is_booked_not_lost() -> None:
    """Water lifted off the map is counted, so mass still balances.

    Without the book, a pump would look like a hole in the mass budget and the
    closed-domain invariant every other result rests on would quietly stop
    meaning anything.
    """
    z, h = _bowls((9, 9), (9, 20))
    h[0, 9, 9] = 3.0
    solver = _closed(z)
    solver.structures = _pump((9, 9), None, q=4.0)
    qx, qy = solver.zeros_flux(1)
    v0 = float(h.sum()) * DX * DX
    for _ in range(120):
        h, qx, qy = solver.step(h, qx, qy, dt=20.0)
    v1 = float(h.sum()) * DX * DX
    assert solver.pumped_volume_m3 > 0.0
    assert abs((v1 + solver.pumped_volume_m3) - v0) / v0 < 1e-3


def test_structures_absent_changes_nothing() -> None:
    """The no-structures path must be exactly what it was before.

    Every existing solver result depends on this, so it is asserted rather
    than assumed.
    """
    z, h = _flat(h0=0.0)
    h[0, 6, 6] = 5.0
    a = _closed(z)
    b = _closed(z)
    b.structures = Structures()                 # present but empty
    ha, hb = h.clone(), h.clone()
    qa = a.zeros_flux(1)
    qb = b.zeros_flux(1)
    for _ in range(50):
        ha, *qa = a.step(ha, *qa, dt=25.0)
        hb, *qb = b.step(hb, *qb, dt=25.0)
    assert torch.allclose(ha, hb, atol=0.0)
