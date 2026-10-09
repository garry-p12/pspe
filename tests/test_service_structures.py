"""`build_structures` — the service's half of the gate and the pump.

`tests/test_structures.py` checks the SOLVER's gate and pump terms: that a
gate drains one way, that a pump cannot lift what is not there, that mass is
conserved. None of that says the service wires them up correctly, and the
wiring is where the interesting mistakes live — which cell is inside, which
is outside, whether a plan that does not build a structure really shuts it,
and whether the batch dimension lines up with the bed batch it has to ride.

That last one is the whole reason `Structures` carries a leading batch
dimension at all: `/plan` solves one bed per candidate plan in a single
timestep loop, so a gate that is built in plan 3 and not in plan 1 has to be
open in row 3 and shut in row 1 of the same tensor. Getting that transposed
would not crash; it would quietly attribute one plan's gate to another.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from service.app import (                                    # noqa: E402
    GATE_UNIT_AREA_M2, PUMP_UNIT_Q_M3S, build_structures,
)


class _Dom:
    """The two things `build_structures` reads off a domain."""

    def __init__(self, z, bounds, dx):
        self.z = z
        self.bounds = bounds
        self.dx = dx


def _dom(h=40, w=40, dx=40.0):
    # A plain tilted plane. The elevations matter only for the gate sill and
    # the pump sump, both of which are read at the inside cell.
    z = torch.linspace(10.0, 0.0, h)[:, None].expand(h, w).clone()
    bounds = {"west": -97.80, "east": -97.70, "south": 30.12, "north": 30.20}
    return _Dom(z, bounds, dx)


def _site(dom, kind, *, frac_lat=0.5, frac_lon=0.5, across_lat=0.01):
    """A measure at the middle of the box, with a section running N-S."""
    b = dom.bounds
    lat = b["south"] + (b["north"] - b["south"]) * frac_lat
    lon = b["west"] + (b["east"] - b["west"]) * frac_lon
    return {
        "kind": kind, "lat": lat, "lon": lon, "width_m": 300.0,
        # A is the landward end by construction, which is the convention
        # `_section_line` establishes and `build_structures` relies on.
        "line": {"a": {"lat": lat - across_lat, "lon": lon},
                 "b": {"lat": lat + across_lat, "lon": lon},
                 "length_m": 800.0},
    }


def test_no_structures_is_none():
    """Levees alone must leave the solver byte-for-byte as it was."""
    dom = _dom()
    sites = [{"kind": "levee", "lat": 30.16, "lon": -97.75, "width_m": 300.0}]
    assert build_structures(dom, sites, torch.ones(3, 1)) is None


def test_a_site_without_a_section_is_skipped():
    """A structure needs two cells. A point carries no direction, and
    guessing one would put the outfall on whichever side rounded first."""
    dom = _dom()
    sites = [{"kind": "gate", "lat": 30.16, "lon": -97.75, "width_m": 300.0}]
    assert build_structures(dom, sites, torch.ones(2, 1)) is None


def test_gate_geometry_and_batching():
    dom = _dom()
    sites = [_site(dom, "gate")]
    P = torch.tensor([[0.0], [1.0], [2.0]])          # not built / one / twin
    st = build_structures(dom, sites, P)

    assert st is not None
    assert st.gate_in is not None and st.gate_out is not None
    assert st.gate_in.shape == (1, 2) and st.gate_out.shape == (1, 2)
    # Inside and outside are different cells, and they straddle the bank
    # rather than sitting on top of each other.
    assert not torch.equal(st.gate_in, st.gate_out)
    assert st.pump_in is None

    # Landward is `line.a`, which is at the LOWER latitude here, so the inside
    # cell is the one further SOUTH -- a larger row index.
    assert int(st.gate_in[0, 0]) > int(st.gate_out[0, 0])

    # One row per plan, scaled by that plan's intensity.
    assert st.gate_area_m2.shape == (3, 1)
    assert st.gate_area_m2[0, 0].item() == pytest.approx(0.0)
    assert st.gate_area_m2[1, 0].item() == pytest.approx(GATE_UNIT_AREA_M2)
    assert st.gate_area_m2[2, 0].item() == pytest.approx(2 * GATE_UNIT_AREA_M2)
    # A plan that does not build it has it shut, which is what lets the whole
    # library share one timestep loop.
    assert st.gate_open.tolist() == [[False], [True], [True]]

    # One unit is one 48-inch barrel, which is what the cost is quoted for.
    assert GATE_UNIT_AREA_M2 == pytest.approx(math.pi * (1.22 / 2) ** 2)

    # The sill sits on the ground it drains.
    r, c = int(st.gate_in[0, 0]), int(st.gate_in[0, 1])
    assert st.gate_invert_m[0].item() == pytest.approx(float(dom.z[r, c]))


def test_pump_geometry_and_hysteresis():
    dom = _dom()
    sites = [_site(dom, "pump")]
    P = torch.tensor([[0.0], [0.5], [3.0]])
    st = build_structures(dom, sites, P)

    assert st is not None
    assert st.gate_in is None
    assert st.pump_q_m3s.shape == (3, 1)
    assert st.pump_q_m3s[1, 0].item() == pytest.approx(0.5 * PUMP_UNIT_Q_M3S)
    assert st.pump_q_m3s[2, 0].item() == pytest.approx(3.0 * PUMP_UNIT_Q_M3S)

    r, c = int(st.pump_in[0, 0]), int(st.pump_in[0, 1])
    assert st.pump_sump_m[0].item() == pytest.approx(float(dom.z[r, c]))
    # It has to stop below where it starts, or it chatters against its own
    # drawdown every step.
    assert float(st.pump_off_m[0]) < float(st.pump_on_m[0])


def test_columns_follow_the_measure_not_its_position():
    """A gate in column 2 must read column 2 of the intensity matrix.

    This is the transposition that would not crash. With a levee first and a
    pump second, a `build_structures` that counted its own structures instead
    of indexing the plan vector would hand the pump the levee's height.
    """
    dom = _dom()
    sites = [
        {"kind": "levee", "lat": 30.16, "lon": -97.75, "width_m": 300.0},
        _site(dom, "pump", frac_lat=0.4),
        _site(dom, "gate", frac_lat=0.6),
    ]
    # Levee 2.0 m, pump 4 m3/s, gate 1 barrel -- all different, so a
    # mis-indexed column is visible.
    P = torch.tensor([[2.0, 4.0, 1.0]])
    st = build_structures(dom, sites, P)

    assert st.pump_q_m3s[0, 0].item() == pytest.approx(4.0 * PUMP_UNIT_Q_M3S)
    assert st.gate_area_m2[0, 0].item() == pytest.approx(1.0 * GATE_UNIT_AREA_M2)


def test_structures_ride_the_solver_and_move_water():
    """End to end: the service's structures, in the service's solver.

    A pit behind a ridge, which is the case a levee cannot answer -- water
    already inside the perimeter. One plan runs the pump and one does not,
    in the same batch, and the pumped row must end shallower.
    """
    from pspe.simulate.flood import FloodConfig, FloodSolver

    h_, w_ = 30, 30
    z = torch.full((h_, w_), 20.0)
    PIT = slice(15, 21)                              # rows 15..20
    z[PIT, 12:18] = 0.0
    dom = _Dom(z, {"west": -97.80, "east": -97.70,
                   "south": 30.12, "north": 30.20}, 40.0)

    b = dom.bounds
    def at(r, c):
        return {"lat": b["north"] - (b["north"] - b["south"]) * r / (h_ - 1),
                "lon": b["west"] + (b["east"] - b["west"]) * c / (w_ - 1)}

    # The wet well goes just inside the rim, so the discharge cell -- three
    # cells away across the section -- lands on the ridge. Placed in the
    # middle of a six-cell pit, BOTH ends fall inside it and the pump then
    # moves water from one pit cell to another, which is a real failure mode
    # and not one this test is for.
    mid = at(16, 14)
    site = {"kind": "pump", **mid, "width_m": 100.0,
            "line": {"a": at(26, 14), "b": at(6, 14), "length_m": 800.0}}

    P = torch.tensor([[0.0], [8.0]])                 # off, and 8 m3/s
    st = build_structures(dom, [site], P)
    assert st is not None
    ri, ro = int(st.pump_in[0, 0]), int(st.pump_out[0, 0])
    assert float(z[ri, 14]) == 0.0, f"wet well at row {ri} is not in the pit"
    assert float(z[ro, 14]) > 0.0, f"outfall at row {ro} is still in the pit"

    cfg = FloodConfig(dx=dom.dx, open_edges=(), manning=0.035, dt_max=60.0)
    solver = FloodSolver(z[None].expand(2, -1, -1).contiguous(), cfg)
    solver.structures = st
    h = torch.zeros(2, h_, w_)
    h[:, PIT, 12:18] = 3.0                           # 3 m standing in the pit
    qx, qy = solver.zeros_flux(2)
    start = float(h[0, PIT, 12:18].sum())
    for _ in range(400):
        dt = solver.adaptive_dt(h)
        h, qx, qy = solver.step(h, qx, qy, dt, z=z[None].expand(2, -1, -1))

    off = float(h[0, PIT, 12:18].sum())
    on = float(h[1, PIT, 12:18].sum())
    assert on < off, f"the pump moved nothing: {on:.3f} vs {off:.3f}"
    assert on < start
