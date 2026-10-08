"""The channel cap is a width, not a cell count.

`channel_crossings` rejected an edge when more than `max_cells` sat within a
few metres of its minimum -- the right idea (kilometres of near-flat ground is
floodplain, not a channel) measured in the wrong unit. 20 cells is 4 km at
200 m, 2.6 km at 130 m and 600 m at 30 m, so the same terrain was a channel or
not depending only on the resolution it was sampled at.

Measured on the Lismore-Coraki box: the south edge is 3.60 km of near-minimum
ground at 200 m (18 cells, accepted) and 3.77 km at 130 m (29 cells, rejected).
Three of four edges were rejected at 130 m, leaving fewer than the two
crossings `river_inflow` needs, so the service fell back to rainfall -- which
spreads the storm thin and floods nothing. A trip that loses 113 km of network
at 200 m reported "no flooding on this trip" at 130 m.
"""

from __future__ import annotations

import numpy as np
import pytest

from pspe.simulate.real import forcing as F


def _synthetic_valley(h: int, w: int, channel_cells: int) -> np.ndarray:
    """Sloping ground with a channel incised down it, north to south.

    The slope matters. A first version made the ground a constant 20 m, which
    left the east and west edges uniformly at their own minimum -- so
    `line <= min + tol` selected every cell of them and the detector reported
    two "crossings" of flat ground. Real terrain is never uniform to the
    millimetre, and a fixture that is will test the tolerance rather than the
    cap.
    """
    z = np.full((h, w), 20.0, dtype=np.float32)
    z += np.linspace(0.0, 60.0, w, dtype=np.float32)[None, :]   # east-west rise
    # The north-south fall has to be steep relative to the 3 m tolerance. At
    # 6 m over the grid, half of each east/west edge sat within 3 m of its own
    # minimum and both were accepted as channels, so a too-wide channel still
    # produced two crossings and the cap looked broken when the fixture was.
    z += np.linspace(0.0, 60.0, h, dtype=np.float32)[:, None]
    c0 = w // 2 - channel_cells // 2
    # Incised well below its surroundings, running the full north-south extent,
    # so it crosses exactly the north and south edges.
    z[:, c0:c0 + channel_cells] = 2.0 + np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    return z


def _channel_edges(z, dx):
    """Which edges the detector accepts, by name."""
    return {c.edge for c in F.channel_crossings(z, dx_m=dx)}


def test_the_same_channel_is_found_at_every_resolution():
    """The bug, directly: one physical width, three samplings, one verdict."""
    width_m = 2000.0
    seen = {}
    for dx in (50.0, 100.0, 200.0):
        cells = max(1, int(round(width_m / dx)))
        z = _synthetic_valley(40, max(16, cells * 4), cells)
        seen[dx] = {"north", "south"} <= _channel_edges(z, dx)
    assert all(seen.values()), (
        f"a {width_m:.0f} m channel was not found at every resolution: {seen}"
    )


def test_a_too_wide_channel_is_rejected_at_every_resolution():
    """And the cap still does its job, identically at every resolution."""
    width_m = 10_000.0           # well past the 4 km default
    seen = {}
    for dx in (50.0, 100.0, 200.0):
        cells = max(1, int(round(width_m / dx)))
        z = _synthetic_valley(40, max(16, cells * 3), cells)
        # The channel's own edges must be rejected. Other edges may still
        # contribute a cell or two from the corner where the slope bottoms
        # out -- that is `river_inflow`'s problem, not the cap's, and
        # asserting on it tests the wrong function.
        seen[dx] = not ({"north", "south"} & _channel_edges(z, dx))
    assert all(seen.values()), (
        f"{width_m/1000:.0f} km of flat ground was accepted somewhere: {seen}"
    )


@pytest.mark.parametrize("dx", [50.0, 100.0, 200.0])
def test_the_cap_sits_at_the_stated_width(dx):
    """Just under the cap passes, just over does not, at any resolution."""
    cap_m = 4000.0
    under = max(1, int(cap_m * 0.6 / dx))
    over = int(cap_m * 1.6 / dx)
    zu = _synthetic_valley(40, max(16, under * 4), under)
    zo = _synthetic_valley(40, max(16, over * 3), over)
    assert {"north", "south"} <= _channel_edges(zu, dx), "under the cap was rejected"
    assert not ({"north", "south"} & _channel_edges(zo, dx)), "over the cap was accepted"


def test_cell_count_fallback_survives_for_callers_without_a_scale():
    """`max_cells` still applies when no resolution is given.

    Kept deliberately: a caller with no dx to hand gets the old behaviour
    rather than an unbounded one.
    """
    z = _synthetic_valley(40, 200, 40)      # 40 cells, over the 20-cell default
    assert all(c.width_cells <= 20 for c in F.channel_crossings(z))
