"""Where water enters a domain, and from which direction.

The ad-hoc solver used to apply rainfall uniformly over the whole box with
every edge open. That is a real question -- what if this much rain fell here --
but it is the wrong one for most cities, and it quietly made mitigation look
useless: a levee redirects water that arrives from somewhere, and has nothing
to block when the rain lands on both sides of it.

Two other drivers matter and are inferable from the terrain alone:

    river     a channel crosses the boundary; discharge enters at the upstream
              crossing and leaves at the downstream one.
    coastal   one edge is the sea; a surge holds a water level against it.

Neither needs the user to draw anything. The DEM already says where the low
ground is, and water does not choose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EDGES = ("north", "south", "east", "west")


def _edge_cells(z: np.ndarray, edge: str) -> np.ndarray:
    """The boundary row or column, north being row 0."""
    if edge == "north":
        return z[0, :]
    if edge == "south":
        return z[-1, :]
    if edge == "west":
        return z[:, 0]
    return z[:, -1]


def _edge_mask(shape: tuple[int, int], edge: str, sel: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    if edge == "north":
        m[0, :] = sel
    elif edge == "south":
        m[-1, :] = sel
    elif edge == "west":
        m[:, 0] = sel
    else:
        m[:, -1] = sel
    return m


@dataclass
class Crossing:
    """Where a channel meets the boundary."""
    edge: str
    mask: np.ndarray       # boundary cells that are channel
    bed_m: float           # mean bed elevation of those cells
    width_cells: int


def channel_crossings(z: np.ndarray, tol_m: float = 3.0,
                      max_cells: int = 20,
                      dx_m: float | None = None,
                      max_width_m: float = 4000.0) -> list[Crossing]:
    """Where the channel meets each edge: cells near that edge's own MINIMUM.

    Measured from the minimum, not the median. Against the median, "well below
    the edge" selects the whole valley floor -- at Cedar Rapids that was 78
    cells, 4.7 km of floodplain, and pouring a river's discharge evenly across
    it is not an inflow, it is rain again. Near the minimum it picks the two to
    six cells that are actually the channel.

    The cap exists because an edge where the ground sits within a few metres of
    its minimum for kilometres is flat floodplain, not a channel, and nothing
    should be injected there.

    **That cap is a WIDTH, not a cell count.** It was `max_cells = 20`, which
    means 4 km at 200 m cells, 2.6 km at 130 m and 600 m at 30 m -- so the same
    terrain was a channel or not depending only on the resolution it was
    sampled at. Measured on the Lismore-Coraki box: the south edge is 3.60 km
    of near-minimum ground at 200 m (18 cells, accepted) and 3.77 km at 130 m
    (29 cells, rejected). Nearly the same river, opposite verdicts. With three
    of four edges rejected at 130 m there were fewer than two crossings, so
    `river_inflow` returned None and the service fell back to rainfall -- which
    spreads the storm thin and floods nothing, turning a route that loses
    113 km of network into "no flooding on this trip".

    `dx_m` is therefore how this should be called. `max_cells` is kept for
    callers that genuinely have no scale to hand, and no longer applies when a
    resolution is given.
    """
    cap = (max_width_m / float(dx_m)) if dx_m else float(max_cells)
    out: list[Crossing] = []
    for edge in EDGES:
        line = np.nan_to_num(_edge_cells(z, edge))
        sel = line <= float(line.min()) + tol_m
        n = int(sel.sum())
        if n == 0 or n > cap:
            continue
        out.append(Crossing(edge=edge, mask=_edge_mask(z.shape, edge, sel),
                            bed_m=float(line[sel].mean()), width_cells=n))
    return out


def _edge_of(lat: float, lon: float, bounds: dict, tol: float = 0.02) -> str | None:
    """Which edge a point sits on, as a fraction of the box's own span."""
    dlon = max(bounds["east"] - bounds["west"], 1e-12)
    dlat = max(bounds["north"] - bounds["south"], 1e-12)
    fx = (lon - bounds["west"]) / dlon
    fy = (lat - bounds["south"]) / dlat
    cand = [("west", abs(fx)), ("east", abs(1 - fx)),
            ("south", abs(fy)), ("north", abs(1 - fy))]
    edge, d = min(cand, key=lambda t: t[1])
    return edge if d <= tol else None


def channel_from_network(ways: list[dict], z: np.ndarray, bounds: dict,
                         dx: float, tol_m: float = 3.0,
                         half_width_cells: int = 3) -> tuple[Crossing, Crossing] | None:
    """Inflow and outflow taken from the MAPPED stream, not from edge minima.

    The edge heuristic below asks which boundary cells sit near their own edge's
    minimum, then calls the two lowest of those the channel. That is a guess
    about which low ground is a river, and it has now failed twice in different
    ways. On a catchment that tilts steadily to the east -- Onion Creek at
    Austin -- the north-east corner is lower than the true western inflow, so it
    injected the Colorado valley corner and the creek never flooded: 1.62 km2
    against a 17.9 km2 mapped floodplain. Rejecting corner-hugging crossings
    does not repair it; measured, that rule moves Onion Creek from the wrong
    edge to a different wrong edge, and at Lismore it rejects all four
    crossings and breaks the case that works.

    The mapped network settles it without guessing. OpenStreetMap has the
    channel, named, nearly everywhere a town is: Onion Creek is recorded
    crossing the west edge and leaving by the east, and the Wilsons River
    crossing north, east and south as it meanders through Lismore. The main
    stem is the named waterway with the most length inside the box, and its
    boundary crossings are the inflow and the outflow.

    Which end is upstream is read off the DEM rather than off the drawing
    direction, because a way's direction is a mapping convention and the bed
    elevation is a measurement: water runs downhill.

    Returns None when nothing is mapped here, so the caller can fall back.
    """
    if z.size == 0:
        return None
    H, W = z.shape
    by_name: dict[str, float] = {}
    pts: dict[str, list[tuple[float, float]]] = {}
    for w in ways:
        name = (w.get("tags") or {}).get("name")
        geom = w.get("geometry") or []
        if not name or len(geom) < 2:
            continue
        length = 0.0
        for i in range(len(geom) - 1):
            a, b = geom[i], geom[i + 1]
            length += math.hypot((b["lon"] - a["lon"]) * 111_320.0
                                 * math.cos(math.radians(a["lat"])),
                                 (b["lat"] - a["lat"]) * 110_540.0)
            e = _edge_of(a["lat"], a["lon"], bounds) or _edge_of(b["lat"], b["lon"], bounds)
            if e:
                pts.setdefault(name, []).append(((a["lat"] + b["lat"]) / 2,
                                                 (a["lon"] + b["lon"]) / 2))
        by_name[name] = by_name.get(name, 0.0) + length
    # The main stem is the one with the most channel inside the box, among
    # those that actually reach the boundary twice. A tributary that enters and
    # stops is not a through-channel and cannot carry a discharge across.
    stems = [(l, n) for n, l in by_name.items() if len(pts.get(n, [])) >= 2]
    if not stems:
        return None
    _, name = max(stems)

    seen: dict[str, tuple[float, float]] = {}
    for lat, lon in pts[name]:
        e = _edge_of(lat, lon, bounds)
        if e and e not in seen:
            seen[e] = (lat, lon)
    if len(seen) < 2:
        return None

    out: list[Crossing] = []
    for edge, (lat, lon) in seen.items():
        col = int(round((lon - bounds["west"])
                        / max(bounds["east"] - bounds["west"], 1e-12) * (W - 1)))
        row = int(round((bounds["north"] - lat)
                        / max(bounds["north"] - bounds["south"], 1e-12) * (H - 1)))
        col, row = min(max(col, 0), W - 1), min(max(row, 0), H - 1)
        line = np.nan_to_num(_edge_cells(z, edge))
        at = col if edge in ("north", "south") else row
        lo = max(at - half_width_cells, 0)
        hi = min(at + half_width_cells + 1, len(line))
        # Widen to the channel actually there: cells near the local minimum
        # around the mapped crossing, not a fixed number of cells.
        local = line[lo:hi]
        sel = np.zeros(len(line), dtype=bool)
        sel[lo:hi] = local <= float(local.min()) + tol_m
        if not sel.any():
            sel[at] = True
        out.append(Crossing(edge=edge, mask=_edge_mask(z.shape, edge, sel),
                            bed_m=float(line[sel].mean()),
                            width_cells=int(sel.sum())))
    if len(out) < 2:
        return None
    inflow = max(out, key=lambda c: c.bed_m)
    outflow = min(out, key=lambda c: c.bed_m)
    return inflow, outflow


def river_inflow(z: np.ndarray, tol_m: float = 3.0,

                 dx_m: float | None = None) -> Crossing | None:
    """The upstream end of the through-channel, or None if there is not one.

    A river needs a way in and a way out, so the two LOWEST crossings are the
    channel and everything above them is a tributary or a road cutting. Taking
    simply the highest crossing picked exactly such a cutting at Cedar Rapids
    -- 238 m, where the real channel runs 217 m in to 208 m out.

    Of the two, upstream is the higher bed: water runs downhill.
    """
    cr = channel_crossings(z, tol_m, dx_m=dx_m)
    if len(cr) < 2:
        return None
    lowest_two = sorted(cr, key=lambda c: c.bed_m)[:2]
    return max(lowest_two, key=lambda c: c.bed_m)


def river_outflow(z: np.ndarray, tol_m: float = 3.0,
                  dx_m: float | None = None) -> Crossing | None:
    """The downstream end, which must stay open however the edges are set."""
    cr = channel_crossings(z, tol_m, dx_m=dx_m)
    if len(cr) < 2:
        return None
    return min(cr, key=lambda c: c.bed_m)


def sea_edge(z: np.ndarray, sea_level_m: float = 2.0,
             min_fraction: float = 0.25) -> Crossing | None:
    """The edge that is open water, if there is one.

    A quarter of an edge sitting within a couple of metres of zero is the sea
    or a tidal estuary; nothing else looks like that. Returned as a crossing so
    a surge can be held against exactly those cells.
    """
    best: Crossing | None = None
    for edge in EDGES:
        line = np.nan_to_num(_edge_cells(z, edge))
        sel = line <= sea_level_m
        frac = sel.mean()
        if frac < min_fraction:
            continue
        c = Crossing(edge=edge, mask=_edge_mask(z.shape, edge, sel),
                     bed_m=float(line[sel].mean()), width_cells=int(sel.sum()))
        if best is None or c.width_cells > best.width_cells:
            best = c
    return best


def inflow_source(mask: np.ndarray, q_m3s: float, dx: float) -> np.ndarray:
    """Discharge as a per-cell source in m/s, for the solver's rain term.

    Q spread over the wetted inflow cells. The solver takes a source in metres
    per second, so this is Q divided by the area it is poured onto.
    """
    n = int(mask.sum())
    if n == 0 or q_m3s <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    src = np.zeros(mask.shape, dtype=np.float32)
    src[mask] = q_m3s / (n * dx * dx)
    return src


def describe(z: np.ndarray, dx: float) -> dict:
    """What this terrain can be forced by, for the interface to offer."""
    riv = river_inflow(z, dx_m=dx)
    out = river_outflow(z, dx_m=dx)
    sea = sea_edge(z)
    cr = channel_crossings(z, dx_m=dx)
    return {
        "river": None if riv is None else {
            "edge": riv.edge, "bed_m": round(riv.bed_m, 1),
            "width_m": round(riv.width_cells * dx),
            "drains_to": None if out is None else out.edge,
            "fall_m": None if out is None else round(riv.bed_m - out.bed_m, 1),
            "crossings": [c.edge for c in cr],
        },
        "coastal": None if sea is None else {
            "edge": sea.edge, "width_m": round(sea.width_cells * dx),
        },
        "relief_m": round(float(np.nanmax(z) - np.nanmin(z)), 1),
        "min_elev_m": round(float(np.nanmin(z)), 1),
    }
