"""Getting out: routing on a road network that the flood is taking apart.

The planner answers "how deep, where" and "what does a levee change". Neither is
the question an operator is actually asked on the night, which is whether the
ambulance can still reach the hospital and by which road. That question is a
shortest path over a graph whose edges go away as the water rises, and it is
worth stating plainly that the path itself is not a hard problem: Dijkstra
solves it exactly on a few hundred edges in under a millisecond, and dressing
that up as learned planning would be inventing a difficulty.

What IS hard, and is the part this framework exists for: the depths are
PREDICTED. A route chosen on predicted depth is a route that can be cut in
reality, and a tool that says "this road is open" on the strength of a
prediction is making a claim it has not earned. So the decision here is not
"shortest path" but "shortest path that stays open under the margin":

    declared open  <=>  predicted max depth along it + q_hat <= what the
                        vehicle can drive through

with q_hat a split-conformal quantile calibrated on held-out plans (see
`eval/run_route_margin.py`). By construction, a route declared open is actually
open with probability at least 1 - delta, and the guarantee is on the scalar the
interface reports rather than on the depth field as a whole -- which is the
distinction Proposition 3 turns on.

Why max depth along the route and not travel time: travel time is a step
function of the plan (a route flips the instant one edge crosses the threshold)
and a margin calibrated on a discontinuous target is dominated by the
discontinuity. Max depth is continuous in the levee heights, so the margin lands
on a well-behaved scalar and the thresholding happens afterwards, once, where it
can be done conservatively.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# Free-flow speeds by OSM class, km/h. Flood routing is not about congestion:
# these set the relative preference for a trunk road over a farm track, which is
# what makes a sensible detour sensible.
SPEED_KMH = {
    "motorway": 100.0, "trunk": 90.0, "primary": 80.0, "secondary": 70.0,
    "tertiary": 60.0, "unclassified": 50.0, "residential": 40.0,
    "service": 25.0, "track": 20.0,
}
DEFAULT_SPEED_KMH = 40.0

# What a vehicle can drive through, metres of standing water. Matching the
# portal's own presets, because a threshold that means one thing in the map and
# another in the router would make the two disagree about the same road.
VEHICLE_DEPTH_M = {"car": 0.25, "truck": 0.45, "none": 0.80}


@dataclass
class Edge:
    """One road segment between two junctions."""
    u: int
    v: int
    length_m: float
    cls: str
    way_id: int
    name: str | None
    #: The geometry, so a chosen route can be drawn and sampled.
    coords: list[tuple[float, float]]

    @property
    def minutes(self) -> float:
        kmh = SPEED_KMH.get(self.cls, DEFAULT_SPEED_KMH)
        return (self.length_m / 1000.0) / kmh * 60.0


@dataclass
class Graph:
    nodes: list[tuple[float, float]] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    #: node index -> list of (edge index, other node index)
    adj: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @property
    def total_km(self) -> float:
        return sum(e.length_m for e in self.edges) / 1000.0


def _seg_m(a: tuple[float, float], b: tuple[float, float], lat: float) -> float:
    dx = (b[0] - a[0]) * 111_320.0 * math.cos(math.radians(lat))
    dy = (b[1] - a[1]) * 110_540.0
    return math.sqrt(dx * dx + dy * dy)


def build_graph(features: Iterable[dict], precision: int = 6) -> Graph:
    """A routable graph from the road GeoJSON the service already fetches.

    Junctions come from coincident vertices, which is sound for Overpass output
    because ways that meet share the node, and was verified on the district
    network before this was relied on: 209 of 212 ways in one connected
    component with no snapping. A way is split at every vertex it shares with
    another way, so a junction in the middle of a road is a junction and not a
    place the router has to drive past.
    """
    g = Graph()
    index: dict[tuple[float, float], int] = {}

    def node(xy: Sequence[float]) -> int:
        k = (round(float(xy[0]), precision), round(float(xy[1]), precision))
        if k not in index:
            index[k] = len(g.nodes)
            g.nodes.append(k)
        return index[k]

    feats = [f for f in features
             if (f.get("geometry") or {}).get("type") == "LineString"
             and len(f["geometry"]["coordinates"]) >= 2]

    # Which vertices are shared, and therefore where ways have to be split.
    seen: dict[tuple[float, float], int] = {}
    for f in feats:
        for xy in f["geometry"]["coordinates"]:
            k = (round(float(xy[0]), precision), round(float(xy[1]), precision))
            seen[k] = seen.get(k, 0) + 1

    for f in feats:
        props = f.get("properties") or {}
        coords = [(float(x), float(y)) for x, y in f["geometry"]["coordinates"]]
        cls = str(props.get("class") or "unclassified")
        way = int(props.get("id") or 0)
        name = props.get("name")

        run: list[tuple[float, float]] = [coords[0]]
        start = node(coords[0])
        for i in range(1, len(coords)):
            run.append(coords[i])
            k = (round(coords[i][0], precision), round(coords[i][1], precision))
            last = i == len(coords) - 1
            if not (last or seen.get(k, 0) > 1):
                continue
            end = node(coords[i])
            if end != start:
                lat = sum(c[1] for c in run) / len(run)
                L = sum(_seg_m(run[j], run[j + 1], lat) for j in range(len(run) - 1))
                if L > 0:
                    e = len(g.edges)
                    g.edges.append(Edge(start, end, L, cls, way, name, list(run)))
                    g.adj.setdefault(start, []).append((e, end))
                    g.adj.setdefault(end, []).append((e, start))
            run = [coords[i]]
            start = end
    return g


def edge_depths(g: Graph, sample: Callable[[float, float], float]) -> list[float]:
    """Deepest water on each edge, from a point sampler over the depth grid.

    The deepest point, not the mean: a road is cut where it goes under, and
    averaging a 1.2 m dip over 3 km of dry bitumen reports a passable road.
    """
    out = []
    for e in g.edges:
        d = 0.0
        for i in range(len(e.coords) - 1):
            a, b = e.coords[i], e.coords[i + 1]
            d = max(d, sample((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        # Vertices as well, so a short edge between two sampled midpoints is
        # not judged on one interior point alone.
        for a in e.coords:
            d = max(d, sample(a[0], a[1]))
        out.append(d)
    return out


def edge_cells(g: Graph, bounds: dict, shape: tuple[int, int]):
    """Grid cells each edge touches, once, for sampling many fields fast.

    `edge_depths` walks every edge's geometry through a Python closure per
    point. That is fine for one field and far too slow for a schedule: a town
    fetched live has a few thousand edges, and thirteen snapshots over them is
    millions of calls. The geometry does not move between snapshots, so the
    cell indices are computed once here and every field after that is a single
    fancy-index and a segmented maximum.

    Returns (flat_index, offsets) where `flat_index[offsets[e]:offsets[e+1]]`
    are the flattened cell indices for edge e. Edges that fall entirely outside
    the box get an empty slice and read as dry, which is what a road outside the
    modelled area has to read as -- the model has nothing to say about it.
    """
    import numpy as np

    h, w = shape
    W, E = bounds["west"], bounds["east"]
    S, N = bounds["south"], bounds["north"]
    flat: list[int] = []
    offs = [0]
    for e in g.edges:
        pts = list(e.coords)
        for i in range(len(e.coords) - 1):
            a, b = e.coords[i], e.coords[i + 1]
            pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        seen = set()
        for lon, lat in pts:
            fx = (lon - W) / (E - W)
            fy = (N - lat) / (N - S)
            if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
                continue
            r = min(h - 1, int(fy * h))
            c = min(w - 1, int(fx * w))
            seen.add(r * w + c)
        flat.extend(sorted(seen))
        offs.append(len(flat))
    return np.asarray(flat, dtype=np.int64), np.asarray(offs, dtype=np.int64)


def depths_from_field(field, flat, offs) -> list[float]:
    """Deepest water per edge for one field, using precomputed cells."""
    import numpy as np

    vals = np.nan_to_num(np.asarray(field, dtype=np.float32).ravel()[flat])
    out = np.zeros(len(offs) - 1, dtype=np.float32)
    for e in range(len(offs) - 1):
        a, b = offs[e], offs[e + 1]
        if b > a:
            out[e] = vals[a:b].max()
    return out.tolist()


def nearest_node(g: Graph, lon: float, lat: float) -> int | None:
    """The junction a place is reached from."""
    best, bi = float("inf"), None
    for i, (x, y) in enumerate(g.nodes):
        if i not in g.adj:
            continue
        d = (x - lon) ** 2 + ((y - lat) * 1.1) ** 2
        if d < best:
            best, bi = d, i
    return bi


@dataclass
class Route:
    """A way out, and what it costs."""
    ok: bool
    minutes: float
    km: float
    #: Edge indices in order, for drawing and for sampling.
    edges: list[int]
    #: Deepest water anywhere on it, under the depths it was planned against.
    max_depth_m: float
    reason: str = ""

    def coords(self, g: Graph) -> list[tuple[float, float]]:
        """The route's geometry, each edge oriented to follow the one before."""
        if not self.edges:
            return []
        out: list[tuple[float, float]] = []
        at = None
        for ei in self.edges:
            e = g.edges[ei]
            seg = list(e.coords)
            if at is None:
                # Orient the first edge by whichever end the second shares.
                if len(self.edges) > 1:
                    nxt = g.edges[self.edges[1]]
                    if e.u in (nxt.u, nxt.v):
                        seg.reverse()
                        at = e.u
                    else:
                        at = e.v
                else:
                    at = e.v
            elif at == e.v:
                seg.reverse()
                at = e.u
            else:
                at = e.v
            out.extend(seg if not out else seg[1:])
        return out

    def names(self, g: Graph) -> list[str]:
        """The named roads it uses, in order, without repeats."""
        out: list[str] = []
        for ei in self.edges:
            nm = g.edges[ei].name
            if nm and (not out or out[-1] != nm):
                out.append(nm)
        return out


def fastest(
    g: Graph, src: int, dst: int, depths: Sequence[float], limit_m: float,
) -> Route:
    """Quickest route using only edges under `limit_m` of water.

    Plain Dijkstra on travel time. Edges over the limit are not penalised, they
    are removed: a road under 40 cm of water is not a slow road, it is a road
    that stops the vehicle, and a cost penalty would let the search buy its way
    through one if the detour were long enough.
    """
    if src == dst:
        return Route(True, 0.0, 0.0, [], 0.0, "already there")
    dist = {src: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    pq = [(0.0, src)]
    done: set[int] = set()
    while pq:
        d, n = heapq.heappop(pq)
        if n in done:
            continue
        done.add(n)
        if n == dst:
            break
        for ei, m in g.adj.get(n, ()):
            if depths[ei] > limit_m or m in done:
                continue
            nd = d + g.edges[ei].minutes
            if nd < dist.get(m, float("inf")):
                dist[m] = nd
                prev[m] = (n, ei)
                heapq.heappush(pq, (nd, m))
    if dst not in dist:
        return Route(False, float("inf"), 0.0, [], float("inf"),
                     f"no route stays under {limit_m:.2f} m")
    path, n = [], dst
    while n != src:
        p, ei = prev[n]
        path.append(ei)
        n = p
    path.reverse()
    km = sum(g.edges[ei].length_m for ei in path) / 1000.0
    return Route(True, dist[dst], km, path,
                 max((depths[ei] for ei in path), default=0.0))


def cut_km(g: Graph, depths: Sequence[float], limit_m: float) -> float:
    """Network length under more water than the vehicle can drive through."""
    return sum(e.length_m for e, d in zip(g.edges, depths) if d > limit_m) / 1000.0


# --------------------------------------------------------------------------
# Time, which is what makes this a planning problem rather than a lookup
# --------------------------------------------------------------------------
#
# A peak-depth field answers "was this road cut at any point", and that is the
# wrong question for someone deciding whether to drive. They are on a given
# road at the time they reach it, and the flood is still moving: a crossing that
# is dry now and 80 cm deep in two hours is passable to someone leaving now and
# a trap for someone leaving at lunchtime. Routing against the peak refuses
# roads that are fine for the next hour; routing against the present refuses
# nothing and sends people into water that arrives while they drive.
#
# So the network carries a SCHEDULE -- depth per edge per snapshot time -- and
# an edge is usable for a crossing that occupies [a, b] only if it stays under
# the limit across that whole window.


@dataclass
class Schedule:
    """Depth per edge over time, from the solver's snapshots.

    `times` are minutes from the start of the run; `depths[k][e]` is edge e's
    deepest water at snapshot k. Between snapshots the depth is taken as the
    larger of the two bracketing values, which is conservative in the only
    direction that matters: it never claims a road is passable during an
    interval it did not look at.
    """
    times: list[float]
    depths: list[list[float]]

    def at(self, ei: int, t: float, tol: float = 1e-6) -> float:
        """Depth on edge ei at time t.

        Exact AT a snapshot, and the enclosing pair's maximum strictly between
        two. The distinction is not pedantic: taking the pair maximum at the
        boundary as well reported a road as flooded at the very moment the
        snapshot says it is dry, which pushed a crossing to the next hour and
        made one test trip arrive at 187 minutes where 127 is correct.
        Conservative inside an interval is right; conservative at a sample we
        actually have is just wrong.
        """
        if not self.times:
            return 0.0
        if t <= self.times[0] + tol:
            return self.depths[0][ei]
        if t >= self.times[-1] - tol:
            return self.depths[-1][ei]
        for k in range(1, len(self.times)):
            if abs(t - self.times[k]) <= tol:
                return self.depths[k][ei]
            if t < self.times[k]:
                return max(self.depths[k - 1][ei], self.depths[k][ei])
        return self.depths[-1][ei]

    def worst_between(self, ei: int, a: float, b: float) -> float:
        """Deepest water on edge ei while a crossing occupies [a, b].

        Includes both endpoints and every snapshot inside the window, so a
        crossing that starts dry and ends deep is judged on the deep end.
        """
        worst = max(self.at(ei, a), self.at(ei, b))
        for k, t in enumerate(self.times):
            if a <= t <= b and self.depths[k][ei] > worst:
                worst = self.depths[k][ei]
        return worst

    @property
    def horizon_min(self) -> float:
        return self.times[-1] if self.times else 0.0

    def peak(self) -> list[float]:
        """Deepest water per edge over the whole run: the static view."""
        if not self.depths:
            return []
        n = len(self.depths[0])
        return [max(row[e] for row in self.depths) for e in range(n)]


def _earliest_start(
    sched: Schedule, ei: int, t: float, minutes: float, limit_m: float,
    allow_wait: bool,
) -> float | None:
    """Earliest time at or after `t` that a crossing of `ei` can be made.

    `None` if there is no such time within the schedule. Candidate starts are
    `t` and each snapshot boundary after it, which is exhaustive given that the
    schedule is piecewise constant between snapshots.
    """
    if sched.worst_between(ei, t, t + minutes) <= limit_m:
        return t
    if not allow_wait:
        return None
    for b in sched.times:
        if b <= t:
            continue
        if sched.worst_between(ei, b, b + minutes) <= limit_m:
            return b
    return None


def fastest_at(
    g: Graph, src: int, dst: int, sched: Schedule, limit_m: float,
    depart_min: float = 0.0, allow_wait: bool = True,
) -> Route:
    """Earliest arrival for someone leaving `src` at `depart_min`.

    Label-setting on arrival time. Whether the first settled label is optimal
    turns on something easy to get wrong, so it is worth stating: travel time
    per edge does not depend on when you enter it, but EDGE AVAILABILITY IS NOT
    MONOTONE -- our runs include the recession, so a crossing that is shut at
    two hours can be open at six. With waiting forbidden, arriving somewhere
    earlier is then not always better (arrive at noon and the only open road is
    the slow one; arrive at four and the fast one has drained), and
    earliest-arrival labelling is a heuristic rather than an optimum.

    Waiting is therefore allowed, which is both the realistic option -- a
    driver can hold at a junction, and the interface already tells them to hold
    before setting out -- and the one that restores exactness: when waiting is
    permitted, an earlier arrival dominates a later one, because the later
    arrival can always be reproduced by waiting. `allow_wait=False` keeps the
    stricter no-holding behaviour, which is then a conservative heuristic and
    is labelled as one.
    """
    if src == dst:
        return Route(True, 0.0, 0.0, [], 0.0, "already there")
    arrive = {src: depart_min}
    prev: dict[int, tuple[int, int]] = {}
    pq: list[tuple[float, int]] = [(depart_min, src)]
    done: set[int] = set()
    while pq:
        t, n = heapq.heappop(pq)
        if n in done:
            continue
        done.add(n)
        if n == dst:
            break
        for ei, m in g.adj.get(n, ()):
            if m in done:
                continue
            start = _earliest_start(sched, ei, t, g.edges[ei].minutes,
                                    limit_m, allow_wait)
            if start is None:
                continue
            t2 = start + g.edges[ei].minutes
            if t2 < arrive.get(m, float("inf")):
                arrive[m] = t2
                prev[m] = (n, ei)
                heapq.heappush(pq, (t2, m))
    if dst not in arrive:
        return Route(False, float("inf"), 0.0, [], float("inf"),
                     f"no route stays under {limit_m:.2f} m leaving at "
                     f"{depart_min:.0f} min")
    path, n = [], dst
    while n != src:
        p, ei = prev[n]
        path.append(ei)
        n = p
    path.reverse()
    km = sum(g.edges[ei].length_m for ei in path) / 1000.0
    # The water the trip actually meets, which is not the same as the deepest
    # water these roads see: that is the whole point of scheduling the crossing.
    # Replaying the path has to replay any holding too, or the depths are read
    # at times the driver is not there.
    t = depart_min
    worst = 0.0
    for ei in path:
        start = _earliest_start(sched, ei, t, g.edges[ei].minutes,
                                limit_m, allow_wait)
        t = start if start is not None else t
        t2 = t + g.edges[ei].minutes
        worst = max(worst, sched.worst_between(ei, t, t2))
        t = t2
    return Route(True, arrive[dst] - depart_min, km, path, worst)


def latest_departure(
    g: Graph, src: int, dst: int, sched: Schedule, limit_m: float,
    step_min: float = 15.0, horizon_min: float | None = None,
) -> tuple[float | None, Route | None]:
    """The last moment someone can leave and still get through.

    This is the number an operator actually needs, and it cannot be read off a
    peak-depth map at all: "you have until 14:45" is a decision, "this road
    floods at some point" is not. Scanned rather than solved backwards, because
    the scan costs one label-setting pass per departure slot and stays correct
    when the edge windows are non-convex -- which they are, since water rises
    and recedes.
    """
    horizon = sched.horizon_min if horizon_min is None else horizon_min
    best_t, best_r = None, None
    last = 0.0
    t = 0.0
    while t <= horizon:
        r = fastest_at(g, src, dst, sched, limit_m, depart_min=t)
        if r.ok:
            best_t, best_r = t, r
        last = t
        t += step_min
    if best_t is None:
        return None, None
    # A trip that is still possible at the end of the modelled window has no
    # deadline that this model knows about, and must not be given one. Returning
    # the horizon here made the interface say "leave within 12 hours, after that
    # every route is cut" about a scenario in which nothing floods at all --
    # a statement the run does not support in either direction, since it has
    # nothing to say about hour thirteen.
    if best_t >= last:
        return None, best_r
    return best_t, best_r


def cut_at(
    g: Graph, route: Route, sched: Schedule, limit_m: float,
    depart_min: float = 0.0,
) -> list[dict]:
    """Where and when a given route meets water, in order along it.

    The explanation the operator reads is built from this: which road, how deep,
    how far along, and how long after leaving. A route that is open still has
    entries here when it passes through water it can cross, and those are worth
    naming -- "35 cm across Wyrallah Road 12 minutes in" is the difference
    between a route someone trusts and one they turn back from.
    """
    out = []
    t, km = depart_min, 0.0
    for ei in route.edges:
        e = g.edges[ei]
        start = _earliest_start(sched, ei, t, e.minutes, limit_m, True)
        if start is not None:
            t = start
        t2 = t + e.minutes
        d = sched.worst_between(ei, t, t2)
        if d > 0.05:
            out.append({
                "name": e.name, "class": e.cls,
                "depth_m": round(d, 3),
                "km_along": round(km, 2),
                "minutes_after_leaving": round(t - depart_min, 1),
                "passable": d <= limit_m,
            })
        t = t2
        km += e.length_m / 1000.0
    return out
