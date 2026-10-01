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
