import type { Box } from "./types";
import type { InspectGrids } from "./inspect";
import { sampleAt } from "./inspect";

/**
 * Getting out: routing on a network the flood is taking apart.
 *
 * This runs in the browser rather than on the service, because everything it
 * needs is already here. The roads are fetched for the map, the depth grid for
 * the chosen option is in the planner's response, and Dijkstra over a few
 * hundred edges is a fraction of a frame. A round-trip would add a second of
 * latency to a question the client can answer instantly, and would need the
 * network posted back to a service that already sent the water.
 *
 * It is a deliberate mirror of `pspe/plan/routes.py` -- same speeds, same
 * vehicle depths, same "remove the edge, do not penalise it" rule -- and
 * `portal/scripts/verify-route-parity.mjs` checks the two against each other on
 * the same inputs, because a router that disagreed with the one the paper's
 * numbers came from would make the portal a different tool wearing the same
 * name.
 */

/** Free-flow speeds by OSM class, km/h. Mirrors SPEED_KMH in routes.py. */
const SPEED_KMH: Record<string, number> = {
  motorway: 100, trunk: 90, primary: 80, secondary: 70,
  tertiary: 60, unclassified: 50, residential: 40, service: 25, track: 20,
};
const DEFAULT_SPEED_KMH = 40;

export interface RouteEdge {
  u: number;
  v: number;
  lengthM: number;
  cls: string;
  name: string | null;
  coords: [number, number][];
  minutes: number;
}

export interface RouteGraph {
  nodes: [number, number][];
  edges: RouteEdge[];
  adj: Map<number, [number, number][]>;
  totalKm: number;
}

function segM(a: [number, number], b: [number, number], lat: number): number {
  const dx = (b[0] - a[0]) * 111320 * Math.cos((lat * Math.PI) / 180);
  const dy = (b[1] - a[1]) * 110540;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * A routable graph from the road GeoJSON.
 *
 * Junctions come from coincident vertices, which holds for Overpass output
 * because ways that meet share the node. Ways are split at every shared vertex,
 * so a T-junction in the middle of a road is a junction and not somewhere the
 * router has to drive past.
 */
export function buildGraph(
  roads: GeoJSON.FeatureCollection | null,
  precision = 6,
): RouteGraph | null {
  if (!roads?.features?.length) return null;
  const nodes: [number, number][] = [];
  const index = new Map<string, number>();
  const key = (xy: [number, number]) =>
    `${xy[0].toFixed(precision)},${xy[1].toFixed(precision)}`;
  const node = (xy: [number, number]) => {
    const k = key(xy);
    let i = index.get(k);
    if (i === undefined) {
      i = nodes.length;
      index.set(k, i);
      nodes.push([Number(xy[0].toFixed(precision)), Number(xy[1].toFixed(precision))]);
    }
    return i;
  };

  // Narrowed once, here, so the rest of the builder works with plain arrays
  // instead of re-asserting the geometry type at every use.
  const feats: { props: Record<string, unknown>; coords: [number, number][] }[] = [];
  for (const f of roads.features) {
    if (f.geometry?.type !== "LineString") continue;
    const coords = f.geometry.coordinates as [number, number][];
    if (coords.length < 2) continue;
    feats.push({ props: (f.properties ?? {}) as Record<string, unknown>, coords });
  }

  // Which vertices more than one way visits, and therefore where to split.
  const seen = new Map<string, number>();
  for (const f of feats) {
    for (const xy of f.coords) {
      const k = key(xy);
      seen.set(k, (seen.get(k) ?? 0) + 1);
    }
  }

  const edges: RouteEdge[] = [];
  const adj = new Map<number, [number, number][]>();
  const link = (n: number, e: number, other: number) => {
    const l = adj.get(n);
    if (l) l.push([e, other]); else adj.set(n, [[e, other]]);
  };

  for (const f of feats) {
    const cls = typeof f.props.class === "string" ? f.props.class : "unclassified";
    const name = typeof f.props.name === "string" ? f.props.name : null;
    const coords = f.coords;
    let run: [number, number][] = [coords[0]];
    let start = node(coords[0]);
    for (let i = 1; i < coords.length; i++) {
      run.push(coords[i]);
      const last = i === coords.length - 1;
      if (!last && (seen.get(key(coords[i])) ?? 0) <= 1) continue;
      const end = node(coords[i]);
      if (end !== start) {
        const lat = run.reduce((s, c) => s + c[1], 0) / run.length;
        let L = 0;
        for (let j = 0; j < run.length - 1; j++) L += segM(run[j], run[j + 1], lat);
        if (L > 0) {
          const kmh = SPEED_KMH[cls] ?? DEFAULT_SPEED_KMH;
          const e = edges.length;
          edges.push({
            u: start, v: end, lengthM: L, cls, name,
            coords: [...run], minutes: (L / 1000) / kmh * 60,
          });
          link(start, e, end);
          link(end, e, start);
        }
      }
      run = [coords[i]];
      start = end;
    }
  }
  return {
    nodes, edges, adj,
    totalKm: edges.reduce((s, e) => s + e.lengthM, 0) / 1000,
  };
}

/**
 * Deepest water on each edge.
 *
 * The deepest point, not the mean: a road is cut where it goes under, and
 * averaging a 1.2 m dip over 3 km of dry bitumen reports a passable road.
 */
export function edgeDepths(g: RouteGraph, grids: InspectGrids, b: Box): number[] {
  return g.edges.map((e) => {
    let d = 0;
    for (let i = 0; i < e.coords.length - 1; i++) {
      const a = e.coords[i];
      const c = e.coords[i + 1];
      const s = sampleAt(grids, b, (a[0] + c[0]) / 2, (a[1] + c[1]) / 2);
      if (s && s.depth_m > d) d = s.depth_m;
    }
    for (const a of e.coords) {
      const s = sampleAt(grids, b, a[0], a[1]);
      if (s && s.depth_m > d) d = s.depth_m;
    }
    return d;
  });
}

export function nearestNode(g: RouteGraph, lon: number, lat: number): number | null {
  let best = Infinity, bi: number | null = null;
  for (const [i] of g.adj) {
    const [x, y] = g.nodes[i];
    const d = (x - lon) ** 2 + ((y - lat) * 1.1) ** 2;
    if (d < best) { best = d; bi = i; }
  }
  return bi;
}

/** A binary min-heap keyed on travel time. Lazy deletion: a node can be pushed
 *  more than once and the stale entries are skipped on pop. */
class MinHeap {
  private k: number[] = [];
  private v: number[] = [];

  get size(): number { return this.k.length; }

  push(key: number, val: number): void {
    this.k.push(key); this.v.push(val);
    let i = this.k.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.k[p] <= this.k[i]) break;
      this.swap(p, i); i = p;
    }
  }

  pop(): [number, number] | null {
    if (!this.k.length) return null;
    const top: [number, number] = [this.k[0], this.v[0]];
    const lk = this.k.pop()!, lv = this.v.pop()!;
    if (this.k.length) {
      this.k[0] = lk; this.v[0] = lv;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.k.length && this.k[l] < this.k[m]) m = l;
        if (r < this.k.length && this.k[r] < this.k[m]) m = r;
        if (m === i) break;
        this.swap(m, i); i = m;
      }
    }
    return top;
  }

  private swap(a: number, b: number): void {
    [this.k[a], this.k[b]] = [this.k[b], this.k[a]];
    [this.v[a], this.v[b]] = [this.v[b], this.v[a]];
  }
}

export interface Route {
  ok: boolean;
  minutes: number;
  km: number;
  edges: number[];
  maxDepthM: number;
  /** The geometry, each edge oriented to follow the one before it. */
  coords: [number, number][];
  /** Named roads used, in order, without repeats. */
  via: string[];
}

const NO_ROUTE: Route = {
  ok: false, minutes: Infinity, km: 0, edges: [], maxDepthM: Infinity,
  coords: [], via: [],
};

/**
 * Quickest route using only edges under `limitM` of water.
 *
 * Edges over the limit are removed, not penalised: a road under 40 cm of water
 * is not a slow road, it is a road that stops the vehicle, and a cost penalty
 * would let the search buy its way through one if the detour were long enough.
 */
export function fastest(
  g: RouteGraph, src: number, dst: number, depths: number[], limitM: number,
): Route {
  if (src === dst) {
    return { ok: true, minutes: 0, km: 0, edges: [], maxDepthM: 0, coords: [], via: [] };
  }
  const dist = new Map<number, number>([[src, 0]]);
  const prev = new Map<number, [number, number]>();
  const done = new Set<number>();
  // A real heap. The first version re-sorted a plain array on every pop, on the
  // premise that these graphs hold a few hundred nodes -- which is true of the
  // district's 347 edges and false of a town fetched live: Lismore's 12 km box
  // comes back from Overpass with 2,319. Re-sorting thousands of entries
  // thousands of times is seconds of main thread for a search that should cost
  // a fraction of a frame.
  const pq = new MinHeap();
  pq.push(0, src);
  while (pq.size) {
    const [d, n] = pq.pop()!;
    if (done.has(n)) continue;
    done.add(n);
    if (n === dst) break;
    for (const [ei, m] of g.adj.get(n) ?? []) {
      if (depths[ei] > limitM || done.has(m)) continue;
      const nd = d + g.edges[ei].minutes;
      if (nd < (dist.get(m) ?? Infinity)) {
        dist.set(m, nd);
        prev.set(m, [n, ei]);
        pq.push(nd, m);
      }
    }
  }
  if (!dist.has(dst)) return NO_ROUTE;

  const path: number[] = [];
  let n = dst;
  while (n !== src) {
    const [p, ei] = prev.get(n)!;
    path.push(ei);
    n = p;
  }
  path.reverse();

  // Orient each edge to continue from the one before, so the drawn line is
  // continuous rather than a set of segments that happen to touch.
  const coords: [number, number][] = [];
  let at: number | null = null;
  for (let i = 0; i < path.length; i++) {
    const e = g.edges[path[i]];
    const seg = [...e.coords];
    if (at === null) {
      if (path.length > 1) {
        const nxt = g.edges[path[1]];
        if (e.u === nxt.u || e.u === nxt.v) { seg.reverse(); at = e.u; } else { at = e.v; }
      } else { at = e.v; }
    } else if (at === e.v) { seg.reverse(); at = e.u; } else { at = e.v; }
    if (coords.length) coords.push(...seg.slice(1)); else coords.push(...seg);
  }

  const via: string[] = [];
  for (const ei of path) {
    const nm = g.edges[ei].name;
    if (nm && via[via.length - 1] !== nm) via.push(nm);
  }
  return {
    ok: true,
    minutes: dist.get(dst)!,
    km: path.reduce((s, ei) => s + g.edges[ei].lengthM, 0) / 1000,
    edges: path,
    maxDepthM: path.reduce((m, ei) => Math.max(m, depths[ei]), 0),
    coords,
    via,
  };
}

/** Network length under more water than the vehicle can drive through. */
/**
 * Kilometres of network a vehicle cannot cross, counting WHOLE edges.
 *
 * An edge is the unit a router can refuse, so if any sample on it is over
 * the limit the whole edge is unusable and its whole length is cut. The
 * Forecast tab's road list answers a different question -- how much of the
 * tarmac is under water -- and counts only the sub-segments that are, which
 * is why the two figures differ by about four times on the same flood. Both
 * are right for their own question; neither should be read as the other.
 */
export function cutKm(g: RouteGraph, depths: number[], limitM: number): number {
  let km = 0;
  g.edges.forEach((e, i) => { if (depths[i] > limitM) km += e.lengthM / 1000; });
  return km;
}

/** Total length of the routing graph, for the denominator. */
export function networkKm(g: RouteGraph): number {
  let km = 0;
  g.edges.forEach((e) => { km += e.lengthM / 1000; });
  return km;
}
