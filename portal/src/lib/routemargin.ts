import { bandAt, type Band, type CalibrationPoint } from "./margin";
import type { Route, RouteGraph } from "./routes";

/**
 * "This road stays open" is a claim, so it gets a margin.
 *
 * The planner's options are each a full solve, so for an option that was
 * actually solved the deepest water on a road is known, not predicted, and no
 * margin is owed. The margin is owed for the plan the greedy search RECOMMENDS,
 * which is almost never one of the solved combinations -- that one is a
 * surrogate prediction, and declaring a road passable on the strength of a
 * prediction is exactly the claim this framework exists to make honest.
 *
 * So: for the route the operator asked about,
 *
 *   alpha[j]  metres of depth that site j alone takes off this route (exact,
 *             from the single-measure solves)
 *   m_hat     the saturating surrogate's prediction for the recommended plan,
 *             the same form the depth planner uses
 *   q_hat     split-conformal quantile of |m - m_hat| over the solved
 *             multi-site combinations for THIS route
 *
 * and the route is called open only when `m_hat + q_hat` is still under what
 * the vehicle can drive through. The scalar carrying the margin is the deepest
 * water on the route, which is continuous in the levee heights -- travel time
 * is a step function of them, and a margin calibrated on a discontinuity is
 * dominated by it.
 *
 * `eval/run_route_margin.py` is where this is measured against held-out plans,
 * including against the field-error margin that Proposition 3 rules out.
 */

/** Deepest water on a route, under one option's depth field. */
export function routeMaxDepth(route: Route, edgeDepths: number[]): number {
  return route.edges.reduce((m, ei) => Math.max(m, edgeDepths[ei] ?? 0), 0);
}

export interface RouteCalibration {
  /** Metres of depth each site alone takes off this route. */
  alpha: number[];
  /** One point per solved multi-site combination. */
  points: CalibrationPoint[];
  /** Deepest water on the route if nothing is built. */
  baseMaxDepthM: number;
}

/**
 * Read the solved option set as a calibration sample for one route.
 *
 * `depthsFor` hands back the per-edge depths under an option, which the caller
 * memoises: it is a few hundred grid samples per option and there is no reason
 * to repeat them every time the confidence slider moves.
 */
export function calibrateRoute(
  route: Route,
  options: { heights: number[] }[],
  depthsFor: (i: number) => number[] | null,
  maxHeightM: number,
): RouteCalibration | null {
  const base = options.findIndex((o) => !o.heights.some((h) => h > 0));
  const bd = base >= 0 ? depthsFor(base) : null;
  if (!bd) return null;
  const k = options[base].heights.length;
  const baseMaxDepthM = routeMaxDepth(route, bd);

  const alpha = new Array<number>(k).fill(0);
  const measured = new Map<number, number>();
  for (let i = 0; i < options.length; i++) {
    const d = depthsFor(i);
    if (!d) continue;
    const m = routeMaxDepth(route, d);
    measured.set(i, m);
    const active = options[i].heights
      .map((h, j) => ({ h, j })).filter((x) => x.h > 0);
    if (active.length === 1) {
      // Exact: one solve, one measure. Positive means the levee makes this
      // road shallower; negative happens and is not an error -- a wall can
      // push water somewhere it was not going.
      alpha[active[0].j] = baseMaxDepthM - m;
    }
  }

  const points: CalibrationPoint[] = [];
  for (const [i, m] of measured) {
    const parts = options[i].heights
      .map((h, j) => (h > 0 ? alpha[j] * (h / maxHeightM) : null))
      .filter((v): v is number => v !== null);
    if (parts.length < 2) continue;      // carries no interaction error
    const lead = Math.max(...parts);
    points.push({
      n_active: parts.length,
      lead,
      rest: parts.reduce((s, v) => s + v, 0) - lead,
      // In reduction space, so the residual is the surrogate's error on the
      // quantity it predicts rather than on an offset of it.
      measured: baseMaxDepthM - m,
    });
  }
  return { alpha, points, baseMaxDepthM };
}

/** The surrogate's predicted reduction for a height vector. Mirrors surrogate.effect. */
export function predictedReduction(
  alpha: number[], heights: number[], maxHeightM: number, S: number,
): number {
  const parts = heights
    .map((h, j) => (h > 0 ? alpha[j] * (h / maxHeightM) : null))
    .filter((v): v is number => v !== null);
  if (!parts.length) return 0;
  const lead = Math.max(...parts);
  return lead + S * Math.tanh((parts.reduce((s, v) => s + v, 0) - lead) / S);
}

export interface RouteVerdict {
  /** Deepest water expected on the route under this plan. */
  expectedM: number;
  /** The same, with the margin added: what the claim is made against. */
  worstCaseM: number;
  /** What the vehicle can drive through. */
  limitM: number;
  /** Open on the point estimate alone. */
  openNominal: boolean;
  /** Open once the margin is applied — the only one worth telling anyone. */
  openWithMargin: boolean;
  band: Band;
  baseMaxDepthM: number;
}

export function routeVerdict(
  cal: RouteCalibration,
  S: number,
  heights: number[],
  maxHeightM: number,
  limitM: number,
  confidencePct: number,
): RouteVerdict {
  const band = bandAt(cal.points, S, confidencePct);
  const expectedM = cal.baseMaxDepthM -
    predictedReduction(cal.alpha, heights, maxHeightM, S);
  // An unattainable band is zero, so the worst case collapses onto the point
  // estimate. That is the honest failure mode and the caller has to say so:
  // `band.attainable` false means "this is a guess", not "this is safe".
  const worstCaseM = expectedM + band.band;
  return {
    expectedM, worstCaseM, limitM,
    openNominal: expectedM <= limitM,
    openWithMargin: worstCaseM <= limitM,
    band,
    baseMaxDepthM: cal.baseMaxDepthM,
  };
}

/** Where a route first goes under, for saying which road is the problem. */
export function firstCut(
  g: RouteGraph, route: Route, edgeDepths: number[], limitM: number,
): { name: string | null; depthM: number; km: number } | null {
  let km = 0;
  for (const ei of route.edges) {
    const d = edgeDepths[ei] ?? 0;
    if (d > limitM) {
      return { name: g.edges[ei].name, depthM: d, km };
    }
    km += g.edges[ei].lengthM / 1000;
  }
  return null;
}
