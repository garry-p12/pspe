"use client";

import { useEffect, useMemo } from "react";
import type { Box } from "@/lib/types";
import type { InspectGrids } from "@/lib/inspect";
import {
  buildGraph, cutKm, edgeDepths, fastest, nearestNode, type Route,
} from "@/lib/routes";
import { DEPTH_PRESETS } from "@/lib/roadquery";
import {
  calibrateRoute, firstCut, routeVerdict, type RouteVerdict,
} from "@/lib/routemargin";

/**
 * Can people still get out, and by which road.
 *
 * The rest of the panel answers "how deep" and "what does a wall change".
 * Neither is the question anyone is actually asked on the night. This one is:
 * the ambulance has to reach the hospital, the town has to empty northwards,
 * and what matters is whether the road is under water and what the detour
 * costs.
 *
 * No abstractions in the copy. "Stays open" and "cut at 60 cm", not
 * passability or conformal anything -- the guarantee is stated as "9 times out
 * of 10", which is what the number means.
 */

export interface RoutePlan {
  /** The recommended height vector. */
  heights: number[];
  max_height_m: number;
  saturation_S: number;
  confidence_pct: number;
  options: { heights: number[]; depth_u8?: string }[];
  /** True when the service solved the recommended plan in its own right. */
  chosen_solved?: boolean;
}

function mins(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return v < 60 ? `${Math.round(v)} min`
    : `${Math.floor(v / 60)} h ${Math.round(v % 60)} min`;
}

export function RoutePanel({
  roads, grids, bounds, from, to, vehicle, onVehicle, plan, planGridFor,
  chosenGrid, onRoutes,
}: {
  roads: GeoJSON.FeatureCollection | null;
  /** The field currently being drawn: do-nothing, or a chosen option. */
  grids: InspectGrids | null;
  bounds: Box | null;
  /** Where people are leaving from — the town. */
  from: { lat: number; lon: number; name: string } | null;
  /** Where they are heading. Null means "wherever the water does not reach". */
  to: { lat: number; lon: number; name: string } | null;
  vehicle: number;
  onVehicle: (m: number) => void;
  /** The recommended plan, for the margin. Null in the archive view. */
  plan?: RoutePlan | null;
  /** An option's depth grid, for calibrating this route. */
  planGridFor?: (i: number) => InspectGrids | null;
  /** The recommended plan's own solved field, when the service ran it.
   *
   *  This is what section 5.13 turns on. The surrogate's error on the deepest
   *  water along a route is 0.16-0.33 m, which is the size of the passability
   *  threshold itself, so a predicted route verdict cannot be stated with a
   *  useful margin. Where the plan has been solved there is nothing to predict
   *  and nothing to hedge, and the panel says which of the two it is showing. */
  chosenGrid?: InspectGrids | null;
  /** Hand the routes up so the map can draw them. */
  onRoutes?: (r: { quickest: Route | null; robust: Route | null;
                   dest: { lat: number; lon: number } | null } | null) => void;
}) {
  const g = useMemo(() => buildGraph(roads), [roads]);

  const depths = useMemo(
    () => (g && grids && bounds ? edgeDepths(g, grids, bounds) : null),
    [g, grids, bounds],
  );

  const answer = useMemo(() => {
    if (!g || !depths || !bounds || !from) return null;
    const src = nearestNode(g, from.lon, from.lat);
    if (src == null) return null;

    // Is this place cut off at all? If the water does not reach the roads
    // around the origin, there is no evacuation to plan and saying so is the
    // answer. The first version asked for "the nearest dry junction" and got
    // back the origin itself, then reported "no trip to assess" -- which reads
    // as a broken tool rather than as good news.
    const atOrigin = Math.max(
      ...(g.adj.get(src) ?? []).map(([ei]) => depths[ei]), 0);

    // Where to. Given a destination, the nearest junction to it. Otherwise dry
    // ground, as close as possible but far enough that the trip is a trip:
    // a shorter drive through a flood is a safer one, and a destination 200 m
    // away is not an evacuation.
    const MIN_TRIP_KM = 2.0;
    let dst: number | null = null;
    if (to) {
      dst = nearestNode(g, to.lon, to.lat);
    } else {
      let best = Infinity;
      for (const [n] of g.adj) {
        const deep = Math.max(...(g.adj.get(n) ?? []).map(([ei]) => depths[ei]), 0);
        if (deep > 0.05) continue;
        const [x, y] = g.nodes[n];
        const km = Math.hypot((x - from.lon) * 97, (y - from.lat) * 111);
        if (km < MIN_TRIP_KM) continue;
        if (km < best) { best = km; dst = n; }
      }
    }
    if (dst == null || dst === src) return null;

    const quickest = fastest(g, src, dst, depths, vehicle);
    // What the drive looks like with no water at all, as the thing to compare
    // the flooded trip against: "12 min becomes 31" is the cost of the flood.
    const dry = fastest(g, src, dst, depths.map(() => 0), vehicle);
    const cut = quickest.ok ? null : firstCut(g, dry, depths, vehicle);

    // The margin, for the recommended plan only. An option that was solved
    // needs no margin on this road -- its depth there is a solve, not a
    // prediction -- and saying "9 times out of 10" over a known number would be
    // borrowing credibility the claim does not need.
    // Solved beats predicted. One route query against the recommended plan's
    // own field, no surrogate and no margin, and the copy below says so.
    let solvedM: number | null = null;
    if (chosenGrid && (quickest.ok || dry.ok)) {
      const sd = edgeDepths(g, chosenGrid, bounds);
      const route = quickest.ok ? quickest : dry;
      solvedM = route.edges.reduce((m, ei) => Math.max(m, sd[ei] ?? 0), 0);
    }

    let verdict: RouteVerdict | null = null;
    if (!chosenGrid && plan?.options?.length && planGridFor && dry.ok) {
      const cache = new Map<number, number[] | null>();
      const depthsFor = (i: number) => {
        if (!cache.has(i)) {
          const og = planGridFor(i);
          cache.set(i, og ? edgeDepths(g, og, bounds) : null);
        }
        return cache.get(i) ?? null;
      };
      const route = quickest.ok ? quickest : dry;
      const cal = calibrateRoute(route, plan.options, depthsFor, plan.max_height_m);
      if (cal && cal.points.length) {
        verdict = routeVerdict(cal, plan.saturation_S, plan.heights,
                               plan.max_height_m, vehicle, plan.confidence_pct);
      }
    }

    // The route that still works if the estimate is off by the margin: every
    // road treated as that much deeper than predicted. Same construction as
    // `eval/run_route_margin.py`, so the minutes quoted here are the minutes
    // the paper reports.
    const band = verdict?.band.attainable ? verdict.band.band : 0;
    const robust = band > 0
      ? fastest(g, src, dst, depths.map((d) => d + band), vehicle)
      : null;

    return {
      quickest: quickest.ok ? quickest : null,
      dry,
      robust: robust?.ok ? robust : null,
      band,
      cut,
      verdict,
      solvedM,
      atOrigin,
      dest: { lat: g.nodes[dst][1], lon: g.nodes[dst][0] },
      cutNetworkKm: cutKm(g, depths, vehicle),
    };
  }, [g, depths, bounds, from, to, vehicle, plan, planGridFor, chosenGrid]);

  // Hand the geometry up for drawing. In an effect, not in render: calling the
  // parent's setter while rendering is a state update during another
  // component's render pass, which React is right to complain about.
  useEffect(() => {
    onRoutes?.(answer
      ? { quickest: answer.quickest, robust: answer.robust, dest: answer.dest }
      : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [answer]);

  if (!roads) {
    return (
      <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
        <h3 className="eyebrow">Getting out</h3>
        <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
          Open the roads view first, so the network is loaded.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
      <div className="flex items-baseline justify-between">
        <h3 className="eyebrow">Getting out</h3>
        <span className="anno">
          {g ? `${g.edges.length} road segments` : "no network"}
        </span>
      </div>

      <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
        {from ? `From ${from.name}` : "No starting point"}
        {to ? ` to ${to.name}` : " to the nearest dry ground"}.
      </p>

      <div className="mt-2.5 flex flex-wrap gap-1">
        {DEPTH_PRESETS.map((p) => (
          <button
            key={p.m}
            className="pill"
            data-on={Math.abs(vehicle - p.m) < 1e-6}
            onClick={() => onVehicle(p.m)}
            title={p.note}
          >
            {p.label}
          </button>
        ))}
      </div>
      <p className="anno mt-1.5">
        Water deeper than {(vehicle * 100).toFixed(0)} cm stops this vehicle
      </p>

      {!answer && (
        <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">
          {g && depths && depths.every((d) => d <= vehicle)
            ? `Nothing here is cut off: no road in this area is under more than
               ${(vehicle * 100).toFixed(0)} cm of water in this scenario, so
               there is no evacuation to plan.`
            : "No trip to assess here yet — pick a place first."}
        </p>
      )}

      {answer && (
        <>
          <div className="mt-3 border-t border-line-soft pt-3">
            {answer.quickest ? (
              <>
                <div className="flex items-baseline justify-between">
                  <span className="text-[13px] text-ink">Road out is open</span>
                  <span className="tnum text-[18px] font-semibold text-ink">
                    {mins(answer.quickest.minutes)}
                  </span>
                </div>
                <p className="anno mt-1">
                  {answer.quickest.km.toFixed(1)} km
                  {answer.quickest.via.length > 0 &&
                    ` · via ${answer.quickest.via.slice(0, 3).join(", ")}`}
                </p>
                {answer.dry.ok &&
                 answer.quickest.minutes - answer.dry.minutes > 0.5 && (
                  <p className="anno mt-1">
                    {mins(answer.dry.minutes)} on a dry day — the flood adds{" "}
                    {mins(answer.quickest.minutes - answer.dry.minutes)}
                  </p>
                )}
              </>
            ) : (
              <>
                <span className="chip-harm text-[13px]">No way out by road</span>
                {answer.cut && (
                  <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
                    {answer.cut.name ?? "An unnamed road"} goes under{" "}
                    {(answer.cut.depthM * 100).toFixed(0)} cm of water
                    {answer.cut.km > 0.1 &&
                      ` about ${answer.cut.km.toFixed(1)} km along`}
                    , and every other road out is cut too.
                  </p>
                )}
              </>
            )}
            <p className="anno mt-1.5">
              {answer.cutNetworkKm.toFixed(0)} km of the local network is under
              more water than this vehicle can cross
              {answer.atOrigin <= 0.05 && answer.cutNetworkKm < 0.1 &&
                " — none of it near here"}
            </p>
          </div>

          {answer.solvedM != null && (
            <div className="mt-3 border-t border-line-soft pt-3">
              <h4 className="eyebrow">If the recommended walls are built</h4>
              <div className="mt-1.5 flex items-baseline justify-between">
                <span className="text-[13px] text-ink-mute">
                  Deepest water on this road
                </span>
                <span className="tnum text-[15px] font-semibold text-ink">
                  {(answer.solvedM * 100).toFixed(0)} cm
                </span>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed text-ink">
                {answer.solvedM <= vehicle
                  ? "Stays open. This is a full run of this exact plan, not an estimate, so there is no margin to add."
                  : "Still under water after the walls. These walls are not what keeps this road open."}
              </p>
            </div>
          )}

          {answer.verdict && (
            <div className="mt-3 border-t border-line-soft pt-3">
              <h4 className="eyebrow">If the recommended walls are built</h4>
              <div className="mt-1.5 flex items-baseline justify-between">
                <span className="text-[13px] text-ink-mute">
                  Deepest water on this road
                </span>
                <span className="tnum text-[15px] font-semibold text-ink">
                  {(answer.verdict.expectedM * 100).toFixed(0)} cm
                </span>
              </div>
              <div className="mt-1 flex items-baseline justify-between">
                <span className="text-[13px] text-ink-mute">
                  Allowing for what the estimate can get wrong
                </span>
                <span className="tnum text-[15px] font-semibold text-ink">
                  {(answer.verdict.worstCaseM * 100).toFixed(0)} cm
                </span>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed">
                {answer.verdict.band.attainable ? (
                  answer.verdict.openWithMargin ? (
                    <span className="text-ink">
                      Stays open, {plan?.confidence_pct ?? 90}% of the time.
                      Plan on it.
                    </span>
                  ) : answer.verdict.openNominal ? (
                    <span className="text-ink">
                      Open on the best estimate, but not with enough room to
                      promise it. Treat this road as one that may be cut.
                    </span>
                  ) : (
                    <span className="text-ink">
                      Still under water after the walls. These walls are not
                      what keeps this road open.
                    </span>
                  )
                ) : (
                  <span className="text-ink-mute">
                    Not enough solved combinations on this road to put a figure
                    on what the estimate can get wrong
                    {answer.verdict.band.needed > 0 &&
                      ` — ${answer.verdict.band.needed} are needed and ${answer.verdict.band.have} were usable`}
                    . The depth above is an estimate with nothing behind it.
                  </span>
                )}
              </p>
              <p className="anno mt-1.5 leading-relaxed">
                Without the walls this road sits under{" "}
                {(answer.verdict.baseMaxDepthM * 100).toFixed(0)} cm at its
                worst point.
              </p>
            </div>
          )}

          {answer.robust && answer.quickest && (
            <div className="mt-3 border-t border-line-soft pt-3">
              <div className="flex items-baseline justify-between">
                <span className="text-[13px] text-ink-mute">
                  Route that still works if the water is deeper than estimated
                </span>
                <span className="tnum text-[15px] font-semibold text-ink">
                  {mins(answer.robust.minutes)}
                </span>
              </div>
              <p className="anno mt-1">
                {answer.robust.minutes - answer.quickest.minutes < 0.5
                  ? "The quickest way out is already the safe one."
                  : `${mins(answer.robust.minutes - answer.quickest.minutes)} longer than the quickest way out, allowing every road to be ${(answer.band * 100).toFixed(0)} cm deeper than predicted.`}
              </p>
            </div>
          )}

          <p className="anno mt-2.5 leading-relaxed">
            Speeds are free-flow by road class, not a traffic model. A road is
            judged by its deepest point, which is where a vehicle stops.
          </p>
        </>
      )}
    </div>
  );
}
