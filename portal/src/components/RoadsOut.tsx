"use client";

import { useMemo } from "react";
import { DEPTH_PRESETS, queryRoads } from "@/lib/roadquery";
import type { InspectGrids } from "@/lib/inspect";
import type { Box } from "@/lib/types";

/**
 * Which roads out are closed, deepest first.
 *
 * The question behind every other panel is eventually this one: can people
 * leave. The depth field answers it only after someone has picked a threshold,
 * because what stops a car does not stop a truck and what stops a truck does
 * not stop a helicopter — so the threshold is a control and not a constant,
 * and the heading says which vehicle the list is about.
 *
 * Sorted by the DEEPEST water on each road rather than by how much of it is
 * under. A road is closed by its worst point: a kilometre of dry tarmac with
 * thirty centimetres across the middle of it is a closed road, and a long
 * stretch under a centimetre is not.
 */
export function RoadsOut({
  roads, grids, bounds, place, vehicle, onVehicle, onQuery,
}: {
  roads: GeoJSON.FeatureCollection | null;
  grids: InspectGrids | null;
  bounds: Box | null;
  place: string | null;
  vehicle: number;
  onVehicle: (m: number) => void;
  onQuery?: (q: ReturnType<typeof queryRoads>) => void;
}) {
  const q = useMemo(
    () => queryRoads(roads, grids, bounds, vehicle),
    [roads, grids, bounds, vehicle],
  );
  const preset = DEPTH_PRESETS.reduce((a, p) =>
    Math.abs(p.m - vehicle) < Math.abs(a.m - vehicle) ? p : a, DEPTH_PRESETS[1]);

  if (!roads || !grids || !bounds) return null;

  return (
    <div className="border-b border-line-soft px-5 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="eyebrow">
          Roads out{place ? ` of ${place.split(",")[0]}` : ""}
        </h3>
        <span className="anno">
          {preset.label.toLowerCase()} · closed over {(vehicle * 100).toFixed(0)} cm
        </span>
      </div>

      <div className="mt-1.5 flex flex-wrap gap-1">
        {DEPTH_PRESETS.map((d) => (
          <button key={d.m} className="pill" data-on={Math.abs(vehicle - d.m) < 1e-6}
                  onClick={() => { onVehicle(d.m); onQuery?.(q); }} title={d.note}>
            {d.label}
          </button>
        ))}
      </div>

      {!q || q.roads.length === 0 ? (
        <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
          No named road in this area is under more than {(vehicle * 100).toFixed(0)} cm.
        </p>
      ) : (
        <>
          <ul className="mt-2 flex flex-col">
            {q.roads.filter((r) => r.name).slice(0, 8).map((r) => (
              <li key={r.id}
                  className="flex items-baseline gap-2 border-b border-line-soft py-1.5 last:border-0">
                <span aria-hidden
                      className={`h-[9px] w-[9px] shrink-0 rounded-[2px] ${
                        r.maxDepthM > vehicle ? "bg-bad" : "border border-line bg-bg"}`} />
                <span className="flex-1 truncate text-[13px] text-ink">{r.name}</span>
                <span className="tnum text-[13px] text-ink-mute">
                  {(r.maxDepthM * 100).toFixed(0)} cm
                </span>
                <span className={`w-[52px] text-right text-[12px] ${
                  r.maxDepthM > vehicle ? "font-medium text-bad" : "text-ink-faint"}`}>
                  {r.maxDepthM > vehicle ? "Closed" : "Open"}
                </span>
              </li>
            ))}
          </ul>
          <p className="anno mt-1.5 leading-relaxed">
            {q.cutKm.toFixed(1)} km of {q.totalKm.toFixed(0)} km cut · deepest first,
            because a road is closed by its worst point
          </p>
        </>
      )}
    </div>
  );
}
