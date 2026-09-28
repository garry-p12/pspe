"use client";

import { useEffect, useMemo, useState } from "react";
import type { Box } from "@/lib/types";
import type { InspectGrids } from "@/lib/inspect";
import { DEPTH_PRESETS, queryRoads, type RoadQuery } from "@/lib/roadquery";

export function RoadQueryPanel({
  roads,
  grids,
  bounds,
  onQuery,
}: {
  roads: GeoJSON.FeatureCollection | null;
  grids: InspectGrids | null;
  bounds: Box | null;
  onQuery: (q: RoadQuery | null) => void;
}) {
  const [idx, setIdx] = useState(1); // default: car
  const preset = DEPTH_PRESETS[idx];

  const q = useMemo(
    () => queryRoads(roads, grids, bounds, preset.m),
    [roads, grids, bounds, preset.m],
  );

  // Hand the result up so the map can colour the network.
  //
  // In an effect, not a useMemo. Calling the parent's setter while this
  // component renders updates another component mid-render, which React is
  // entitled to defer or drop -- and did: the map never received a cut map, so
  // every road drew at the default colour and the network looked uniform in
  // both the district and an ad-hoc area. An effect runs after commit, when
  // setting parent state is legal.
  useEffect(() => { onQuery(q); }, [q, onQuery]);

  if (!roads || !grids) {
    return (
      <div className="border-b border-line-soft p-4">
        <p className="eyebrow mb-1">Road access</p>
        <p className="text-[13px] text-ink-faint">Loading network…</p>
      </div>
    );
  }

  return (
    <div className="border-b border-line-soft p-4">
      <p className="eyebrow mb-2">Road access in this event</p>

      <div className="mb-3 grid grid-cols-4 gap-1">
        {DEPTH_PRESETS.map((p, i) => (
          <button
            key={p.m}
            onClick={() => setIdx(i)}
            className={`rounded-md border px-1 py-1.5 text-[12px] leading-tight transition-colors ${
              i === idx
                ? "border-accent/60 bg-accent/10 text-accent"
                : "border-line text-ink-mute hover:text-ink"
            }`}
            title={`${p.note} — above ${(p.m * 100).toFixed(0)} cm`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <p className="mb-3 text-[12px] text-ink-faint">
        {preset.note} · above {(preset.m * 100).toFixed(0)} cm
      </p>

      {q && (
        <>
          <div className="flex items-baseline gap-2">
            <span className="tnum text-[24px] font-semibold leading-none text-bad">
              {q.cutKm.toFixed(0)} km
            </span>
            <span className="text-[13px] text-ink-mute">
              cut of {q.totalKm.toFixed(0)} km
            </span>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-bg-inset">
            <div
              className="h-full rounded-full bg-bad"
              style={{ width: `${Math.min(100, (q.cutKm / q.totalKm) * 100)}%` }}
            />
          </div>
          <p className="mt-1 text-[12px] text-ink-faint">
            {((q.cutKm / q.totalKm) * 100).toFixed(0)}% of the mapped network
          </p>

          {q.roads.length > 0 && (
            <div className="mt-3">
              <p className="eyebrow mb-1.5">Worst affected</p>
              <ul className="flex flex-col gap-1">
                {q.roads
                  .filter((r) => r.name)
                  .slice(0, 7)
                  .map((r) => (
                    <li key={r.id} className="flex items-baseline gap-2 text-[13px]">
                      <span className="flex-1 truncate text-ink-mute">{r.name}</span>
                      <span className="tnum shrink-0 text-ink-faint">
                        {r.cutKm.toFixed(1)} km
                      </span>
                      <span className="tnum w-9 shrink-0 text-right text-[11px] text-ink-faint">
                        {((r.cutKm / r.totalKm) * 100).toFixed(0)}%
                      </span>
                    </li>
                  ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
