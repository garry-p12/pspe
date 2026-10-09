"use client";

import { useCallback, useMemo, useRef, useState } from "react";

/**
 * One street, and which floors the water got over.
 *
 * Every other view in this portal answers about ground: how deep here, how much
 * land, which roads. A resident asks about a floor, and the two are not the
 * same question — on 2,924 surveyed Lismore properties the median floor stands
 * 1.95 m above its own front gate, so the same flood is ankle-deep in one house
 * and waist-deep in its neighbour.
 *
 * Two things here are load-bearing and both are stated on the panel rather
 * than buried.
 *
 * **The floor height is assumed.** Austin publishes the ground at the
 * structure, from lidar, and not the height of the floor above it; the FEMA
 * elevation certificates that carry the real number are not bulk-public. A
 * slab-on-grade default of 0.17 m is used and every count that depends on it
 * says so. Where houses are not on slabs this assumption is worth metres.
 *
 * **The water is downscaled, not solved here.** A street-sized box has no
 * upstream in it: measured, 50 mm/h for six hours over a 2 km box put 10 of
 * 576 houses over their floors, and three blocks sitting entirely inside
 * FEMA's 1% floodplain behaved the same way, because the rain drains out of
 * the sides. The catchment is solved at its own scale and the water SURFACE
 * from it — which carries the hydraulic gradient — is laid over bare-earth
 * terrain at street resolution. Depth is that surface minus the fine ground.
 */

export interface StreetBuilding {
  id: string;
  ring: [number, number][];
  address: string | null;
  ground_m: number;
  ground_source: string;
  height_m: number | null;
  floor_m: number;
  depth_on_ground_m: number;
  water_surface_m: number | null;
  over_floor_m: number | null;
  over_floor_series: number[];
  dem_ground_m: number;
}

export interface StreetResult {
  ok: boolean;
  reason?: string;
  name: string;
  bounds: { west: number; south: number; east: number; north: number };
  dx_m: number;
  grid: [number, number];
  terrain_source: string;
  buildings: StreetBuilding[];
  count: number;
  over_floor: number;
  deepest_on_ground_m: number;
  snapshot_hours: number[];
  floor_above_ground_m: number;
  floor_source: string;
  driver: string;
  driver_note: string;
  depth_png: string;
  terrain_png: string;
  depth_vmax_m: number;
  frames: { t_h: number; depth_png: string }[];
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-line bg-bg-raised px-3 py-2.5">
      <p className="anno">{label}</p>
      <p className="tnum mt-0.5 text-[20px] font-semibold leading-none text-ink">{value}</p>
      {sub && <p className="anno mt-1">{sub}</p>}
    </div>
  );
}

export function StreetPanel({
  result, busy, elapsed, err, frame, onFrame, selected, onSelect, onSolve,
  canSolve, place, onBack,
}: {
  result: StreetResult | null;
  busy: boolean;
  elapsed: number;
  err: string | null;
  /** Which solved instant is on screen, or null for the peak. */
  frame: number | null;
  onFrame: (i: number | null) => void;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onSolve: (p: { driver: string; storm: number; q: number | null }) => void;
  canSolve: boolean;
  /** The study this block sits in, for the breadcrumb out of it. */
  place: string | null;
  onBack: () => void;
}) {
  const [q, setQ] = useState("");
  // River by default, and that is a physical claim rather than a preference.
  // Onion Creek drains roughly 850 km2; no box this view solves contains that,
  // so rain falling inside the box is not where a street's water comes from.
  // Measured: a rain-forced catchment put 2 of 245 houses over their floors
  // here, and the same block forced down the channel put 361 of 574.
  const [driver, setDriver] = useState("river");
  const [storm, setStorm] = useState(0);
  const [qText, setQText] = useState("");
  const listRef = useRef<HTMLUListElement>(null);

  const rows = useMemo(() => {
    if (!result) return [];
    const needle = q.trim().toLowerCase();
    return result.buildings
      .filter((b) => (b.over_floor_m ?? -1) > -Infinity)
      .filter((b) => !needle || (b.address ?? b.id).toLowerCase().includes(needle))
      .sort((a, b) => (b.over_floor_m ?? -99) - (a.over_floor_m ?? -99))
      .slice(0, 200);
  }, [result, q]);

  const sel = useMemo(
    () => result?.buildings.find((b) => b.id === selected) ?? null,
    [result, selected],
  );

  /** What the chosen instant says, falling back to the peak. */
  const at = useCallback(
    (b: StreetBuilding) =>
      frame !== null && b.over_floor_series[frame] !== undefined
        ? b.over_floor_series[frame]
        : (b.over_floor_m ?? -b.floor_m),
    [frame],
  );

  // What the block is called, from the addresses on it. The design puts a
  // street name in the breadcrumb and in the heading; the city's address
  // points give one, so the heading can say where you are rather than "this
  // street". Two names where the block spans a corner, which is how anyone
  // describes a corner out loud.
  const streetName = useMemo(() => {
    const counts = new Map<string, number>();
    for (const b of result?.buildings ?? []) {
      const a = b.address;
      if (!a) continue;
      const st = a.replace(/^\s*[\d-]+\s+/, "").trim();
      if (st) counts.set(st, (counts.get(st) ?? 0) + 1);
    }
    const top = [...counts.entries()].sort((x, y) => y[1] - x[1]).slice(0, 2);
    if (!top.length) return null;
    return top.length > 1 && top[1][1] >= top[0][1] * 0.4
      ? `${top[0][0]} & ${top[1][0]}`
      : top[0][0];
  }, [result]);

  const grounds = (result?.buildings ?? []).map((b) => b.ground_m);

  return (
    <div className="px-5 pb-5 pt-4">
      {/* Back out to the whole valley. The street is a zoom, so the way out
          of it is a zoom out, and the design puts that above the heading
          rather than leaving the reader to find the map chip again. */}
      <button onClick={onBack}
              className="mb-1 text-[12px] text-ink-faint hover:text-ink">
        ← {place ? place.split(",")[0] : "Whole valley"}
      </button>
      <h2 className="text-[19px] font-semibold leading-tight text-ink">
        {streetName ?? "This street"}
      </h2>
      {grounds.length > 0 && (
        <p className="anno mt-0.5">
          {grounds.length} buildings · ground {Math.min(...grounds).toFixed(1)}–
          {Math.max(...grounds).toFixed(1)} m NAVD88
        </p>
      )}
      <p className="mb-3 mt-2 text-[13px] leading-relaxed text-ink-mute">
        Every building in the block, and whether the water got over its floor —
        not over the ground outside it.
      </p>

      {!result && !busy && (
        <>
          <div>
            <h3 className="eyebrow">Where the water comes from</h3>
            <div className="mt-1.5 flex gap-1">
              {(["river", "rain"] as const).map((k) => (
                <button key={k} className="pill" data-on={driver === k}
                        onClick={() => setDriver(k)}>
                  {k === "river" ? "Down the channel" : "Rain on the box"}
                </button>
              ))}
            </div>
            <p className="anno mt-1.5 leading-relaxed">
              {driver === "river"
                ? "A street floods from upstream. The catchment this view solves is a few kilometres across and the creek above it drains hundreds, so the water arrives as discharge rather than as rain landing here."
                : "Rain on the solved box only. Honest for a cloudburst on a flat suburb, and it under-floods a creek: measured, 2 of 245 houses over their floors against 361 of 574 forced down the channel."}
            </p>
          </div>

          {driver === "river" && (
            <div className="mt-3">
              <h3 className="eyebrow">How big an event</h3>
              <div className="mt-1.5 flex gap-1">
                {["As gauged", "2.5x", "4x"].map((lab, i) => (
                  <button key={lab} className="pill" data-on={storm === i}
                          onClick={() => setStorm(i)}>{lab}</button>
                ))}
              </div>
              <label htmlFor="street-q" className="eyebrow mt-3 block">
                Discharge coming down the creek
              </label>
              <div className="mt-1.5 flex items-center gap-2">
                <input
                  id="street-q"
                  value={qText}
                  onChange={(e) => setQText(e.target.value.replace(/[^0-9.]/g, ""))}
                  placeholder="estimate from the terrain"
                  inputMode="decimal"
                  className="w-36 rounded-md border border-line bg-bg-inset px-2.5 py-1.5
                             text-[13px] text-ink placeholder:text-ink-faint
                             focus:border-accent/60 focus:outline-none"
                />
                <span className="text-[13px] text-ink-mute">m³/s</span>
              </div>
              <p className="anno mt-1.5 leading-relaxed">
                Onion Creek peaked near 3,400 m³/s in October 2013, when this
                neighbourhood went under. Left empty, bankfull is estimated from
                the channel geometry, which on bare earth reads high.
              </p>
              <p className="anno mt-1.5 leading-relaxed">
                Worth knowing before reading a count off this: at 3,400 m³/s
                with no multiplier, this model puts <em>no</em> house in the
                block over its floor, and it takes about 2.5× that to reproduce
                the inundation 2013 actually caused. The gap is the model&apos;s,
                not the creek&apos;s — a 14 km box truncates a catchment that
                drains hundreds of square kilometres, and nothing here knows
                about the bridges and culverts that backed the water up. Treat
                the multiplier as the calibration knob it is.
              </p>
            </div>
          )}
        </>
      )}

      {!result && (
        <button
          onClick={() => onSolve({
            driver, storm,
            q: Number(qText) > 0 ? Number(qText) : null,
          })}
          disabled={busy || !canSolve}
          className="mt-4 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium text-bg
                     transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {busy ? `Solving the catchment, then the street… ${elapsed}s`
            : canSolve ? "Open the street view"
              : "Solve a place first"}
        </button>
      )}
      {busy && (
        <p className="anno mt-1.5 leading-relaxed">
          The catchment is solved first and the street downscaled from it. A
          street-sized box has no upstream in it, so rain falling on one drains
          out of the sides and floods nothing.
        </p>
      )}
      {err && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">Could not open this street</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{err}</p>
        </div>
      )}

      {result?.ok && (
        <>
          <p className="anno mb-2">
            {result.count} buildings · ground{" "}
            {Math.min(...result.buildings.map((b) => b.ground_m)).toFixed(1)}–
            {Math.max(...result.buildings.map((b) => b.ground_m)).toFixed(1)} m ·
            solved at {result.dx_m.toFixed(0)} m
          </p>

          <div className="grid grid-cols-2 gap-2">
            <Tile
              label="Deepest on the ground"
              value={`${result.deepest_on_ground_m.toFixed(2)} m`}
              sub="anywhere in the block"
            />
            <Tile
              label="Over floor level"
              value={`${result.over_floor} of ${result.count}`}
              sub={`floor assumed +${result.floor_above_ground_m.toFixed(2)} m`}
            />
          </div>

          {/* ---- time ------------------------------------------------- */}
          {result.snapshot_hours.length > 1 && (
            <div className="mt-3">
              <label htmlFor="street-t" className="flex items-baseline justify-between">
                <span className="eyebrow">When</span>
                <span className="tnum text-[13px] text-ink">
                  {frame === null ? "peak"
                    : `${result.snapshot_hours[frame].toFixed(1)} h`}
                </span>
              </label>
              <input
                id="street-t"
                type="range"
                min={-1}
                max={result.snapshot_hours.length - 1}
                step={1}
                value={frame ?? -1}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  onFrame(v < 0 ? null : v);
                }}
                className="mt-1 w-full accent-ink"
              />
              <p className="anno">
                {frame === null
                  ? "the worst it ever gets, which is what a flood map shows"
                  : `${result.snapshot_hours[frame].toFixed(1)} hours into the storm`}
              </p>
            </div>
          )}

          {/* ---- the selected building -------------------------------- */}
          {sel && (
            <div className="mt-3 rounded-lg border border-line bg-bg-raised p-3">
              <div className="flex items-start justify-between gap-2">
                <p className="text-[14px] font-semibold text-ink">
                  {sel.address ?? sel.id}
                </p>
                <button onClick={() => onSelect(null)}
                        className="text-[12px] text-ink-faint hover:text-ink">clear</button>
              </div>
              <p className="anno mt-0.5">
                slab on ground, assumed · ground {sel.ground_m.toFixed(2)} m from the{" "}
                {sel.ground_source}
              </p>
              <div className="mt-2">
                {at(sel) > 0 ? (
                  <>
                    <span className="chip-harm">Over floor</span>{" "}
                    <span className="tnum text-[13px] font-medium text-ink">
                      {at(sel).toFixed(2)} m of water inside
                    </span>
                  </>
                ) : (
                  <>
                    <span className="chip-good">Floor above water</span>{" "}
                    <span className="tnum text-[13px] text-ink-mute">
                      {Math.abs(at(sel)).toFixed(2)} m to spare
                    </span>
                  </>
                )}
              </div>
              <p className="anno mt-1.5 leading-relaxed">
                Floor {result.floor_above_ground_m.toFixed(2)} m above the ground ·{" "}
                {sel.depth_on_ground_m.toFixed(2)} m on the ground outside
              </p>
            </div>
          )}

          {/* ---- on this street --------------------------------------- */}
          <div className="mt-4">
            <div className="flex items-baseline justify-between gap-2">
              <h3 className="eyebrow">On this street</h3>
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="filter"
                aria-label="Filter buildings"
                className="w-24 rounded-md border border-line bg-bg-inset px-2 py-0.5
                           text-[12px] text-ink placeholder:text-ink-faint
                           focus:border-accent/60 focus:outline-none"
              />
            </div>
            <ul ref={listRef} className="mt-1.5 flex max-h-[320px] flex-col gap-0.5 overflow-y-auto">
              {rows.map((b) => {
                const v = at(b);
                return (
                  <li key={b.id}>
                    <button
                      onClick={() => onSelect(b.id === selected ? null : b.id)}
                      aria-pressed={b.id === selected}
                      className={`flex w-full items-baseline gap-2 rounded px-2 py-1 text-left
                                  text-[13px] ${b.id === selected
                                    ? "bg-ink text-bg"
                                    : "text-ink-mute hover:bg-bg-inset"}`}
                    >
                      <span aria-hidden className={`h-[9px] w-[9px] shrink-0 rounded-[2px] ${
                        v > 0 ? "bg-bad" : "border border-line bg-bg"}`} />
                      <span className="flex-1 truncate">{b.address ?? b.id}</span>
                      <span className="tnum">
                        {v > 0 ? `+${v.toFixed(2)} m` : "dry"}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>

          <details className="fold mt-3 border-t border-line-soft">
            <summary>What the floor number rests on</summary>
            <p className="anno pb-2 leading-relaxed">{result.floor_source}</p>
            <p className="anno pb-2 leading-relaxed">
              The water surface comes from a catchment solve and is laid over
              bare-earth terrain at {result.dx_m.toFixed(0)} m. The terrain is
              {result.terrain_source === "3dep" ? " USGS 3DEP bare earth" : " Copernicus GLO-30"}.
              A street-sized box cannot be rain-forced: it has no upstream in it.
            </p>
            <p className="anno pb-2 leading-relaxed">{result.driver_note}</p>
          </details>
        </>
      )}
    </div>
  );
}
