"use client";

import { Explain } from "./Explain";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OptionRow } from "./OptionList";
import { ForecastPanel } from "./ForecastPanel";
import { ObservePanel } from "./ObservePanel";
import { decodeGrids as decode, type InspectGrids } from "@/lib/inspect";

/** What a shipped study says about itself: where, which storm, and how well
 *  it did against an independent map of the same flood. */
export interface StudyMeta {
  id: string;
  place: string;
  short: string;
  event: string;
  evidence: string;
  terrain: string;
  caveats: string[];
  skill: {
    csi: number; pod: number; far: number; threshold_m: number;
    ours_km2: number; reference: string; reference_km2: number;
    reference_polygons: number;
  } | null;
}

export interface AnalysisResult {
  name: string;
  /** Which elevation model the solve actually ran on. */
  terrain_source?: string;
  terrain_source_label?: string;
  study?: StudyMeta;
  driver?: string;
  driver_note?: string;
  roads?: GeoJSON.FeatureCollection | null;
  roads_note?: string | null;
  inspect?: {
    meta: { grid: [number, number]; depth_vmax_m: number;
            elev_min_m: number; elev_max_m: number };
    depth_u8: string;
    elev_u16: string;
  } | null;
  bounds: { west: number; south: number; east: number; north: number };
  grid: [number, number];
  dx_m: number;
  epsg: number;
  extent_km: [number, number];
  elevation_range_m: [number, number];
  flooded_km2: number;
  peak_depth_m: number;
  mean_depth_wet_m: number;
  depth_vmax_m: number;
  depth_png: string;
  terrain_png: string;
  forecast: { total_mm: number | null; peak_discharge_m3s: number | null };
  timing: { dem_seconds: number; seconds: number; steps: number };
  storm: { rain_mm_h: number; storm_hours: number; total_mm: number };
  /** Instantaneous depth through the run, where the solve was asked for it.
   *  Present in the shipped study bundle; absent from an ad-hoc analysis,
   *  which asks for the peak only because the user is waiting on it. */
  frames?: { t_h: number; flooded_km2: number; peak_depth_m: number;
             depth_png: string }[];
  levees?: { lat: number; lon: number }[];
  cached?: boolean;
}

interface Place { name: string; short: string; lat: number; lon: number; kind: string }

/** Storms a planner would actually test against, not abstract return periods. */
const STORMS = [
  { mm_h: 20, hours: 6, label: "Heavy rain", note: "120 mm" },
  { mm_h: 50, hours: 6, label: "Severe storm", note: "300 mm" },
  { mm_h: 80, hours: 6, label: "Extreme", note: "480 mm" },
] as const;

export interface PlacedLevee { lat: number; lon: number; height_m: number; width_m: number }

/** One solved option: its ranking numbers, and the field behind them. */
export interface AnywhereOption extends OptionRow {
  area_flooded_km2: number;
  peak_depth_m: number;
  /** The option's own depth grid, quantised like the baseline's. */
  depth_u8?: string;
}

/** A plan built for a place that has no precomputed anything. */
export interface AnywherePlan {
  ok: boolean;
  reason?: string;
  heights: number[];
  options: AnywhereOption[];
  /** Ground and baseline depth for the planning grid, which is coarser than
   *  the single solve's. An option's depth is read against this. */
  inspect?: {
    meta: { grid: [number, number]; depth_vmax_m: number;
            elev_min_m: number; elev_max_m: number };
    depth_u8: string;
    elev_u16: string;
  } | null;
  base_area_km2: number;
  base_peak_depth_m: number;
  cost_aud: number;
  budget_aud: number;
  band_pct: number;
  reduction_pct: number;
  guaranteed_pct: number;
  confidence_pct: number;
  band_note: string;
  /** The fitted saturation and the height cap, so a route's own margin can be
   *  calibrated client-side against the same surrogate the depth claim uses. */
  saturation_S: number;
  max_height_m: number;
  /** The recommended plan, solved in its own right. Present whenever the
   *  greedy search landed somewhere the library did not already cover. */
  chosen_solved?: boolean;
  chosen_reduction_pct?: number | null;
  surrogate_overclaim_pct?: number | null;
  chosen_depth_u8?: string | null;
  scenarios_solved: number;
  timing: { seconds: number };
  attribution: { site: number; height_m: number; marginal_pct: number; alone_pct: number }[];
  harmful: { site: number; alone_pct: number }[];
}

export function AnywherePanel({
  onResult,
  onPreview,
  onPlace,
  onGrids,
  onHeights,
  section,
  initial,
}: {
  /** Which of the three panel sections is open. Everything below the place
   *  chrome belongs to exactly one of them; before this, all three rendered
   *  the same thing and switching tabs changed nothing. */
  section: "forecast" | "roads" | "plan";
  onResult: (r: AnalysisResult | null) => void;
  onPreview: (c: { lat: number; lon: number } | null) => void;
  levees: PlacedLevee[];
  onLevees: (l: PlacedLevee[]) => void;
  placing: boolean;
  onPlacing: (v: boolean) => void;
  /** The town the user settled on, so the map can mark it. */
  onPlace?: (p: { lat: number; lon: number; name: string } | null) => void;
  /** Depth + ground grids for the analysed box, so the map can draw the flood
   *  as a body of water here too rather than only on the archive scene. */
  onGrids?: (g: InspectGrids | null) => void;
  /** Heights, per placed site, of whatever is currently drawn, so the map can
   *  show which sites are built in it and how high. */
  onHeights?: (h: number[] | null) => void;
  /** The way out, so the map can draw it. */
  onRoutes?: (r: unknown) => void;
  /** A study the portal opened on, already solved and shipped with the app.
   *
   *  Seeded by DERIVATION rather than by an effect that calls setState: the
   *  panel reads `res` and `chosen`, and those now fall back to the study
   *  until the user searches for somewhere else. An effect would have put one
   *  render on screen with the study's map and the panel's empty state. */
  initial?: (AnalysisResult & { study?: StudyMeta }) | null;
}) {
  const [q, setQ] = useState("");
  const [places, setPlaces] = useState<Place[]>([]);
  const [picked, setPicked] = useState<Place | null>(null);
  const [storm, setStorm] = useState(1);
  const [driver, setDriver] = useState<"rain" | "river" | "coastal">("river");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [ownRes, setRes] = useState<AnalysisResult | null>(null);
  /** The study until the user picks somewhere, then whatever they solved. */
  const res = ownRes ?? initial ?? null;
  const initialPlace = useMemo<Place | null>(() => {
    if (!initial) return null;
    const b = initial.bounds;
    return { name: initial.study?.place ?? initial.name,
             short: initial.study?.short ?? initial.name,
             lat: (b.south + b.north) / 2, lon: (b.west + b.east) / 2,
             kind: "study" };
  }, [initial]);
  const chosen = picked ?? initialPlace;
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Debounced place search.
  useEffect(() => {
    if (q.trim().length < 3) { setPlaces([]); return; }
    const id = setTimeout(() => {
      fetch(`/api/geocode?q=${encodeURIComponent(q)}`)
        .then((r) => r.json())
        .then((d) => setPlaces(d.results ?? []))
        .catch(() => setPlaces([]));
    }, 350);
    return () => clearTimeout(id);
  }, [q]);

  /** The 12 km box this place is modelled in. */
  const boundsFor = (p: Place) => {
    const dLat = 0.055;
    const dLon = 0.055 / Math.cos((p.lat * Math.PI) / 180);
    return { west: p.lon - dLon, east: p.lon + dLon,
             south: p.lat - dLat, north: p.lat + dLat };
  };

  // Memoised, and that is load-bearing rather than an optimisation.
  // RoadQueryPanel reports its query upward from a useMemo, so a bounds object
  // with a fresh identity on every render makes the query recompute, which
  // sets state in the parent, which renders again -- React #185, an infinite
  // loop. The district never hit it because its bounds come from a manifest
  // that is fetched once and never rebuilt.
  const box = useMemo(() => (chosen ? boundsFor(chosen) : null), [chosen]);

  useEffect(() => {
    onPlace?.(chosen ? { lat: chosen.lat, lon: chosen.lon, name: chosen.short } : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chosen]);

  /** The service sends depth and ground as the same quantised grids the
   *  district ships as files, so the point query and the road scoring are one
   *  piece of code rather than two that drift. */
  const grids = useMemo<InspectGrids | null>(() => decode(res?.inspect), [res]);

  /** The same, for the planning grid, which is coarser than the single solve's
   *  and is the grid every option was solved on. */


  /** The option the user picked, as a field.
   *
   *  Choosing an option used to move a radio button and nothing else, which
   *  made the ranking look decorative -- the whole point of a list of options
   *  is to see what each one does to the water. The solver had already run
   *  every one of them; only the field was being thrown away. */

  /** One option's field, for scoring a route against the solved library. */

  /** Dry ground once this option is built, as paint for the terrain. */
  useEffect(() => {
    onGrids?.(grids);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grids]);

  useEffect(() => {
    onHeights?.(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** One owner for what the map draws. The solve sets `res`; a chosen option
   *  overrides the field on it, leaving its bounds and terrain alone, so the
   *  comparison is the same ground with different water. */
  useEffect(() => {
    if (!res) { onResult(null); return; }
    onResult(res);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [res]);
  /* The road network used to be fetched here, lazily, when the roads or plan
     view opened. Neither can open inside this panel -- it renders only under
     `section === "forecast"` -- so the effect never ran and `onRoads` never
     fired. The map's network comes from the shipped study bundle instead
     (Planner.tsx loads `study.roads` alongside the depth grids). */

  /* `planHere` -- the ad-hoc plan solve -- was only ever called from the
     unreachable `section === "plan"` branch, so `plan` was always null and
     with it `planGrids`, `chosenOption` and `shown`. Planning now lives in
     the Plan tab's levee kit, against a named flood from the gauge record,
     which is a question this panel never asked. */

  const run = useCallback(async (p: Place, lv: PlacedLevee[] = []) => {
    setBusy(true); setErr(null); setElapsed(0);
    timer.current = setInterval(() => setElapsed((t) => t + 1), 1000);
    // A 12 km box: large enough to hold a town and its floodplain, small
    // enough that the solve answers while someone is still looking at it.
    const dLat = 0.055;
    const dLon = 0.055 / Math.cos((p.lat * Math.PI) / 180);
    const s = STORMS[storm];
    try {
      const r = await fetch("/api/analyse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          west: p.lon - dLon, east: p.lon + dLon,
          south: p.lat - dLat, north: p.lat + dLat,
          name: p.short, dx: 60,
          rain_mm_h: s.mm_h, storm_hours: s.hours, run_hours: 12,
          /* Ask for the frames too. Without this the response carries an
             empty `frames` array, the scrubber keeps playing the SHIPPED
             study's eighteen hours, and the timeline and the map end up
             describing two different floods on the same screen. */
          snapshots: 9,
          driver,
          levees: lv,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      setRes(d);
    } catch (e) {
      setErr(String((e as Error).message));
      setRes(null);
    } finally {
      setBusy(false);
      if (timer.current) clearInterval(timer.current);
    }
  }, [storm, driver]);

  return (
    <div className="px-5 pb-5 pt-4">
      <h2 className="eyebrow mb-2">Model anywhere</h2>

      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search a town, river or region…"
        className="w-full rounded-md border border-line bg-bg-inset px-2.5 py-1.5 text-[13px] text-ink placeholder:text-ink-faint focus:border-accent/60 focus:outline-none"
      />

      {/* Choosing a place sets the query to its own name, which then re-matches
          and shows the suggestion again under the input. Suppress the list once
          it is only offering back what has already been chosen. */}
      {places.length > 0 && !busy && !(chosen && q === chosen.short) && (
        <ul className="mt-1.5 flex flex-col gap-0.5">
          {places.map((p) => (
            <li key={`${p.lat},${p.lon}`}>
              <button
                onMouseEnter={() => onPreview({ lat: p.lat, lon: p.lon })}
                onClick={() => { setPicked(p); setPlaces([]); setQ(p.short); run(p); }}
                className="w-full truncate rounded px-2 py-1 text-left text-[13px] text-ink-mute hover:bg-bg-inset hover:text-ink"
                title={p.name}
              >
                {p.short}
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* What drives the flood. Asked before the solve because it changes what
          is solved, not how it is drawn -- and because "rain everywhere" is
          the assumption that quietly made every levee useless at a river
          city. */}
      <div className="mt-3 grid grid-cols-3 gap-1">
        {([["river", "River"], ["coastal", "Coastal"], ["rain", "Rainfall"]] as const)
          .map(([k, label]) => (
            <button
              key={k}
              onClick={() => setDriver(k)}
              className={`rounded-md border px-1 py-1.5 text-[13px] transition-colors ${
                driver === k
                  ? "border-ink bg-bg-inset font-medium text-ink"
                  : "border-line text-ink-mute hover:text-ink"
              }`}
            >
              {label}
            </button>
          ))}
      </div>
      <p className="anno mt-1.5 flex items-center gap-1.5">
        What drives the water
        <Explain title="What drives the flood">
          <p>
            <b>River</b> holds the discharge as a stage where the channel
            enters the box. <b>Coastal</b> holds a surge against the
            open-water edge. <b>Rainfall</b> falls evenly over the whole
            area — a levee has little to block when the water lands on both
            sides of it.
          </p>
          <p>
            River and coastal fall back to rainfall where there is no
            channel or no coast. The line under the storm pills says which
            one actually ran.
          </p>
        </Explain>
      </p>

      <div className="mt-3 grid grid-cols-3 gap-1">
        {STORMS.map((s, i) => (
          <button
            key={s.label}
            onClick={() => setStorm(i)}
            className={`rounded-md border px-1 py-1.5 text-[12px] transition-colors ${
              i === storm
                ? "border-accent/60 bg-accent/10 text-accent"
                : "border-line text-ink-mute hover:text-ink"
            }`}
            title={s.note}
          >
            {s.label}
            <span className="block text-[11px] opacity-70">{s.note}</span>
          </button>
        ))}
      </div>
      {/* What the service ACTUALLY did, which is not always what was asked: a
          river driver over terrain with no channel falls back to rainfall, and
          the user should be told rather than left to infer it. */}
      {res?.driver_note && !busy && (
        <p className="anno mt-2 leading-relaxed">{res.driver_note}</p>
      )}

      {chosen && !busy && (
        <button
          onClick={() => run(chosen)}
          className="mt-3 w-full rounded-md border border-accent/50 bg-accent/10 px-3 py-1.5 text-[13px] text-accent transition-colors hover:bg-accent/20"
        >
          Re-run {chosen.short}
        </button>
      )}

      {busy && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] text-ink">Modelling {chosen?.short}…</p>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">
            About a minute.
          </p>
          <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-bg">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-1000"
              style={{ width: `${Math.min(96, elapsed * 1.6)}%` }}
            />
          </div>
          <p className="tnum mt-1 text-[11px] text-ink-faint">{elapsed}s</p>
        </div>
      )}

      {err && (
        <div className="mt-3 rounded-lg border border-bad/40 bg-bad/5 p-3">
          <p className="text-[13px] font-medium text-bad">Could not model that area</p>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-mute">{err}</p>
        </div>
      )}

      {/* ---- FORECAST: this place's own weather, not the district's ----- */}
      {section === "forecast" && chosen && (
        <div className="-mx-5 mt-4 border-t border-line">
          <ForecastPanel lat={chosen.lat} lon={chosen.lon} place={chosen.short} />
          <ObservePanel bounds={box} />
        </div>
      )}

      {/* ---- ROADS: the real network, scored against the real depths ---- */}
      {/* Everything the planner reached through this panel under
          `section === "roads"` and `section === "plan"` used to live here:
          candidate sites clicked onto the map, the option library, the
          budget slider, the road-cut summary and the evacuation route.
          None of it could render.

          `AnywherePanel` is mounted only when `tab === "anywhere"`, and
          `tab` is derived -- `anywhere` happens only when
          `section === "forecast"`. A branch inside it asking for
          `section === "plan"` is asking for two things that cannot both be
          true, so 272 lines of it were dead from the day the navigation
          became three tabs.

          What was worth keeping has been moved rather than deleted: the
          evacuation answer is now `RoutePanel` in the Roads tab, on the
          shipped study's own network (Planner.tsx). The road-cut summary was
          already duplicated by `RoadsOut` in this column, which also drives
          the map's road colours. The option library and the click-to-place
          site flow belong to the levee kit, which replaced them. */}


      {/* ---- what doing nothing costs here ---------------------------- */}
      {section === "forecast" && res?.study && !ownRes && !busy && (
        <div className="mx-4 mt-4 rounded-lg border border-line bg-bg-raised p-3">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="chip-harm">{res.study.evidence}</span>
            <span className="text-[13px] font-semibold text-ink">{res.study.place}</span>
          </div>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
            {res.study.event}, solved here on {res.study.terrain}. This is our
            solver&apos;s answer, not a reference flood.
          </p>
          {res.study.skill && (
            <>
              <div className="mt-2 border-t border-line-soft pt-2">
                <div className="flex items-baseline justify-between py-0.5">
                  <span className="text-[13px] text-ink-mute">
                    Overlap with FEMA&apos;s 1% floodplain
                  </span>
                  <span className="tnum text-[13px] font-medium text-ink">
                    CSI {res.study.skill.csi.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between py-0.5">
                  <span className="text-[13px] text-ink-mute">Of it we find</span>
                  <span className="tnum text-[13px] font-medium text-ink">
                    {(res.study.skill.pod * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-baseline justify-between py-0.5">
                  <span className="text-[13px] text-ink-mute">
                    Of what we call wet, outside it
                  </span>
                  <span className="tnum text-[13px] font-medium text-ink">
                    {(res.study.skill.far * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <p className="anno mt-1.5 leading-relaxed">
                {res.study.skill.ours_km2} km² against {res.study.skill.reference_km2} km²
                over {res.study.skill.reference_polygons} mapped polygons. Total extent
                agrees far better than the overlap does, which is why the overlap is the
                number shown.
              </p>
            </>
          )}
          <details className="fold mt-1">
            <summary>What this answer is not</summary>
            <ul className="flex flex-col gap-1 pb-2">
              {res.study.caveats.map((c) => (
                <li key={c} className="text-[13px] leading-relaxed text-ink-mute">{c}</li>
              ))}
            </ul>
          </details>
        </div>
      )}


      {/* ---- candidate sites ------------------------------------------- */}

      {/* ---- budget ----------------------------------------------------- */}


      {/* ---- the recommendation, and what it is worth ------------------- */}
      {/* A recommendation is only a recommendation if it beats doing nothing by
          more than the tool's own resolution. Below that, saying "0.1%, at
          least 90% of the time" dresses up a null result as a plan. */}


      {/* ---- can people get out ---------------------------------------- */}

      {/* ---- the action ------------------------------------------------- */}
    </div>
  );
}
