"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { describe, formatChange, MATERIAL_PCT, OptionList, type OptionRow } from "./OptionList";
import { ForecastPanel } from "./ForecastPanel";
import { RoutePanel } from "./RoutePanel";
import { ObservePanel } from "./ObservePanel";
import { RoadQueryPanel } from "./RoadQueryPanel";
import { decodeGrids as decode, type InspectGrids } from "@/lib/inspect";
import { depthTexture, withDepth } from "@/lib/depthtex";
import type { RoadQuery } from "@/lib/roadquery";

export interface AnalysisResult {
  name: string;
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
  levees?: { lat: number; lon: number }[];
  cached?: boolean;
}

interface Place { name: string; short: string; lat: number; lon: number; kind: string }

/** Area, at a precision that does not round a flood away. */
function km2(v: number): string {
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

/** Storms a planner would actually test against, not abstract return periods. */
const STORMS = [
  { mm_h: 20, hours: 6, label: "Heavy rain", note: "120 mm — a wet day" },
  { mm_h: 50, hours: 6, label: "Severe storm", note: "300 mm — major flooding" },
  { mm_h: 80, hours: 6, label: "Extreme", note: "480 mm — near the 2022 event" },
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
  levees,
  onLevees,
  placing,
  onPlacing,
  onPlace,
  onGrids,
  onHeights,
  onRoutes,
  section,
  onRoadQuery,
  onRoads,
}: {
  /** Which of the three panel sections is open. Everything below the place
   *  chrome belongs to exactly one of them; before this, all three rendered
   *  the same thing and switching tabs changed nothing. */
  section: "forecast" | "roads" | "plan";
  /** Road cuts here, so the map can colour the network. */
  onRoadQuery: (q: RoadQuery | null) => void;
  /** Hand the fetched network up so the map can draw it. */
  onRoads?: (r: GeoJSON.FeatureCollection | null) => void;
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
}) {
  const [q, setQ] = useState("");
  const [places, setPlaces] = useState<Place[]>([]);
  const [chosen, setChosen] = useState<Place | null>(null);
  const [storm, setStorm] = useState(1);
  const [plan, setPlan] = useState<AnywherePlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [budgetM, setBudgetM] = useState(20);
  const [driver, setDriver] = useState<"rain" | "river" | "coastal">("river");
  const [planChosen, setPlanChosen] = useState<string>("");
  /** What the vehicle being routed can drive through, metres. */
  const [vehicle, setVehicle] = useState(0.25);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [res, setRes] = useState<AnalysisResult | null>(null);
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

  const [baseline, setBaseline] = useState<AnalysisResult | null>(null);

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
  const planGrids = useMemo<InspectGrids | null>(
    () => (plan?.ok ? decode(plan.inspect) : null), [plan]);

  const chosenOption = useMemo(
    () => (plan?.ok ? plan.options.find((o) => o.id === planChosen) ?? null : null),
    [plan, planChosen]);

  /** The option the user picked, as a field.
   *
   *  Choosing an option used to move a radio button and nothing else, which
   *  made the ranking look decorative -- the whole point of a list of options
   *  is to see what each one does to the water. The solver had already run
   *  every one of them; only the field was being thrown away. */
  const shown = useMemo<InspectGrids | null>(() => {
    if (!planGrids || !chosenOption?.depth_u8) return null;
    return withDepth(planGrids, chosenOption.depth_u8);
  }, [planGrids, chosenOption]);

  /** One option's field, for scoring a route against the solved library. */
  const planGridFor = useCallback((i: number) => {
    const u8 = plan?.options?.[i]?.depth_u8;
    return planGrids && u8 ? withDepth(planGrids, u8) : null;
  }, [plan, planGrids]);

  /** Dry ground once this option is built, as paint for the terrain. */
  const shownPng = useMemo(() => (shown ? depthTexture(shown) : null), [shown]);

  useEffect(() => {
    onGrids?.(shown ?? grids);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown, grids]);

  useEffect(() => {
    onHeights?.(chosenOption ? chosenOption.heights : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chosenOption]);

  /** One owner for what the map draws. The solve sets `res`; a chosen option
   *  overrides the field on it, leaving its bounds and terrain alone, so the
   *  comparison is the same ground with different water. */
  useEffect(() => {
    if (!res) { onResult(null); return; }
    onResult(shown && shownPng ? { ...res, depth_png: shownPng } : res);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [res, shown, shownPng]);

  /** Roads are fetched when the roads view is opened, not with the solve:
   *  Overpass is slow and unreliable, and it has no business delaying a flood
   *  answer or failing one. */
  const [roads, setRoads] = useState<GeoJSON.FeatureCollection | null>(null);
  const [roadsNote, setRoadsNote] = useState<string | null>(null);
  const [roadsBusy, setRoadsBusy] = useState(false);
  const roadsFor = useRef<string>("");

  useEffect(() => {
    if ((section !== "roads" && section !== "plan") || !box) return;
    const b = box;
    const key = `${b.west.toFixed(4)},${b.south.toFixed(4)}`;
    if (roadsFor.current === key) return;
    roadsFor.current = key;
    setRoads(null); setRoadsNote(null); setRoadsBusy(true);
    fetch("/api/roads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(b),
    })
      .then((r) => r.json())
      .then((d) => {
        setRoads(d.roads ?? null);
        setRoadsNote(d.ok ? null : (d.reason ?? "road data unavailable"));
        onRoads?.(d.roads ?? null);
      })
      .catch((e) => setRoadsNote(String(e.message ?? e)))
      .finally(() => setRoadsBusy(false));
  }, [section, box, onRoads]);

  const planHere = useCallback(async (p: Place, lv: PlacedLevee[]) => {
    setPlanning(true); setPlanErr(null); setPlan(null); setPlanChosen("");
    try {
      // Same box as the single solve, coarser: every candidate option is solved
      // in one batch, so the grid buys the option count back.
      const dLat = 0.055;
      const dLon = 0.055 / Math.cos((p.lat * Math.PI) / 180);
      const s = STORMS[storm];
      const r = await fetch("/api/plan-anywhere", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          west: p.lon - dLon, east: p.lon + dLon,
          south: p.lat - dLat, north: p.lat + dLat,
          name: p.short, dx: 90,
          rain_mm_h: s.mm_h, storm_hours: s.hours, run_hours: 10,
          driver,
          sites: lv.map((l) => ({ lat: l.lat, lon: l.lon, width_m: l.width_m })),
          max_height_m: 3.0, budget_aud: budgetM * 1e6, delta: 0.1,
          protect: { lat: p.lat, lon: p.lon, radius_m: 2500 },
        }),
      });
      const out: AnywherePlan = await r.json();
      if (!r.ok || !out.ok) throw new Error(out.reason ?? `service returned ${r.status}`);
      setPlan(out);
    } catch (e) {
      setPlanErr(e instanceof Error ? e.message : "could not plan here");
    } finally {
      setPlanning(false);
    }
  }, [storm, budgetM, driver]);

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
          driver,
          levees: lv,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      setRes(d);
      if (lv.length === 0) setBaseline(d);
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
                onClick={() => { setChosen(p); setPlaces([]); setQ(p.short); run(p); }}
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
      <p className="anno mt-1.5 leading-relaxed">
        {driver === "river"
          ? "Discharge held as a stage where the channel enters, from the flood forecast for this reach. Falls back to rainfall if no channel crosses this area."
          : driver === "coastal"
            ? "A surge held against the open-water edge. Falls back to rainfall if this area has no coast."
            : "Rain falling evenly over the whole area. A levee has little to block when the water lands on both sides of it."}
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
          </button>
        ))}
      </div>
      <p className="anno mt-1.5">{STORMS[storm].note}</p>
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
          Re-run {chosen.short} with this storm
        </button>
      )}

      {busy && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] text-ink">Modelling {chosen?.short}…</p>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">
            Fetching terrain, then solving the shallow-water equations over the
            catchment. About a minute.
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
      {section === "roads" && chosen && res && !busy && roadsBusy && (
        <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
          <p className="text-[13px] text-ink">Fetching the road network…</p>
          <p className="anno mt-1.5 leading-relaxed">
            OpenStreetMap, over this box. Usually under a minute.
          </p>
        </div>
      )}

      {section === "roads" && chosen && res && !busy && !roadsBusy && (
        grids && roads?.features?.length ? (
          <div className="-mx-5 mt-4 border-t border-line">
            <RoadQueryPanel
              roads={roads}
              grids={grids}
              bounds={box}
              onQuery={onRoadQuery}
            />
          </div>
        ) : (
          <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
            <h3 className="eyebrow">Roads unavailable here</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
              {roadsNote ??
                "OpenStreetMap returned no roads for this box."}
            </p>
          </div>
        )
      )}

      {/* ---- what doing nothing costs here ---------------------------- */}
      {section === "plan" && res && !busy && chosen && (
        <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
          <div className="flex items-baseline justify-between">
            <span className="text-[13px] font-medium text-ink">{chosen.short}</span>
            <span className="text-[13px] text-ink-mute">Do nothing</span>
          </div>
          <div className="mt-2.5 flex gap-7">
            <div>
              <p className="tnum text-[26px] font-semibold leading-none text-ink">
                {km2(res.flooded_km2)} km²
              </p>
              <p className="anno mt-1.5">land flooded</p>
            </div>
            <div>
              <p className="tnum text-[26px] font-semibold leading-none text-ink">
                {res.peak_depth_m.toFixed(1)} m
              </p>
              <p className="anno mt-1.5">deepest</p>
            </div>
          </div>
          <p className="anno mt-3">
            {res.extent_km[0].toFixed(1)}x{res.extent_km[1].toFixed(1)} km at{" "}
            {res.dx_m.toFixed(0)} m · terrain from Copernicus DEM · solved in{" "}
            {res.timing.seconds.toFixed(0)}s
          </p>
        </div>
      )}

      {/* ---- candidate sites ------------------------------------------- */}
      {section === "plan" && res && !busy && chosen && (
        <div className="mt-5">
          <div className="flex items-baseline justify-between">
            <h3 className="eyebrow">Candidate sites</h3>
            {levees.length > 0 && (
              <button
                onClick={() => { onLevees([]); setPlan(null); }}
                className="text-[13px] text-ink-mute hover:text-ink"
              >
                clear
              </button>
            )}
          </div>
          <button
            onClick={() => onPlacing(!placing)}
            className={`mt-2 w-full rounded-lg border px-3 py-2.5 text-[13px] transition-colors ${
              placing ? "border-ink bg-bg-inset text-ink" : "border-line text-ink-mute hover:text-ink"
            }`}
          >
            {placing ? "Click the map to place a site" : "Add a candidate site"}
          </button>
          <p className="anno mt-2">
            {levees.length} placed · 2.5 m high, 600 m long
            {levees.length < 4 && " · four or more are needed before a 90% margin can be calibrated"}
          </p>
        </div>
      )}

      {/* ---- budget ----------------------------------------------------- */}
      {section === "plan" && res && !busy && chosen && levees.length >= 2 && (
        <div className="mt-5">
          <div className="flex items-baseline justify-between">
            <label htmlFor="any-budget" className="eyebrow">Budget</label>
            <span className="tnum text-[15px] font-semibold text-ink">A${budgetM}M</span>
          </div>
          <input
            id="any-budget"
            type="range" min={2} max={60} step={1} value={budgetM}
            onChange={(e) => setBudgetM(Number(e.target.value))}
            className="mt-2 w-full accent-[var(--ink)]"
          />
          {plan?.ok && (
            <OptionList
              options={plan.options}
              budgetM={budgetM}
              chosen={planChosen}
              onChoose={setPlanChosen}
            />
          )}
          {chosenOption && plan && (
            <div className="mt-3 rounded-xl border border-ink/40 bg-bg-inset px-4 py-3.5">
              <div className="flex items-baseline justify-between">
                <h3 className="eyebrow">On the map now</h3>
                <button
                  onClick={() => setPlanChosen("")}
                  className="text-[13px] text-ink-mute hover:text-ink"
                >
                  back to do nothing
                </button>
              </div>
              <p className="mt-1.5 text-[13px] text-ink">
                {describe(chosenOption).title} · {describe(chosenOption).detail}
              </p>
              <div className="mt-2.5 flex gap-7">
                <div>
                  <p className="tnum text-[22px] font-semibold leading-none text-ink">
                    {km2(chosenOption.area_flooded_km2)} km²
                  </p>
                  <p className="anno mt-1.5">
                    flooded · was {km2(plan.base_area_km2)}
                  </p>
                </div>
                <div>
                  <p className="tnum text-[22px] font-semibold leading-none text-ink">
                    {chosenOption.peak_depth_m.toFixed(1)} m
                  </p>
                  <p className="anno mt-1.5">
                    deepest · was {plan.base_peak_depth_m.toFixed(1)}
                  </p>
                </div>
              </div>
              {/* Two identical-looking columns are not a readout. An option
                  that does nothing has to say so, or the numbers invite the
                  reader to find a difference that is not there. */}
              {Math.abs(chosenOption.core_reduction_pct ?? 0) < MATERIAL_PCT && (
                <p className="anno mt-2.5 leading-relaxed">
                  This option leaves the water where it is. The markers show
                  where it would be built, and what it would cost.
                </p>
              )}
              {!chosenOption.depth_u8 && (
                <p className="anno mt-2.5 leading-relaxed">
                  This option was solved before the map could show it. Solve
                  again to see its water.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {planErr && (
        <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">{planErr}</p>
      )}

      {/* ---- the recommendation, and what it is worth ------------------- */}
      {/* A recommendation is only a recommendation if it beats doing nothing by
          more than the tool's own resolution. Below that, saying "0.1%, at
          least 90% of the time" dresses up a null result as a plan. */}
      {section === "plan" && plan?.ok && plan.reduction_pct < MATERIAL_PCT && (
        <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
          <h3 className="eyebrow">No measure here is worth building</h3>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
            The best allocation of A${budgetM}M changes flood depth by{" "}
            {plan.reduction_pct.toFixed(2)}%, which this model cannot
            distinguish from doing nothing.
          </p>
          <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
            That is often the right answer rather than a failure. The storm is
            applied as rain falling across the whole area, and a levee can only
            redirect water that arrives from somewhere — it has nothing to block
            when the rain lands on both sides of it. Embankments earn their cost
            where terrain funnels water along a path, not on flat ground that
            floods everywhere at once.
          </p>
          <p className="anno mt-2.5 leading-relaxed">
            {plan.scenarios_solved} options solved on this terrain in{" "}
            {plan.timing.seconds.toFixed(0)}s.
          </p>
        </div>
      )}

      {section === "plan" && plan?.ok && plan.reduction_pct >= MATERIAL_PCT && (
        <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
          <h3 className="eyebrow">Best plan for this budget</h3>
          <ul className="mt-2 space-y-0.5">
            {plan.attribution.map((a) => (
              <li key={a.site} className="flex justify-between text-[13px]">
                <span className="text-ink">Site {a.site} at {a.height_m.toFixed(1)} m</span>
                <span className="tnum text-ink-mute">{formatChange(a.marginal_pct)}</span>
              </li>
            ))}
          </ul>
          <div className="mt-3 border-t border-line-soft pt-3">
            <div className="flex items-baseline justify-between">
              <span className="text-[13px] text-ink-mute">
                {plan.chosen_solved ? "Measured for this exact plan" : "Estimated"}
              </span>
              <span className="tnum text-[18px] font-semibold text-ink">
                {(plan.chosen_solved && plan.chosen_reduction_pct != null
                  ? plan.chosen_reduction_pct : plan.reduction_pct).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-[13px] text-ink-mute">
                At least, {plan.confidence_pct}% of the time
              </span>
              <span className="tnum text-[18px] font-semibold text-ink">
                {plan.guaranteed_pct.toFixed(1)}%
              </span>
            </div>
          </div>
          {plan.harmful.length > 0 && (
            <p className="anno mt-3 leading-relaxed">
              Rejected:{" "}
              {plan.harmful.map((h) => `site ${h.site} (${formatChange(h.alone_pct)})`).join(", ")}
              {" "}— measured as deepening flooding here.
            </p>
          )}
          {plan.chosen_solved && plan.surrogate_overclaim_pct != null &&
           Math.abs(plan.surrogate_overclaim_pct) >= 0.5 && (
            <p className="anno mt-2.5 leading-relaxed">
              The fast estimate put this at {plan.reduction_pct.toFixed(1)}%
              {plan.surrogate_overclaim_pct > 0 ? ", which is " : ", which is "}
              {Math.abs(plan.surrogate_overclaim_pct).toFixed(1)} points{" "}
              {plan.surrogate_overclaim_pct > 0 ? "more" : "less"} than this plan
              actually achieves. The figure above is a full run of this exact
              plan, not the estimate the search used to find it.
            </p>
          )}
          <p className="anno mt-2 leading-relaxed">
            {plan.scenarios_solved} options solved on this terrain in{" "}
            {plan.timing.seconds.toFixed(0)}s. {plan.band_note}
          </p>
        </div>
      )}

      {/* ---- can people get out ---------------------------------------- */}
      {section === "plan" && res && !busy && chosen && (
        <RoutePanel
          roads={roads}
          grids={shown ?? grids}
          bounds={box}
          from={{ lat: chosen.lat, lon: chosen.lon, name: chosen.short }}
          to={null}
          vehicle={vehicle}
          onVehicle={setVehicle}
          plan={plan?.ok ? {
            heights: plan.heights,
            max_height_m: plan.max_height_m,
            saturation_S: plan.saturation_S,
            confidence_pct: plan.confidence_pct,
            options: plan.options,
            chosen_solved: plan.chosen_solved,
          } : null}
          planGridFor={planGridFor}
          chosenGrid={
            plan?.ok && plan.chosen_solved && plan.chosen_depth_u8 && planGrids
              ? withDepth(planGrids, plan.chosen_depth_u8) : null
          }
          onRoutes={onRoutes}
        />
      )}

      {/* ---- the action ------------------------------------------------- */}
      {section === "plan" && res && !busy && chosen && (
        <div className="sticky bottom-0 -mx-5 mt-5 border-t border-line bg-bg px-5 py-3.5">
          <button
            onClick={() => planHere(chosen, levees)}
            disabled={planning || levees.length < 2}
            className="w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                       text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {planning
              ? "Solving every option…"
              : levees.length < 2
                ? "Place at least two candidate sites"
                : `Find the best plan for A$${budgetM}M`}
          </button>
        </div>
      )}
    </div>
  );
}
