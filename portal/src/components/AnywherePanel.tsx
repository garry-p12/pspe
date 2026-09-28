"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { formatChange, MATERIAL_PCT, OptionList, type OptionRow } from "./OptionList";

export interface AnalysisResult {
  name: string;
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

/** Storms a planner would actually test against, not abstract return periods. */
const STORMS = [
  { mm_h: 20, hours: 6, label: "Heavy rain", note: "120 mm — a wet day" },
  { mm_h: 50, hours: 6, label: "Severe storm", note: "300 mm — major flooding" },
  { mm_h: 80, hours: 6, label: "Extreme", note: "480 mm — near the 2022 event" },
] as const;

export interface PlacedLevee { lat: number; lon: number; height_m: number; width_m: number }

/** A plan built for a place that has no precomputed anything. */
export interface AnywherePlan {
  ok: boolean;
  reason?: string;
  heights: number[];
  options: OptionRow[];
  base_area_km2: number;
  base_peak_depth_m: number;
  cost_aud: number;
  budget_aud: number;
  band_pct: number;
  reduction_pct: number;
  guaranteed_pct: number;
  confidence_pct: number;
  band_note: string;
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
}: {
  onResult: (r: AnalysisResult | null) => void;
  onPreview: (c: { lat: number; lon: number } | null) => void;
  levees: PlacedLevee[];
  onLevees: (l: PlacedLevee[]) => void;
  placing: boolean;
  onPlacing: (v: boolean) => void;
}) {
  const [q, setQ] = useState("");
  const [places, setPlaces] = useState<Place[]>([]);
  const [chosen, setChosen] = useState<Place | null>(null);
  const [storm, setStorm] = useState(1);
  const [plan, setPlan] = useState<AnywherePlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [budgetM, setBudgetM] = useState(20);
  const [planChosen, setPlanChosen] = useState<string>("");
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

  const planHere = useCallback(async (p: Place, lv: PlacedLevee[]) => {
    setPlanning(true); setPlanErr(null); setPlan(null);
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
  }, [storm, budgetM]);

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
          levees: lv,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      setRes(d); onResult(d);
      if (lv.length === 0) setBaseline(d);
    } catch (e) {
      setErr(String((e as Error).message));
      setRes(null); onResult(null);
    } finally {
      setBusy(false);
      if (timer.current) clearInterval(timer.current);
    }
  }, [storm, onResult]);

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
      <p className="mt-1.5 text-[12px] text-ink-faint">{STORMS[storm].note}</p>

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

      {/* ---- what doing nothing costs here ---------------------------- */}
      {res && !busy && chosen && (
        <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
          <div className="flex items-baseline justify-between">
            <span className="text-[13px] font-medium text-ink">{chosen.short}</span>
            <span className="text-[13px] text-ink-mute">Do nothing</span>
          </div>
          <div className="mt-2.5 flex gap-7">
            <div>
              <p className="tnum text-[26px] font-semibold leading-none text-ink">
                {res.flooded_km2.toFixed(0)} km²
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
      {res && !busy && chosen && (
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
      {res && !busy && chosen && levees.length >= 2 && (
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
        </div>
      )}

      {planErr && (
        <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">{planErr}</p>
      )}

      {/* ---- the recommendation, and what it is worth ------------------- */}
      {/* A recommendation is only a recommendation if it beats doing nothing by
          more than the tool's own resolution. Below that, saying "0.1%, at
          least 90% of the time" dresses up a null result as a plan. */}
      {plan?.ok && plan.reduction_pct < MATERIAL_PCT && (
        <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
          <h3 className="eyebrow">No measure here is worth building</h3>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
            The best allocation of A${budgetM}M changes flood depth by{" "}
            {plan.reduction_pct.toFixed(2)}%, which this model cannot
            distinguish from doing nothing. Try candidate sites closer to where
            the water actually runs, or a larger budget.
          </p>
          <p className="anno mt-2.5 leading-relaxed">
            {plan.scenarios_solved} options solved on this terrain in{" "}
            {plan.timing.seconds.toFixed(0)}s.
          </p>
        </div>
      )}

      {plan?.ok && plan.reduction_pct >= MATERIAL_PCT && (
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
              <span className="text-[13px] text-ink-mute">Estimated</span>
              <span className="tnum text-[18px] font-semibold text-ink">
                {plan.reduction_pct.toFixed(1)}%
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
          <p className="anno mt-2 leading-relaxed">
            {plan.scenarios_solved} options solved on this terrain in{" "}
            {plan.timing.seconds.toFixed(0)}s. {plan.band_note}
          </p>
        </div>
      )}

      {/* ---- the action ------------------------------------------------- */}
      {res && !busy && chosen && (
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
