"use client";

import { useCallback, useEffect, useRef, useState } from "react";

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
          max_height_m: 3.0, budget_aud: 20e6, delta: 0.1,
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
  }, [storm]);

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
    <div className="border-b border-line-soft p-4">
      <p className="eyebrow mb-2">Model anywhere</p>

      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search a town, river or region…"
        className="w-full rounded-md border border-line bg-bg-inset px-2.5 py-1.5 text-[12.5px] text-ink placeholder:text-ink-faint focus:border-accent/60 focus:outline-none"
      />

      {places.length > 0 && !busy && (
        <ul className="mt-1.5 flex flex-col gap-0.5">
          {places.map((p) => (
            <li key={`${p.lat},${p.lon}`}>
              <button
                onMouseEnter={() => onPreview({ lat: p.lat, lon: p.lon })}
                onClick={() => { setChosen(p); setPlaces([]); setQ(p.short); run(p); }}
                className="w-full truncate rounded px-2 py-1 text-left text-[11.5px] text-ink-mute hover:bg-bg-inset hover:text-ink"
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
            className={`rounded-md border px-1 py-1.5 text-[10.5px] transition-colors ${
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
      <p className="mt-1.5 text-[10.5px] text-ink-faint">{STORMS[storm].note}</p>

      {chosen && !busy && (
        <button
          onClick={() => run(chosen)}
          className="mt-3 w-full rounded-md border border-accent/50 bg-accent/10 px-3 py-1.5 text-[12px] text-accent transition-colors hover:bg-accent/20"
        >
          Re-run {chosen.short} with this storm
        </button>
      )}

      {busy && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[12px] text-ink">Modelling {chosen?.short}…</p>
          <p className="mt-1 text-[11px] leading-relaxed text-ink-faint">
            Fetching terrain, then solving the shallow-water equations over the
            catchment. About a minute.
          </p>
          <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-bg">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-1000"
              style={{ width: `${Math.min(96, elapsed * 1.6)}%` }}
            />
          </div>
          <p className="tnum mt-1 text-[10px] text-ink-faint">{elapsed}s</p>
        </div>
      )}

      {err && (
        <div className="mt-3 rounded-lg border border-bad/40 bg-bad/5 p-3">
          <p className="text-[12px] font-medium text-bad">Could not model that area</p>
          <p className="mt-1 text-[11px] leading-relaxed text-ink-mute">{err}</p>
        </div>
      )}

      {res && !busy && chosen && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <div className="mb-2 flex items-center justify-between">
            <p className="eyebrow">Mitigation</p>
            {levees.length > 0 && (
              <button
                onClick={() => { onLevees([]); run(chosen, []); }}
                className="text-[10.5px] text-ink-faint hover:text-ink"
              >
                clear
              </button>
            )}
          </div>
          <button
            onClick={() => onPlacing(!placing)}
            className={`w-full rounded-md border px-2 py-1.5 text-[11.5px] transition-colors ${
              placing
                ? "border-accent/60 bg-accent/15 text-accent"
                : "border-line text-ink-mute hover:text-ink"
            }`}
          >
            {placing ? "Click the map to place a levee" : "Add a levee"}
          </button>
          {levees.length > 0 && (
            <>
              <p className="tnum mt-2 text-[11px] text-ink-mute">
                {levees.length} levee{levees.length > 1 ? "s" : ""} · 2.5 m high,
                600 m long
              </p>
              <button
                onClick={() => run(chosen, levees)}
                className="mt-2 w-full rounded-md border border-accent/50 bg-accent/10 px-2 py-1.5 text-[11.5px] text-accent hover:bg-accent/20"
              >
                Model with these levees
              </button>

              {/* The framework, not just a solve. Treats the placed markers as
                  CANDIDATE sites, builds a scenario library for them on the
                  spot, and returns a plan with a calibrated margin. Four sites
                  is where a 90% margin becomes attainable; below that the
                  service refuses to state one and says so. */}
              <button
                onClick={() => planHere(chosen, levees)}
                disabled={planning || levees.length < 2}
                className="mt-2 w-full rounded-md border border-line px-2 py-1.5
                           text-[11.5px] text-ink transition-colors
                           hover:border-accent/50 disabled:opacity-50"
              >
                {planning
                  ? "Solving every option…"
                  : `Plan the best use of a budget (${levees.length} site${levees.length > 1 ? "s" : ""})`}
              </button>
              {levees.length < 4 && (
                <p className="mt-1 text-[10px] leading-snug text-ink-faint">
                  Four or more candidate sites are needed before a 90% margin can
                  be calibrated.
                </p>
              )}
              {planErr && (
                <p className="mt-1 text-[10.5px] leading-snug text-ink-mute">{planErr}</p>
              )}
              {plan?.ok && (
                <div className="mt-2 border-t border-line-soft pt-2">
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10.5px] text-ink-faint">Estimated</span>
                    <span className="tnum text-[13px] font-semibold text-ink">
                      {plan.reduction_pct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10.5px] text-ink-faint">
                      At least, {plan.confidence_pct}% of the time
                    </span>
                    <span className="tnum text-[13px] font-semibold text-accent">
                      {plan.guaranteed_pct.toFixed(1)}%
                    </span>
                  </div>
                  <ul className="mt-1.5 space-y-0.5">
                    {plan.attribution.map((a) => (
                      <li key={a.site} className="flex justify-between text-[10.5px]">
                        <span className="text-ink-mute">
                          Site {a.site} at {a.height_m.toFixed(1)} m
                        </span>
                        <span className="tnum text-ink-mute">+{a.marginal_pct.toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>
                  {plan.harmful.length > 0 && (
                    <p className="mt-1 text-[10px] leading-snug text-ink-faint">
                      Rejected:{" "}
                      {plan.harmful.map((h) => `site ${h.site} (${h.alone_pct.toFixed(1)}%)`).join(", ")}
                      {" "}— measured as deepening flooding here.
                    </p>
                  )}
                  <p className="mt-1 text-[10px] leading-snug text-ink-faint">
                    {plan.scenarios_solved} options solved on this terrain in{" "}
                    {plan.timing.seconds}s. {plan.band_note}
                  </p>
                </div>
              )}
              {baseline && res && res.levees && res.levees.length > 0 && (
                <div className="mt-2 border-t border-line-soft pt-2">
                  <p className="text-[11.5px] leading-relaxed">
                    {res.flooded_km2 < baseline.flooded_km2 - 0.05 ? (
                      <span className="text-ok">
                        Reduces flooding by{" "}
                        {(baseline.flooded_km2 - res.flooded_km2).toFixed(2)} km²
                      </span>
                    ) : res.flooded_km2 > baseline.flooded_km2 + 0.05 ? (
                      <span className="text-bad">
                        Increases flooding by{" "}
                        {(res.flooded_km2 - baseline.flooded_km2).toFixed(2)} km² —
                        this holds water in
                      </span>
                    ) : (
                      <span className="text-ink-mute">
                        No material change ({baseline.flooded_km2.toFixed(1)} →{" "}
                        {res.flooded_km2.toFixed(1)} km²)
                      </span>
                    )}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {res && !busy && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="mb-2 text-[12.5px] font-semibold text-ink">{res.name}</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="eyebrow mb-0.5">Area flooded</p>
              <p className="tnum text-[18px] font-semibold leading-none text-bad">
                {res.flooded_km2.toFixed(1)} km²
              </p>
            </div>
            <div>
              <p className="eyebrow mb-0.5">Deepest</p>
              <p className="tnum text-[18px] font-semibold leading-none text-ink">
                {res.peak_depth_m.toFixed(1)} m
              </p>
            </div>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-ink-mute">
            {res.extent_km[0]}×{res.extent_km[1]} km at {res.dx_m} m ·
            ground {res.elevation_range_m[0].toFixed(0)}–
            {res.elevation_range_m[1].toFixed(0)} m
          </p>
          <p className="mt-2 border-t border-line-soft pt-2 text-[10.5px] text-ink-faint">
            Terrain from Copernicus DEM · solved in{" "}
            {(res.timing.dem_seconds + res.timing.seconds).toFixed(0)}s
            {res.cached ? " (cached)" : ""}
          </p>
        </div>
      )}
    </div>
  );
}
