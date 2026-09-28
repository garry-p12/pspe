"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { PlannerMap, type MeasureMarker } from "./PlannerMap";
import { ForecastPanel } from "./ForecastPanel";
import { MeasureBuilder } from "./MeasureBuilder";
import { RoadQueryPanel } from "./RoadQueryPanel";
import { AnywherePanel, type AnalysisResult, type PlacedLevee } from "./AnywherePanel";
import { PlanPanel } from "./PlanPanel";
import { ObservePanel, type Observation } from "./ObservePanel";
import type { RoadQuery } from "@/lib/roadquery";
import { loadManifest } from "@/lib/data";
import type { Manifest } from "@/lib/types";
import { eventLabel, km, money, type OpsPayload } from "@/lib/ops";
import { useRegion } from "./RegionContext";
import {
  loadInspect, nearestRoad, sampleAt,
  type InspectGrids, type PointInfo,
} from "@/lib/inspect";

function Metric({
  label, value, sub, tone = "default",
}: {
  label: string; value: string; sub?: string;
  tone?: "default" | "good" | "bad";
}) {
  const c = { default: "text-ink", good: "text-ok", bad: "text-bad" }[tone];
  return (
    <div>
      <p className="eyebrow mb-1">{label}</p>
      <p className={`tnum text-[21px] font-semibold leading-none ${c}`}>{value}</p>
      {sub && <p className="mt-1 text-[11px] text-ink-faint">{sub}</p>}
    </div>
  );
}

export function Planner() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [ops, setOps] = useState<OpsPayload | null>(null);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [roads, setRoads] = useState<GeoJSON.FeatureCollection | null>(null);
  const [scale, setScale] = useState(1);
  const [budget, setBudget] = useState(20);
  const [chosen, setChosen] = useState<string>("base");
  const [focus, setFocus] = useState<"region" | "town">("region");
  const [grids, setGrids] = useState<InspectGrids | null>(null);
  const [point, setPoint] = useState<PointInfo | null>(null);
  const [custom, setCustom] = useState<number[]>([]);
  const [mode, setMode] = useState<"options" | "build">("options");
  const [roadQ, setRoadQ] = useState<RoadQuery | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [tab, setTab] = useState<"district" | "anywhere">("district");
  const [placed, setPlaced] = useState<PlacedLevee[]>([]);
  const [placing, setPlacing] = useState(false);
  const [observed, setObserved] = useState<Observation | null>(null);
  const { setRegion } = useRegion();

  // Keep the header honest about which place is on screen.
  useEffect(() => {
    setRegion(tab === "anywhere" ? (analysis?.name ?? "Anywhere") : null);
  }, [tab, analysis?.name, setRegion]);

  useEffect(() => {
    loadManifest().then(setManifest).catch(() => {});
    fetch("/data/roads.geojson").then((r) => r.json()).then(setRoads).catch(() => {});
    loadInspect().then(setGrids).catch(() => {});
    fetch("/api/options")
      .then(async (r) => {
        if (!r.ok) {
          const e = await r.json().catch(() => ({}));
          throw new Error(e.detail ?? `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((d: OpsPayload) => { setOps(d); setScale(d.event_scales.includes(1) ? 1 : d.event_scales[0]); })
      .catch((e) => setUnavailable(String(e.message ?? e)));
  }, []);

  const forScale = useMemo(
    () => (ops?.options ?? []).filter((o) => o.event_scale === scale),
    [ops, scale],
  );
  const base = forScale.find((o) => o.id === "base");
  const current = forScale.find((o) => o.id === chosen) ?? base;

  const affordable = useMemo(
    () => forScale.filter((o) => o.id === "base" || o.cost_aud <= budget * 1e6),
    [forScale, budget],
  );

  // Best value among affordable options that actually help.
  const recommended = useMemo(() => {
    const helpful = affordable.filter(
      (o) => (o.core_reduction_pct ?? 0) > 1 && o.id !== "base",
    );
    // Best value is reduction per dollar, not the largest reduction: the most
    // expensive option here buys LESS than one costing half as much.
    helpful.sort(
      (a, b) =>
        b.cost_aud / Math.max(b.core_reduction_pct ?? 1, 1e-6) >
        a.cost_aud / Math.max(a.core_reduction_pct ?? 1, 1e-6) ? -1 : 1,
    );
    return helpful[0];
  }, [affordable]);

  useEffect(() => {
    const n = manifest?.levee_geometry?.sites.length ?? 0;
    if (n && custom.length !== n) setCustom(Array(n).fill(0));
  }, [manifest, custom.length]);

  const markers: MeasureMarker[] = useMemo(() => {
    const geo = manifest?.levee_geometry;
    if (!geo) return [];
    const heights = mode === "build" ? custom : (current?.heights ?? []);
    return geo.sites.map((s) => {
      const single = forScale.find((o) => o.id === `s${s.id}`);
      return {
        id: s.id,
        lon: s.lon,
        lat: s.lat,
        selected: (heights[s.id] ?? 0) > 0,
        worsens: Boolean(single?.makes_worse),
      };
    });
  }, [manifest, current, forScale, mode, custom]);

  const onMarker = useCallback((id: number) => {
    setChosen((c) => (c === `s${id}` ? "base" : `s${id}`));
    setFocus("town");
  }, []);

  const onPick = useCallback(
    (lon: number, lat: number) => {
      if (tab === "anywhere") {
        if (!placing) return;
        setPlaced((l) => [...l, { lat, lon, height_m: 2.5, width_m: 600 }]);
        setPlacing(false);
        return;
      }
      if (!grids || !manifest) return;
      setPoint(sampleAt(grids, manifest.bounds, lon, lat));
    },
    [grids, manifest, tab, placing],
  );

  const near = useMemo(
    () => (point ? nearestRoad(roads, point.lon, point.lat) : null),
    [point, roads],
  );

  const frame = manifest
    ? manifest.frames.reduce((a, b) => (b.wet_cells > a.wet_cells ? b : a), manifest.frames[0]).i
    : 0;

  return (
    <div className="flex h-full min-h-0">
      {/* ---- decision panel ------------------------------------------- */}
      <aside className="flex w-[370px] shrink-0 flex-col overflow-y-auto border-r border-line bg-bg-raised">
        <div className="flex gap-1.5 border-b border-line-soft px-4 py-3">
          {(["district", "anywhere"] as const).map((k) => (
            <button
              key={k}
              onClick={() => { setTab(k); if (k === "district") setAnalysis(null); }}
              className={`flex-1 rounded-md border px-2 py-1.5 text-[11.5px] transition-colors ${
                tab === k
                  ? "border-accent/60 bg-accent/10 text-accent"
                  : "border-line text-ink-mute hover:text-ink"
              }`}
            >
              {k === "district" ? "This district" : "Anywhere"}
            </button>
          ))}
        </div>

        {tab === "anywhere" && (
          <AnywherePanel
            onResult={setAnalysis}
            onPreview={() => {}}
            levees={placed}
            onLevees={setPlaced}
            placing={placing}
            onPlacing={setPlacing}
          />
        )}

        {tab === "district" && <>
        <ForecastPanel />
        <RoadQueryPanel
          roads={roads}
          grids={grids}
          bounds={manifest?.bounds ?? null}
          onQuery={setRoadQ}
        />
        <div className="border-b border-line-soft p-4">
          <p className="eyebrow mb-1.5">Plan against</p>
          <div className="flex gap-1.5">
            {(ops?.event_scales ?? [1]).map((s) => (
              <button
                key={s}
                onClick={() => setScale(s)}
                className={`flex-1 rounded-md border px-2 py-1.5 text-[11.5px] transition-colors ${
                  scale === s
                    ? "border-accent/60 bg-accent/10 text-accent"
                    : "border-line text-ink-mute hover:text-ink"
                }`}
              >
                {eventLabel(s)}
              </button>
            ))}
          </div>
          {ops && (
            <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
              Capital works are appraised against design events, not next week&apos;s
              weather. A road is treated as cut at{" "}
              {(ops.cut_depth_m * 100).toFixed(0)} cm of water.
            </p>
          )}
        </div>

        <div className="border-b border-line-soft p-4">
          <div className="mb-2 flex items-baseline justify-between">
            <p className="eyebrow">Capital budget</p>
            <p className="tnum text-[13px] font-semibold text-ink">A${budget}M</p>
          </div>
          <input
            type="range" min={2} max={60} step={1} value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
            className="w-full accent-[var(--accent)]"
          />
        </div>

        <ObservePanel
          bounds={manifest?.bounds ?? null}
          onObservation={setObserved}
        />

        <PlanPanel
          budgetM={budget}
          eventScale={scale}
          crestLengths={(ops?.sites ?? []).map((s) => s.crest_length_m)}
          onPlan={(h) => { setMode("build"); setCustom(h); }}
        />

        </>}

        {tab === "district" && point && (
          <div className="border-b border-line-soft p-4">
            <div className="mb-2 flex items-start justify-between gap-2">
              <p className="eyebrow">Selected location</p>
              <button
                onClick={() => setPoint(null)}
                className="text-[11px] text-ink-faint hover:text-ink"
              >
                clear
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Metric
                label="Peak flood depth"
                value={point.flooded ? `${point.depth_m.toFixed(2)} m` : "dry"}
                tone={point.depth_m > 1 ? "bad" : point.flooded ? "default" : "good"}
                sub="in the 2022 event"
              />
              <Metric
                label="Ground level"
                value={`${point.elev_m.toFixed(1)} m`}
                sub="above sea level"
              />
            </div>
            {near && (
              <p className="mt-3 text-[11.5px] leading-relaxed text-ink-mute">
                Nearest road: <span className="text-ink">{near.name}</span>
                {near.km < 3 ? ` · ${(near.km * 1000).toFixed(0)} m away` : ""}
              </p>
            )}
            <p className="tnum mt-1 text-[10.5px] text-ink-faint">
              {point.lat.toFixed(4)}, {point.lon.toFixed(4)}
            </p>
          </div>
        )}

        {tab === "district" && !point && (
          <div className="border-b border-line-soft px-4 py-3">
            <p className="text-[11.5px] leading-relaxed text-ink-faint">
              <span className="text-ink-mute">Click anywhere on the map</span> to
              see how deep the water got there in 2022, or click a marker to
              assess a mitigation measure.
            </p>
          </div>
        )}

        {tab === "district" && unavailable && (
          <div className="m-4 rounded-lg border border-warn/40 bg-warn/5 p-3">
            <p className="text-[12px] font-medium text-warn">Options not available yet</p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-ink-mute">
              {unavailable} The flood model runs on a cluster and its results are
              published here when complete.
            </p>
          </div>
        )}

        {tab === "district" && ops && (
          <>
            <div className="flex gap-1.5 border-b border-line-soft px-4 py-3">
              {(["options", "build"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`flex-1 rounded-md border px-2 py-1.5 text-[11.5px] transition-colors ${
                    mode === m
                      ? "border-accent/60 bg-accent/10 text-accent"
                      : "border-line text-ink-mute hover:text-ink"
                  }`}
                >
                  {m === "options" ? "Appraised options" : "Build your own"}
                </button>
              ))}
            </div>

            {mode === "build" && manifest?.levee_geometry && (
              <MeasureBuilder
                siteCount={manifest.levee_geometry.sites.length}
                siteElev={manifest.levee_geometry.sites.map((s) => s.elev)}
                heights={custom}
                onHeights={setCustom}
                eventScale={scale}
                crestLengths={ops.sites.map((s) => s.crest_length_m)}
                budgetM={budget}
              />
            )}

            {mode === "options" && (
            <div className="border-b border-line-soft p-4">
              <p className="eyebrow mb-2">Mitigation options</p>
              <ul className="flex flex-col gap-1.5">
                {affordable.map((o) => {
                  const worse = (o.core_reduction_pct ?? 0) < -0.5;
                  const sel = o.id === chosen;
                  const prot = o.road_protected_km ?? 0;
                  return (
                    <li key={o.id}>
                      <button
                        onClick={() => { setChosen(o.id); setFocus("town"); }}
                        className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                          sel ? "border-accent/60 bg-accent/10"
                              : "border-line bg-bg-inset hover:border-line-soft"
                        }`}
                      >
                        <div className="flex items-baseline gap-2">
                          <span className="flex-1 truncate text-[12.5px] text-ink">
                            {o.name}
                          </span>
                          <span className="tnum text-[11px] text-ink-faint">
                            {o.id === "base" ? "—" : money(o.cost_aud)}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-2">
                          {o.id === "base" ? (
                            <span className="text-[11px] text-ink-faint">
                              {km(o.road_cut_km)} of road cut in this event
                            </span>
                          ) : (o.core_reduction_pct ?? 0) < -0.5 ? (
                            <span className="text-[11px] font-medium text-bad">
                              ⚠ deepens flooding {Math.abs(o.core_reduction_pct ?? 0).toFixed(0)}%
                            </span>
                          ) : (
                            <span className="text-[11px] text-ok">
                              cuts flooding {(o.core_reduction_pct ?? 0).toFixed(0)}%
                            </span>
                          )}
                          {recommended?.id === o.id && (
                            <span className="ml-auto rounded bg-ok/15 px-1.5 py-[1px] text-[9.5px] font-semibold tracking-wide text-ok">
                              BEST VALUE
                            </span>
                          )}
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
            )}

            {mode === "options" && current && (
              <div className="p-4">
                <p className="eyebrow mb-3">Projected impact</p>
                <div className="grid grid-cols-2 gap-4">
                  <Metric
                    label="Road cut"
                    value={km(current.road_cut_km)}
                    sub={`of ${ops.road_network_km.toFixed(0)} km network`}
                    tone={current.makes_worse ? "bad" : "default"}
                  />
                  <Metric
                    label="Land flooded"
                    value={`${current.area_flooded_km2.toFixed(1)} km²`}
                  />
                  {current.id !== "base" && (
                    <>
                      <Metric
                        label="Change vs no action"
                        value={`${(current.road_cut_change_km ?? 0) >= 0 ? "+" : ""}${km(current.road_cut_change_km ?? 0, 2)}`}
                        tone={current.makes_worse ? "bad" : "good"}
                      />
                      <Metric
                        label="Cost"
                        value={money(current.cost_aud)}
                        sub={current.cost_per_km_protected
                          ? `${money(current.cost_per_km_protected)} per km kept open`
                          : "no road kept open"}
                      />
                    </>
                  )}
                </div>

                {current.makes_worse && (
                  <div className="mt-4 rounded-lg border border-bad/40 bg-bad/5 p-3">
                    <p className="text-[12px] font-semibold text-bad">
                      This option increases flooding
                    </p>
                    <p className="mt-1 text-[11.5px] leading-relaxed text-ink-mute">
                      It sits on a drainage path. Walling it off holds water in
                      the floodplain rather than keeping it out, cutting{" "}
                      {km(current.road_cut_change_km ?? 0, 2)} more road than
                      taking no action at all.
                    </p>
                  </div>
                )}

                {current.named_roads_cut.length > 0 && (
                  <div className="mt-4">
                    <p className="eyebrow mb-2">Roads affected</p>
                    <ul className="flex flex-col gap-1">
                      {current.named_roads_cut.slice(0, 6).map((r) => (
                        <li key={r.name} className="flex items-baseline gap-2 text-[11.5px]">
                          <span className="flex-1 truncate text-ink-mute">{r.name}</span>
                          <span className="tnum text-ink-faint">{km(r.km, 1)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {ops.cost_note && (
                  <p className="mt-4 border-t border-line-soft pt-3 text-[10.5px] leading-relaxed text-ink-faint">
                    {ops.cost_note}
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </aside>

      {/* ---- map ------------------------------------------------------- */}
      <div className="relative min-w-0 flex-1">
        {manifest ? (
          <PlannerMap
            manifest={manifest}
            frame={frame}
            roads={tab === "district" ? roads : null}
            markers={markers}
            onMarker={onMarker}
            onPick={onPick}
            pin={point ? { lon: point.lon, lat: point.lat } : null}
            cutFraction={tab === "district" ? (roadQ?.cutFraction ?? null) : null}
            placed={tab === "anywhere" ? placed : []}
            observed={observed}
              analysis={analysis}
            focus={focus}
          />
        ) : (
          <div className="h-full w-full animate-pulse bg-bg-inset" />
        )}
        <div className="absolute left-4 top-4 flex gap-1.5">
          {(["region", "town"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFocus(f)}
              // These float over a LIGHT map, so they invert: the panel's own
              // dark ground becomes the chip, and selection is a solid fill.
              className={`rounded-md border px-2.5 py-1 text-[11px] backdrop-blur transition-colors ${
                focus === f
                  ? "border-ink bg-ink text-bg font-medium"
                  : "border-ink-faint/40 bg-bg/80 text-ink hover:bg-bg"
              }`}
            >
              {f === "region" ? "Whole valley" : "Town"}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
