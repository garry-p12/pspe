"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { PlannerMap, type MeasureMarker } from "./PlannerMap";
import { ForecastPanel } from "./ForecastPanel";
import { MeasureBuilder } from "./MeasureBuilder";
import { RoadQueryPanel } from "./RoadQueryPanel";
import { AnywherePanel, type AnalysisResult, type PlacedLevee } from "./AnywherePanel";
import { PlanPanel } from "./PlanPanel";
import { PlanSection } from "./PlanSection";
import { ObservePanel, type Observation } from "./ObservePanel";
import { RoutePanel } from "./RoutePanel";
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
      <p className={`tnum text-[24px] font-semibold leading-none ${c}`}>{value}</p>
      {sub && <p className="mt-1 text-[12px] text-ink-faint">{sub}</p>}
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
  /** Water drawn as a body, or painted flat over the terrain. */
  const [solid, setSolid] = useState(true);
  const [place, setPlace] = useState<{ lat: number; lon: number; name: string } | null>(null);
  const [adHocGrids, setAdHocGrids] = useState<InspectGrids | null>(null);
  /** Heights of the option being drawn for an arbitrary place, per placed site. */
  const [adHocHeights, setAdHocHeights] = useState<number[] | null>(null);
  /** The way out and, where the margin allows one, the way out that survives
   *  the depth being underestimated. Drawn on the map. */
  /** What the vehicle being routed can drive through, metres, for the district
   *  view. The Anywhere view keeps its own, because the two are looking at
   *  different places and a threshold carried across would be answering about
   *  the wrong one. */
  const [vehicle, setVehicle] = useState(0.25);
  const [routeLines, setRouteLines] = useState<{
    forTab: "district" | "anywhere";
    quickest: { coords: [number, number][]; minutes: number } | null;
    robust: { coords: [number, number][]; minutes: number } | null;
    dest: { lat: number; lon: number } | null;
  } | null>(null);
  const [point, setPoint] = useState<PointInfo | null>(null);
  const [custom, setCustom] = useState<number[]>([]);
  const [mode, setMode] = useState<"options" | "build">("options");
  const [roadQ, setRoadQ] = useState<RoadQuery | null>(null);
  const [adHocRoads, setAdHocRoads] = useState<GeoJSON.FeatureCollection | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [tab, setTab] = useState<"district" | "anywhere">("district");
  const [placed, setPlaced] = useState<PlacedLevee[]>([]);
  const [placing, setPlacing] = useState(false);
  const [observed, setObserved] = useState<Observation | null>(null);
  // The panel used to be one continuous scroll: forecast, roads, design event,
  // budget, options, impact, all stacked. Everything was on screen and nothing
  // was findable. Three sections match the three questions an operator
  // actually arrives with -- what is coming, what does it cut, what do we
  // build -- and only one is ever open.
  const [section, setSection] = useState<"forecast" | "roads" | "plan">("forecast");

  useEffect(() => {
    if (section === "plan") setFocus("town");
    else if (section === "forecast") setFocus("region");
  }, [section]);

  // Put the view in the address bar so a plan can be sent to someone. Read once
  // on mount, written on change. Deliberately only the few things that identify
  // what you are looking at -- not every transient bit of panel state.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const b = Number(q.get("budget"));
    if (Number.isFinite(b) && b >= 2 && b <= 60) setBudget(b);
    const sec = q.get("view");
    if (sec === "forecast" || sec === "roads" || sec === "plan") setSection(sec);
  }, []);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    q.set("budget", String(budget));
    q.set("view", section);
    window.history.replaceState(null, "", `?${q.toString()}`);
  }, [budget, section]);
  const [planTrigger, setPlanTrigger] = useState(0);
  const [planning, setPlanning] = useState(false);
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
      <aside className="flex w-[380px] shrink-0 flex-col overflow-y-auto border-r border-line bg-bg">
        <div className="px-5 pt-4">
          <div className="seg" role="tablist">
            {(["forecast", "roads", "plan"] as const).map((k) => (
              <button
                key={k}
                role="tab"
                aria-selected={section === k}
                data-on={section === k}
                onClick={() => setSection(k)}
              >
                {k === "forecast" ? "Forecast" : k === "roads" ? "Roads" : "Plan"}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <span className="text-[13px] text-ink-mute">Area</span>
          <div className="flex gap-1">
            {(["district", "anywhere"] as const).map((k) => (
              <button
                key={k}
                className="pill"
                data-on={tab === k}
                onClick={() => { setTab(k); if (k === "district") setAnalysis(null); }}
              >
                {k === "district" ? "This district" : "Anywhere"}
              </button>
            ))}
          </div>
        </div>

        {tab === "anywhere" && (
          <AnywherePanel
            section={section}
            onResult={setAnalysis}
            onRoadQuery={setRoadQ}
            onRoads={setAdHocRoads}
            onPreview={() => {}}
            levees={placed}
            onLevees={setPlaced}
            placing={placing}
            onPlacing={setPlacing}
            onPlace={setPlace}
            onGrids={setAdHocGrids}
            onHeights={setAdHocHeights}
            onRoutes={(r) => setRouteLines(
              r ? { ...(r as object), forTab: "anywhere" } as typeof routeLines : null)}
          />
        )}

        {tab === "district" && section === "forecast" && <>
        <ForecastPanel />
        <ObservePanel
          bounds={manifest?.bounds ?? null}
          onObservation={setObserved}
        />
        </>}

        {tab === "district" && section === "roads" && <>
        <RoadQueryPanel
          roads={roads}
          grids={grids}
          bounds={manifest?.bounds ?? null}
          onQuery={setRoadQ}
        />
        </>}

        {tab === "district" && section === "plan" && ops && (
          <PlanSection
            ops={ops}
            budgetM={budget}
            onBudget={setBudget}
            scale={scale}
            onScale={setScale}
            chosen={chosen}
            onChoose={(id) => { setChosen(id); setFocus("town"); setMode("options"); }}
            onBuildYourOwn={() => setMode("build")}
            onFindPlan={() => setPlanTrigger((n) => n + 1)}
            planning={planning}
          >
            <PlanPanel
              budgetM={budget}
              eventScale={scale}
              crestLengths={(ops.sites ?? []).map((s) => s.crest_length_m)}
              onPlan={(h) => { setMode("build"); setCustom(h); }}
              onShowHarm={(site) => {
                // Put ONLY the rejected wall on the map. The depth layer then
                // shows where that water ends up instead of at the settlement,
                // which is the thing the number cannot convey.
                const h = (ops.sites ?? []).map((_, i) => (i === site ? 3.0 : 0));
                setMode("build"); setCustom(h);
              }}
              trigger={planTrigger}
              onBusy={setPlanning}
            />
          </PlanSection>
        )}

        {tab === "district" && section === "plan" && point && (
          <div className="border-b border-line-soft p-4">
            <div className="mb-2 flex items-start justify-between gap-2">
              <p className="eyebrow">Selected location</p>
              <button
                onClick={() => setPoint(null)}
                className="text-[12px] text-ink-faint hover:text-ink"
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
              <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">
                Nearest road: <span className="text-ink">{near.name}</span>
                {near.km < 3 ? ` · ${(near.km * 1000).toFixed(0)} m away` : ""}
              </p>
            )}
            <p className="tnum mt-1 text-[12px] text-ink-faint">
              {point.lat.toFixed(4)}, {point.lon.toFixed(4)}
            </p>
          </div>
        )}

        {tab === "district" && section === "plan" && !point && (
          <div className="border-b border-line-soft px-4 py-3">
            <p className="text-[13px] leading-relaxed text-ink-faint">
              <span className="text-ink-mute">Click anywhere on the map</span> to
              see how deep the water got there in 2022, or click a marker to
              assess a mitigation measure.
            </p>
          </div>
        )}

        {/* Can people get out. No margin here: the district's options come
            from a precomputed library and the field drawn is the reference
            solution, so the honest answer is the route under that field and
            what the flood costs in minutes -- not a guarantee about a plan
            this tab cannot solve on demand (see section 5.13). */}
        {tab === "district" && section === "plan" && manifest && (
          <div className="border-b border-line-soft px-1 pb-2">
            <RoutePanel
              roads={roads}
              grids={grids}
              bounds={manifest.bounds}
              from={manifest.levee_geometry?.settlement
                ? { lat: manifest.levee_geometry.settlement.lat,
                    lon: manifest.levee_geometry.settlement.lon,
                    name: "the settlement" }
                : null}
              to={null}
              vehicle={vehicle}
              onVehicle={setVehicle}
              plan={null}
              onRoutes={(r) => setRouteLines(
                r ? { ...(r as object), forTab: "district" } as typeof routeLines : null)}
            />
          </div>
        )}

        {tab === "district" && section === "plan" && unavailable && (
          <div className="m-4 rounded-lg border border-warn/40 bg-warn/5 p-3">
            <p className="text-[13px] font-medium text-warn">Options not available yet</p>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
              {unavailable} The flood model runs on a cluster and its results are
              published here when complete.
            </p>
          </div>
        )}

        {tab === "district" && section === "plan" && ops && mode === "build"
          && manifest?.levee_geometry && (
          <div className="px-5 pb-5">
            <MeasureBuilder
              siteCount={manifest.levee_geometry.sites.length}
              siteElev={manifest.levee_geometry.sites.map((s) => s.elev)}
              heights={custom}
              onHeights={setCustom}
              eventScale={scale}
              crestLengths={ops.sites.map((s) => s.crest_length_m)}
              budgetM={budget}
            />
          </div>
        )}

        {tab === "district" && section === "plan" && ops && (
          <>
            {mode === "options" && current && chosen !== "base" && (
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
                    <p className="text-[13px] font-semibold text-bad">
                      This option increases flooding
                    </p>
                    <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
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
                        <li key={r.name} className="flex items-baseline gap-2 text-[13px]">
                          <span className="flex-1 truncate text-ink-mute">{r.name}</span>
                          <span className="tnum text-ink-faint">{km(r.km, 1)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {ops.cost_note && (
                  <p className="mt-4 border-t border-line-soft pt-3 text-[12px] leading-relaxed text-ink-faint">
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
            roads={tab === "district" ? roads : adHocRoads}
            markers={tab === "district" ? markers : []}
            onMarker={onMarker}
            onPick={onPick}
            pin={point ? { lon: point.lon, lat: point.lat } : null}
            cutFraction={roadQ?.cutFraction ?? null}
            cutSignature={roadQ ? `${roadQ.threshold}:${roadQ.cutKm.toFixed(2)}` : "none"}
            placed={tab === "anywhere" ? placed : []}
            placedHeights={tab === "anywhere" ? adHocHeights : null}
            routes={routeLines?.forTab === tab ? routeLines : null}
            place={tab === "anywhere" ? place : null}
            observed={observed}
              analysis={analysis}
            focus={focus}
            grids={tab === "anywhere" ? adHocGrids : grids}
            heights={mode === "build" ? custom : (current?.heights ?? [])}
            solid={solid}
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
              className={`rounded-md border px-2.5 py-1 text-[12px] backdrop-blur transition-colors ${
                focus === f
                  ? "border-ink bg-ink text-bg font-medium"
                  : "border-ink-faint/40 bg-bg/80 text-ink hover:bg-bg"
              }`}
            >
              {f === "region" ? "Whole valley" : "Town"}
            </button>
          ))}
          {/* The solid flood hides the ground under it, which some people want
              to see. Plain wording: this is about what the water looks like,
              not about rendering. */}
          <button
            onClick={() => setSolid((v) => !v)}
            title={solid
              ? "Show the ground under the water instead"
              : "Show the water as a body, with its real depth"}
            className={`rounded-md border px-2.5 py-1 text-[12px] backdrop-blur transition-colors ${
              solid
                ? "border-ink bg-ink text-bg font-medium"
                : "border-ink-faint/40 bg-bg/80 text-ink hover:bg-bg"
            }`}
          >
            Water depth
          </button>
        </div>
      </div>
    </div>
  );
}
