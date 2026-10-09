"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { PlannerMap, type MeasureMarker } from "./PlannerMap";
import { ForecastPanel } from "./ForecastPanel";
import { LayersPanel, DEFAULT_LAYERS, type LayerState,
         type GaugeFeed } from "./LayersPanel";
import { GaugeHeadline } from "./GaugeHeadline";
import { MeasureBuilder } from "./MeasureBuilder";
import { RoadQueryPanel } from "./RoadQueryPanel";
import { AnywherePanel, type AnalysisResult, type PlacedLevee,
         type StudyMeta } from "./AnywherePanel";
import { PlanPanel } from "./PlanPanel";
import { PlanSection } from "./PlanSection";
import { ObservePanel, type Observation } from "./ObservePanel";
import { RoutePanel } from "./RoutePanel";
import { TripPanel, type TripResult } from "./TripPanel";
import { SectionPanel, type SectionEnds, type SectionOverlay } from "./SectionPanel";
import { StreetPanel, type StreetResult } from "./StreetPanel";
import type { RoadQuery } from "@/lib/roadquery";
import { loadManifest } from "@/lib/data";
import type { Manifest } from "@/lib/types";
import { eventLabel, km, money, type OpsPayload } from "@/lib/ops";
import { useRegion } from "./RegionContext";
import {
  decodeGrids, loadInspect, nearestRoad, sampleAt,
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
    forTab: "district" | "anywhere" | "trip" | "section" | "street";
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
  // Austin opens the portal, not the Richmond archive.
  //
  // The archive was the default because it was the only place with reference
  // depths. Austin has no archive -- it has a bare-earth DEM, a regulatory
  // floodplain to be scored against, and gauges -- so the default study here
  // is our own solver, run once at build time and shipped with its skill
  // score, rather than a reference flood being replayed. The two are labelled
  // differently on purpose: Richmond is a VALIDATED MODEL, Austin a
  // PROJECTION that recovers about half of FEMA's mapped 1% floodplain.
  const [tab, setTab] = useState<"district" | "anywhere" | "trip" | "section" | "street">("anywhere");
  const [study, setStudy] = useState<(AnalysisResult & { study?: StudyMeta }) | null>(null);
  const [studyErr, setStudyErr] = useState(false);
  /** What is drawn over the terrain, and the live feed behind one of them. */
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const [gaugeFeed, setGaugeFeed] = useState<GaugeFeed | null>(null);
  /** A trip's own result: its own box, its own flood, its own network. */
  const [trip, setTrip] = useState<TripResult | null>(null);
  /** The two ends of a cross-section, picked on the map rather than searched:
   *  a section is drawn across a floodplain, which is a place no gazetteer has
   *  a name for. */
  const [sectionEnds, setSectionEnds] = useState<SectionEnds>({ a: null, b: null });
  /** Whether the next map click goes to the section. Armed, not latched: a
   *  complete pair disarms it by itself, so the click after the second end --
   *  typically on the map, to look at something -- cannot throw the line away,
   *  and nothing has to be unset in an effect. */
  const [sectionArmed, setSectionArmed] = useState(true);
  const [sectionOverlay, setSectionOverlay] = useState<SectionOverlay | null>(null);
  /** The street view: one block, its buildings, and which floors got wet. */
  const [street, setStreet] = useState<StreetResult | null>(null);
  const [streetBusy, setStreetBusy] = useState(false);
  const [streetErr, setStreetErr] = useState<string | null>(null);
  const [streetSec, setStreetSec] = useState(0);
  const [streetFrame, setStreetFrame] = useState<number | null>(null);
  const [streetSel, setStreetSel] = useState<string | null>(null);
  /** Where the block sits. Clicking the map in this tab moves it. */
  const [streetAt, setStreetAt] = useState<{ lat: number; lon: number } | null>(null);
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
  const { setRegion, setLive } = useRegion();

  // The header says LIVE only when something on the page really is. A badge
  // over a replayed archive would be a lie, and the Richmond study has no
  // gauge feed at all.
  useEffect(() => {
    const n = gaugeFeed?.gauges.length ?? 0;
    setLive(n
      ? `${n} gauge${n === 1 ? "" : "s"} every ${gaugeFeed?.cadence_min ?? 15} min`
      : null);
  }, [gaugeFeed, setLive]);

  // Keep the header honest about which place is on screen.
  useEffect(() => {
    setRegion(tab === "district"
      ? null
      : (analysis?.name ?? study?.study?.place ?? "Anywhere"));
  }, [tab, analysis?.name, study, setRegion]);

  useEffect(() => {
    // The default study ships with the app, so the first view is a solved
    // flood rather than a search box. The archive still loads underneath:
    // the map's terrain and the district tab need it either way.
    fetch("/data/austin/study.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => {
        setStudy(d);
        setAdHocRoads(d.roads ?? null);
        // The grids too, not just the roads: the street view picks its block
        // from the solved depth field, and without them it opens on the
        // middle of the box, which is wherever the box happened to be drawn.
        setAdHocGrids(decodeGrids(d.inspect as Parameters<typeof decodeGrids>[0]));
      })
      .catch(() => setStudyErr(true));
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

  // The block the street view opens on: wherever the user last clicked in
  // this tab, else the middle of whatever place is solved.
  const streetCentre = useMemo(() => {
    if (streetAt) return streetAt;
    const b = analysis?.bounds ?? study?.bounds;
    if (!b) return null;
    // Open on the WETTEST part of the solved field, not the middle of the box.
    // The middle is wherever the box happened to be drawn; measured, it put
    // the first street view on a block with 0 of 245 houses over their floors
    // while the same study had neighbourhoods under a metre. Weighted by
    // depth, so a single deep channel cell does not drag the block into the
    // creek either.
    const g = adHocGrids;
    if (g) {
      const [h, w] = g.meta.grid;
      // The wettest WINDOW, not the centroid of all wet cells. A centroid
      // averages two separate flooded reaches into the dry ground between
      // them: measured, it opened the street view on a block with 0 of 561
      // houses over their floors while the same study had neighbourhoods
      // under water. A window finds a place that is actually wet.
      const km = 110.574 * (b.north - b.south) / h;        // km per row
      const win = Math.max(2, Math.round(0.8 / Math.max(km, 1e-6)));
      let best = -1, br = 0, bc = 0;
      for (let r = win; r < h - win; r += Math.max(1, win >> 1)) {
        for (let c = win; c < w - win; c += Math.max(1, win >> 1)) {
          let sum = 0;
          for (let dr = -win; dr <= win; dr += 1) {
            for (let dc = -win; dc <= win; dc += 1) {
              sum += Math.min(g.depth[(r + dr) * w + (c + dc)], 80);
            }
          }
          if (sum > best) { best = sum; br = r; bc = c; }
        }
      }
      if (best > 0) {
        return {
          lon: b.west + (b.east - b.west) * bc / (w - 1),
          lat: b.north - (b.north - b.south) * br / (h - 1),
        };
      }
    }
    return { lat: (b.south + b.north) / 2, lon: (b.west + b.east) / 2 };
  }, [streetAt, analysis, study, adHocGrids]);

  const solveStreet = useCallback(async (
    opts: { driver: string; storm: number; q: number | null },
  ) => {
    if (!streetCentre) return;
    setStreetBusy(true); setStreetErr(null); setStreet(null);
    setStreetSel(null); setStreetFrame(null); setStreetSec(0);
    const tick = setInterval(() => setStreetSec((n) => n + 1), 1000);
    // A block, not a suburb: ~1.7 x 1.7 km, which is what a 10 m grid can
    // carry and about as much street as anyone reads at once.
    const h = 0.0085;
    try {
      const r = await fetch("/api/buildings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          west: streetCentre.lon - h, east: streetCentre.lon + h,
          south: streetCentre.lat - h * 0.86, north: streetCentre.lat + h * 0.86,
          name: analysis?.name ?? study?.study?.place ?? "Street",
          dx: 10, snapshots: 13, max_buildings: 600,
          parent_km: 14, parent_dx: 40, run_hours: 18,
          driver: opts.driver,
          ...(opts.q ? { river_q_m3s: opts.q } : {}),
          // The storm presets are multiples of the discharge, so "as gauged"
          // has to be 1x: the rain figures only matter on the rain driver.
          rain_mm_h: [20, 50, 80][opts.storm] ?? 50, storm_hours: 6,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      if (!d.ok) throw new Error(d.reason ?? "this block could not be solved");
      setStreet(d);
    } catch (e) {
      setStreetErr(e instanceof Error ? e.message : "could not open this street");
    } finally {
      setStreetBusy(false);
      clearInterval(tick);
    }
  }, [streetCentre, analysis, study]);

  const streetBuildings = useMemo(() => {
    if (!street?.ok) return null;
    return street.buildings.map((b) => ({
      id: b.id,
      ring: b.ring,
      height_m: b.height_m,
      over: streetFrame !== null && b.over_floor_series[streetFrame] !== undefined
        ? b.over_floor_series[streetFrame]
        : (b.over_floor_m ?? -1),
      selected: b.id === streetSel,
    }));
  }, [street, streetFrame, streetSel]);

  const sectionPicking =
    sectionArmed && !(sectionEnds.a && sectionEnds.b);

  const onPick = useCallback(
    (lon: number, lat: number) => {
      // A section's two ends come off the map in order: first click is A,
      // second is B, and a third starts a new line rather than silently
      // moving one of the two -- which is what "pick two points" means to
      // everyone who has used a ruler tool.
      if (tab === "street") {
        setStreetAt({ lat, lon });
        return;
      }
      if (tab === "section") {
        if (!sectionPicking) return;
        setSectionEnds((e) => (e.a ? { a: e.a, b: { lat, lon } }
                                   : { a: { lat, lon }, b: null }));
        return;
      }
      if (tab === "anywhere") {
        if (!placing) return;
        setPlaced((l) => [...l, { lat, lon, height_m: 2.5, width_m: 600 }]);
        setPlacing(false);
        return;
      }
      if (!grids || !manifest) return;
      setPoint(sampleAt(grids, manifest.bounds, lon, lat));
    },
    [grids, manifest, tab, placing, sectionPicking],
  );

  /** The box the live feeds are asked about: whatever place is on screen. */
  const liveBounds = useMemo(() => {
    const b = analysis?.bounds ?? trip?.bounds ?? study?.bounds ?? manifest?.bounds;
    return b ? { west: b.west, south: b.south, east: b.east, north: b.north } : null;
  }, [analysis, trip, study, manifest]);

  const near = useMemo(
    () => (point ? nearestRoad(roads, point.lon, point.lat) : null),
    [point, roads],
  );

  // The trip ships the same quantised grids an ad-hoc analysis does, so the
  // body of water and the click-anywhere depth work there too.
  const tripGrids = useMemo(
    () => decodeGrids(trip?.inspect as Parameters<typeof decodeGrids>[0]),
    [trip]);

  const frame = manifest
    ? manifest.frames.reduce((a, b) => (b.wet_cells > a.wet_cells ? b : a), manifest.frames[0]).i
    : 0;

  return (
    <div className="flex h-full min-h-0">
      {/* ---- decision panel ------------------------------------------- */}
      <aside className="flex w-[380px] shrink-0 flex-col overflow-y-auto border-r border-line bg-bg">
        {/* Which study. A different axis from the three sections below it --
            those ask what you want to know, this asks about where -- so it
            sits above them rather than becoming a fifth area pill. */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line px-5 py-2.5">
          <span className="text-[13px] text-ink-mute">Study</span>
          <button
            className="pill whitespace-nowrap"
            data-on={tab !== "district"}
            onClick={() => { setTab("anywhere"); setAnalysis(null); }}
          >
            Austin, Texas
          </button>
          <button
            className="pill whitespace-nowrap"
            data-on={tab === "district"}
            onClick={() => { setTab("district"); setAnalysis(null); setTrip(null); }}
          >
            Richmond Valley, NSW
          </button>
          {studyErr && (
            <span className="anno">default study not built — run scripts/build_austin_study.py</span>
          )}
        </div>

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

        <div className="border-b border-line px-5 py-3">
          <span className="text-[13px] text-ink-mute">Area</span>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {(["district", "anywhere", "trip", "section", "street"] as const).map((k) => (
              <button
                key={k}
                className="pill whitespace-nowrap"
                data-on={tab === k}
                onClick={() => {
                  setTab(k);
                  if (k === "district") setAnalysis(null);
                  if (k !== "trip") setTrip(null);
                  if (k === "section") setSectionArmed(true);
                }}
              >
                {k === "district" ? "This district"
                  : k === "anywhere" ? "Anywhere"
                    : k === "trip" ? "Plan a trip"
                      : k === "section" ? "Cross-section" : "Street"}
              </button>
            ))}
          </div>
        </div>

        {tab === "trip" && (
          <TripPanel
            onResult={setTrip}
            onRoutes={(r) => setRouteLines(
              r ? { ...(r as object), forTab: "trip" } as typeof routeLines : null)}
          />
        )}

        {section === "forecast" && (
          <>
            <GaugeHeadline feed={gaugeFeed} />
            <LayersPanel
              bounds={liveBounds}
              layers={layers}
              onLayers={setLayers}
              onGauges={setGaugeFeed}
              modelRunNote={analysis
                ? `solved at ${analysis.dx_m.toFixed(0)} m · ${analysis.flooded_km2} km² over 10 cm`
                : undefined}
            />
          </>
        )}

        {tab === "street" && (
          <StreetPanel
            result={street}
            busy={streetBusy}
            elapsed={streetSec}
            err={streetErr}
            frame={streetFrame}
            onFrame={setStreetFrame}
            selected={streetSel}
            onSelect={setStreetSel}
            onSolve={solveStreet}
            canSolve={Boolean(streetCentre)}
          />
        )}

        {tab === "section" && (
          <SectionPanel
            ends={sectionEnds}
            picking={sectionPicking}
            onPicking={setSectionArmed}
            onClearEnds={() => setSectionEnds({ a: null, b: null })}
            onOverlay={setSectionOverlay}
          />
        )}

        {tab === "anywhere" && (
          <AnywherePanel
            initial={study}
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
            roads={tab === "district" ? roads
              : tab === "trip" ? (trip?.roads ?? null) : adHocRoads}
            focusBounds={tab === "street" && street?.ok ? street.bounds : null}
            markers={tab === "district" ? markers : []}
            pins={tab === "trip" && trip
              ? [{ lon: trip.origin.lon, lat: trip.origin.lat, label: "from" },
                 { lon: trip.destination.lon, lat: trip.destination.lat, label: "to" }]
              : null}
            onMarker={onMarker}
            onPick={onPick}
            pin={point ? { lon: point.lon, lat: point.lat } : null}
            cutFraction={roadQ?.cutFraction ?? null}
            cutSignature={roadQ ? `${roadQ.threshold}:${roadQ.cutKm.toFixed(2)}` : "none"}
            placed={tab === "anywhere" ? placed : []}
            placedHeights={tab === "anywhere" ? adHocHeights : null}
            routes={routeLines?.forTab === tab ? routeLines : null}
            section={tab === "section" ? sectionOverlay : null}
            buildings={tab === "street" ? streetBuildings : null}
            onBuilding={setStreetSel}
            gauges={layers.gauges ? (gaugeFeed?.gauges ?? null) : null}
            radar={layers.radar ? liveBounds : null}
            place={tab === "anywhere" ? place : null}
            observed={observed}
              analysis={tab === "trip"
                ? (trip ? { bounds: trip.bounds, depth_png: trip.depth_png,
                            terrain_png: trip.terrain_png } : null)
                : tab === "street" && street?.ok
                  ? { bounds: street.bounds,
                      depth_png: (streetFrame !== null && street.frames[streetFrame])
                        ? street.frames[streetFrame].depth_png
                        : street.depth_png,
                      terrain_png: street.terrain_png }
                  : tab === "district" ? null : analysis}
            focus={focus}
            // No solid water body in the street view: the grids are the
            // study's 40 m field over the whole study box, and drawn under a
            // camera zoomed to one block it covers everything in sight. The
            // block's own 10 m field arrives as the depth overlay instead.
            grids={tab === "district" ? grids
              : tab === "trip" ? tripGrids
                : tab === "street" ? null : adHocGrids}
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
