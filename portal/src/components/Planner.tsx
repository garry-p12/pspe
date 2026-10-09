"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PlannerMap } from "./PlannerMap";
import { LayersPanel, DEFAULT_LAYERS, type LayerState,
         type GaugeFeed } from "./LayersPanel";
import { GaugeHeadline } from "./GaugeHeadline";
import { Hydrograph } from "./Hydrograph";
import { RoadsOut } from "./RoadsOut";
import { AnywherePanel, type AnalysisResult, type PlacedLevee,
         type StudyMeta } from "./AnywherePanel";
import { TripPanel, type TripResult } from "./TripPanel";
import { SectionPanel, type SectionEnds, type SectionOverlay } from "./SectionPanel";
import { SectionStrip } from "./SectionStrip";
import { wallAt, crestToHold, bestStation, reaches, channelBelts,
         type SectionResult } from "@/lib/section";
import { ScenarioPicker } from "./ScenarioPicker";
import { RunScrubber, type Frame } from "./RunScrubber";
import { scenariosFrom, type PeaksFeed, type Scenario } from "@/lib/scenarios";
import { StreetPanel, type StreetResult } from "./StreetPanel";
import { LeveeKit, PIECES, pieceOf, costOf, type SitesFeed, type Placed,
         type PieceId } from "./LeveeKit";
import type { RoadQuery } from "@/lib/roadquery";
import { useRegion } from "./RegionContext";
import { decodeGrids, sampleAt,
         type InspectGrids, type PointInfo } from "@/lib/inspect";

export function Planner() {
  const [focus, setFocus] = useState<"region" | "town">("region");
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
    forTab: "anywhere" | "trip" | "section" | "street" | "kit";
    quickest: { coords: [number, number][]; minutes: number } | null;
    robust: { coords: [number, number][]; minutes: number } | null;
    dest: { lat: number; lon: number } | null;
  } | null>(null);
  const [point, setPoint] = useState<PointInfo | null>(null);
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
  /** Which planning tool the Plan tab is showing. */
  const [planMode, setPlanMode] = useState<"kit" | "section">("kit");
  /** The street zoom. A map control, not a panel -- which is where the design
   *  puts it and where anyone reading a map expects a zoom to be. */
  const [streetOn, setStreetOn] = useState(false);
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
  /** The cross-section view: a chosen bank, a solved line, and a wall on it. */
  const [secSite, setSecSite] = useState<number | null>(null);
  const [secRes, setSecRes] = useState<SectionResult | null>(null);
  const [secBusy, setSecBusy] = useState(false);
  const [secSec, setSecSec] = useState(0);
  const [secErr, setSecErr] = useState<string | null>(null);
  const [secStation, setSecStation] = useState<number | null>(null);
  const [secCrest, setSecCrest] = useState<number | null>(null);
  /** The street view: one block, its buildings, and which floors got wet. */
  const [street, setStreet] = useState<StreetResult | null>(null);
  const [streetBusy, setStreetBusy] = useState(false);
  const [streetErr, setStreetErr] = useState<string | null>(null);
  const [streetSec, setStreetSec] = useState(0);
  const [streetFrame, setStreetFrame] = useState<number | null>(null);
  const [streetSel, setStreetSel] = useState<string | null>(null);
  /** Where the block sits. Clicking the map in this tab moves it. */
  const [streetAt, setStreetAt] = useState<{ lat: number; lon: number } | null>(null);

  /** The levee kit: where a wall would pay, what is placed, what it buys. */
  const [kit, setKit] = useState<SitesFeed | null>(null);
  const [kitBusy, setKitBusy] = useState(false);
  const [kitSec, setKitSec] = useState(0);
  const [kitErr, setKitErr] = useState<string | null>(null);
  const [placedPieces, setPlacedPieces] = useState<Placed[]>([]);
  const [armed, setArmed] = useState<PieceId | null>(null);
  const [kitBudget, setKitBudget] = useState(20);
  const [kitPlan, setKitPlan] = useState<Record<string, unknown> | null>(null);
  const [kitPlanBusy, setKitPlanBusy] = useState(false);
  const [showPlan, setShowPlan] = useState(true);
  /** Which flood the plan is judged against. See `lib/scenarios`: the study's
   *  own design rainstorm is the wrong question to put a levee in front of,
   *  because rain lands on both sides of a wall. */
  const [peaks, setPeaks] = useState<PeaksFeed | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  /** Whether the Plan tab has already gone looking for its sites. */
  const autoSites = useRef(false);
  /** Which hour of the solved run the map is showing. Null is the peak, which
   *  is where everything sits until someone presses play. */
  const [runFrame, setRunFrame] = useState<number | null>(null);
  const [placed, setPlaced] = useState<PlacedLevee[]>([]);
  const [placing, setPlacing] = useState(false);
  // The panel used to be one continuous scroll: forecast, roads, design event,
  // budget, options, impact, all stacked. Everything was on screen and nothing
  // was findable. Three sections match the three questions an operator
  // actually arrives with -- what is coming, what does it cut, what do we
  // build -- and only one is ever open.
  const [section, setSection] = useState<"forecast" | "roads" | "plan">("forecast");

  // One place decides which view is live: the three tabs the design has, plus
  // the two controls that are not tabs. Everything downstream still reads
  // `tab`, so the panels and the map did not change when the nav did.
  const tab: "anywhere" | "trip" | "section" | "street" | "kit" =
    streetOn ? "street"
      : section === "roads" ? "trip"
        : section === "plan" ? planMode
          : "anywhere";
  const setTab = (t: "anywhere" | "trip" | "section" | "street" | "kit") => {
    if (t === "street") { setStreetOn(true); return; }
    setStreetOn(false);
    if (t === "trip") setSection("roads");
    else if (t === "kit" || t === "section") { setSection("plan"); setPlanMode(t); }
    else setSection("forecast");
  };

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
    if (Number.isFinite(b) && b >= 2 && b <= 80) setKitBudget(b);
    const sec = q.get("view");
    if (sec === "forecast" || sec === "roads" || sec === "plan") setSection(sec);
  }, []);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    q.set("budget", String(kitBudget));
    q.set("view", section);
    window.history.replaceState(null, "", `?${q.toString()}`);
  }, [kitBudget, section]);
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
    setRegion(analysis?.name ?? study?.study?.place ?? null);
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
  }, []);

  // Best value among affordable options that actually help.
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

  /** The solved run, hour by hour. Nine instantaneous depth fields the
   *  solver computed on its way to the peak and used to throw away. */
  const runFrames: Frame[] = useMemo(
    () => (study?.frames ?? []) as Frame[], [study]);

  /**
   * The flood the map paints, and which solve it came from.
   *
   * Four answers, and the order is the point. A trip and a street view each
   * solve their own box and must draw it. The Plan tab draws the SCENARIO's
   * flood -- without that a panel headed "9 Sep 1921 flood" sat over a map
   * of a 50 mm/h design storm, naming one flood and showing another. The
   * scrubber, when it is on an hour, draws that hour. Everything else is the
   * shipped study's peak.
   */
  const mapField = useMemo(() => {
    if (tab === "trip") {
      return trip ? { bounds: trip.bounds, depth_png: trip.depth_png,
                      terrain_png: trip.terrain_png } : null;
    }
    if (tab === "street" && street?.ok) {
      return { bounds: street.bounds,
               depth_png: (streetFrame !== null && street.frames[streetFrame])
                 ? street.frames[streetFrame].depth_png
                 : street.depth_png,
               terrain_png: street.terrain_png };
    }
    if (section === "plan" && kit?.ok && kit.depth_png && kit.bounds) {
      return { bounds: kit.bounds, depth_png: kit.depth_png,
               terrain_png: study?.terrain_png ?? "" };
    }
    if (runFrame !== null && runFrames[runFrame] && study) {
      return { bounds: study.bounds,
               depth_png: runFrames[runFrame].depth_png,
               terrain_png: study.terrain_png };
    }
    return analysis ?? (study
      ? { bounds: study.bounds, depth_png: study.depth_png,
          terrain_png: study.terrain_png }
      : null);
  }, [tab, section, trip, street, streetFrame, kit, runFrame, runFrames,
      analysis, study]);

  /**
   * Is every road out cut?
   *
   * Measured, not asserted: the road query already scores every named road
   * in the box against the vehicle depth, so this is the share of named
   * roads whose worst point is under it. It is the fact that decides whether
   * a levee is the right instrument at all -- a place nobody can leave needs
   * a route before it needs a wall.
   */
  const noWayOut = useMemo(() => {
    const named = roadQ?.roads.filter((r) => r.name) ?? [];
    return named.length >= 3 && named.every((r) => r.maxDepthM > (roadQ?.threshold ?? 0));
  }, [roadQ]);

  const kitBox = useMemo(() => {
    const b = analysis?.bounds ?? study?.bounds;
    return b ? { west: b.west, south: b.south, east: b.east, north: b.north } : null;
  }, [analysis, study]);

  // The scenario list, from the reach's own gauge record.
  //
  // Fetched once per box, alongside the site search, because a plan panel that
  // opens on a design rainstorm is answering the wrong question and the right
  // one needs the USGS peak record to exist first.
  const scenarioName = analysis?.name ?? study?.name ?? "";
  useEffect(() => {
    if (!kitBox || section !== "plan" || peaks) return;
    let dead = false;
    void (async () => {
      try {
        const r = await fetch("/api/peaks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...kitBox, reach: scenarioName }),
        });
        const d = await r.json();
        if (!dead) setPeaks(d as PeaksFeed);
      } catch {
        if (!dead) setPeaks({ ok: false, reason: "no gauge record", gauges: [],
                              source: "" });
      }
    })();
    return () => { dead = true; };
  }, [kitBox, section, peaks, scenarioName]);

  const { list: scenarios, gauge: peakGauge } = useMemo(
    () => scenariosFrom(peaks,
                        study?.storm?.rain_mm_h ?? 50,
                        study?.storm?.storm_hours ?? 6),
    [peaks, study]);

  // The first RECORDED flood, where there is one. Not the study's storm: a
  // levee under uniform rain holds water in, and opening on that makes the
  // whole panel read as if walls do not work.
  const scenario: Scenario = useMemo(
    () => scenarios.find((x) => x.id === scenarioId) ?? scenarios[0],
    [scenarios, scenarioId]);

  /** What every solve on this tab is driven by. One object, so the sites, the
   *  section and the plan cannot end up disagreeing about the flood. */
  const forcing = useMemo(() => ({
    driver: scenario.driver,
    river_q_m3s: scenario.river_q_m3s,
    rain_mm_h: study?.storm?.rain_mm_h ?? 50,
    storm_hours: study?.storm?.storm_hours ?? 6,
  }), [scenario, study]);

  // A changed scenario is a changed flood, so everything downstream of it
  // goes with it. Done in the handler rather than in an effect watching the
  // id: an effect would be a second render pass to undo what the first one
  // just drew, and there is no external system here to synchronise with.
  const chooseScenario = useCallback((id: string) => {
    setScenarioId(id);
    autoSites.current = false;
    setKit(null); setKitErr(null); setPlacedPieces([]); setKitPlan(null);
    setSecSite(null); setSecRes(null); setSecErr(null);
  }, []);

  /** The plan, as a file. See the button's comment: there is no account here
   *  and no server to save to, so what Save produces is the thing someone
   *  could actually reproduce the run from. */
  const savePlan = useCallback(() => {
    if (!kit?.ok) return;
    const doc = {
      saved_at: new Date().toISOString(),
      place: analysis?.name ?? study?.name ?? "",
      bounds: kitBox,
      scenario: {
        id: scenario.id, label: scenario.label, note: scenario.note,
        driver: scenario.driver, river_q_m3s: scenario.river_q_m3s,
      },
      gauge: peakGauge
        ? { site: peakGauge.site, name: peakGauge.name, years: peakGauge.years,
            fit: peakGauge.fit }
        : null,
      budget_usd: kitBudget * 1e6,
      measures: placedPieces.map((pp) => {
        const site = kit.sites[pp.site];
        const piece = pieceOf(pp.piece)!;
        return {
          piece: piece.id, label: piece.label, kind: piece.kind,
          intensity: piece.intensity, cost_usd: Math.round(costOf(pp, kit.sites)),
          site: { name: site.name, lat: site.lat, lon: site.lon,
                  length_m: site.length_m, needs_m: site.needs_m },
        };
      }),
      solved: kitPlan ?? null,
      rates: "USACE construction rates escalated to 2025 x 0.88 Texas; "
           + "see portal/src/lib/rates.ts for each figure and its source.",
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `plan-${(doc.place.split(",")[0] || "area")
      .toLowerCase().replace(/\W+/g, "-")}-${scenario.id.replace(/\W+/g, "")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [kit, kitBox, kitBudget, kitPlan, placedPieces, scenario, peakGauge,
      analysis, study]);

  const findSites = useCallback(async () => {
    if (!kitBox) return;
    setKitBusy(true); setKitErr(null); setKit(null); setKitSec(0);
    setPlacedPieces([]); setKitPlan(null);
    const tick = setInterval(() => setKitSec((n) => n + 1), 1000);
    try {
      const r = await fetch("/api/sites", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...kitBox, name: analysis?.name ?? "Area", dx: 40,
          ...forcing, run_hours: 18, max_sites: 6, shortfall_m: 2.0,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      if (!d.ok) throw new Error(d.reason ?? "no sites here");
      setKit(d);
    } catch (e) {
      setKitErr(e instanceof Error ? e.message : "could not find sites");
    } finally { setKitBusy(false); clearInterval(tick); }
  }, [kitBox, analysis, forcing]);

  // The Plan tab finds its own sites.
  //
  // It used to open on a button and a paragraph explaining what the button
  // would do. But the sites are not a choice the planner makes -- they come
  // out of the terrain and the solved flood, the same way the flood field
  // does -- so asking for a click before showing them put a turnstile in
  // front of the one thing the tab exists to show. The design has the tray
  // and the outlines already on screen. This is what puts them there.
  useEffect(() => {
    if (section !== "plan" || !kitBox) return;
    if (kit || kitBusy || kitErr || autoSites.current) return;
    autoSites.current = true;
    void findSites();
  }, [section, kitBox, kit, kitBusy, kitErr, findSites]);

  const solveKitPlan = useCallback(async () => {
    if (!kitBox || !kit?.ok || placedPieces.length < 2) return;
    setKitPlanBusy(true);
    try {
      // Only the sites with something on them go to the planner: the library
      // it enumerates is exponential in the number of sites, and asking it to
      // consider six when two are in play pays for combinations nobody chose.
      const chosen = placedPieces.map((p) => ({
        site: kit.sites[p.site], piece: pieceOf(p.piece)! }));
      const maxH = Math.max(...chosen.map((c) => c.piece.intensity), 1);
      const r = await fetch("/api/plan-anywhere", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          // 60 m and twelve hours, not 40 and eighteen. This solves one bed
          // per candidate combination, and at 40 m over the whole study box it
          // measured seven minutes for two sites -- which is not a thing
          // anyone will sit through, and the margin needs more sites than
          // that before it can be calibrated at all.
          ...kitBox, name: analysis?.name ?? "Area", dx: 60,
          ...forcing, run_hours: 12,
          // The alignment goes with the site. Without it the solver falls
          // back to a line through the point along the local contour, which
          // is a guess where `/sites` already measured the answer.
          // Each measure carries what it IS and what it costs. `kind` tells
          // the solver whether to sweep a crest into the bed or to open an
          // exchange between two cells; `cost_model` and `unit_cost` are the
          // rates from `lib/rates`, sent rather than duplicated, so the
          // service has no construction price of its own to drift from.
          sites: chosen.map((c) => ({
            lat: c.site.lat, lon: c.site.lon, width_m: c.site.width_m,
            outline: c.site.outline, line: c.site.line,
            kind: c.piece.kind,
            cost_model: c.piece.cost_model,
            unit_cost: c.piece.unit,
          })),
          // What the plan is bought to protect. Without this the service
          // scores the mean depth over the WHOLE box, and two one-kilometre
          // walls on a 130 km2 domain moved it 0.04% -- a true number about
          // the wrong question. Behind the walls is where the money is spent.
          protect: {
            lat: chosen.reduce((a, c) => a + c.site.lat, 0) / chosen.length,
            lon: chosen.reduce((a, c) => a + c.site.lon, 0) / chosen.length,
            radius_m: 2000,
          },
          max_height_m: maxH,
          budget_usd: kitBudget * 1e6,
          delta: 0.1,
        }),
      });
      const d = await r.json();
      if (!r.ok || !d.ok) throw new Error(d.reason ?? d.detail ?? `HTTP ${r.status}`);
      setKitPlan(d);
    } catch (e) {
      setKitErr(e instanceof Error ? e.message : "could not solve this plan");
    } finally { setKitPlanBusy(false); }
  }, [kitBox, kit, placedPieces, kitBudget, analysis, forcing]);

  const kitImpact = useMemo(() => {
    const d = kitPlan as {
      options?: { id: string; heights: number[]; area_flooded_km2: number;
                  core_reduction_pct: number }[];
      band_pct?: number; reduction_pct?: number;
      band_attainable?: boolean; band_note?: string;
    } | null;
    if (!d?.options?.length) return null;
    const base = d.options.find((o) => o.id === "base") ?? d.options[0];
    // The best option the solve actually produced within budget, which is the
    // honest "with plan": the greedy allocation can propose heights no solved
    // option carries, and showing its predicted area beside a solved baseline
    // would compare a prediction with a measurement.
    const best = d.options.reduce((a, o) =>
      o.core_reduction_pct > a.core_reduction_pct ? o : a, base);
    return {
      base: { area_km2: base.area_flooded_km2, cut_km: null, homes: null },
      with: { area_km2: best.area_flooded_km2, cut_km: null, homes: null },
      // The service says whether the margin was ATTAINABLE, and that is not
      // the same as whether a number came back: split conformal needs enough
      // calibration points before any finite order statistic carries the
      // stated rate, and `margins.py` refuses rather than returning one it
      // cannot justify. Showing `band_pct` without checking this would quote a
      // guarantee the framework declined to make.
      band_pct: d.band_attainable === false ? null
        : (typeof d.band_pct === "number" ? d.band_pct : null),
      reduction_pct: best.core_reduction_pct,
      band_reason: d.band_note
        ?? "A margin needs enough calibrated combinations before it means 90%.",
    };
  }, [kitPlan]);

  const kitSites = useMemo(() => {
    if (!kit?.ok) return null;
    return kit.sites.map((s, i) => {
      const on = placedPieces.find((p) => p.site === i);
      const piece = on && PIECES.find((x) => x.id === on.piece);
      return {
        lat: s.lat, lon: s.lon, name: s.name, outline: s.outline,
        taken: Boolean(on),
        label: piece
          ? (piece.kind === "levee"
            ? `${piece.label} ${piece.intensity.toFixed(1)} m`
            : piece.label)
          : `${s.name ?? "bank"} · ${(s.length_m / 1000).toFixed(1)} km`,
      };
    });
  }, [kit, placedPieces]);

  const solveSection = useCallback(async (
    opts: { driver: string; storm: number; q: number | null },
  ) => {
    if (!sectionEnds.a || !sectionEnds.b) return;
    setSecBusy(true); setSecErr(null); setSecRes(null); setSecSec(0);
    setSecStation(null); setSecCrest(null);
    const tick = setInterval(() => setSecSec((n) => n + 1), 1000);
    const s = [20, 50, 80][opts.storm] ?? 50;
    try {
      const r = await fetch("/api/section", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          a: sectionEnds.a, b: sectionEnds.b, name: "Cross-section",
          // Sample at roughly the model's own spacing. A fixed 140 points put
          // a sample every 6 m on an 800 m line read off a 50 m grid, which
          // is eight samples per cell: eight copies of the same number, drawn
          // as if they were eight measurements.
          samples: Math.max(24, Math.min(240, Math.round(
            Math.hypot(
              (sectionEnds.b.lon - sectionEnds.a.lon) * 111320
                * Math.cos((sectionEnds.a.lat * Math.PI) / 180),
              (sectionEnds.b.lat - sectionEnds.a.lat) * 110540,
            ) / 30))),
          rain_mm_h: s, storm_hours: 6, run_hours: 12,
          driver: opts.driver,
          ...(opts.driver === "river" && opts.q ? { river_q_m3s: opts.q } : {}),
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      if (!d.ok) throw new Error(d.reason ?? "this line could not be solved");
      setSecRes(d);
    } catch (e) {
      setSecErr(e instanceof Error ? e.message : "could not cut this section");
    } finally { setSecBusy(false); clearInterval(tick); }
  }, [sectionEnds]);

  // The wall's default position: the highest ground that still faces the
  // water, at the lowest crest that holds there. Derived rather than seeded,
  // so a new solve or a new bank never renders one line's wall on another
  // line's ground.
  const secDefault = useMemo(() => {
    if (!secRes?.ok) return null;
    const st = bestStation(secRes.profile, "a");
    const c = crestToHold(secRes.profile, st, 0.3, "a");
    return { station: st, crest: c ?? secRes.profile[st].ground_m + 0.5 };
  }, [secRes]);

  const secStationIdx = secStation ?? secDefault?.station ?? 0;
  const secCrestM = secCrest ?? secDefault?.crest ?? 0;
  const secBankLen = secSite !== null && kit?.ok
    ? kit.sites[secSite]?.length_m ?? 0 : 0;
  // Where the channel is on this line, so a wall cannot be dropped in it.
  const secBelts = useMemo(() => {
    if (!secRes?.ok) return [];
    return reaches(secRes.profile).flatMap((r) => channelBelts(secRes.profile, r));
  }, [secRes]);
  const secWall = useMemo(
    () => (secRes?.ok
      ? wallAt(secRes.profile, secStationIdx, secCrestM, secBankLen || 500, 2600,
               "a", secBelts)
      : null),
    [secRes, secStationIdx, secCrestM, secBankLen, secBelts],
  );

  /** The box the scene is built on. The map used to be gated on the Richmond
   *  manifest, so deleting that one file rendered a grey pulse for Austin. */
  const mapBounds = useMemo(
    () => analysis?.bounds ?? study?.bounds ?? null,
    [analysis, study]);

  const sectionPicking =
    sectionArmed && !(sectionEnds.a && sectionEnds.b);

  const chooseSectionSite = useCallback((i: number | null) => {
    setSecSite(i);
    setSecRes(null); setSecErr(null); setSecStation(null); setSecCrest(null);
    const line = i !== null && kit?.ok ? kit.sites[i]?.line : null;
    setSectionEnds(line ? { a: line.a, b: line.b } : { a: null, b: null });
  }, [kit]);

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
      if (!adHocGrids || !mapBounds) return;
      setPoint(sampleAt(adHocGrids, mapBounds, lon, lat));
    },
    [adHocGrids, mapBounds, tab, placing, sectionPicking],
  );

  /** The box the live feeds are asked about: whatever place is on screen. */
  const liveBounds = useMemo(() => {
    const b = analysis?.bounds ?? trip?.bounds ?? study?.bounds;
    if (!b) return null;
    // Padded well beyond the solved box, and that is not slop. The gauge that
    // tells you what is coming is UPSTREAM, which is outside the box by
    // definition: the Austin study misses Onion Creek at US 183 -- the one
    // gauge here with published flood levels -- by 0.002 degrees of longitude.
    const pad = 0.12;
    return { west: b.west - pad, south: b.south - pad,
             east: b.east + pad, north: b.north + pad };
  }, [analysis, trip, study]);

  // The gauge the strip under the map plots: one with published flood levels
  // if there is one, because a stage without a threshold beside it does not
  // say whether anything is wrong.
  // Dots for every gauge, READINGS for a few. Twenty-seven callouts over one
  // city is not a map, it is a wall of boxes with a city behind it. The ones
  // that get a label are the ones with flood levels -- a stage with a
  // threshold beside it is the only kind that means anything on its own --
  // and then the largest flows.
  const labelledGauges = useMemo(() => {
    const gs = gaugeFeed?.gauges ?? [];
    const rank = (g: typeof gs[number]) =>
      (g.flood_levels_m ? 1e9 : 0) + (g.q_m3s ?? 0);
    const keep = new Set([...gs].sort((a, b) => rank(b) - rank(a))
                                .slice(0, 6).map((g) => g.site));
    return gs.map((g) => ({ ...g, labelled: keep.has(g.site) }));
  }, [gaugeFeed]);

  const hydroGauge = useMemo(() => {
    const gs = gaugeFeed?.gauges ?? [];
    return gs.find((g) => g.series.length > 2 && g.flood_levels_m)
      ?? gs.find((g) => g.series.length > 2)
      ?? null;
  }, [gaugeFeed]);

  /** The top of the depth ramp on screen, for the legend. */
  const depthVmax = useMemo(() => {
    if (tab === "street" && street?.ok) return street.depth_vmax_m;
    if (analysis) return analysis.depth_vmax_m;
    if (study) return study.depth_vmax_m;
    return null;
  }, [tab, street, analysis, study]);

  // The trip ships the same quantised grids an ad-hoc analysis does, so the
  // body of water and the click-anywhere depth work there too.
  const tripGrids = useMemo(
    () => decodeGrids(trip?.inspect as Parameters<typeof decodeGrids>[0]),
    [trip]);

  return (
    <div className="flex h-full min-h-0">
      {/* ---- decision panel ------------------------------------------- */}
      <aside className="flex w-[380px] shrink-0 flex-col overflow-y-auto border-r border-line bg-bg">
        {studyErr && (
          <div className="border-b border-line bg-bg-inset px-5 py-3">
            <p className="text-[13px] font-medium text-ink">No study is built</p>
            <p className="anno mt-1 leading-relaxed">
              The portal ships one solved place and it is not there. Build it
              with scripts/build_austin_study.py.
            </p>
          </div>
        )}
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

        {/* Three tabs and nothing else above the content.
            There was a "Study" row with two places and an "Area" row with six
            chips: 180 px of navigation before anything is said, and two of
            the six -- Street and Cross-section -- are not areas at all. One
            study needs no study switch. Street is a zoom, so it belongs on
            the map with the other zooms. And the two planning tools are what
            the Plan tab is FOR, so they are steps inside it. */}
        {/* Which flood, before which tool. A levee that holds against one
            event is not a levee that holds, and the panel cannot say what a
            wall buys until it has been told what it is buying against. */}
        {section === "plan" && (
          <ScenarioPicker
            list={scenarios}
            chosen={scenario}
            gauge={peakGauge}
            disabled={kitBusy || kitPlanBusy || secBusy}
            onChoose={(sc) => chooseScenario(sc.id)}
          />
        )}

        {section === "plan" && (
          <div className="border-b border-line px-5 py-3">
            <div className="flex flex-wrap gap-1">
              {(["kit", "section"] as const).map((k) => (
                <button key={k} className="pill whitespace-nowrap"
                        data-on={planMode === k}
                        onClick={() => { setPlanMode(k);
                                         if (k === "section") setSectionArmed(true); }}>
                  {k === "kit" ? "Build a plan" : "Cross-section"}
                </button>
              ))}
            </div>
          </div>
        )}

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
            {/* Three days of river, with the gauge reading it belongs to.
                It used to run the full width under the map, where it sat
                beside a play button that could not move it. */}
            {hydroGauge && (
              <div className="border-b border-line px-5 pb-3">
                <Hydrograph gauge={hydroGauge} />
              </div>
            )}
            <LayersPanel
              bounds={liveBounds}
              layers={layers}
              onLayers={setLayers}
              onGauges={setGaugeFeed}
              modelRunNote={analysis
                ? `solved at ${analysis.dx_m.toFixed(0)} m · ${analysis.flooded_km2} km² over 10 cm`
                : undefined}
            />
            {layers.roads && (
              <RoadsOut
                roads={adHocRoads}
                grids={adHocGrids}
                bounds={mapBounds}
                place={analysis?.name ?? study?.study?.short ?? null}
                vehicle={vehicle}
                onVehicle={setVehicle}
                onQuery={setRoadQ}
              />
            )}
          </>
        )}

        {tab === "kit" && (
          <LeveeKit
            feed={kit}
            busy={kitBusy}
            elapsed={kitSec}
            err={kitErr}
            onFindSites={findSites}
            canFind={Boolean(kitBox)}
            placed={placedPieces}
            onPlaced={setPlacedPieces}
            budgetM={kitBudget}
            onBudget={setKitBudget}
            armed={armed}
            onArm={setArmed}
            impact={kitImpact}
            planBusy={kitPlanBusy}
            onPlan={solveKitPlan}
            noWayOut={noWayOut}
            onSave={savePlan}
          />
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
            onClearEnds={() => { setSectionEnds({ a: null, b: null }); setSecRes(null); }}
            onOverlay={setSectionOverlay}
            sites={kit}
            sitesBusy={kitBusy}
            onFindSites={findSites}
            canFind={Boolean(kitBox)}
            chosenSite={secSite}
            onChooseSite={chooseSectionSite}
            result={secRes}
            busy={secBusy}
            elapsed={secSec}
            err={secErr}
            onSolve={solveSection}
            stationIdx={secStationIdx}
            crestM={secCrestM}
            wall={secWall}
            onCrest={setSecCrest}
            onSnap={() => {
              if (!secRes?.ok) return;
              const st = bestStation(secRes.profile, "a");
              setSecStation(st);
              setSecCrest(crestToHold(secRes.profile, st, 0.3, "a")
                          ?? secRes.profile[st].ground_m + 0.5);
            }}
            homesKeptDry={null}
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

      </aside>

      {/* ---- map, and the section under it ----------------------------- */}
      <div className="flex min-w-0 flex-1 flex-col">
      <div className="relative min-h-0 flex-1">
        {mapBounds ? (
          <PlannerMap
            bounds={mapBounds}
            roads={tab === "trip" ? (trip?.roads ?? null) : adHocRoads}
            focusBounds={tab === "street" && street?.ok ? street.bounds : null}
            pins={tab === "trip" && trip
              ? [{ lon: trip.origin.lon, lat: trip.origin.lat, label: "from" },
                 { lon: trip.destination.lon, lat: trip.destination.lat, label: "to" }]
              : null}
            onPick={onPick}
            pin={point ? { lon: point.lon, lat: point.lat } : null}
            cutFraction={roadQ?.cutFraction ?? null}
            cutSignature={roadQ ? `${roadQ.threshold}:${roadQ.cutKm.toFixed(2)}` : "none"}
            placed={tab === "anywhere" ? placed : []}
            placedHeights={tab === "anywhere" ? adHocHeights : null}
            routes={routeLines?.forTab === tab ? routeLines : null}
            section={tab === "section" ? sectionOverlay : null}
            buildings={tab === "street" ? streetBuildings : null}
            sites={tab === "kit" ? kitSites : null}
            onSite={(i) => {
              if (armed) {
                setPlacedPieces((l) => [...l.filter((p) => p.site !== i),
                                        { site: i, piece: armed }]);
                setArmed(null);
              } else {
                setPlacedPieces((l) => l.filter((p) => p.site !== i));
              }
            }}
            onBuilding={setStreetSel}
            gauges={layers.gauges ? labelledGauges : null}
            radar={layers.radar ? liveBounds : null}
            place={tab === "anywhere" ? place : null}
              analysis={mapField}
            focus={focus}
            // No solid water body in the street view: the grids are the
            // study's 40 m field over the whole study box, and drawn under a
            // camera zoomed to one block it covers everything in sight. The
            // block's own 10 m field arrives as the depth overlay instead.
            grids={tab === "trip" ? tripGrids
              : tab === "street" ? null
                : section === "plan" ? null
                  : runFrame !== null ? null : adHocGrids}
            solid={solid}
          />
        ) : (
          <div className="h-full w-full animate-pulse bg-bg-inset" />
        )}
        {/* The depth scale. A blue wash over a satellite photo is not
            self-explanatory, and the one number that makes it readable -- what
            the darkest blue means -- was nowhere on screen. */}
        {depthVmax !== null && (
          <div className="absolute right-4 top-4 rounded-md border border-ink-faint/40
                          bg-bg/90 px-2.5 py-1.5 backdrop-blur">
            <p className="anno mb-1">Water depth</p>
            <div className="h-[7px] w-[150px] rounded-[2px]"
                 style={{ background: "linear-gradient(90deg,#60c8ff,#1f8fd6,#04247f)" }} />
            <div className="mt-1 flex w-[150px] justify-between">
              {[0, depthVmax / 2, depthVmax].map((v, i) => (
                <span key={i} className="anno">
                  {i === 0 ? "0 m" : `${v.toFixed(1)}${i === 2 ? " m+" : ""}`}
                </span>
              ))}
            </div>
          </div>
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
          {/* Street sits with the other two because it is the third zoom of
              the same place, not a different question -- which is how the
              wireframe has it and how anyone reading a map expects it. */}
          <button
            aria-label="Zoom to street level"
            onClick={() => setTab(tab === "street" ? "anywhere" : "street")}
            className={`rounded-md border px-2.5 py-1 text-[12px] backdrop-blur transition-colors ${
              tab === "street"
                ? "border-ink bg-ink text-bg font-medium"
                : "border-ink-faint/40 bg-bg/80 text-ink hover:bg-bg"
            }`}
          >
            Street
          </button>
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

        {/* Before / With plan, on the map.
            It was a pair of pills in the panel, which is where the design
            does NOT put it: what the toggle switches is the water on the
            map, so the control belongs there, centred, the way the design
            has it. */}
        {section === "plan" && kitImpact && (
          <div className="absolute left-1/2 top-3 z-10 flex -translate-x-1/2 gap-0
                          overflow-hidden rounded-md border border-ink-faint/40
                          bg-bg/85 backdrop-blur">
            {([false, true] as const).map((v) => (
              <button
                key={String(v)}
                onClick={() => setShowPlan(v)}
                aria-pressed={showPlan === v}
                className={`px-3 py-1 text-[12px] transition-colors ${
                  showPlan === v ? "bg-ink font-medium text-bg"
                    : "text-ink hover:bg-bg"}`}
              >
                {v ? "With plan" : "Before"}
              </button>
            ))}
          </div>
        )}

        {/* The scrubber, floating on the map rather than stacked under it.
            It is on the map because what it moves is on the map: nine
            instantaneous depth fields across the solved run. The gauge
            hydrograph is NOT here, and that is the point -- the gauge runs
            on a wall clock and the model on its own, and one play button
            over two clocks would be a control over nothing. */}
        {runFrames.length > 1 && section === "forecast" && tab !== "street" && (
          <div className="pointer-events-none absolute inset-x-4 bottom-4 z-10">
            <RunScrubber
              frames={runFrames}
              index={runFrame}
              onIndex={setRunFrame}
              stormHours={study?.storm?.storm_hours ?? 6}
              peakKm2={study?.flooded_km2 ?? 0}
            />
          </div>
        )}
      </div>

      {/* The section runs the full width under the map, because that is the
          shape it is read in: a long thin drawing of ground and water, with
          the map above it showing where the line was cut. In the side panel
          it was 340 px wide and a kilometre of floodplain had to fit in it. */}

      {tab === "section" && secRes?.ok && secWall && (
        <div className="shrink-0 border-t border-line bg-bg">
          <SectionStrip
            res={secRes}
            stationIdx={secStationIdx}
            crestM={secCrestM}
            wall={secWall}
            landward="a"
            title={(secSite !== null && kit?.ok
                    ? kit.sites[secSite]?.name : null) ?? "this line"}
            onStation={setSecStation}
            onCrest={setSecCrest}
          />
        </div>
      )}
      </div>
    </div>
  );
}
