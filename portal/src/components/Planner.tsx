"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PlannerMap, STREET_VE } from "./PlannerMap";
import { LayersPanel, DEFAULT_LAYERS, type LayerState,
         type GaugeFeed } from "./LayersPanel";
import { GaugeHeadline, pickGauge } from "./GaugeHeadline";
import { Hydrograph } from "./Hydrograph";
import { RoadsOut } from "./RoadsOut";
import { queryRoads } from "@/lib/roadquery";
import { AnywherePanel, type AnalysisResult, type PlacedLevee,
         type StudyMeta } from "./AnywherePanel";
import { TripPanel, type TripResult } from "./TripPanel";
import { RoutePanel } from "./RoutePanel";
import { SectionPanel, type SectionEnds, type SectionOverlay } from "./SectionPanel";
import { SectionStrip } from "./SectionStrip";
import { OperatePanel, type OperateReply } from "./OperatePanel";
import { wallAt, crestToHold, bestStation, reaches, channelBelts,
         type SectionResult } from "@/lib/section";
import { ScenarioPicker } from "./ScenarioPicker";
import { EARTH_FILL } from "@/lib/rates";
import { RunScrubber, type Frame } from "./RunScrubber";
import { scenariosFrom, type PeaksFeed, type Scenario } from "@/lib/scenarios";
import { StreetPanel, type StreetResult } from "./StreetPanel";
import { LeveeKit, PIECES, pieceOf, costOf, type SitesFeed, type Placed,
         type PieceId } from "./LeveeKit";
import { useRegion } from "./RegionContext";
import { withDepth } from "@/lib/depthtex";
import { decodeGrids, sampleAt,
         type InspectGrids, type PointInfo } from "@/lib/inspect";

/** A 1x1 fully transparent PNG, for drawing the ground with no flood on it.
 *
 *  Generated and decoded rather than copied: the first constant here was a
 *  "transparent" PNG off the shelf whose single pixel is RGBA (0, 0, 255,
 *  127) -- half-opaque blue -- so switching the depth layer off would have
 *  washed the entire map blue. Checking that a PNG is VALID is not checking
 *  that it is transparent. This one's IDAT inflates to `00 00 00 00 00`:
 *  filter 0, then RGBA all zero. */
const BLANK_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNg"
  + "AAIAAAUAAXpeqz8AAAAASUVORK5CYII=";

export function Planner() {
  const [focus, setFocus] = useState<"region" | "town">("region");
  /** Water drawn as a body, or painted flat over the terrain. */
  /* Off by default. A flood map is read as a map -- north up, flat, filling
     its pane -- and the solid body is the opt-in second reading, which is why
     it now brings the pitch with it. It was on by default and invisible: the
     columns were drawn from z = 0 while the terrain mesh put the ground six
     hundred metres up, so the chip had never shown anything. */
  const [solid, setSolid] = useState(false);
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
  /* `roadQ` is derived further down, not stored. The only other writer was
     `RoadQueryPanel`, which renders inside `AnywherePanel` under
     `section === "roads"` while `AnywherePanel` itself needs
     `section === "forecast"` -- conditions that cannot both hold, so that
     panel is unreachable and never set it. */
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
  const [planMode, setPlanMode] = useState<"kit" | "section" | "operate">("kit");
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
  /** How far the street block has been turned, in degrees. A model on a
   *  table is turned to be read; the two buttons over the map step this. */
  const [streetSpin, setStreetSpin] = useState(0);
  /** Which house the camera is centred on, when that is not the densest
   *  cluster. Picking from the LIST is navigation -- the list is a
   *  directory, sorted by depth, and its top row is often nowhere near the
   *  street on screen. Picking on the MAP is not: the reader is already
   *  looking at the thing they clicked, and moving the camera under them
   *  throws away the view they built. */
  const [streetGoTo, setStreetGoTo] = useState<string | null>(null);

  /** The levee kit: where a wall would pay, what is placed, what it buys. */
  const [kit, setKit] = useState<SitesFeed | null>(null);
  /* Operating what was placed. `/plan` decides the design; this runs it.
     Kept beside the kit's state because it reads the same placed pieces --
     a schedule for structures that are not on the map is not a question. */
  const [op, setOp] = useState<OperateReply | null>(null);
  const [opBusy, setOpBusy] = useState(false);
  const [opSec, setOpSec] = useState(0);
  const [opPolicy, setOpPolicy] = useState<string | null>(null);
  const [opFrame, setOpFrame] = useState<number | null>(null);
  const [kitBusy, setKitBusy] = useState(false);
  const [kitSec, setKitSec] = useState(0);
  const [kitErr, setKitErr] = useState<string | null>(null);
  const [placedPieces, setPlacedPiecesRaw] = useState<Placed[]>([]);
  const [armed, setArmed] = useState<PieceId | null>(null);
  const [kitBudget, setKitBudget] = useState(20);
  const [kitPlan, setKitPlan] = useState<Record<string, unknown> | null>(null);
  const [kitPlanBusy, setKitPlanBusy] = useState(false);
  /* Changing the plan throws away the answer to the old one. "Solve this
     plan" used to unmount itself the moment a result arrived, so after one
     solve you could add, remove or clear pieces and the readout -- "Land
     flooded 6.35 -> 6.12 km2, at least 12.3% less flooding" -- stayed on
     screen describing a plan that was no longer on the map, with no way to
     re-run it short of changing the scenario. */
  const setPlacedPieces = useCallback(
    (next: Placed[] | ((l: Placed[]) => Placed[])) => {
      setKitPlan(null);
      setPlacedPiecesRaw(next);
    }, []);
  const [showPlan, setShowPlan] = useState(true);
  /** Which flood the plan is judged against. See `lib/scenarios`: the study's
   *  own design rainstorm is the wrong question to put a levee in front of,
   *  because rain lands on both sides of a wall. */
  const [peaks, setPeaks] = useState<PeaksFeed | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  /** Whether the Plan tab has already gone looking for its sites, and for
   *  which flood. */
  const autoSites = useRef(false);
  const solvedFor = useRef<string | null>(null);
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
  const tab: "anywhere" | "trip" | "section" | "street" | "kit" | "operate" =
    streetOn ? "street"
      : section === "roads" ? "trip"
        : section === "plan" ? planMode
          : "anywhere";
  const setTab = (t: "anywhere" | "trip" | "section" | "street" | "kit"
                   | "operate") => {
    // A depth reading belongs to the view it was taken in. Carrying it across
    // is how a number sampled from the study's peak field ends up floating
    // over a street solve or a different scenario's flood.
    setPoint(null);
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

  /* The header says LIVE only when something on the page really is.
     It used to read "27 gauges every 15 min" while nothing re-fetched at
     all -- the cadence of the SOURCE, printed as though it were the
     cadence of this page. The panel now polls, so the badge states when
     the reading on screen was taken, which is the fact a reader needs to
     judge whether it is still worth anything. */
  useEffect(() => {
    const n = gaugeFeed?.gauges.length ?? 0;
    const at = gaugeFeed?.gauges.find((g) => g.at)?.at;
    const read = at
      ? new Date(at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      : null;
    setLive(n
      ? `${n} gauge${n === 1 ? "" : "s"}${read ? ` · read ${read}` : ""}`
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
      // "Still solving" is neither an answer nor a failure. The service runs
      // one solve per distinct block and lets every caller wait on it, so
      // asking again joins that run instead of starting a second.
      if (d.pending) return "pending" as const;
      if (!d.ok) throw new Error(d.reason ?? "this block could not be solved");
      setStreet(d);
    } catch (e) {
      setStreetErr(e instanceof Error ? e.message : "could not open this street");
    } finally {
      setStreetBusy(false);
      clearInterval(tick);
    }
    return "done" as const;
  }, [streetCentre, analysis, study]);

  /** Keep asking until the block answers.
   *
   *  A cold street solve is a parent catchment plus the block itself and runs
   *  past the five minutes a browser's fetch waits for response headers. Each
   *  call waits on the one run the service is already doing, so this loop
   *  costs nothing but patience. */
  const solveStreetUntilDone = useCallback(async (
    opts: { driver: string; storm: number; q: number | null },
  ) => {
    for (let i = 0; i < 8; i++) {
      if (await solveStreet(opts) !== "pending") return;
    }
  }, [solveStreet]);

  /* A new street is a new model on the table: it starts square-on and with
     nothing picked. Adjusted during render rather than in an effect, which
     is React's own prescription for resetting state when an input changes
     -- an effect here would paint the previous street's camera for a frame
     and costs a second render. */
  const [spinFor, setSpinFor] = useState(street);
  if (street !== spinFor) {
    setSpinFor(street);
    setStreetSpin(0);
    setStreetGoTo(null);
  }

  const streetBuildings = useMemo(() => {
    if (!street?.ok) return null;
    return street.buildings.map((b) => ({
      id: b.id,
      ring: b.ring,
      height_m: b.height_m,
      // The ground the SOLVER used, not the one the panel quotes. Depth, and
      // therefore `over`, is measured against the DEM; the city survey is up
      // to 2.1 m away from it on this block. A box standing on the city
      // ground inside a water plane built from the DEM would contradict its
      // own colour. The panel keeps the surveyed figure, which is the better
      // number for a reader; the drawing keeps one frame throughout.
      base_m: b.dem_ground_m,
      over: streetFrame !== null && b.over_floor_series[streetFrame] !== undefined
        ? b.over_floor_series[streetFrame]
        : (b.over_floor_m ?? -1),
      selected: b.id === streetSel,
    }));
  }, [street, streetFrame, streetSel]);

  /**
   * The block the street view draws, as a scene rather than a flood map.
   *
   * Three numbers the map needs and nothing else computes:
   *
   * `datum` — the lowest ground on the block. The terrain decoder returns
   * absolute elevation, so a 1.7 km block of ground at 165-178 m renders as
   * a slab floating 165 m over the basemap (and, at the 5x exaggeration the
   * valley views use, 825 m over it, which is what made this view a blue
   * mountain). Subtracting the datum puts the ground back on the imagery.
   *
   * `waterM` — the water SURFACE for the hour on screen, in the same frame.
   * Measured on Onion Creek it is flat to 0.18 m across the block, which is
   * why one plane is an honest drawing of it and a per-cell wash is not.
   * It is built from `dem_ground_m + floor + over_floor_series[i]` rather
   * than the response's own `water_surface_m`, which is the city survey's
   * ground plus the DEM's depth and so disagrees with the colours by up to
   * half a metre here and 2.1 m at worst.
   *
   * `ceilingM` — the highest thing in the scene, for the vertical scale note.
   */
  const streetScene = useMemo(() => {
    if (!street?.ok || !street.buildings.length) return null;
    const bs = street.buildings;
    const grounds = bs.map((b) => b.dem_ground_m);
    const floor = street.floor_above_ground_m ?? 0.17;
    const surfaces = bs
      .map((b) => {
        const over = streetFrame !== null
          ? b.over_floor_series[streetFrame] : b.over_floor_m;
        return over !== null && over !== undefined && over > 0
          ? b.dem_ground_m + floor + over : null;
      })
      .filter((x): x is number => x !== null)
      .sort((a, b) => a - b);
    const waterM = surfaces.length
      ? surfaces[Math.floor(surfaces.length / 2)] : null;
    const ceilingM = Math.max(
      ...bs.map((b) => b.dem_ground_m + Math.max(b.height_m ?? 4, 2.5)));
    /* The plane's own footprint. A bare quad over the block floods the hill
       as well as the creek, because a terrain mesh carrying a transparent
       texture writes no depth and so occludes nothing. The solver already
       ships the shape of the water for every hour -- the depth PNG, which is
       transparent wherever the ground is dry -- so the plane is that image,
       hung at the water's elevation rather than painted on the ground. */
    const frame = streetFrame !== null ? street.frames[streetFrame] : null;
    const waterPng = frame ? frame.depth_png : street.depth_png;
    const surfacePng = frame ? frame.surface_png : street.surface_png;

    /* What the camera should frame: a STREET, which is what the panel is
       about. The solved box is 1.7 km of catchment either side of the houses
       so the water has somewhere to come from, and framing all of it puts a
       nine-metre house at four pixels -- which is how this view came to show
       a sheet of water with no buildings on it. Framing the footprints'
       extent is no better: they are scattered across the whole box, so their
       bounding box IS the box.

       So frame the house the reader is looking at, with its neighbours
       around it: the selected one, or failing that the worst-affected one,
       which is the row at the top of the list. Everything else is a pan
       away, and the overview is one click on "Whole valley". */
    const centre = (ring: [number, number][]) => {
      let sx = 0, sy = 0;
      for (const [lon, lat] of ring) { sx += lon; sy += lat; }
      return [sx / ring.length, sy / ring.length] as [number, number];
    };
    /* The anchor is the densest place, not the worst-hit one. The deepest
       house on this block turns out to stand alone in woodland, and framing
       it gave a screen of trees with one pink roof in it -- an answer to a
       question nobody asked. A street is where the houses are, so count
       neighbours and take the winner; ties go to the one with more water in
       it, which is the row the panel puts at the top. */
    const cs = bs.map((x) => centre(x.ring));
    const R_LAT = 0.0016;                       // ~175 m
    const R_LON = R_LAT / Math.cos((cs[0][1] * Math.PI) / 180);
    let best = 0, bestN = -1, bestOver = -99;
    for (let i = 0; i < bs.length; i++) {
      let n = 0;
      for (let j = 0; j < bs.length; j++) {
        if (Math.abs(cs[j][0] - cs[i][0]) <= R_LON
          && Math.abs(cs[j][1] - cs[i][1]) <= R_LAT) n++;
      }
      const over = bs[i].over_floor_m ?? -99;
      if (n > bestN || (n === bestN && over > bestOver)) {
        best = i; bestN = n; bestOver = over;
      }
    }
    /* Clicking a house SELECTS it. It used to re-frame on it too, because
       the selection fed this box and this box is the camera -- so every
       click threw away whatever the reader had panned to and flew somewhere
       else, and clicking the worst-hit house (which stands alone in
       woodland here) left a screen of trees. The mock's instruction is
       "click a house", not "go to a house". */
    const goIdx = bs.findIndex((x) => x.id === streetGoTo);
    const [cx, cy] = cs[goIdx >= 0 ? goIdx : best];
    // About 175 m across, then widened to hold whatever neighbours fall
    // inside it -- a house on a big lot should not be framed tighter than a
    // terrace. This is the single biggest lever on whether a house reads as
    // a building or as four pixels of roof.
    const RADIUS_DEG_LAT = 0.0008;
    const RADIUS_DEG_LON = RADIUS_DEG_LAT / Math.cos((cy * Math.PI) / 180);
    let w = cx - RADIUS_DEG_LON, e = cx + RADIUS_DEG_LON;
    let so = cy - RADIUS_DEG_LAT, no = cy + RADIUS_DEG_LAT;
    for (const bb of bs) {
      const [bx, by] = centre(bb.ring);
      if (bx < w || bx > e || by < so || by > no) continue;
      for (const [lon, lat] of bb.ring) {
        if (lon < w) w = lon; if (lon > e) e = lon;
        if (lat < so) so = lat; if (lat > no) no = lat;
      }
    }
    const padLon = Math.max((e - w) * 0.16, 0.0004);
    const padLat = Math.max((no - so) * 0.16, 0.0004);

    /* The datum is the ground under the houses in FRAME, not the lowest
       ground anywhere in the block. The imagery is a flat sheet at z = 0
       and the terrain stands off it; centring the datum on what the camera
       is looking at keeps a house and its own driveway in register. */
    const inFrame = bs.filter((x) => {
      const [bx, by] = centre(x.ring);
      return bx >= w - padLon && bx <= e + padLon
        && by >= so - padLat && by <= no + padLat;
    });
    const near = (inFrame.length ? inFrame : bs).map((x) => x.dem_ground_m);
    const datum = near.reduce((a, x) => a + x, 0) / near.length;

    return {
      datum,
      waterM,
      waterPng,
      surfacePng,
      wetCount: surfaces.length,
      groundLoM: Math.min(...grounds),
      groundHiM: Math.max(...grounds),
      ceilingM,
      blockBox: { west: w - padLon, south: so - padLat,
                  east: e + padLon, north: no + padLat },
      floorM: floor,
    };
  }, [street, streetFrame, streetGoTo]);

  /** The solved run, hour by hour. Nine instantaneous depth fields the
   *  solver computed on its way to the peak and used to throw away. */
  const runFrames: Frame[] = useMemo(
    () => (study?.frames ?? []) as Frame[], [study]);

  /* Both the scrubber and the click readout want the bottom-left corner, and
     the scrubber is painted after it, so a click on the Forecast tab put the
     answer underneath the transport control -- the map looked like it had
     ignored the click. Whoever is second moves up. */
  const scrubberOn = runFrames.length > 1 && section === "forecast"
    && !streetOn;

  /** The two fields the Plan tab's "Before" / "With plan" pair swaps
   *  between: the do-nothing bed and the best solved option, both from the
   *  plan library's own grid. Null until a plan has been solved. */
  const planFields = useMemo(() => {
    const d = kitPlan as {
      bounds?: { west: number; south: number; east: number; north: number };
      depth_png?: string; best_option_depth_png?: string | null;
    } | null;
    if (!d?.bounds || !d.depth_png) return null;
    return { bounds: d.bounds, base_png: d.depth_png,
             with_png: d.best_option_depth_png ?? null };
  }, [kitPlan]);

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
      // Before a trip is solved this returned null, so the Roads tab drew a
      // bare basemap -- and the evacuation answer below it, which is computed
      // from the shipped study's flood, had nothing on screen to stand on.
      // Fall back to that same study field, so the panel and the map are
      // talking about one flood.
      if (trip) {
        return { bounds: trip.bounds, depth_png: trip.depth_png,
                 terrain_png: trip.terrain_png };
      }
      return study ? { bounds: study.bounds, depth_png: study.depth_png,
                       terrain_png: study.terrain_png } : null;
    }
    if (tab === "street" && street?.ok) {
      return { bounds: street.bounds,
               depth_png: (streetFrame !== null && street.frames[streetFrame])
                 ? street.frames[streetFrame].depth_png
                 : street.depth_png,
               terrain_png: street.terrain_png };
    }
    /* Before / With plan. These two buttons wrote `showPlan` and nothing
       read it, so both states painted the baseline and the pair was a
       decoration over a map that never changed. The solved plan library
       holds a bed per option; the service now also returns the field of the
       option the table calls "with plan", so the pair on the map is the pair
       in the table -- and both come from the library's own 60 m grid rather
       than one of them from the 40 m bank search. */
    if (section === "plan" && planFields) {
      return {
        bounds: planFields.bounds,
        depth_png: showPlan && planFields.with_png
          ? planFields.with_png : planFields.base_png,
        terrain_png: study?.terrain_png ?? "",
      };
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
      analysis, study, planFields, showPlan]);

  /** The same terrain with no water painted on it, for when the depth layer
   *  is switched off. The terrain layer needs a texture either way, so the
   *  flood is removed by handing it a transparent one rather than by
   *  unmounting the ground. */
  const dryField = useMemo(
    () => (mapField ? { ...mapField, depth_png: BLANK_PNG } : null),
    [mapField],
  );

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
    // The discharge is the scenario's own, not a base for the storm presets
    // to multiply. Without this flag the service put 3,823 m3/s through its
    // `rain_mm_h / 20` multiple and ran 9,557, while the heading above the
    // map said 3,823.
    river_q_is_design: scenario.river_q_m3s !== null,
    rain_mm_h: study?.storm?.rain_mm_h ?? 50,
    storm_hours: study?.storm?.storm_hours ?? 6,
  }), [scenario, study]);

  // A changed scenario is a changed flood, so everything downstream of it
  // goes with it. Done in the handler rather than in an effect watching the
  // id: an effect would be a second render pass to undo what the first one
  // just drew, and there is no external system here to synchronise with.
  const chooseScenario = useCallback((id: string) => {
    setScenarioId(id);
    setPoint(null);
    autoSites.current = false;
    solvedFor.current = null;
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
          // Twelve hours, the same run the plan solver uses.
          //
          // It was eighteen, which is a correctness problem before it is a
          // speed one: the banks were found in one flood and the plan then
          // judged in another. Measured side by side on Onion Creek under
          // the 2013 peak, twelve and eighteen hours give the identical
          // 6.35 km2 and the identical six banks -- the inflow runs for six
          // hours and the extent has stopped growing long before twelve.
          // What differs is the depth still building in the channel, which
          // is not what a bank search reads.
          //
          // It also takes 72 s instead of over 300, which is the limit a
          // browser's fetch imposes on this page whatever it asks for.
          ...kitBox, name: analysis?.name ?? "Area", dx: 40,
          ...forcing, run_hours: 12, max_sites: 6, shortfall_m: 2.0,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      // "Still solving" is not a failure and not an answer. The service runs
      // one solve per distinct question and lets every caller wait on it, so
      // asking again joins the same run rather than starting a second —
      // which is what made a retry the wrong fix on its own.
      if (d.pending) return "pending" as const;
      if (!d.ok) throw new Error(d.reason ?? "no sites here");
      setKit(d);
    } catch (e) {
      setKitErr(e instanceof Error ? e.message : "could not find sites");
    } finally { setKitBusy(false); clearInterval(tick); }
    return "done" as const;
  }, [kitBox, analysis, forcing]);

  /** Ask until it answers. Each call waits on the one run the service is
   *  already doing; the loop exists because a browser's fetch stops
   *  listening after five minutes and the solve may legitimately be longer
   *  than that on a cold cache. */
  const findSitesUntilDone = useCallback(async () => {
    for (let i = 0; i < 6; i++) {
      if (await findSites() !== "pending") return;
    }
  }, [findSites]);

  // The Plan tab finds its own sites, for the scenario it is actually
  // judging against.
  //
  // It used to open on a button and a paragraph explaining what the button
  // would do. But the sites are not a choice the planner makes -- they come
  // out of the terrain and the solved flood, the same way the flood field
  // does -- so asking for a click before showing them put a turnstile in
  // front of the one thing the tab exists to show.
  //
  // It waits for `peaks`. Without that it fired on the default scenario --
  // the study's design rainstorm -- and when the gauge record arrived a
  // moment later the heading flipped to "31 Oct 2013 flood" over a site set
  // and a map that were still the rainstorm's. The panel named one flood and
  // showed another, which is the exact failure the scenario control was
  // added to fix. Keyed on the scenario rather than on a once-only flag, so
  // a change of flood re-finds the banks it changes.
  useEffect(() => {
    if (section !== "plan" || !kitBox || !peaks) return;
    if (kitBusy || solvedFor.current === scenario.id) return;
    if (kitErr && autoSites.current) return;
    solvedFor.current = scenario.id;
    autoSites.current = true;
    void findSitesUntilDone();
  }, [section, kitBox, peaks, scenario.id, kitBusy, kitErr,
      findSitesUntilDone]);

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
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      // Same shared run as the site search: one solve per distinct plan,
      // and asking again waits on it rather than starting another.
      if (d.pending) return "pending" as const;
      if (!d.ok) throw new Error(d.reason ?? `HTTP ${r.status}`);
      setKitPlan(d);
    } catch (e) {
      setKitErr(e instanceof Error ? e.message : "could not solve this plan");
    } finally { setKitPlanBusy(false); }
    return "done" as const;
  }, [kitBox, kit, placedPieces, kitBudget, analysis, forcing]);

  /* Run the event under each controller.
     The structures are whatever is on the map, so this sends the SAME site
     payload `/plan` sends. A schedule only exists where a gate or a pump was
     placed: a levee has no setting, and the service refuses rather than
     inventing a basin to operate. */
  const runOperate = useCallback(async (budget: number) => {
    if (!kitBox || !kit?.ok || !placedPieces.length) return;
    setOpBusy(true); setOp(null); setOpSec(0); setOpFrame(null);
    const tick = setInterval(() => setOpSec((n) => n + 1), 1000);
    try {
      const chosen = placedPieces.map((p) => ({
        site: kit.sites[p.site], piece: pieceOf(p.piece)! }));
      const payload = {
          ...kitBox, name: analysis?.name ?? "Area", dx: 60,
          ...forcing, run_hours: 12,
          budget_pump_hours: budget,
          horizon: 10, control_minutes: 30,
          sites: chosen.map((c) => ({
            lat: c.site.lat, lon: c.site.lon, width_m: c.site.width_m,
            outline: c.site.outline, line: c.site.line,
            kind: c.piece.kind,
            cost_model: c.piece.cost_model,
            unit_cost: c.piece.unit,
            height_m: c.piece.intensity,
          })),
      };
      const r = await fetch("/api/operate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      let d = (await r.json()) as OperateReply & { detail?: string };
      /* Join the run rather than making the reader click again.
         A real event on real terrain takes minutes, and the service answers
         `pending` while it finishes -- the work is keyed, so asking again
         joins the same run instead of starting a second one. Without this
         the panel simply reported "still running" and stopped. */
      for (let i = 0; i < 12 && d?.pending; i++) {
        await new Promise((ok) => setTimeout(ok, 15_000));
        const again = await fetch("/api/operate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        d = (await again.json()) as OperateReply & { detail?: string };
      }
      setOp(d.ok === undefined && d.detail
        ? { ok: false, reason: d.detail } : d);
      setOpPolicy(d.results?.[0]?.policy ?? null);
    } catch (e) {
      setOp({ ok: false, reason: String((e as Error).message).slice(0, 200) });
    } finally {
      clearInterval(tick);
      setOpBusy(false);
    }
  }, [kitBox, kit, placedPieces, analysis, forcing]);


  /** As `findSitesUntilDone`, for the plan library. */
  const solveKitPlanUntilDone = useCallback(async () => {
    for (let i = 0; i < 8; i++) {
      if (await solveKitPlan() !== "pending") return;
    }
  }, [solveKitPlan]);

  const kitImpact = useMemo(() => {
    const d = kitPlan as {
      options?: { id: string; heights: number[]; area_flooded_km2: number;
                  core_reduction_pct: number }[];
      band_pct?: number; reduction_pct?: number;
      band_attainable?: boolean; band_note?: string;
      protect_basis?: string; protect_km2?: number; dx_m?: number;
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
      // A percentage is meaningless without the area it is a percentage of,
      // and this one used to be averaged over an undisclosed circle.
      protectBasis: d.protect_basis ?? null,
      protectKm2: typeof d.protect_km2 === "number" ? d.protect_km2 : null,
      // The plan library is solved coarser than the bank search, so the
      // areas in this table and the area under the map are two grids of the
      // same flood. Carried so the panel can say which is which.
      dxM: typeof d.dx_m === "number" ? d.dx_m : null,
      sitesDxM: kit?.dx_m ?? null,
    };
  }, [kitPlan, kit]);

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
      ? wallAt(secRes.profile, secStationIdx, secCrestM, secBankLen || 500,
               EARTH_FILL.usd, "a", secBelts)
      : null),
    [secRes, secStationIdx, secCrestM, secBankLen, secBelts],
  );

  /** The box the scene is built on. The map used to be gated on the Richmond
   *  manifest, so deleting that one file rendered a grey pulse for Austin. */
  const mapBounds = useMemo(
    () => analysis?.bounds ?? study?.bounds ?? null,
    [analysis, study]);

  /**
   * Which roads this vehicle cannot cross, and how much of each is cut.
   *
   * Derived here rather than handed up from the panel. `RoadsOut` used to
   * call `onQuery` from its own pill handler, which passed the query
   * computed from the threshold being REPLACED -- so the map's road colours
   * sat one click behind the selected pill, and clicking the same pill twice
   * was what synced them. It also meant every road drew uniform white until
   * someone happened to click a pill at all. `RoadsOut` itself lives under
   * `section === "forecast"`, so that click was only ever available there.
   */
  const roadQ = useMemo(
    () => queryRoads(adHocRoads, adHocGrids, mapBounds, vehicle),
    [adHocRoads, adHocGrids, mapBounds, vehicle],
  );

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
    // Against the whole network, not against the cut list.
    //
    // `queryRoads` pushes a road into `roads` only `if (cut > 0.01)`, so that
    // array contains cut roads and nothing else -- asking whether every entry
    // in it is cut is asking whether every cut road is cut, which is true the
    // moment anything floods. The chip claims "every named road out is cut",
    // so the comparison has to be against every named road there is.
    if (!roadQ || !adHocRoads) return false;
    const namedTotal = adHocRoads.features.filter(
      (f) => (f.properties as { name?: string | null } | null)?.name).length;
    const namedCut = roadQ.roads.filter((r) => r.name).length;
    return namedTotal >= 3 && namedCut >= namedTotal;
  }, [roadQ, adHocRoads]);

  const sectionPicking =
    sectionArmed && !(sectionEnds.a && sectionEnds.b);

  const chooseSectionSite = useCallback((i: number | null) => {
    setSecSite(i);
    setSecRes(null); setSecErr(null); setSecStation(null); setSecCrest(null);
    const line = i !== null && kit?.ok ? kit.sites[i]?.line : null;
    setSectionEnds(line ? { a: line.a, b: line.b } : { a: null, b: null });
  }, [kit]);

  /** Where "getting out" starts from: the middle of the solved place. */
  const studyCentre = useMemo(() => {
    const b = analysis?.bounds ?? study?.bounds;
    const name = analysis?.name ?? study?.study?.short ?? study?.name;
    if (!b || !name) return null;
    return { lat: (b.south + b.north) / 2, lon: (b.west + b.east) / 2, name };
  }, [analysis, study]);

  /** The trip planner owns the map's route lines once it has solved one, so
   *  the two route panels cannot fight over them. */
  const tripRoutesOwned = Boolean(trip);

  /** The evacuation question's own vehicle threshold, separate from the one
   *  the Forecast tab's road list uses. */
  const [outVehicle, setOutVehicle] = useState(0.25);

  /**
   * Is the map still showing the place that shipped with the app?
   *
   * `analysis` is not the test: `AnywherePanel` falls back to the shipped
   * study (`res = ownRes ?? initial`), so it is non-null from the first
   * paint. What distinguishes an ad-hoc solve is that it is a different BOX
   * -- and that matters because `adHocGrids` is replaced by such a solve
   * while `adHocRoads` is not, so anything reading both would be scoring one
   * place's roads against another place's flood.
   */
  const showingShippedStudy = useMemo(() => {
    const a = analysis?.bounds, b = study?.bounds;
    if (!b) return false;
    if (!a) return true;
    return Math.abs(a.west - b.west) < 1e-6 && Math.abs(a.south - b.south) < 1e-6
        && Math.abs(a.east - b.east) < 1e-6 && Math.abs(a.north - b.north) < 1e-6;
  }, [analysis, study]);

  /**
   * Is the road query describing the flood the map is painting?
   *
   * `roadQ` is computed from the shipped study's network and field. That is
   * what the Forecast column's road list is about, and what the Roads tab
   * shows before a trip is solved. The Plan tab paints its scenario's flood
   * and the street view its own block, so study cut-fractions over either
   * would be one flood's colours on another flood's water.
   */
  const roadColoursMatchTheMap =
    showingShippedStudy && (tab === "anywhere" || (tab === "trip" && !trip));

  /** The Plan tab's solved plan, in the shape the route panel wants, so the
   *  route verdict can carry the conformal margin rather than only saying
   *  whether the road is open today. */
  const routePlan = useMemo(() => {
    const d = kitPlan as {
      ok?: boolean; heights?: number[]; max_height_m?: number;
      saturation_S?: number; confidence_pct?: number;
      options?: { heights: number[]; depth_u8?: string }[];
      chosen_solved?: boolean;
    } | null;
    if (!d?.ok || !d.heights || !d.options?.length) return null;
    return {
      heights: d.heights,
      max_height_m: d.max_height_m ?? Math.max(...d.heights, 1),
      saturation_S: d.saturation_S ?? 0,
      confidence_pct: d.confidence_pct ?? 90,
      options: d.options,
      chosen_solved: d.chosen_solved,
    };
  }, [kitPlan]);

  /** The grids for one solved option, for calibrating a route against it.
   *  The plan ships its own coarse ground + baseline depth in `inspect`, and
   *  each option's field is a depth byte-array over that same grid. */
  const routePlanGrids = useMemo(() => {
    const d = kitPlan as { ok?: boolean; inspect?: unknown } | null;
    return d?.ok && d.inspect
      ? decodeGrids(d.inspect as Parameters<typeof decodeGrids>[0])
      : null;
  }, [kitPlan]);

  const routePlanGridFor = useCallback((i: number) => {
    const u8 = routePlan?.options?.[i]?.depth_u8;
    return routePlanGrids && u8 ? withDepth(routePlanGrids, u8) : null;
  }, [routePlan, routePlanGrids]);

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
      // Placing a levee by hand consumes the click; otherwise it falls
      // through to the depth query. It used to RETURN here when not placing,
      // which made `setPoint` unreachable on the tab the portal opens on --
      // so clicking the map on Forecast did nothing at all, and the sampled
      // depth, ground elevation and wet flag were computed nowhere.
      if (tab === "anywhere" && placing) {
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

  // The same gauge the headline speaks for. See `pickGauge`.
  const hydroGauge = useMemo(() => {
    const g = pickGauge(gaugeFeed);
    return g && g.series.length > 2 ? g : null;
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
                // The tabs set `section` directly rather than going through
                // `setTab`, so the depth readout has to be cleared here too:
                // a number sampled from the study's peak field must not
                // follow the reader onto a tab drawing a different flood.
                onClick={() => { setSection(k); setPoint(null); }}
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
              {(["kit", "section", "operate"] as const).map((k) => (
                <button key={k} className="pill whitespace-nowrap"
                        data-on={planMode === k}
                        onClick={() => { setPlanMode(k);
                                         if (k === "section") setSectionArmed(true); }}>
                  {k === "kit" ? "Build a plan"
                    : k === "section" ? "Cross-section" : "Operate"}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Can people get out of THIS place, by road.
            A different question from the trip planner above it, which routes
            between two points someone names. This one asks the one that gets
            asked on the night -- is there any way out of here at all -- and
            answers it from the shipped study's own network and flood, so it
            needs no solve.

            It was unreachable. `RoutePanel` was mounted inside
            `AnywherePanel` under `section === "plan"`, while `AnywherePanel`
            itself renders only when `section === "forecast"` -- conditions
            that cannot both hold, so the whole panel, and with it the only
            route-level statement of the conformal margin, had been dead since
            the nav became three tabs. */}
        {/* Only while the shipped study is what is loaded. An ad-hoc solve
            in Forecast replaces `adHocGrids` but not `adHocRoads`, so the
            panel would score this place's roads against another place's
            flood and cheerfully report that nothing is cut off. */}
        {tab === "trip" && !tripRoutesOwned && showingShippedStudy
          && adHocRoads && adHocGrids && mapBounds && (
          <div className="border-t border-line">
            <RoutePanel
              roads={adHocRoads}
              grids={adHocGrids}
              bounds={mapBounds}
              from={studyCentre}
              to={null}
              // Its own threshold. Sharing Planner's `vehicle` meant moving
              // this row silently retuned the Forecast tab's "Roads out"
              // list, which is a different question about a different
              // network.
              vehicle={outVehicle}
              onVehicle={setOutVehicle}
              // The plan solved in the Plan tab, where there is one. This is
              // what carries the route-level conformal margin -- without it
              // the panel answers "is there a way out" and says nothing
              // about what a plan would change, which is most of why it was
              // worth restoring.
              plan={routePlan}
              planGridFor={routePlanGridFor}
              onRoutes={(r) => setRouteLines(
                r ? { ...(r as object), forTab: "trip" } as typeof routeLines : null)}
            />
          </div>
        )}

        {tab === "trip" && (
          <TripPanel
            onResult={setTrip}
            onRoutes={(r) => setRouteLines(
              r ? { ...(r as object), forTab: "trip" } as typeof routeLines : null)}
          />
        )}

        {/* `section` and `streetOn` are independent -- the street is a zoom,
            not a tab -- so both columns rendered at once and the street panel
            sat below three screens of forecast. The street REPLACES the live
            column while it is open, the way the design has it, and the
            Forecast tab is still what you return to when you zoom back out. */}
        {section === "forecast" && !streetOn && (
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
              reach={analysis?.name ?? study?.name ?? ""}
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
              />
            )}
          </>
        )}

        {tab === "operate" && (
          <OperatePanel
            /* A schedule needs something with a setting on it. */
            canRun={Boolean(kit?.ok) && placedPieces.some((p) => {
              const k = pieceOf(p.piece)?.kind;
              return k === "gate" || k === "pump";
            })}
            busy={opBusy}
            elapsed={opSec}
            reply={op}
            onRun={runOperate}
            onPick={(k) => { setOpPolicy(k); setOpFrame(null); }}
            picked={opPolicy}
            frame={opFrame}
            onFrame={setOpFrame}
          />
        )}

        {tab === "kit" && (
          <LeveeKit
            feed={kit}
            busy={kitBusy}
            elapsed={kitSec}
            err={kitErr}
            onFindSites={findSitesUntilDone}
            // Both: a solved place AND a settled scenario. See the kit's own
            // comment -- a manual search before the flood is chosen answers
            // the wrong question and then relabels it.
            canFind={Boolean(kitBox) && Boolean(peaks)}
            placed={placedPieces}
            onPlaced={setPlacedPieces}
            budgetM={kitBudget}
            onBudget={setKitBudget}
            armed={armed}
            onArm={setArmed}
            impact={kitImpact}
            planBusy={kitPlanBusy}
            onPlan={solveKitPlanUntilDone}
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
            onSelect={(id) => { setStreetSel(id); setStreetGoTo(id); }}
            onSolve={solveStreetUntilDone}
            canSolve={Boolean(streetCentre)}
            place={analysis?.name ?? study?.name ?? null}
            onBack={() => setTab("anywhere")}
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
            sitesErr={kitErr}
            onFindSites={findSitesUntilDone}
            canFind={Boolean(kitBox) && Boolean(peaks)}
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
            roads={tab === "trip" ? (trip?.roads ?? adHocRoads) : adHocRoads}
            focusBounds={tab === "street" && street?.ok
              ? (streetScene?.blockBox ?? street.bounds) : null}
            pins={tab === "trip" && trip
              ? [{ lon: trip.origin.lon, lat: trip.origin.lat, label: "from" },
                 { lon: trip.destination.lon, lat: trip.destination.lat, label: "to" }]
              : null}
            onPick={onPick}
            pin={point ? { lon: point.lon, lat: point.lat } : null}
            // Only where the query and the painted flood are the same
            // flood. Making `roadQ` always-derived fixed the one-click lag
            // and introduced this: the query is always the shipped study's,
            // so the Plan tab coloured roads by the design storm while the
            // water under them was the chosen scenario, and the Roads tab
            // coloured a trip's own network by study cut-fractions looked up
            // on shared OSM way ids. Where they disagree, no colour.
            cutFraction={roadColoursMatchTheMap ? (roadQ?.cutFraction ?? null) : null}
            cutSignature={roadColoursMatchTheMap && roadQ
              ? `${roadQ.threshold}:${roadQ.cutKm.toFixed(2)}` : "none"}
            placed={tab === "anywhere" ? placed : []}
            placedHeights={tab === "anywhere" ? adHocHeights : null}
            routes={routeLines?.forTab === tab ? routeLines : null}
            section={tab === "section" ? sectionOverlay : null}
            buildings={tab === "street" ? streetBuildings : null}
            spin={tab === "street" ? streetSpin : 0}
            scene={tab === "street"
              ? (layers.depth ? streetScene
                : streetScene && { ...streetScene, waterM: null })
              : null}
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
            onBuilding={(id) => { setStreetSel(id); setStreetGoTo(null); }}
            gauges={layers.gauges ? labelledGauges : null}
            radar={layers.radar ? liveBounds : null}
            place={tab === "anywhere" ? place : null}
              /* The street view draws its water as a SURFACE, not as paint
                 on the ground, so the terrain there carries no depth
                 texture: the imagery reads through and the plane is the
                 water. The depth switch still governs whether there is any
                 water in the scene -- it just governs the plane now. */
              analysis={layers.depth && tab !== "street" ? mapField : dryField}
            focus={focus}
            // No solid water body in the street view: the grids are the
            // study's 40 m field over the whole study box, and drawn under a
            // camera zoomed to one block it covers everything in sight. The
            // block's own 10 m field arrives as the depth overlay instead.
            // `layers.depth` was declared, drawn as a switch, and read by
            // nothing -- the toggle moved and the water stayed. It gates the
            // solved field here, which is what its label says it does.
            grids={!layers.depth ? null
              : tab === "trip" ? tripGrids
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
        {depthVmax !== null && layers.depth && tab !== "street" && (
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

        {/* What is under the point you clicked.
            `sampleAt` already returned the depth, the ground elevation and
            whether the cell is wet, and all of it was thrown away except the
            two numbers needed to draw a dot. A map you can click and that
            answers nothing is worse than a map you cannot click. */}
        {/* Only where the sampled field is the one on screen. `onPick`
            samples `adHocGrids` -- the shipped study's peak -- and the Plan
            tab draws its scenario's flood instead, so a reading there would
            be a number from a different flood sitting over this one. */}
        {point && (tab === "anywhere" || (tab === "trip" && !trip)) && (
          <div data-readout
               className={`absolute left-4 z-10 w-[188px] rounded-[10px]
                          border border-line bg-bg/92 px-3 py-2.5 backdrop-blur
                          shadow-[0_2px_8px_rgba(16,20,28,0.12)]
                          ${scrubberOn ? "bottom-[128px]" : "bottom-4"}`}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="eyebrow">Here</span>
              <button onClick={() => setPoint(null)}
                      className="text-[11px] text-ink-faint hover:text-ink">clear</button>
            </div>
            <p className="tnum mt-0.5 text-[17px] font-semibold leading-tight text-ink">
              {point.flooded ? `${point.depth_m.toFixed(2)} m` : "dry"}
            </p>
            <p className="anno leading-snug">
              {point.flooded ? "of water on the ground" : "no water in this cell"}
            </p>
            <p className="anno mt-1 leading-snug">
              ground {point.elev_m.toFixed(1)} m · {point.lat.toFixed(4)},{" "}
              {point.lon.toFixed(4)}
            </p>
          </div>
        )}

        {/* Turn the block. The scene is a model seen from one fixed angle,
            and the house a reader wants is sometimes behind another one. */}
        {tab === "street" && streetBuildings?.length ? (
          <div className="absolute right-4 top-3 z-10 flex gap-1.5">
            {[[-45, "Turn the block left", "\u21ba"],
              [45, "Turn the block right", "\u21bb"]].map(([d, label, glyph]) => (
              <button
                key={label as string}
                onClick={() => setStreetSpin((x) => x + (d as number))}
                aria-label={label as string}
                title={label as string}
                className="grid h-8 w-8 place-items-center rounded-full border
                           border-ink-faint/40 bg-bg/85 text-[15px] text-ink
                           backdrop-blur transition-colors hover:bg-bg"
              >
                {glyph as string}
              </button>
            ))}
          </div>
        ) : null}

        {/* What the colours on a street mean.
            Three swatches, because the street view is the one place the map
            is not a depth ramp: a building is salmon or it is bone, and
            which it is depends on a floor height the reader cannot see.
            Spelling that out on the map is cheaper than making them find it
            in the panel. */}
        {tab === "street" && streetBuildings?.length ? (
          <div className="absolute bottom-4 left-4 z-10 rounded-[10px] border border-line
                          bg-bg/92 px-3 py-2.5 backdrop-blur
                          shadow-[0_2px_8px_rgba(16,20,28,0.12)]">
            {/* Four, not three. The near-black selected state was on the map
                and missing from the key, and the old third swatch was
                #2563a8 -- a blue that matched no pixel on screen, because
                the water is drawn from the depth ramp and not from a flat
                colour. The surface swatch is written from the same ramp. */}
            {[["#f7f5f2", "Floor above water"],
              ["#e8887a", "Water over floor"],
              ["#18181b", "The one you clicked"],
              ["linear-gradient(90deg,#60c8ff,#1f8fd6,#04247f)",
               "Water surface, deeper darker"]].map(([c, label]) => (
              <div key={label} className="flex items-center gap-2 py-[3px]">
                <span className="h-[11px] w-[11px] shrink-0 rounded-[2px]
                                 border border-ink/15"
                      style={{ background: c }} />
                <span className="text-[12px] text-ink">{label}</span>
              </div>
            ))}
            {/* The mock states its vertical scale and so does this. Heights
                are exaggerated to make a waterline on a wall visible at a
                street's width; the plan is not. */}
            <p className="anno mt-1.5 leading-snug">
              heights ×{STREET_VE} · ground, houses and water all on that
              scale
            </p>
            <p className="anno leading-snug">
              floor assumed {((street?.ok && street.floor_above_ground_m) || 0.17)
                .toFixed(2)} m above ground
              {streetScene?.groundLoM != null && streetScene?.groundHiM != null
                ? ` · block falls ${(streetScene.groundHiM - streetScene.groundLoM)
                    .toFixed(1)} m` : ""}
            </p>
          </div>
        ) : null}

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
                title={v && !planFields?.with_png
                  ? "No solved option beat doing nothing, so there is no second field to show"
                  : undefined}
              >
                {v ? "With plan" : "Before"}
              </button>
            ))}
          </div>
        )}
        {/* The honest case. The best solved option can BE the do-nothing bed
            -- on this reach two 2 m levees under the design storm made the
            flooded area slightly worse -- and then the two pills paint the
            same water. Saying so is the difference between a null result and
            a broken control. */}
        {section === "plan" && kitImpact && showPlan && !planFields?.with_png && (
          <div className="absolute left-1/2 top-12 z-10 -translate-x-1/2 rounded-md
                          border border-line bg-bg/92 px-2.5 py-1 backdrop-blur">
            <span className="anno">no option beat doing nothing — same water</span>
          </div>
        )}

        {/* The scrubber, floating on the map rather than stacked under it.
            It is on the map because what it moves is on the map: nine
            instantaneous depth fields across the solved run. The gauge
            hydrograph is NOT here, and that is the point -- the gauge runs
            on a wall clock and the model on its own, and one play button
            over two clocks would be a control over nothing. */}
        {scrubberOn && (
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
