"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { WebMercatorViewport } from "@deck.gl/core";
import type { Layer } from "@deck.gl/core";
import { TerrainLayer, TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer, ColumnLayer, GeoJsonLayer, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { PathStyleExtension } from "@deck.gl/extensions";
import type { InspectGrids } from "@/lib/inspect";
import { waterColumns } from "@/lib/watermesh";
import type { Box } from "@/lib/types";
import { ELEVATION_DECODER } from "@/lib/data";

const IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export interface MeasureMarker {
  id: number;
  lon: number;
  lat: number;
  selected: boolean;
  worsens: boolean;
}

/** The zoom that makes a box COVER its pane, not fit inside it.
 *
 * `fitBounds` fits, which letterboxes: a wide box in a tall pane leaves the
 * page showing above and below it, and the study box then reads as a
 * rectangle of model floating on the world rather than as the map. Covering
 * means the shorter axis overflows and the imagery reaches every edge.
 */
/** Vertical exaggeration for the street scene, and nothing else.
 *
 * At true scale a six-metre house against two metres of water is six pixels
 * against two, which is geometrically honest and visually nothing: the whole
 * point of this view -- how far up the wall the water came -- disappears
 * into the thickness of a line. Four is enough to read and small enough that
 * nobody mistakes the block for a mountain, and the legend states it, which
 * is the trade the design mock makes explicitly with its "1 m = 18 px".
 *
 * It multiplies HEIGHTS, never slopes that carry meaning on their own: the
 * water surface and the ground go through the same transform, so the
 * shoreline stays where the two really cross.
 */
export const STREET_VE = 4;

function coverZoom(
  box: { west: number; south: number; east: number; north: number },
  size: { w: number; h: number } | null,
  fallback = 11.6,
): number {
  if (!size || size.w < 2 || size.h < 2) return fallback;
  const vp = new WebMercatorViewport({ width: size.w, height: size.h });
  const fit = vp.fitBounds([[box.west, box.south], [box.east, box.north]],
                           { padding: 0 });
  // Measure how big the box actually is at the fitted zoom, then zoom in by
  // whichever axis still has slack.
  const at = new WebMercatorViewport({
    width: size.w, height: size.h,
    longitude: fit.longitude, latitude: fit.latitude, zoom: fit.zoom,
  });
  const a = at.project([box.west, box.south]);
  const b2 = at.project([box.east, box.north]);
  const pw = Math.abs(b2[0] - a[0]);
  const ph = Math.abs(b2[1] - a[1]);
  if (pw < 1 || ph < 1) return fit.zoom;
  const grow = Math.max(size.w / pw, size.h / ph);
  return fit.zoom + Math.log2(Math.max(grow, 1));
}

export function PlannerMap({
  bounds,
  roads,
  onPick,
  pin,
  cutFraction,
  cutSignature,
  analysis,
  observed,
  placed,
  placedHeights,
  pins,
  routes,
  section,
  gauges,
  radar,
  buildings,
  onBuilding,
  scene = null,
  sites,
  onSite,
  focusBounds,
  place,
  focus,
  spin = 0,
  grids,
  heights,
  solid = true,
}: {
  /** The study's own box. This was `manifest`, the Richmond archive, which
   *  made every other place a special case layered over a valley in NSW. */
  bounds: Box;
  roads: GeoJSON.FeatureCollection | null;
  onPick?: (lon: number, lat: number) => void;
  pin?: { lon: number; lat: number } | null;
  cutFraction?: Map<number, number> | null;
  /** Changes whenever the cut query changes; deck.gl needs a primitive. */
  cutSignature?: string;
  /** An on-demand result for an arbitrary place, drawn instead of the archive. */
  analysis?: {
    bounds: { west: number; south: number; east: number; north: number };
    depth_png: string;
    terrain_png: string;
  } | null;
  /** What the satellite last saw, drawn over the model's own account. */
  observed?: { observed_png?: string; bounds?: number[] } | null;
  /** Levees the user has dropped on an ad-hoc analysis. */
  placed?: { lat: number; lon: number }[];
  /** Height built at each placed site in the option being drawn, so a site
   *  that is part of the chosen plan does not look like one that is not. */
  placedHeights?: number[] | null;
  /** Labelled points: the two ends of a trip, which the route runs between. */
  pins?: { lon: number; lat: number; label: string }[] | null;
  /** The way out, and the way out that survives being wrong about the depth. */
  routes?: {
    quickest: { coords: [number, number][]; minutes: number } | null;
    robust: { coords: [number, number][]; minutes: number } | null;
    dest: { lat: number; lon: number } | null;
  } | null;
  /** A line cut across the floodplain, and the stretch of it a levee covers.
   *  `crest` is given as fractions along the line rather than coordinates: the
   *  panel works in distance along the section, and converting once here keeps
   *  one definition of where the wall is. */
  section?: {
    a: { lat: number; lon: number };
    b: { lat: number; lon: number };
    /** Where the wall sits, as a fraction along the line. A section is cut
     *  ACROSS a bank, so the wall is a point on it and not a stretch of it. */
    wallAt: number | null;
    verdict: string;
  } | null;
  /** Live river gauges, drawn as readings rather than as dots.
   *
   *  A gauge is the one thing on this map that is an instrument, so it is
   *  labelled with its value and its trend instead of being left as a symbol
   *  the reader has to click. */
  gauges?: {
    site: string; name: string; lat: number; lon: number;
    stage_m: number | null; trend_cm_h: number | null;
    /** Whether this one gets its reading drawn beside it. */
    labelled?: boolean;
  }[] | null;
  /** NOAA base reflectivity over the analysed box, if the user asked for it. */
  radar?: { west: number; south: number; east: number; north: number } | null;
  /** The street view's buildings, extruded and coloured by whether the water
   *  got over the floor rather than over the ground outside it. */
  buildings?: {
    id: string;
    ring: [number, number][];
    height_m: number | null;
    /** The solver's ground under this footprint, metres. The box stands on
     *  it, so that it and the water plane are in one frame. */
    base_m: number;
    over: number;                      // metres over the floor; <= 0 is dry
    selected: boolean;
  }[] | null;
  onBuilding?: (id: string) => void;
  /** The street block as a scene: the datum the whole thing is drawn
   *  relative to, and the water surface for the hour on screen. Null
   *  everywhere but the street view. */
  scene?: {
    datum: number;
    waterM: number | null;
    /** The solved water for the hour on screen, as an image: transparent
     *  where the ground is dry, so the sheet has the flood's own shape. */
    waterPng: string;
    /** The water's SURFACE as terrain-RGB: the height the sheet hangs at,
     *  cell by cell, which on a block that falls thirteen metres is not one
     *  level. */
    surfacePng: string;
    groundLoM: number;
    groundHiM: number;
    ceilingM: number;
    blockBox: { west: number; south: number; east: number; north: number };
    /** The assumed floor height above ground, for the probe stick. */
    floorM: number;
  } | null;
  /** A box the view should sit on, overriding the analysed one. The street
   *  view solves a block inside the study's box and the camera has to follow
   *  it; without this the scene stays where the study was and the buildings
   *  render off screen. */
  focusBounds?: { west: number; south: number; east: number; north: number } | null;
  /** Candidate levee alignments: the ground that only just fails to hold the
   *  water. Drawn as outlines because they are places to build, not things
   *  that are built. */
  sites?: {
    lat: number; lon: number; name: string | null;
    outline: [number, number][]; taken: boolean; label: string;
  }[] | null;
  onSite?: (i: number) => void;
  /** The town the user searched for, marked like the archive settlement. */
  place?: { lat: number; lon: number; name: string } | null;
  focus: "region" | "town";
  /** Degrees added to the street scene's bearing, from the two turn
   *  buttons. A model on a table is turned to be read. */
  spin?: number;
  /** Depth and ground grids, for drawing the flood as a body of water. */
  grids?: InspectGrids | null;
  /** Chosen levee heights, drawn as walls of that height. */
  heights?: number[] | null;
  /** False falls back to the flat painted flood. */
  solid?: boolean;
}) {
  const b = bounds;
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const ab = focusBounds ?? analysis?.bounds;
  // Only the street view is pitched; see the camera memo.
  /* A body of water is only a body from an angle. Seen from a flat, north-up
     camera the column layer is just a blockier copy of the painted wash, and
     an opaque one -- it hides the imagery the wash sits on. So the chip that
     asks for the water "as a body, with its real depth" tilts the scene as
     well as building it, and turning it off puts the map back flat. */
  const tilted = Boolean(buildings?.length) || Boolean(solid && grids);
  const view = useMemo(() => {
    // FLAT and north-up, except in the street view.
    //
    // The scene was pitched 48 degrees and rotated -18 for every view, which
    // turns a city into a tilted slab with the page showing round its cut
    // edges, and turns the study box into an obvious rectangle of model
    // floating over the world. A flood map is read as a map: north up, filling
    // its pane. Pitch is worth paying for in exactly one place -- the street
    // view, where the thing being read IS the height of buildings against a
    // water surface -- and nowhere else.
    const base = tilted
      ? { pitch: 48, bearing: -18 + spin, transitionDuration: 900 }
      : { pitch: 0, bearing: 0, transitionDuration: 900 };
    if (ab) {
      /* "Whole valley" / "Town" were a pair of pills that set `focus`, which
         arrived here, was typed, was listed in this memo's dependency array
         -- and was never read. Both pills produced the same camera, so the
         only thing that moved was the highlight. "Town" is one zoom step in
         on the same centre: the study box is about twelve kilometres across
         and a reader looking at a street wants the neighbourhood, not the
         catchment. The street view keeps its own fixed zoom. */
      /* Frame the block, rather than trusting a literal. 15.4 was tuned for
         a mesh that stood 800 m in the air and had to be backed away from;
         with the street scene drawn at true scale the camera can simply fit
         what it is looking at. A tilted view loses the far edge to the
         horizon, so give the pitch a little room. */
      const cover = focusBounds
        ? coverZoom(ab, size, 15.4) - (tilted ? 0.55 : 0)
        : coverZoom(ab, size);
      return {
        ...base,
        longitude: (ab.west + ab.east) / 2,
        latitude: (ab.south + ab.north) / 2,
        zoom: !focusBounds && focus === "town" ? cover + 1.3 : cover,
      };
    }
    const c = {
      ...base,
      longitude: (b.west + b.east) / 2,
      latitude: (b.south + b.north) / 2,
      zoom: 9.6,
    };
    if (!size) return c;
    return { ...c, zoom: coverZoom(b, size, 9.6) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b, size, focus, ab, focusBounds, tilted, spin]);

  const cutKey = cutFraction ? `${cutFraction.size}:${cutSignature}` : "none";
  // Primitive keys for the array props, for the same reason cutKey exists: the
  // layer memo has to rebuild when a height vector or a depth field changes,
  // and `heights` arrives with a fresh identity on every render, so depending
  // on the array itself would rebuild every layer continuously.
  const heightKey = heights?.join(",") ?? "none";
  const placedKey = placedHeights?.join(",") ?? "none";
  const routeKey = routes
    ? `${routes.quickest?.minutes ?? -1}:${routes.robust?.minutes ?? -1}:` +
      `${routes.quickest?.coords.length ?? 0}:${routes.dest?.lat ?? 0}`
    : "none";
  const buildingKey = buildings
    ? `${buildings.length}:${buildings.filter((b) => b.over > 0).length}:` +
      `${buildings.find((b) => b.selected)?.id ?? ""}`
    : "none";
  const siteKey = sites
    ? sites.map((x) => `${x.lat.toFixed(4)}${x.taken ? "*" : ""}`).join(",")
    : "none";
  // The scene is an object literal rebuilt on every render of the parent, so
  // the layer memo needs a primitive to compare. The water level is the part
  // that moves -- it is what the clock changes.
  const sceneKey = scene
    ? `${scene.datum}:${scene.waterM?.toFixed(2) ?? "dry"}:${scene.waterPng.length}:${scene.surfacePng.length}`
    : "none";
  const gaugeKey = gauges
    ? gauges.map((g) => `${g.site}:${g.stage_m ?? ""}:${g.labelled ? 1 : 0}`).join(",")
    : "none";
  const radarKey = radar
    ? `${radar.west.toFixed(3)},${radar.south.toFixed(3)},${radar.east.toFixed(3)},${radar.north.toFixed(3)}`
    : "none";
  const sectionKey = section
    ? `${section.a.lat},${section.a.lon},${section.b.lat},${section.b.lon},` +
      `${section.wallAt ?? -1},${section.verdict}`
    : "none";

  const layers = useMemo(() => {
    const out: Layer[] = [];
    const street = Boolean(scene);
    /* Vertical scale, and the datum it is measured from.
       The decoder returns ABSOLUTE elevation, and multiplying every term by
       five -- the offset included -- multiplies the elevation rather than
       the relief. On the valley views that is the established look and the
       lift is lost in a twelve-kilometre frame. On a 1.7 km block of ground
       at 165-178 m it put the mesh 825 m into the air, where it swallowed
       the camera, hid the houses standing at z = 0 underneath it, and turned
       one street into a mountain. The street view therefore draws at TRUE
       scale and subtracts the block's own lowest ground, so the ground sits
       on the imagery and a 6 m house next to 3 m of water is 6 m next to
       3 m. Everything else in that scene -- the boxes, the water plane --
       is offset by the same datum. */
    const exag = scene ? STREET_VE : 5;
    const datum = scene ? scene.datum : 0;
    /* ONE decoder, shared by the ground and by the water. If these two ever
       drift apart the shoreline stops being the place where the surfaces
       cross and becomes an artefact of two different vertical frames, which
       is unreadable and silently wrong. */
    const dec = {
      rScaler: ELEVATION_DECODER.rScaler * exag,
      gScaler: ELEVATION_DECODER.gScaler * exag,
      bScaler: ELEVATION_DECODER.bScaler * exag,
      offset: ELEVATION_DECODER.offset * exag - datum * exag,
    };
    const up = (m: number) => (m - datum) * exag;

    out.push(
      new TileLayer({
        id: "imagery",
        data: IMAGERY,
        minZoom: 0,
        // Esri's imagery reaches z19+ over cities but stops at z17-18 over
        // farmland and remote ground, and a request past the end of a service
        // does not fail -- it returns a grey "Map data not available" tile,
        // which then gets drawn as though it were the world. Capping here makes
        // deck.gl magnify the deepest real tile instead: slightly soft close in,
        // but always imagery. This tool opens anywhere on Earth, so the cap has
        // to hold for the thinnest coverage, not the best.
        maxZoom: 18,
        tileSize: 256,
        renderSubLayers: (props) => {
          const { boundingBox } = props.tile;
          return new BitmapLayer({
            id: `${props.id}-b`,
            image: props.data as string,
            bounds: [boundingBox[0][0], boundingBox[0][1],
                     boundingBox[1][0], boundingBox[1][1]],
            /* Dimmed under the street scene so a bone house and a salmon
               one read as objects against it rather than as two more
               patches of photograph. It is also the first thing drawn and
               nothing should ever be hidden behind it, so it never writes
               depth. */
            opacity: street ? 0.5 : 0.92,
            parameters: { depthWriteEnabled: false },
          });
        },
      }) as unknown as Layer,
    );
    /* The block as an object on a table, which is what the design mock
       draws and what makes one street legible: everything outside the frame
       fades to the page colour (that mask is pushed last, below) and the
       ground gets a visible lip here, so the block reads as a slab rather
       than as a photograph that happens to stop. */
    if (scene) {
      const k = scene.blockBox;
      // Under the lowest ground in frame, so only the lip is ever seen.
      const plinthTop = up(scene.groundLoM) - 3;
      const ring = [[k.west, k.south], [k.west, k.north],
                    [k.east, k.north], [k.east, k.south]] as [number, number][];
      out.push(
        new PolygonLayer({
          id: "street-plinth",
          // Below everything, showing only its lip. Its top face used to sit
          // at z = 0, exactly where the imagery sheet is, so the slab won
          // the coplanar fight and the block lost its photograph.
          data: [{ ring }],
          getPolygon: (d: { ring: [number, number][] }) =>
            d.ring.map(([lon, lat]) =>
              [lon, lat, plinthTop - 34] as [number, number, number]),
          extruded: true,
          getElevation: 34,
          filled: true,
          stroked: false,
          getFillColor: [206, 201, 192, 255],
          material: { ambient: 0.6, diffuse: 0.5, shininess: 1,
                      specularColor: [0, 0, 0] },
        }) as unknown as Layer,
      );
    }

    // The solved study IS the terrain layer. It used to be an override on a
    // shipped Richmond raster, which is why every other place rendered as a
    // patch on a valley in New South Wales.
    const terr = analysis
      ? { elev: `data:image/png;base64,${analysis.terrain_png}`,
          tex: `data:image/png;base64,${analysis.depth_png}`,
          bb: [analysis.bounds.west, analysis.bounds.south,
               analysis.bounds.east, analysis.bounds.north] as [number, number, number, number] }
      : null;
    if (terr) out.push(
      new TerrainLayer({
        id: analysis ? "terrain-adhoc" : "terrain",
        elevationData: terr.elev,
        texture: terr.tex,
        bounds: terr.bb,
        elevationDecoder: dec,
        /* In the street scene the ground is a matte model surface, not a
           photograph. The texture there is the blank PNG -- the depth ramp
           is drawn as a water SURFACE instead -- and a mesh with a
           transparent texture does not show what is under it: deck.gl's
           mesh shader has no alpha discard, so it paints the material's own
           colour. Rather than fight that, name the colour: a warm stone the
           houses and the water both read against, lit so the relief shows.
           The satellite photograph stays as the context around the block. */
        color: street ? [216, 211, 201, 255] : [255, 255, 255, 255],
        material: street
          ? { ambient: 0.62, diffuse: 0.55, shininess: 2,
              specularColor: [0, 0, 0] }
          : { ambient: 0.55, diffuse: 0.6, shininess: 6,
              specularColor: [30, 45, 60] },
      }) as unknown as Layer,
    );
    if (roads) {
      out.push(
        new GeoJsonLayer({
          id: "roads",
          data: roads,
          stroked: true,
          filled: false,
          lineWidthUnits: "pixels",
          // Severity carried twice -- width AND colour -- because on satellite
          // imagery a thin red line over a brown field is not a signal. A cut
          // road is the heaviest thing on the map; an open one recedes.
          // Severity carried twice -- width AND colour -- because a thin red
          // line over brown farmland is not a signal.
          getLineWidth: (f) => {
            const props = (f as GeoJSON.Feature).properties;
            const frac = props?.id != null
              ? (cutFraction?.get(props.id as number) ?? 0) : 0;
            const c = props?.class;
            const base = c === "motorway" || c === "trunk" || c === "primary"
              ? 2.4 : c === "secondary" || c === "tertiary" ? 1.8 : 1.0;
            if (frac > 0.5) return base + 2.6;
            if (frac > 0.05) return base + 1.2;
            return base;
          },
          getLineColor: (f) => {
            const props = (f as GeoJSON.Feature).properties;
            const frac = props?.id != null
              ? (cutFraction?.get(props.id as number) ?? 0) : 0;
            if (frac > 0.5) return [239, 68, 68, 255];
            if (frac > 0.05) return [251, 191, 36, 240];
            return [255, 255, 255, 110];
          },
          // A PRIMITIVE trigger, not the Map itself.
          //
          // deck.gl caches the evaluated attribute buffers; a changed accessor
          // closure alone does not re-evaluate them, updateTriggers has to say
          // so. Handing it the cut Map looked right and did not work -- every
          // road kept the colour it was given on the first build, when the map
          // was still null, so the network drew uniformly white while the panel
          // correctly reported 72 km cut. A string that changes whenever the
          // query changes is compared the way deck.gl expects.
          updateTriggers: { getLineColor: cutKey, getLineWidth: cutKey },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }
    // The Perceive stage, drawn ON TOP of the model rather than instead of it.
    // The point of showing it is the disagreement: where the model paints water
    // and the instrument does not, and the reverse.
    if (observed?.observed_png && observed.bounds?.length === 4) {
      const ob = observed.bounds as number[];
      out.push(
        new BitmapLayer({
          id: "observed-sar",
          image: `data:image/png;base64,${observed.observed_png}`,
          bounds: [ob[0], ob[1], ob[2], ob[3]] as [number, number, number, number],
          opacity: 0.8,
        }) as unknown as Layer,
      );
    }

    // ---- the flood as a body of water, and the levees as walls ----------
    // Drawn wherever the depth and ground grids are available. The ad-hoc
    // analysis ships them too (service returns `inspect` alongside the PNGs),
    // so a town the user searched for gets the same body of water as the
    // archive scene rather than a flat painted one.
    if (solid && grids) {
      const wb = analysis ? analysis.bounds : b;
      const { cells, cellSizeM } = waterColumns(grids, wb, { step: 3, minDepth: 0.15 });
      out.push(
        new ColumnLayer({
          id: "water-body",
          data: cells,
          diskResolution: 4,
          // A diskResolution-4 column is a SQUARE inscribed in `radius`, so its
          // side is radius*sqrt(2). Tiling cells of `cellSizeM` therefore needs
          // radius = cellSizeM/sqrt(2) ~ 0.707, and `angle` to turn the square
          // off its default 45 deg so edges meet the grid. At 0.62 the squares
          // left gaps and the flood rendered as a halftone screen.
          radius: cellSizeM * 0.73,
          angle: 45,
          extruded: true,
          pickable: false,
          /* Stand the column ON THE GROUND. `waterColumns` has always
             computed each cell's ground height and the layer threw it away:
             a 2-element position puts every column at z = 0 while the
             terrain mesh draws the ground at 5x its 138-246 m elevation, so
             the whole body of water sat six hundred metres underneath the
             landscape it belonged to and the "Water depth" chip looked like
             a control that did nothing. The base is exaggerated by the same
             5 the terrain uses; the height is not, because `elevationScale`
             already applies it and the colour ramp reads the true depth. */
          getPosition: (d: { position: [number, number]; base: number }) =>
            [d.position[0], d.position[1], d.base * 5] as [number, number, number],
          getElevation: (d: { depth: number }) => d.depth,
          elevationScale: 5,          // matches the terrain's exaggeration
          // Shallow water reads lighter; deep water darker and more opaque, so
          // depth is legible without a legend.
          getFillColor: (d: { depth: number }) => {
            const t = Math.min(1, d.depth / 6);
            return [38 + 20 * (1 - t), 110 + 70 * (1 - t), 190 + 40 * (1 - t), 150 + 80 * t];
          },
          material: { ambient: 0.4, diffuse: 0.7, shininess: 120,
                      specularColor: [160, 190, 220] },
        }) as unknown as Layer,
      );
    }

    if (pin) {
      out.push(
        new ScatterplotLayer({
          id: "pin",
          data: [pin],
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 7,
          getFillColor: [255, 255, 255, 230],
          stroked: true,
          getLineColor: [8, 11, 16, 220],
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }
    // Sites the user placed on a town they chose. Numbered for the same reason
    // the archive sites are: the panel refers to them by number, and an
    // unlabelled dot cannot be matched to a row.
    if (placed?.length) {
      // Which of them the drawn option actually builds. Without this the dots
      // look identical whatever is selected, so the map gave no sign that a
      // choice had been made -- only the water changed, and on flat ground it
      // can change very little.
      const built = (i: number) => (placedHeights?.[i] ?? 0) > 0;
      const rows = placed.map((d, i) => ({ ...d, n: i, h: placedHeights?.[i] ?? 0 }));
      const builtKey = placedHeights ? placedHeights.join(",") : "none";
      out.push(
        new ScatterplotLayer({
          id: "placed-levees",
          data: rows,
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: (d: { n: number }) => (built(d.n) ? 17 : 13),
          // Built reads as a solid wall colour; a site left out of this option
          // recedes to an outline, still numbered so the rows still match.
          getFillColor: (d: { n: number }) =>
            built(d.n) ? [250, 204, 21, 245] : [56, 189, 248, 150],
          stroked: true,
          getLineColor: (d: { n: number }) =>
            built(d.n) ? [24, 24, 27, 235] : [255, 255, 255, 200],
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          updateTriggers: {
            getFillColor: builtKey, getLineColor: builtKey, getRadius: builtKey,
          },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "placed-levee-labels",
          data: rows,
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: (d: { n: number; h: number }) =>
            d.h > 0 ? `${d.n} · ${d.h.toFixed(1)}m` : String(d.n),
          getColor: (d: { n: number }) =>
            built(d.n) ? [24, 24, 27, 255] : [255, 255, 255, 255],
          updateTriggers: { getText: builtKey, getColor: builtKey },
          getSize: 14,
          fontWeight: 700,
          getTextAnchor: "middle",
          getAlignmentBaseline: "center",
          outlineWidth: 2,
          outlineColor: [10, 14, 20, 200],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // The two ends of a trip. Labelled, because "from" and "to" on a map are
    // not interchangeable and a route drawn between two identical dots makes
    // the reader work out which way they are going.
    if (pins?.length) {
      out.push(
        new ScatterplotLayer({
          id: "trip-pins",
          data: pins,
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 8,
          getFillColor: (d: { label: string }) =>
            d.label === "from" ? [255, 255, 255, 245] : [34, 197, 94, 250],
          stroked: true,
          getLineColor: [10, 14, 20, 230],
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          updateTriggers: { getFillColor: pins.length },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "trip-pin-labels",
          data: pins,
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: (d: { label: string }) => d.label,
          getSize: 12,
          getColor: [255, 255, 255, 255],
          fontWeight: 700,
          getTextAnchor: "start",
          getPixelOffset: [12, 0],
          outlineWidth: 2,
          outlineColor: [10, 14, 20, 220],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // ---- the way out ---------------------------------------------------
    // Drawn over the water on purpose. A route is an instruction, and an
    // instruction that disappears under the thing it is routing around is not
    // one. The robust route sits underneath as a wider, paler line, so where
    // the two coincide it reads as one road with a halo and where they diverge
    // the detour is obvious without a legend.
    if (routes?.robust?.coords?.length &&
        routes.robust.minutes !== routes.quickest?.minutes) {
      out.push(
        new PathLayer({
          id: "route-robust",
          data: [{ path: routes.robust.coords }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [250, 204, 21, 210],
          getWidth: 9,
          widthUnits: "pixels",
          capRounded: true,
          jointRounded: true,
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }
    if (routes?.quickest?.coords?.length) {
      out.push(
        new PathLayer({
          id: "route-quickest",
          data: [{ path: routes.quickest.coords }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [34, 197, 94, 245],
          getWidth: 5,
          widthUnits: "pixels",
          capRounded: true,
          jointRounded: true,
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }
    if (routes?.dest) {
      out.push(
        new ScatterplotLayer({
          id: "route-dest",
          data: [routes.dest],
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 7,
          getFillColor: [34, 197, 94, 255],
          stroked: true,
          getLineColor: [255, 255, 255, 240],
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "route-dest-label",
          data: [routes.dest],
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: () => "out",
          getSize: 12,
          getColor: [255, 255, 255, 255],
          fontWeight: 700,
          getTextAnchor: "start",
          getPixelOffset: [11, 0],
          outlineWidth: 2,
          outlineColor: [10, 14, 20, 220],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // ---- the cross-section --------------------------------------------
    // The line is thin and the WALL is thick, because the line is a question
    // and the wall is the answer: an operator glancing at the map should see
    // how far the embankment runs without reading the panel. The halo is not
    // decoration -- this sits on satellite imagery that is sometimes pale sand
    // and sometimes dark water, and a single-colour line disappears into one
    // or the other.
    if (section) {
      const { a: sa, b: sb } = section;
      const at = (f: number): [number, number] => [
        sa.lon + (sb.lon - sa.lon) * f,
        sa.lat + (sb.lat - sa.lat) * f,
      ];
      out.push(
        new PathLayer({
          id: "section-line-halo",
          data: [{ path: [[sa.lon, sa.lat], [sb.lon, sb.lat]] }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [10, 14, 20, 200],
          getWidth: 5,
          widthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new PathLayer({
          id: "section-line",
          data: [{ path: [[sa.lon, sa.lat], [sb.lon, sb.lat]] }],
          getPath: (d: { path: [number, number][] }) => d.path,
          getColor: [255, 255, 255, 235],
          getWidth: 2,
          widthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
      if (section.wallAt !== null) {
        // The wall as a marker on the line, in the same ochre the section
        // draws it in, so the two pictures of the same object agree. A wall
        // that is overtopped is drawn hollow rather than merely paler: it
        // should not read as a solid thing on a map skimmed at a distance.
        const holds = section.verdict === "holds";
        out.push(
          new ScatterplotLayer({
            id: "section-wall",
            data: [{ p: at(section.wallAt) }],
            radiusUnits: "pixels",
            getPosition: (d: { p: [number, number] }) => d.p,
            getRadius: 9,
            filled: holds,
            getFillColor: [200, 127, 62, 250],
            stroked: true,
            getLineColor: [138, 85, 41, 250],
            getLineWidth: 3,
            lineWidthUnits: "pixels",
            parameters: { depthCompare: "always" as const },
          }) as unknown as Layer,
        );
      }
      const ends = [{ ...sa, label: "A" }, { ...sb, label: "B" }];
      out.push(
        new ScatterplotLayer({
          id: "section-ends",
          data: ends,
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 7,
          getFillColor: [255, 255, 255, 245],
          stroked: true,
          getLineColor: [10, 14, 20, 235],
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "section-end-labels",
          data: ends,
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: (d: { label: string }) => d.label,
          getSize: 12,
          getColor: [255, 255, 255, 255],
          fontWeight: 700,
          getTextAnchor: "start",
          getPixelOffset: [11, 0],
          outlineWidth: 2,
          outlineColor: [10, 14, 20, 220],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // ---- the street's buildings -----------------------------------------
    // Extruded to their real heights where the city publishes one, and
    // coloured by the thing the panel is about: water over the FLOOR, not over
    // the ground outside. A building standing in half a metre of water with a
    // floor above it is a different answer from one with half a metre inside,
    // and a map that colours both by depth cannot tell them apart.
    if (buildings?.length) {
      out.push(
        new PolygonLayer({
          id: "street-buildings",
          data: buildings,
          pickable: true,
          extruded: true,
          wireframe: false,
          /* Stand each box on its own ground. deck.gl's polygon tesselator
             reads a third ordinate when the ring carries one and the
             extrusion shader adds `elevation` ON TOP of it, so a 3-element
             ring is the supported way to give a building a base. Without it
             every house stood at z = 0 while the terrain it belongs to was
             drawn hundreds of metres above, which is why no building was
             visible in this view at all. */
          getPolygon: (d: { ring: [number, number][]; base_m: number }) =>
            d.ring.map(([lon, lat]) =>
              [lon, lat, up(d.base_m)] as [number, number, number]),
          getElevation: (d: { height_m: number | null }) =>
            Math.max(d.height_m ?? 4, 2.5) * exag,
          getFillColor: (d: { over: number; selected: boolean }) =>
            d.selected ? [24, 24, 27, 255]
              : d.over > 0 ? [232, 136, 122, 242]
                : [247, 245, 242, 240],
          getLineColor: [10, 14, 20, 120],
          getLineWidth: 0.6,
          lineWidthUnits: "pixels",
          stroked: true,
          material: { ambient: 0.55, diffuse: 0.65, shininess: 24,
                      specularColor: [255, 255, 255] },
          onClick: ({ object }) => {
            const o = object as { id: string } | undefined;
            if (o && onBuilding) onBuilding(o.id);
          },
          updateTriggers: {
            getFillColor: buildingKey,
            getElevation: buildingKey,
            getPolygon: `${buildingKey}:${datum}`,
          },
        }) as unknown as Layer,
      );
    }

    /* ---- the water, as a surface ----------------------------------------
       The street view used to show the flood the way every other view does:
       a depth ramp painted on the ground. At sixteen metres over a block the
       ramp saturates, so the answer to "whose floor did it get over" was one
       flat sheet of dark blue with the houses invisible underneath it.

       What the question actually needs is a LEVEL, and the thing a reader
       is looking for -- which roofs stand clear of it -- is then geometry
       rather than colour. It is a SURFACE and not one plane because this
       block falls thirteen metres across 1.7 km and the water follows that
       gradient; a single level would be wrong by metres at one end. (An
       earlier revision drew a flat quad and the hill did not hide it. The
       reason was never established -- the guess written here, that a mesh
       carrying a transparent texture writes no depth, is false: deck.gl's
       mesh shader has no alpha discard. The surface made the question moot,
       so the quad's behaviour is unexplained rather than understood.) */
    if (scene?.waterM != null && analysis) {
      const wb = analysis.bounds;
      out.push(
        new TerrainLayer({
          id: "street-water-surface",
          // The same mesh machinery as the ground, carrying the water's own
          // height instead: the sheet hangs at the solved surface rather
          // than lying on the floor, so a house standing in it is cut at the
          // waterline and one standing clear of it keeps its own colour.
          elevationData: `data:image/png;base64,${scene.surfacePng}`,
          // Transparent wherever the ground is dry, so the sheet has the
          // flood's shape and stops at its edge.
          texture: `data:image/png;base64,${scene.waterPng}`,
          bounds: [wb.west, wb.south, wb.east, wb.north],
          elevationDecoder: dec,
          // Translucent enough to read the ground and the submerged half of
          // a house through it, opaque enough to be a surface and not a
          // tint. Flat-lit: water is not a hillside.
          opacity: 0.62,
          material: { ambient: 0.95, diffuse: 0.05, shininess: 1,
                      specularColor: [0, 0, 0] },
        }) as unknown as Layer,
      );
    }

    /* The streets, named. The mock labels its river; nothing in this
       pipeline ships a channel centreline -- `build_forcing` keeps an
       inflow EDGE mask, not a line, and fetching waterways would put an
       Overpass round trip on a solve that already runs past the browser's
       five-minute wait. What the browser does already hold is the named
       road network, so the block gets its street names instead, and the
       water's own shape shows where the creek runs. */
    if (scene && roads) {
      const k = scene.blockBox;
      const seen = new Set<string>();
      const marks: { at: [number, number]; text: string }[] = [];
      for (const f of roads.features) {
        const nm = (f.properties as { name?: string } | null)?.name;
        if (!nm || nm === "Unnamed road" || seen.has(nm)) continue;
        const g = f.geometry;
        if (g?.type !== "LineString") continue;
        const cs = g.coordinates as [number, number][];
        const inside = cs.filter(([lon, lat]) =>
          lon >= k.west && lon <= k.east && lat >= k.south && lat <= k.north);
        if (inside.length < 2) continue;
        seen.add(nm);
        marks.push({ at: inside[Math.floor(inside.length / 2)], text: nm });
        if (marks.length >= 6) break;
      }
      if (marks.length) out.push(
        new TextLayer({
          id: "street-road-labels",
          data: marks,
          getPosition: (d: { at: [number, number] }) =>
            [d.at[0], d.at[1], 0] as [number, number, number],
          getText: (d: { text: string }) => d.text,
          getSize: 11,
          billboard: true,
          getColor: [34, 40, 50, 230],
          outlineWidth: 2.5,
          outlineColor: [255, 255, 255, 215],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    /* The number, as a picture. The panel says "+4.36 m of water inside"
       and until now nothing on the map said it: the house was salmon, and
       salmon is a yes/no. This is a stick from the selected house's floor
       up to the water on it, drawn over everything so it cannot be lost
       inside its own building, with the figure beside it. */
    const picked = scene && buildings?.find((x) => x.selected);
    if (scene && picked) {
      let sx = 0, sy = 0;
      for (const [lon, lat] of picked.ring) { sx += lon; sy += lat; }
      const at: [number, number] = [sx / picked.ring.length,
                                    sy / picked.ring.length];
      const floorM = picked.base_m + scene.floorM;
      const topM = floorM + Math.max(picked.over, 0);
      const wet = picked.over > 0;
      out.push(
        new PathLayer({
          id: "street-probe",
          data: [{ path: [[at[0], at[1], up(floorM)],
                          [at[0], at[1], up(wet ? topM : floorM + 1.2)]] }],
          getPath: (d: { path: [number, number, number][] }) => d.path,
          getColor: wet ? [24, 24, 27, 240] : [70, 80, 95, 200],
          widthUnits: "pixels",
          getWidth: 2,
          capRounded: true,
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "street-probe-label",
          data: [{ at, text: wet
            ? `+${picked.over.toFixed(2)} m over the floor`
            : "floor above the water" }],
          getPosition: (d: { at: [number, number] }) =>
            [d.at[0], d.at[1], up(wet ? topM : floorM + 1.2)] as
              [number, number, number],
          getText: (d: { text: string }) => d.text,
          getSize: 12,
          billboard: true,
          getColor: [255, 255, 255, 255],
          getPixelOffset: [0, -12],
          outlineWidth: 2.5,
          outlineColor: [10, 14, 20, 235],
          fontSettings: { sdf: true },
          characterSet: "+-.0123456789 mabcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // ---- where a wall would pay -----------------------------------------
    // Outlines, not fills: these are candidates. A taken one is drawn solid
    // and dark so the plan reads at a glance against the places still open.
    if (sites?.length) {
      const rows = sites.map((s, i) => ({ ...s, i }));
      // An alignment, drawn as one. These used to be a cloud of dots -- one
      // per cell of the candidate run, in the order a flood fill happened to
      // visit them, which is all the geometry the service returned. The
      // service now returns the run's centreline, so the thing on screen is
      // the line a wall would be built along: a white casing so it reads over
      // dark water, and a dashed core because a candidate is not yet a levee.
      // A taken site goes solid.
      out.push(
        new PathLayer({
          id: "site-casing",
          data: rows,
          pickable: true,
          getPath: (d: { outline: [number, number][] }) => d.outline,
          getColor: [255, 255, 255, 210],
          getWidth: (d: { taken: boolean }) => (d.taken ? 13 : 10),
          widthUnits: "pixels",
          capRounded: true,
          jointRounded: true,
          onClick: ({ object }) => {
            const o = object as { i: number } | undefined;
            if (o && onSite) onSite(o.i);
          },
          updateTriggers: { getWidth: siteKey },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new PathLayer({
          id: "site-line",
          data: rows,
          getPath: (d: { outline: [number, number][] }) => d.outline,
          getColor: (d: { taken: boolean }) =>
            (d.taken ? [24, 24, 27, 255] : [24, 24, 27, 190]),
          getWidth: (d: { taken: boolean }) => (d.taken ? 7 : 3),
          widthUnits: "pixels",
          capRounded: true,
          jointRounded: true,
          getDashArray: (d: { taken: boolean }) => (d.taken ? [0, 0] : [7, 5]),
          dashJustified: true,
          dashGapPickable: true,
          extensions: [new PathStyleExtension({ dash: true, highPrecisionDash: true })],
          updateTriggers: {
            getColor: siteKey, getWidth: siteKey, getDashArray: siteKey,
          },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "site-labels",
          data: rows,
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: (d: { label: string }) => d.label,
          getSize: 11,
          getColor: [16, 20, 28, 255],
          background: true,
          getBackgroundColor: [255, 255, 255, 235],
          backgroundPadding: [6, 4, 6, 4],
          getPixelOffset: [0, -16],
          getTextAnchor: "middle",
          getAlignmentBaseline: "bottom",
          fontWeight: 600,
          fontSettings: { sdf: true },
          updateTriggers: { getText: siteKey },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // ---- rain radar ----------------------------------------------------
    // One image over the analysed box rather than a tile pyramid: the box is a
    // few kilometres across, the product refreshes every six minutes, and a
    // single exportImage is one request where tiles would be dozens.
    if (radar) {
      out.push(
        new BitmapLayer({
          id: "radar",
          bounds: [radar.west, radar.south, radar.east, radar.north],
          image:
            "https://mapservices.weather.noaa.gov/eventdriven/rest/services/radar/" +
            "radar_base_reflectivity_time/ImageServer/exportImage" +
            `?bbox=${radar.west},${radar.south},${radar.east},${radar.north}` +
            "&bboxSR=4326&imageSR=4326&size=1024,1024&format=png32" +
            "&transparent=true&f=image",
          opacity: 0.5,
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // ---- river gauges --------------------------------------------------
    // Drawn with their reading attached. The value and the hourly trend are
    // what an operator reads off a gauge; the dot on its own says only that
    // one exists.
    if (gauges?.length) {
      const withStage = gauges.filter((g) => g.stage_m !== null);
      out.push(
        new ScatterplotLayer({
          id: "gauge-dots",
          data: withStage,
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 6,
          getFillColor: [255, 255, 255, 250],
          stroked: true,
          getLineColor: [10, 14, 20, 235],
          getLineWidth: 2.5,
          lineWidthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "gauge-readings",
          data: withStage.filter((g) => g.labelled !== false),
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: (d: { name: string; stage_m: number | null; trend_cm_h: number | null }) => {
            const nm = d.name.split(",")[0].slice(0, 28);
            const tr = d.trend_cm_h === null ? ""
              : `  ${d.trend_cm_h >= 0 ? "\u2191" : "\u2193"}${Math.abs(d.trend_cm_h).toFixed(0)}`;
            return `${nm}\n${(d.stage_m ?? 0).toFixed(2)} m${tr}`;
          },
          getSize: 11,
          getColor: [16, 20, 28, 255],
          background: true,
          getBackgroundColor: [255, 255, 255, 240],
          backgroundPadding: [7, 5, 7, 5],
          getPixelOffset: [0, -26],
          getTextAnchor: "middle",
          getAlignmentBaseline: "bottom",
          // The arrows must be in the atlas or deck.gl drops them SILENTLY.
          // Without this the readings rendered as "4.27 m  2" -- the trend
          // number with its arrow missing, which reads as a second quantity
          // nobody can identify, and "0.59 m  0" reads as a measurement of
          // zero. A dropped glyph is worse than a missing feature.
          characterSet: [...new Set(
            "0123456789.,-+ m³/s↑↓abcdefghijklmnopqrstuvwxyz"
            + "ABCDEFGHIJKLMNOPQRSTUVWXYZ'()".split(""))],
          lineHeight: 1.25,
          fontWeight: 600,
          fontSettings: { sdf: true },
          updateTriggers: { getText: gaugeKey },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }

    // The place the user searched for, marked the same way the archive town is.
    if (place) {
      out.push(
        new ScatterplotLayer({
          id: "place-ring",
          data: [place],
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 26,
          filled: false,
          stroked: true,
          getLineColor: [255, 255, 255, 220],
          getLineWidth: 2.5,
          lineWidthUnits: "pixels",
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
        new TextLayer({
          id: "place-label",
          data: [place],
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: (d: { name: string }) => d.name,
          getSize: 13,
          getColor: [255, 255, 255, 245],
          fontWeight: 700,
          getPixelOffset: [0, -36],
          getTextAnchor: "middle",
          outlineWidth: 3,
          outlineColor: [10, 14, 20, 220],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }
    /* The mask, last of all. Drawn before the water it was painted over,
       and the solved domain ran on past the block as a blue wall. It is a
       polygon with a hole in it -- the outer ring is the world, the inner
       ring is the block -- so it covers everything the reader should not be
       looking at and nothing they should. */
    if (scene) {
      const k = scene.blockBox;
      const g = 0.25;
      out.push(
        new PolygonLayer({
          id: "street-surround",
          data: [{ rings: [
            [[k.west - g, k.south - g], [k.east + g, k.south - g],
             [k.east + g, k.north + g], [k.west - g, k.north + g]],
            [[k.west, k.south], [k.west, k.north],
             [k.east, k.north], [k.east, k.south]],
          ] }],
          getPolygon: (d: { rings: [number, number][][] }) => d.rings,
          filled: true,
          stroked: false,
          extruded: false,
          getFillColor: [232, 229, 222, 255],
          parameters: { depthCompare: "always" as const,
                        depthWriteEnabled: false },
        }) as unknown as Layer,
      );
    }

    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b, roads, pin, cutFraction, cutKey,
      observed, analysis, placed, place, solid, grids,
      heightKey, placedKey, routeKey, sectionKey, gaugeKey, radarKey,
      buildingKey, onBuilding, siteKey, onSite, pins,
      sceneKey, focusBounds]);

  if (!size) return <div ref={host} className="h-full w-full bg-bg-inset" />;

  return (
    <div ref={host} className="relative h-full w-full">
      <DeckGL
        initialViewState={view}
        controller={{ dragRotate: true }}
        layers={layers}
        getCursor={({ isHovering, isDragging }) =>
          isDragging ? "grabbing" : isHovering ? "pointer" : "crosshair"}
        onClick={(info) => {
          // A click on a measure is handled by that layer; a click anywhere
          // else is a location query, which is the question a planner asks
          // most often: how deep did it get *here*.
          if (info.layer?.id === "measures") return;
          if (info.coordinate && onPick) {
            onPick(info.coordinate[0], info.coordinate[1]);
          }
        }}
        /* What the cursor is actually over. This cast to `MeasureMarker`
           outlived the layer it was written for: the only pickable layers
           now are the street buildings and the candidate bank alignments,
           so hovering a house said "Measure undefined / Click to add". */
        getTooltip={({ object, layer }) => {
          if (!object) return null;
          const id = layer?.id;
          let html: string | null = null;
          if (id === "street-buildings") {
            const bld = object as { id?: string; over?: number };
            const over = bld.over ?? 0;
            html = `<b>${bld.id ?? "building"}</b><br/>${
              over > 0 ? `${over.toFixed(2)} m of water over the floor`
                : "floor above the water"}`;
          } else if (id === "site-casing") {
            const st = object as { name?: string | null; label?: string;
                                   taken?: boolean };
            html = `<b>${st.name || "unnamed bank"}</b><br/>${
              st.label ?? ""}${st.taken ? "" : " · click to build here"}`;
          }
          if (!html) return null;
          return {
            html,
            style: { background: "#11151c", color: "#e8edf4",
                     border: "1px solid #1d232d", borderRadius: "8px",
                     padding: "7px 9px", fontSize: "12px" },
          };
        }}
      />
      <p className="pointer-events-none absolute bottom-1 right-2 text-[11px] text-ink-faint">
        Imagery © Esri · Roads © OpenStreetMap contributors
      </p>
    </div>
  );
}
