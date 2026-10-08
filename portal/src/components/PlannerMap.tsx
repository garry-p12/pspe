"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { WebMercatorViewport } from "@deck.gl/core";
import type { Layer } from "@deck.gl/core";
import { TerrainLayer, TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer, ColumnLayer, GeoJsonLayer, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import type { InspectGrids } from "@/lib/inspect";
import { leveeWalls, waterColumns } from "@/lib/watermesh";
import type { Manifest } from "@/lib/types";
import { ELEVATION_DECODER, frameUrl } from "@/lib/data";

const IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export interface MeasureMarker {
  id: number;
  lon: number;
  lat: number;
  selected: boolean;
  worsens: boolean;
}

export function PlannerMap({
  manifest,
  frame,
  roads,
  markers,
  onMarker,
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
  place,
  focus,
  grids,
  heights,
  solid = true,
}: {
  manifest: Manifest;
  frame: number;
  roads: GeoJSON.FeatureCollection | null;
  markers: MeasureMarker[];
  onMarker: (id: number) => void;
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
    crest: { from: number; to: number } | null;
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
  }[] | null;
  /** NOAA base reflectivity over the analysed box, if the user asked for it. */
  radar?: { west: number; south: number; east: number; north: number } | null;
  /** The town the user searched for, marked like the archive settlement. */
  place?: { lat: number; lon: number; name: string } | null;
  focus: "region" | "town";
  /** Depth and ground grids, for drawing the flood as a body of water. */
  grids?: InspectGrids | null;
  /** Chosen levee heights, drawn as walls of that height. */
  heights?: number[] | null;
  /** False falls back to the flat painted flood. */
  solid?: boolean;
}) {
  const b = manifest.bounds;
  const ext = manifest.frame_ext ?? "png";
  const settle = manifest.levee_geometry?.settlement;
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

  const ab = analysis?.bounds;
  const view = useMemo(() => {
    const base = { pitch: 48, bearing: -18, transitionDuration: 900 };
    if (ab) {
      return {
        ...base,
        longitude: (ab.west + ab.east) / 2,
        latitude: (ab.south + ab.north) / 2,
        zoom: 11.6,
      };
    }
    if (focus === "town" && settle) {
      return { ...base, longitude: settle.lon, latitude: settle.lat, zoom: 12.3 };
    }
    const c = {
      ...base,
      longitude: (b.west + b.east) / 2,
      latitude: (b.south + b.north) / 2,
      zoom: 9.6,
    };
    if (!size) return c;
    const vp = new WebMercatorViewport({ width: size.w, height: size.h });
    const fit = vp.fitBounds([[b.west, b.south], [b.east, b.north]], { padding: 70 });
    return { ...c, zoom: fit.zoom + 0.5 };
  }, [b, size, focus, settle, ab]);

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
  const gaugeKey = gauges
    ? gauges.map((g) => `${g.site}:${g.stage_m ?? ""}`).join(",")
    : "none";
  const radarKey = radar
    ? `${radar.west.toFixed(3)},${radar.south.toFixed(3)},${radar.east.toFixed(3)},${radar.north.toFixed(3)}`
    : "none";
  const sectionKey = section
    ? `${section.a.lat},${section.a.lon},${section.b.lat},${section.b.lon},` +
      `${section.crest?.from ?? -1},${section.crest?.to ?? -1},${section.verdict}`
    : "none";

  const layers = useMemo(() => {
    const out: Layer[] = [];
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
            opacity: 0.92,
          });
        },
      }) as unknown as Layer,
    );
    // An ad-hoc analysis replaces the archive layer entirely: its terrain and
    // flood are self-contained, so nothing from the Richmond bundle is mixed in.
    const terr = analysis
      ? { elev: `data:image/png;base64,${analysis.terrain_png}`,
          tex: `data:image/png;base64,${analysis.depth_png}`,
          bb: [analysis.bounds.west, analysis.bounds.south,
               analysis.bounds.east, analysis.bounds.north] as [number, number, number, number] }
      : { elev: "/data/terrain.png", tex: frameUrl(frame, ext),
          bb: [b.west, b.south, b.east, b.north] as [number, number, number, number] };
    out.push(
      new TerrainLayer({
        id: analysis ? "terrain-adhoc" : "terrain",
        elevationData: terr.elev,
        texture: terr.tex,
        bounds: terr.bb,
        elevationDecoder: {
          rScaler: ELEVATION_DECODER.rScaler * 5,
          gScaler: ELEVATION_DECODER.gScaler * 5,
          bScaler: ELEVATION_DECODER.bScaler * 5,
          offset: ELEVATION_DECODER.offset * 5,
        },
        material: { ambient: 0.55, diffuse: 0.6, shininess: 6,
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
          getPosition: (d: { position: [number, number] }) => d.position,
          // Stand the column on the ground and give it the water's own height.
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
          updateTriggers: { getFillColor: [frame], getElevation: [frame] },
        }) as unknown as Layer,
      );
    }

    if (solid && heights?.some((h) => h > 0) && manifest.levee_geometry?.sites) {
      const walls = leveeWalls(manifest.levee_geometry.sites, heights, {});
      out.push(
        new PolygonLayer({
          id: "levee-walls",
          data: walls,
          extruded: true,
          pickable: false,
          getPolygon: (d: { polygon: [number, number][] }) => d.polygon,
          getElevation: (d: { height: number }) => d.height,
          elevationScale: 5,
          getFillColor: [238, 238, 232, 255],
          getLineColor: [40, 40, 40, 255],
          material: { ambient: 0.6, diffuse: 0.5, shininess: 10,
                      specularColor: [60, 60, 60] },
          updateTriggers: { getElevation: [heights.join(",")] },
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
      if (section.crest) {
        const crest = [at(section.crest.from), at(section.crest.to)];
        // A crest that is overtopped is drawn narrower, so a wall that does
        // not do its job does not read as the heaviest thing on the map.
        const holds = section.verdict === "holds";
        out.push(
          new PathLayer({
            id: "section-crest-halo",
            data: [{ path: crest }],
            getPath: (d: { path: [number, number][] }) => d.path,
            getColor: [10, 14, 20, 220],
            getWidth: holds ? 13 : 9,
            widthUnits: "pixels",
            capRounded: true,
            parameters: { depthCompare: "always" as const },
          }) as unknown as Layer,
          new PathLayer({
            id: "section-crest",
            data: [{ path: crest }],
            getPath: (d: { path: [number, number][] }) => d.path,
            getColor: holds ? [255, 255, 255, 250] : [255, 255, 255, 170],
            getWidth: holds ? 8 : 4,
            widthUnits: "pixels",
            capRounded: true,
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
          data: withStage,
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
    if (markers.length) {
      out.push(
        new ScatterplotLayer<MeasureMarker>({
          id: "measures",
          data: markers,
          pickable: true,
          radiusUnits: "pixels",
          getPosition: (d) => [d.lon, d.lat],
          getRadius: (d) => (d.selected ? 20 : 16),
          getFillColor: (d) =>
            d.selected
              ? d.worsens
                ? [248, 113, 113, 240]
                : [56, 189, 248, 240]
              : [148, 163, 184, 170],
          stroked: true,
          getLineColor: (d) => (d.selected ? [255, 255, 255, 235] : [8, 11, 16, 190]),
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          onClick: ({ object }) => object && onMarker(object.id),
          updateTriggers: { getRadius: markers, getFillColor: markers, getLineColor: markers },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    const settle = analysis ? null : manifest.levee_geometry?.settlement;
    if (settle) {
      out.push(
        new ScatterplotLayer({
          id: "settlement-ring",
          data: [settle],
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
          id: "settlement-label",
          data: [settle],
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getText: () => "the town",
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

      out.push(
        new TextLayer<MeasureMarker>({
          id: "measure-labels",
          data: markers,
          pickable: false,
          getPosition: (d) => [d.lon, d.lat],
          getText: (d) => String(d.id),
          getSize: 15,
          outlineWidth: 2,
          outlineColor: [10, 14, 20, 200],
          fontSettings: { sdf: true },
          getColor: (d) => (d.selected ? [255, 255, 255, 255] : [255, 255, 255, 225]),
          fontWeight: 700,
          getTextAnchor: "middle",
          getAlignmentBaseline: "center",
          parameters: { depthCompare: "always" as const },
          updateTriggers: { getColor: markers },
        }) as unknown as Layer,
      );
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b, frame, ext, roads, markers, onMarker, pin, cutFraction, cutKey,
      observed, analysis, placed, place, settle, solid, grids,
      heightKey, placedKey, routeKey, sectionKey, gaugeKey, radarKey, pins]);

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
        getTooltip={({ object }) => {
          const m = object as MeasureMarker | undefined;
          if (!m) return null;
          return {
            html: `<b>Measure ${m.id}</b><br/>${m.selected ? "In plan" : "Click to add"}`,
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
