"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { WebMercatorViewport } from "@deck.gl/core";
import type { Layer } from "@deck.gl/core";
import { TerrainLayer, TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer, GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
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
  analysis,
  observed,
  placed,
  focus,
}: {
  manifest: Manifest;
  frame: number;
  roads: GeoJSON.FeatureCollection | null;
  markers: MeasureMarker[];
  onMarker: (id: number) => void;
  onPick?: (lon: number, lat: number) => void;
  pin?: { lon: number; lat: number } | null;
  cutFraction?: Map<number, number> | null;
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
  focus: "region" | "town";
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

  const layers = useMemo(() => {
    const out: Layer[] = [];
    out.push(
      new TileLayer({
        id: "imagery",
        data: IMAGERY,
        minZoom: 0,
        maxZoom: 19,
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
          getLineWidth: (f) => {
            const c = (f as GeoJSON.Feature).properties?.class;
            return c === "primary" || c === "secondary" ? 2.2 : 1.1;
          },
          // Red where the road is cut at the chosen depth, pale where it is
          // passable. A planner reads the network, not the depth field.
          getLineColor: (f) => {
            const id = (f as GeoJSON.Feature).properties?.id as number | undefined;
            const frac = id != null ? (cutFraction?.get(id) ?? 0) : 0;
            if (frac > 0.5) return [248, 113, 113, 235];
            if (frac > 0.05) return [251, 191, 36, 215];
            return [232, 237, 244, 120];
          },
          updateTriggers: { getLineColor: cutFraction },
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
    if (placed?.length) {
      out.push(
        new ScatterplotLayer({
          id: "placed-levees",
          data: placed,
          radiusUnits: "pixels",
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 9,
          getFillColor: [56, 189, 248, 235],
          stroked: true,
          getLineColor: [255, 255, 255, 220],
          getLineWidth: 2,
          lineWidthUnits: "pixels",
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
          getRadius: (d) => (d.selected ? 15 : 11),
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
    }
    return out;
  }, [b, frame, ext, roads, markers, onMarker, pin, cutFraction, analysis, placed]);

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
      <p className="pointer-events-none absolute bottom-1 right-2 text-[9.5px] text-ink-faint">
        Imagery © Esri · Roads © OpenStreetMap contributors
      </p>
    </div>
  );
}
