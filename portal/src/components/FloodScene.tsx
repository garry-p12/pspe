"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { WebMercatorViewport } from "@deck.gl/core";
import { TerrainLayer, TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import type { Layer } from "@deck.gl/core";
import type { Manifest } from "@/lib/types";
import { ELEVATION_DECODER, frameUrl } from "@/lib/data";

/** Esri World Imagery, drawn by deck.gl's own TileLayer.
 *
 * Two dead ends got us here. MapLibre's vector basemap needs a Web Worker built
 * with `new Worker(new URL(...), import.meta.url)`, which Next's bundler fails
 * to resolve in dev *and* production. CARTO's raster tiles avoid the worker but
 * now stamp "API KEY REQUIRED" across every tile. Esri's imagery service is open
 * and key-free, and for a page about one real floodplain, aerial imagery reads
 * far better than a street map: the viewer can see the river, the farmland and
 * the towns the water is about to cover.
 *
 * Note the {y}/{x} ordering -- this service is not {x}/{y} like most.
 */
const BASEMAP_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export interface LeveeSite {
  id: number;
  lon: number;
  lat: number;
  elev: number;
  reductionPct: number;
}

export type ViewPreset = "extent" | "settlement";

export function FloodScene({
  manifest,
  frame,
  exaggeration,
  sites,
  selected,
  onSelect,
  showTerrain,
  preset,
}: {
  manifest: Manifest;
  frame: number;
  exaggeration: number;
  sites?: LeveeSite[];
  selected?: number | null;
  onSelect?: (id: number | null) => void;
  showTerrain: boolean;
  preset: ViewPreset;
}) {
  const b = manifest.bounds;
  const ext = manifest.frame_ext ?? "png";
  const bounds = useMemo(
    () => [b.west, b.south, b.east, b.north] as [number, number, number, number],
    [b],
  );

  // Warm the next few frames so scrubbing does not flash empty.
  const cache = useRef<Set<string>>(new Set());
  useEffect(() => {
    for (let k = 1; k <= 4; k++) {
      const i = Math.min(frame + k, manifest.frames.length - 1);
      const u = frameUrl(i, ext);
      if (!cache.current.has(u)) {
        cache.current.add(u);
        const img = new Image();
        img.src = u;
      }
    }
  }, [frame, ext, manifest.frames.length]);

  const layers = useMemo(() => {
    const out: Layer[] = [];
    out.push(
      new TileLayer({
        id: "basemap",
        data: BASEMAP_TILES,
        minZoom: 0,
        maxZoom: 19,
        tileSize: 256,
        renderSubLayers: (props) => {
          const { boundingBox } = props.tile;
          return new BitmapLayer({
            id: `${props.id}-bmp`,
            image: props.data as string,
            bounds: [
              boundingBox[0][0], boundingBox[0][1],
              boundingBox[1][0], boundingBox[1][1],
            ],
            opacity: 0.9,
          });
        },
      }) as unknown as Layer,
    );
    out.push(
      new TerrainLayer({
        id: `terrain-${showTerrain ? "3d" : "flat"}`,
        elevationData: "/data/terrain.png",
        texture: frameUrl(frame, ext),
        bounds,
        elevationDecoder: {
          ...ELEVATION_DECODER,
          rScaler: ELEVATION_DECODER.rScaler * (showTerrain ? exaggeration : 0),
          gScaler: ELEVATION_DECODER.gScaler * (showTerrain ? exaggeration : 0),
          bScaler: ELEVATION_DECODER.bScaler * (showTerrain ? exaggeration : 0),
          offset: showTerrain ? ELEVATION_DECODER.offset * exaggeration : 0,
        },
        material: { ambient: 0.5, diffuse: 0.6, shininess: 8, specularColor: [40, 60, 80] },
        opacity: 1,
      }) as unknown as Layer,
    );
    if (sites?.length) {
      out.push(
        new ScatterplotLayer<LeveeSite>({
          id: "levee-sites",
          data: sites,
          pickable: true,
          radiusUnits: "pixels",
          getPosition: (d) => [d.lon, d.lat],
          getRadius: (d) => (d.id === selected ? 13 : 9),
          getFillColor: (d) =>
            d.reductionPct < -0.5
              ? [248, 113, 113, 230]
              : d.reductionPct > 3
                ? [52, 211, 153, 230]
                : [148, 163, 184, 210],
          getLineColor: (d) => (d.id === selected ? [255, 255, 255, 255] : [10, 13, 18, 200]),
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          stroked: true,
          onClick: ({ object }) =>
            onSelect?.(object && object.id === selected ? null : (object?.id ?? null)),
          updateTriggers: { getRadius: selected, getLineColor: selected },
          // The markers are geographic points at z=0 while the terrain mesh is
          // lifted by the exaggeration slider, so depth-testing buries them
          // inside the hillside. They are annotations, not scene geometry.
          // (luma.gl v9 spells this depthCompare, not depthTest.)
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
      out.push(
        new TextLayer<LeveeSite>({
          id: "levee-labels",
          data: sites,
          getPosition: (d) => [d.lon, d.lat],
          getText: (d) => `S${d.id}`,
          getSize: 11,
          getColor: [232, 237, 244, 230],
          getPixelOffset: [0, -16],
          fontFamily: "monospace",
          characterSet: "auto",
          outlineWidth: 2,
          outlineColor: [8, 11, 16, 220],
          fontSettings: { sdf: true },
          parameters: { depthCompare: "always" as const },
        }) as unknown as Layer,
      );
    }
    return out;
  }, [bounds, frame, ext, exaggeration, showTerrain, sites, selected, onSelect]);

  // Fit the 32 km extent rather than guessing a zoom: a hard-coded level framed
  // the terrain as a small tile in the middle of an empty viewport.
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Two framings, because the story lives at two scales. "extent" shows the
  // whole 32 km event; "settlement" frames the 1.4 km ring of levee sites,
  // which otherwise collapses into an unreadable cluster of overlapping dots.
  const settle = manifest.levee_geometry?.settlement;
  const viewState = useMemo(() => {
    const base = {
      pitch: showTerrain ? 50 : 0,
      bearing: showTerrain ? -20 : 0,
      transitionDuration: 900,
    };
    if (preset === "settlement" && settle) {
      return {
        ...base,
        longitude: settle.lon,
        latitude: settle.lat,
        zoom: showTerrain ? 12.4 : 12.9,
      };
    }
    const centre = {
      ...base,
      longitude: (b.west + b.east) / 2,
      latitude: (b.south + b.north) / 2,
      zoom: 9.6,
    };
    if (!size) return centre;
    const vp = new WebMercatorViewport({ width: size.w, height: size.h });
    const fit = vp.fitBounds([[b.west, b.south], [b.east, b.north]], {
      padding: showTerrain ? 90 : 30,
    });
    // fitBounds solves for an unpitched camera; tilting pushes the far edge
    // away, so the 3D view needs to come IN rather than out.
    return { ...centre, zoom: fit.zoom + (showTerrain ? 0.55 : 0.1) };
  }, [b, size, showTerrain, preset, settle]);

  if (!size) {
    return <div ref={hostRef} className="h-full w-full animate-pulse bg-bg-inset" />;
  }

  return (
    <div ref={hostRef} className="relative h-full w-full">
    <DeckGL
      initialViewState={viewState}
      controller={{ dragRotate: true, touchRotate: true }}
      layers={layers}
      getTooltip={({ object }) => {
        const s = object as LeveeSite | undefined;
        if (!s) return null;
        return {
          html: `<div style="font:12px var(--font-sans-stack),sans-serif">
            <b>Site ${s.id}</b><br/>perimeter ${s.elev.toFixed(1)} m<br/>
            ${s.reductionPct >= 0 ? "reduces" : "<b style='color:#f87171'>worsens</b>"} flooding ${Math.abs(s.reductionPct).toFixed(2)}%
          </div>`,
          style: {
            background: "#11151c",
            color: "#e8edf4",
            border: "1px solid #1d232d",
            borderRadius: "8px",
            padding: "8px 10px",
          },
        };
      }}
    >
    </DeckGL>
      <p className="pointer-events-none absolute bottom-1 right-2 text-[9.5px] text-ink-faint">
        Imagery © Esri, Maxar, Earthstar Geographics
      </p>
    </div>
  );
}
