import type { Box } from "./types";

/** Point-query grids: peak depth over the event, and ground elevation. */
export interface InspectMeta {
  grid: [number, number];
  depth_vmax_m: number;
  elev_min_m: number;
  elev_max_m: number;
}

export interface InspectGrids {
  meta: InspectMeta;
  depth: Uint8Array;
  elev: Uint16Array;
}

export async function loadInspect(): Promise<InspectGrids> {
  const [meta, d, e] = await Promise.all([
    fetch("/data/inspect.json").then((r) => r.json() as Promise<InspectMeta>),
    fetch("/data/peak_depth.u8").then((r) => r.arrayBuffer()),
    fetch("/data/elevation.u16").then((r) => r.arrayBuffer()),
  ]);
  return { meta, depth: new Uint8Array(d), elev: new Uint16Array(e) };
}

export interface PointInfo {
  lon: number;
  lat: number;
  depth_m: number;
  elev_m: number;
  flooded: boolean;
}

/** Sample both grids at a geographic point. */
export function sampleAt(
  g: InspectGrids,
  b: Box,
  lon: number,
  lat: number,
): PointInfo | null {
  const [h, w] = g.meta.grid;
  const fx = (lon - b.west) / (b.east - b.west);
  const fy = (b.north - lat) / (b.north - b.south);
  if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
  const c = Math.min(w - 1, Math.floor(fx * w));
  const r = Math.min(h - 1, Math.floor(fy * h));
  const i = r * w + c;
  const depth = (g.depth[i] / 255) * g.meta.depth_vmax_m;
  const elev =
    g.meta.elev_min_m +
    (g.elev[i] / 65535) * (g.meta.elev_max_m - g.meta.elev_min_m);
  return { lon, lat, depth_m: depth, elev_m: elev, flooded: depth > 0.1 };
}

/** Nearest named road to a point, for orientation. */
export function nearestRoad(
  roads: GeoJSON.FeatureCollection | null,
  lon: number,
  lat: number,
): { name: string; km: number } | null {
  if (!roads) return null;
  let best: { name: string; km: number } | null = null;
  for (const f of roads.features) {
    const name = (f.properties as { name?: string } | null)?.name;
    if (!name || f.geometry.type !== "LineString") continue;
    for (const [x, y] of f.geometry.coordinates as [number, number][]) {
      // Local-plane approximation: fine at this latitude over a few km.
      const dx = (x - lon) * 97.0;
      const dy = (y - lat) * 111.0;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (!best || d < best.km) best = { name, km: d };
    }
  }
  return best;
}
