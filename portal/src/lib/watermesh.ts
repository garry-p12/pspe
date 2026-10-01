import type { Box } from "./types";
import type { InspectGrids } from "./inspect";

/**
 * The flood as a body of water, not a blue picture.
 *
 * The map draws real terrain from the DEM, but the flood has been a flat image
 * draped over it. That throws away the one quantity with genuine vertical
 * extent in this scene: the terrain relief is 220 m over 32 km (1:146, flatter
 * than it sounds once projected), while the water reaches 19 m deep. The water
 * is the three-dimensional thing here, and drawing it flat hides that.
 *
 * Each cell becomes a column standing on the ground at its own depth, so a
 * levee reads as a wall holding a volume back rather than a marker beside a
 * blue shape.
 *
 * Decimated deliberately: a column per DEM cell would be ~250k draws for no
 * visual gain at the zooms this is viewed from.
 */

export interface WaterCell {
  position: [number, number];
  /** Ground height in metres, where the column starts. */
  base: number;
  /** Water depth in metres, the column's height. */
  depth: number;
}

export function waterColumns(
  g: InspectGrids,
  b: Box,
  opts: { step?: number; minDepth?: number; exaggeration?: number } = {},
): { cells: WaterCell[]; cellSizeM: number } {
  const step = opts.step ?? 3;
  const minDepth = opts.minDepth ?? 0.15;
  const exag = opts.exaggeration ?? 1;
  const [rows, cols] = g.meta.grid;
  const dv = g.meta.depth_vmax_m / 255;
  const ev = (g.meta.elev_max_m - g.meta.elev_min_m) / 65535;

  const cells: WaterCell[] = [];
  for (let r = 0; r < rows; r += step) {
    for (let c = 0; c < cols; c += step) {
      const i = r * cols + c;
      const depth = g.depth[i] * dv;
      if (depth < minDepth) continue;
      const elev = g.meta.elev_min_m + g.elev[i] * ev;
      // Cell centre, not corner: a column drawn from the corner sits half a
      // cell off its own ground sample and the surface looks sheared.
      const lon = b.west + ((c + 0.5) / cols) * (b.east - b.west);
      const lat = b.north - ((r + 0.5) / rows) * (b.north - b.south);
      cells.push({ position: [lon, lat], base: elev * exag, depth: depth * exag });
    }
  }
  // Metres per decimated cell, for the column footprint.
  const midLat = (b.north + b.south) / 2;
  const widthM = (b.east - b.west) * 111_320 * Math.cos((midLat * Math.PI) / 180);
  return { cells, cellSizeM: (widthM / cols) * step };
}

/**
 * A levee as a wall of its chosen height, standing on the ground.
 *
 * A proposed levee has been a dot on the map. "Site 3 at 3.0 m" is a number
 * until you can see a 3 m wall against water that reaches 4 m, which is the
 * comparison the whole decision turns on.
 */
export interface LeveeWall {
  polygon: [number, number][];
  base: number;
  height: number;
  site: number;
}

export function leveeWalls(
  sites: { id: number; lon: number; lat: number; elev: number }[],
  heights: number[],
  opts: { lengthM?: number; exaggeration?: number } = {},
): LeveeWall[] {
  const lengthM = opts.lengthM ?? 1750;
  const exag = opts.exaggeration ?? 1;
  const out: LeveeWall[] = [];
  for (const s of sites) {
    const h = heights[s.id] ?? 0;
    if (!(h > 0)) continue;
    const dLat = (lengthM / 2) / 111_320;
    const dLon = dLat / Math.cos((s.lat * Math.PI) / 180);
    // A thin quad across the approach, oriented NE-SW, which is the direction
    // the Richmond floodwater arrives from.
    const wLat = 60 / 111_320;
    const wLon = wLat / Math.cos((s.lat * Math.PI) / 180);
    out.push({
      site: s.id,
      base: s.elev * exag,
      height: h * exag,
      polygon: [
        [s.lon - dLon - wLon, s.lat - dLat + wLat],
        [s.lon + dLon - wLon, s.lat + dLat + wLat],
        [s.lon + dLon + wLon, s.lat + dLat - wLat],
        [s.lon - dLon + wLon, s.lat - dLat - wLat],
      ],
    });
  }
  return out;
}
