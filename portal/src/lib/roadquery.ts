import type { Bounds } from "./types";
import type { InspectGrids } from "./inspect";
import { sampleAt } from "./inspect";

export interface CutRoad {
  id: number;
  name: string | null;
  cls: string;
  cutKm: number;
  totalKm: number;
}

export interface RoadQuery {
  threshold: number;
  cutKm: number;
  totalKm: number;
  roads: CutRoad[];
  /** Per-feature fraction cut, for colouring the map. */
  cutFraction: Map<number, number>;
}

function segKm(a: [number, number], b: [number, number]): number {
  // Local-plane approximation; exact enough at this latitude over a few km.
  const dx = (b[0] - a[0]) * 97.0;
  const dy = (b[1] - a[1]) * 111.0;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Which roads are impassable at a given depth.
 *
 * A planner's question is "what gets cut", not "what is the depth field". The
 * threshold is theirs to set: a passenger car is stopped by far less water than
 * a truck or an ambulance, so the answer depends on who has to get through.
 */
export function queryRoads(
  roads: GeoJSON.FeatureCollection | null,
  grids: InspectGrids | null,
  bounds: Bounds | null,
  threshold: number,
): RoadQuery | null {
  if (!roads || !grids || !bounds) return null;
  const out: CutRoad[] = [];
  const cutFraction = new Map<number, number>();
  let cutKm = 0;
  let totalKm = 0;

  for (const f of roads.features) {
    if (f.geometry.type !== "LineString") continue;
    const props = f.properties as { id: number; name: string | null; class: string } | null;
    if (!props) continue;
    const coords = f.geometry.coordinates as [number, number][];
    let cut = 0;
    let tot = 0;
    for (let i = 0; i < coords.length - 1; i++) {
      const a = coords[i];
      const b = coords[i + 1];
      const L = segKm(a, b);
      tot += L;
      const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const s = sampleAt(grids, bounds, mid[0], mid[1]);
      if (s && s.depth_m > threshold) cut += L;
    }
    totalKm += tot;
    cutKm += cut;
    if (tot > 0) cutFraction.set(props.id, cut / tot);
    if (cut > 0.01) {
      out.push({ id: props.id, name: props.name, cls: props.class, cutKm: cut, totalKm: tot });
    }
  }

  out.sort((a, b) => b.cutKm - a.cutKm);
  return { threshold, cutKm, totalKm, roads: out, cutFraction };
}

/** Vehicle-relevant thresholds, so the control means something operationally. */
export const DEPTH_PRESETS = [
  { m: 0.10, label: "Any water", note: "surface water present" },
  { m: 0.25, label: "Car", note: "most cars stall or float" },
  { m: 0.45, label: "4WD / truck", note: "heavy vehicles only" },
  { m: 0.80, label: "Impassable", note: "no road vehicle" },
] as const;
