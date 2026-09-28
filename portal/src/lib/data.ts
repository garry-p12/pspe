import type { Manifest, Metrics } from "./types";

/** The bundle is static and versioned with the build, so a plain fetch is fine. */
export async function loadManifest(): Promise<Manifest> {
  const r = await fetch("/data/manifest.json");
  if (!r.ok) throw new Error(`manifest: ${r.status}`);
  return r.json();
}

export async function loadMetrics(): Promise<Metrics> {
  const r = await fetch("/data/metrics.json");
  if (!r.ok) return {};
  return r.json();
}

export function frameUrl(i: number, ext = "png"): string {
  return `/data/depth/${String(i).padStart(4, "0")}.${ext}`;
}

/** Mapbox terrain-RGB decoder, matching scripts/build_portal_assets.py. */
export const ELEVATION_DECODER = {
  rScaler: 6553.6,
  gScaler: 25.6,
  bScaler: 0.1,
  offset: -10000,
} as const;

export function fmtHours(h: number): string {
  const d = Math.floor(h / 24);
  const r = Math.round(h % 24);
  return d > 0 ? `day ${d + 1}, ${String(r).padStart(2, "0")}:00` : `day 1, ${String(r).padStart(2, "0")}:00`;
}

export function fmt(n: number, dp = 2): string {
  return n.toLocaleString(undefined, {
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });
}
