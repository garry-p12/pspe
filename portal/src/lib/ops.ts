/** The operational contract: what the planner shows, with no physics in it. */

export interface NamedRoad { name: string; km: number }

export interface Option {
  event_scale: number;
  id: string;
  name: string;
  heights: number[];
  cost_aud: number;
  road_cut_km: number;
  area_flooded_km2: number;
  named_roads_cut: NamedRoad[];
  road_cut_change_km?: number;
  local_road_cut_km?: number;
  core_reduction_pct?: number;
  protects_to_km?: number;
  displaces_from_km?: number | null;
  rings?: { from_km: number; to_km: number; change_m: number; protected: boolean }[];
  area_change_km2?: number;
  makes_worse?: boolean;
  road_protected_km?: number;
  cost_per_km_protected?: number | null;
}

export interface SiteInfo { id: number; elev_m: number; crest_length_m: number }

export interface OpsPayload {
  region: string;
  reference_event: string;
  cut_depth_m: number;
  dx_m: number;
  cost_note?: string;
  sites: SiteInfo[];
  event_scales: number[];
  options: Option[];
  road_network_km: number;
}

export const EVENT_LABEL: Record<string, string> = {
  "0.8": "Smaller event",
  "1": "2022 flood (recorded)",
  "1.25": "Larger event",
};

export function eventLabel(scale: number): string {
  return EVENT_LABEL[String(scale)] ?? `${scale}× event`;
}

export function money(aud: number): string {
  if (aud >= 1e6) return `A$${(aud / 1e6).toFixed(1)}M`;
  if (aud >= 1e3) return `A$${Math.round(aud / 1e3)}k`;
  return `A$${Math.round(aud)}`;
}

export function km(v: number, dp = 1): string {
  return `${v.toFixed(dp)} km`;
}
