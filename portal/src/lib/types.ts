/** Shapes of the precomputed bundle in `public/data`. */

export type Badge = "OBSERVED" | "VALIDATED MODEL" | "PROJECTION";

export interface FrameMeta {
  i: number;
  t_seconds: number;
  t_hours: number;
  wet_cells: number;
  max_depth: number;
  mean_depth_wet: number;
}

export interface Bounds {
  west: number; east: number; south: number; north: number;
  epsg: number; extent_km: number; place: string; event: string;
}

export interface Provenance {
  badge: Badge;
  source: string;
  note: string;
}

export interface LeveeGeometry {
  settlement: { row: number; col: number; lon: number; lat: number; elev: number };
  radius_cells: number;
  cell_m: number;
  sites: { id: number; row: number; col: number; elev: number; lon: number; lat: number }[];
}

export interface Manifest {
  levee_geometry?: LeveeGeometry;
  generated: string;
  bounds: Bounds;
  grid: [number, number];
  native_grid: [number, number];
  native_dx_m: number;
  raster_dx_m: number;
  dem_range_m: [number, number];
  depth_vmax_m: number;
  frame_ext?: string;
  frames: FrameMeta[];
  provenance: Record<string, Provenance>;
  caveats: string[];
  metrics?: string[];
}

/** One frame of the solver-vs-reference comparison (runs/floodcast_validate). */
export interface ValidationFrame {
  frame: number;
  t_hours: number;
  ref_wet: number;
  our_wet: number;
  rmse_wet: number;
  bias_wet: number;
  ["csi@0.01"]: number;
  ["csi@0.05"]: number;
}

export interface Validation {
  event: string;
  dx: number;
  grid: [number, number];
  window_hours: [number, number];
  unmodelled_forcing_frac: number;
  frames: ValidationFrame[];
  mean_csi: Record<string, number>;
  mean_rmse_wet: number;
  note: string;
}

export interface SiteCandidate {
  row: number; col: number; elev: number;
  total: number; pre: number; routed: number;
  rain_driven: number; direct_rain_cap: number;
  controllable: number; controllable_frac: number;
}

export interface LeveeSingle {
  site: number; elev: number; depth: number; reduction_pct: number;
}

export interface Sites {
  screened: number;
  top: SiteCandidate[];
  site_used_in_5_6a: SiteCandidate | null;
  levee_test: {
    site: [number, number];
    do_nothing: number;
    singles: LeveeSingle[];
    all_sites: number;
    best_single_pct: number;
    all_sites_pct: number;
  };
}

export interface PrecondRow {
  q_log_sigma: number;
  epsilon: number;
  sigma: number;
  rho: number;
  bias: number;
  z_delta: number;
  required_margin: number;
  headroom: number;
  span: number;
  f_fraction: number;
  margin_pct_of_span: number;
  precondition: "PASS" | "FAIL";
}

export interface ElasticityRow {
  inflow: number;
  elasticity: number;
  berm_rows_wet: number;
  span: number;
  verdict: string;
}

export interface Metrics {
  validation?: Validation;
  validation_openbc?: Validation;
  sites?: Sites;
  precond_synthetic?: PrecondRow[];
  elasticity?: { rows: ElasticityRow[] };
  span_synthetic?: unknown;
}
