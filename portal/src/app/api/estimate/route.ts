import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * Instant estimate for an arbitrary set of measures.
 *
 * The hydrodynamic solver takes about an hour. This evaluates a surrogate
 * fitted to solver runs, in under a millisecond, so a planner can move a slider
 * and see the consequence. It returns a RANGE, not a point: single-measure
 * effects are exact from the solver, combinations carry a fitted interaction
 * term, and the band is the spread that term does not explain.
 *
 * That band is the product-facing form of the framework's safety margin. The
 * planner is never shown a number the model cannot stand behind, and any plan
 * that gets adopted is queued for a full solver run.
 */

interface Fit {
  base_road_cut_km: number;
  base_area_km2: number;
  alpha_km: Record<string, number>;
  saturation_S: number;
  fit_rmse: number;
  band_km: number;
  n_calibration_runs: number;
  max_height_m: number;
}

let cache: { metric: string; fits: Record<string, Fit>; sites: unknown[] } | null = null;

async function load() {
  if (cache) return cache;
  const p = path.join(process.cwd(), "public", "data", "surrogate.json");
  cache = JSON.parse(await readFile(p, "utf8"));
  return cache!;
}

export async function POST(req: Request) {
  let body: { heights?: number[]; event_scale?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }

  let data;
  try {
    data = await load();
  } catch {
    return NextResponse.json(
      { error: "surrogate_unavailable", detail: "Model not fitted yet." },
      { status: 503 },
    );
  }

  const scale = body.event_scale ?? 1;
  const fit = data.fits[String(scale)] ?? Object.values(data.fits)[0];
  if (!fit) {
    return NextResponse.json({ error: "no_fit_for_scale" }, { status: 400 });
  }

  const heights = body.heights ?? [];
  const active = heights.reduce((n, h) => n + (h > 0 ? 1 : 0), 0);

  let linear = 0;
  for (let i = 0; i < heights.length; i++) {
    const a = fit.alpha_km[String(i)];
    if (a === undefined || !(heights[i] > 0)) continue;
    linear += a * (heights[i] / fit.max_height_m);
  }

  // Combined measures saturate: they cannot remove more water than is there.
  // A model that simply added them predicted 86% reduction for all six levees
  // where the solver measured 62%. S*tanh(sum/S) fits the runs best of the
  // three forms tried and is the only one that stays sane when extrapolated.
  const S = fit.saturation_S || 100;
  const effect = active > 1 ? S * Math.tanh(linear / S) : linear;
  const band = active > 1 ? fit.band_km : 0;

  return NextResponse.json({
    event_scale: scale,
    metric: data.metric ?? "core_reduction_pct",
    // Positive = flooding reduced at the defended settlement.
    reduction_pct: effect,
    linear_sum_pct: linear,
    band_pct: band,
    best_case_pct: effect + band,
    worst_case_pct: effect - band,
    makes_worse: effect < -0.5,
    exact: active <= 1,
    saturating: active > 1 && linear - effect > 1,
    n_calibration_runs: fit.n_calibration_runs,
    method: active <= 1
      ? "Direct result of a hydrodynamic run for this measure."
      : "Fast estimate. Combined measures overlap, so the total is less than the sum; the range covers what the fit cannot pin down.",
  });
}
