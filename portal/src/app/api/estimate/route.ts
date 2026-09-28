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
 * The band is a SPLIT-CONFORMAL quantile over held-out solver runs -- the same
 * `pspe.plan.margins` the research code uses -- so the number here and the
 * number in the paper mean the same thing. It replaced a normal-approximation
 * interval, 1.645 * RMSE * sqrt(1 + 1/n), which assumed Gaussian residuals that
 * nothing established and, fitted on six runs, reported HALF the honest width:
 * +/-10.5 points against +/-20.2. A margin fitted under a convenient
 * assumption is not a margin.
 *
 * Conformal also refuses rather than guessing. Below n = 1/delta - 1 no finite
 * order statistic carries the stated rate, and the response says so instead of
 * printing a confident-looking number.
 */

interface Fit {
  base_road_cut_km: number;
  base_area_km2: number;
  alpha_km: Record<string, number>;
  saturation_S: number;
  fit_rmse: number;
  band_km: number;
  delta?: number;
  band_attainable?: boolean;
  band_note?: string;
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

  // lead + S*tanh(rest/S): saturating the whole sum made the model jump between
  // one measure and two, so adding a levee that helps could lower the answer.
  const parts: number[] = [];
  for (let i = 0; i < heights.length; i++) {
    const a = fit.alpha_km[String(i)];
    if (a === undefined || !(heights[i] > 0)) continue;
    parts.push(a * (heights[i] / fit.max_height_m));
  }
  const active = parts.length;
  const linear = parts.reduce((s, v) => s + v, 0);
  const S = fit.saturation_S || 100;
  const effect = active ? Math.max(...parts) + S * Math.tanh((linear - Math.max(...parts)) / S) : 0;
  const attainable = fit.band_attainable !== false;
  const band = active > 1 && attainable ? fit.band_km : 0;
  const conf = fit.delta ? Math.round(100 * (1 - fit.delta)) : null;

  return NextResponse.json({
    event_scale: scale,
    metric: data.metric ?? "core_reduction_pct",
    // Positive = flooding reduced at the defended settlement.
    reduction_pct: effect,
    linear_sum_pct: linear,
    band_pct: band,
    best_case_pct: effect + band,
    worst_case_pct: effect - band,
    // What the tool will stand behind, as opposed to its best guess.
    guaranteed_pct: effect - band,
    confidence_pct: conf,
    delta: fit.delta ?? null,
    band_attainable: attainable,
    band_note: fit.band_note ?? null,
    makes_worse: effect < -0.5,
    exact: active <= 1,
    saturating: active > 1 && linear - effect > 1,
    n_calibration_runs: fit.n_calibration_runs,
    method: active <= 1
      ? "Direct result of a hydrodynamic run for this measure."
      : "Fast estimate. Combined measures overlap, so the total is less than the sum; the range covers what the fit cannot pin down.",
  });
}
