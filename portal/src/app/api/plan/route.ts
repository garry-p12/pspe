import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * Decision-time planning: spend a budget, under a guarantee.
 *
 * This is the framework's own loop, at the speed an operator needs it. The
 * hydrodynamic solver takes tens of minutes, so the search runs against a
 * surrogate fitted to solver runs (milliseconds), the answer is reported with a
 * split-conformal margin calibrated on held-out solver runs, and the plan that
 * gets adopted is queued for a full solver run that verifies it.
 *
 * Plan in the model, bound the error against reality, verify what you commit
 * to. Previously this endpoint did not exist and the interface ranked a fixed
 * list of precomputed options, which is a catalogue, not a planner.
 *
 * The search is a greedy allocation in height increments. It is not the paper's
 * gradient planner and does not pretend to be: over six sites and a handful of
 * increments the space is small enough that greedy with a full re-evaluation at
 * each step is near-exhaustive, and it is auditable, which matters more in a
 * tool someone spends public money on than sophistication does.
 */

interface Fit {
  alpha_km: Record<string, number>;
  saturation_S: number;
  band_km: number;
  max_height_m: number;
  n_calibration_runs: number;
  delta?: number;
  band_attainable?: boolean;
  band_note?: string;
}

let cache: { metric?: string; fits: Record<string, Fit> } | null = null;

async function load() {
  if (cache) return cache;
  const p = path.join(process.cwd(), "public", "data", "surrogate.json");
  cache = JSON.parse(await readFile(p, "utf8"));
  return cache!;
}

function surrogate(fit: Fit, heights: number[]): { effect: number; linear: number; active: number } {
  // lead + S*tanh(rest/S). Saturating the WHOLE sum makes the model jump at the
  // boundary between one measure and two, so adding a levee that helps could
  // lower the prediction -- measured at -1.3 points for site 5 beside site 3.
  // Only what is stacked on top of the largest single contribution saturates,
  // which keeps single measures exact and the model continuous.
  const parts: number[] = [];
  for (let i = 0; i < heights.length; i++) {
    const a = fit.alpha_km[String(i)];
    if (a === undefined || !(heights[i] > 0)) continue;
    parts.push(a * (heights[i] / fit.max_height_m));
  }
  const linear = parts.reduce((s, v) => s + v, 0);
  if (parts.length === 0) return { effect: 0, linear: 0, active: 0 };
  const lead = Math.max(...parts);
  const S = fit.saturation_S || 100;
  return { effect: lead + S * Math.tanh((linear - lead) / S), linear, active: parts.length };
}

export async function POST(req: Request) {
  const body = (await req.json()) as {
    budget_aud?: number;
    event_scale?: number;
    cost_per_m_per_m?: number;
    crest_lengths_m?: number[];
    step_m?: number;
    max_height_m?: number;
  };

  let data;
  try {
    data = await load();
  } catch {
    return NextResponse.json({ error: "no_surrogate" }, { status: 503 });
  }

  const fit = data.fits[String(body.event_scale ?? 1)] ?? Object.values(data.fits)[0];
  if (!fit) return NextResponse.json({ error: "no_fit_for_scale" }, { status: 400 });

  const lengths = body.crest_lengths_m ?? [];
  const k = Object.keys(fit.alpha_km).length;
  if (lengths.length < k) {
    return NextResponse.json({ error: "need_crest_lengths" }, { status: 400 });
  }

  const unit = body.cost_per_m_per_m ?? 2600;
  const maxH = body.max_height_m ?? fit.max_height_m;
  const step = body.step_m ?? 0.5;
  const budget = body.budget_aud ?? 20e6;

  const heights = new Array(k).fill(0);
  const cost = (h: number[]) => h.reduce((s, hi, i) => s + hi * lengths[i] * unit, 0);

  // Greedy: take the affordable increment that buys the most per dollar, stop
  // when nothing affordable improves the objective. Sites that make flooding
  // worse are never selected, because the increment is chosen by measured
  // effect rather than by assuming a levee helps.
  const trace: { site: number; to_m: number; effect: number; spent: number }[] = [];
  for (;;) {
    let best = { gain: 1e-9, site: -1, spent: 0, effect: 0 };
    const current = surrogate(fit, heights).effect;
    const spentNow = cost(heights);
    for (let i = 0; i < k; i++) {
      if (heights[i] + step > maxH + 1e-9) continue;
      const trial = heights.slice();
      trial[i] += step;
      const spent = cost(trial);
      if (spent > budget) continue;
      const effect = surrogate(fit, trial).effect;
      const perDollar = (effect - current) / (spent - spentNow || 1);
      if (effect > current && perDollar > best.gain) {
        best = { gain: perDollar, site: i, spent, effect };
      }
    }
    if (best.site < 0) break;
    heights[best.site] += step;
    trace.push({ site: best.site, to_m: heights[best.site], effect: best.effect, spent: best.spent });
  }

  const { effect } = surrogate(fit, heights);
  const active = heights.filter((h) => h > 0).length;

  // EXPLAIN, by counterfactual rather than by narration.
  //
  // Section 4.4 killed this project's first explanation module: briefs scored
  // well against the model and carried ZERO state-specific information, which a
  // permutation control exposed -- a brief written for one state scored just as
  // well on another. The lesson is that an explanation has to be something that
  // CHANGES when the situation changes.
  //
  // A leave-one-out marginal does by construction. "Remove this levee from THIS
  // plan and you lose X" depends on what else is in the plan, so it cannot be
  // reused for a different plan, and it is checkable against a solver run.
  // The gap between a measure's standalone effect and its marginal contribution
  // here is the overlap, which is the honest answer to why measures do not add.
  const attribution = heights
    .map((h, i) => {
      if (!(h > 0)) return null;
      const without = heights.slice();
      without[i] = 0;
      const marginal = effect - surrogate(fit, without).effect;
      const alone = (fit.alpha_km[String(i)] ?? 0) * (h / fit.max_height_m);
      return {
        site: i,
        height_m: h,
        // What this measure adds given everything else already built.
        marginal_pct: marginal,
        // What it would achieve on its own, straight from a solver run.
        alone_pct: alone,
        // Positive means the rest of the plan already captures part of it.
        overlap_pct: alone - marginal,
        cost_aud: h * lengths[i] * unit,
      };
    })
    .filter(Boolean) as {
      site: number; height_m: number; marginal_pct: number;
      alone_pct: number; overlap_pct: number; cost_aud: number;
    }[];
  attribution.sort((a, b) => b.marginal_pct - a.marginal_pct);

  // Sites the solver measured as HARMFUL, stated whether or not they were
  // chosen. A planner that silently omits them looks like it never considered
  // them; the backwater effect is a real result and belongs on screen.
  const harmful = Object.entries(fit.alpha_km)
    .filter(([, a]) => a < 0)
    .map(([i, a]) => ({ site: Number(i), alone_pct: a }))
    .sort((x, y) => x.alone_pct - y.alone_pct);
  // The margin bounds the surrogate's interaction error. A single measure is a
  // direct solver result and carries none.
  const attainable = fit.band_attainable !== false;
  const band = active > 1 && attainable ? fit.band_km : 0;
  const conf = fit.delta ? Math.round(100 * (1 - fit.delta)) : null;

  return NextResponse.json({
    event_scale: body.event_scale ?? 1,
    metric: data.metric ?? "core_reduction_pct",
    heights,
    cost_aud: cost(heights),
    budget_aud: budget,
    reduction_pct: effect,
    // What the tool will stand behind, rather than its best guess.
    guaranteed_pct: effect - band,
    band_pct: band,
    delta: fit.delta ?? null,
    confidence_pct: conf,
    band_attainable: attainable,
    band_note: fit.band_note ?? null,
    n_calibration_runs: fit.n_calibration_runs,
    steps: trace,
    attribution,
    harmful,
    verify: {
      required: active > 0,
      why: "The plan was chosen inside a surrogate. Adopting it should be conditional on a full hydrodynamic run of this exact height vector.",
    },
    method: attainable
      ? `Greedy allocation against a surrogate fitted to ${fit.n_calibration_runs} solver runs, reported with a split-conformal margin at ${conf ?? 90}% confidence.`
      : `Greedy allocation against a surrogate. ${fit.band_note ?? ""} No margin is applied, so the figure is a point estimate only.`,
  });
}
