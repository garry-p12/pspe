import { NextResponse } from "next/server";
import { bandAt, bandForPlan, maxConfidencePct, runsNeeded, saturating,
         type CalibrationPoint, type PlanBand } from "@/lib/margin";
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
  calibration?: CalibrationPoint[];
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
  // The form itself is `saturating` in lib/margin.ts, shared with the route
  // margin and parity-checked against pspe.plan.surrogate.saturating. This
  // function was a third copy of it, which meant the district tab's displayed
  // estimate came from code no test could reach.
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
  return { effect: saturating(lead, linear - lead, S), linear, active: parts.length };
}

export async function POST(req: Request) {
  const body = (await req.json()) as {
    budget_aud?: number;
    event_scale?: number;
    cost_per_m_per_m?: number;
    crest_lengths_m?: number[];
    step_m?: number;
    max_height_m?: number;
    /** How sure the user asked to be, 50-99. Drives the margin, not the plan. */
    confidence_pct?: number;
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

  // The requested confidence, clamped, and the stored calibration set. Both are
  // needed by the guarantee search below as well as by the band reported at the
  // end, so they are declared once here rather than twice.
  const askedConf = typeof body.confidence_pct === "number"
    ? Math.min(99.9, Math.max(50, body.confidence_pct))
    : fit.delta ? 100 * (1 - fit.delta) : 90;
  const cal = fit.calibration ?? [];

  // ----------------------------------------------------------------------- //
  // The SECOND plan: the one with the best guarantee, not the best estimate.
  //
  // The greedy search above maximises the surrogate's PREDICTION. That is what
  // a planner does when its margin is a single pooled number, because a
  // constant offset cannot change an argmax -- and it is why the margin was
  // decorative here. Conditioned on the stratum a plan lands in (lib/margin
  // `bandForPlan`) the band stops being constant, and the two objectives come
  // apart:
  //
  //   budget  best predicted            best guaranteed
  //   A$10M   1 site at 2.0 m, 31.0     2 sites, 27.2 -> floor 17.2
  //           (partial height: NO VALID BAND)
  //   A$15M   1 site at 3.0 m, 46.5     same plan, floor 46.5
  //   A$20M   2 sites, 52.1 -> 42.0     1 site at 3.0 m, floor 46.5
  //   A$30M   3 sites, 52.5             1 site at 3.0 m, floor 46.5
  //           (stratum too thin: NO VALID BAND)
  //
  // At A$20M the guaranteed plan predicts 5.6 points LESS and promises 4.5
  // points MORE, because its prediction is a solver measurement rather than an
  // extrapolation. At A$10M and A$30M the prediction-optimal plan cannot be
  // guaranteed at all. This is the framework's own thesis applied to the
  // portal's own task: bound the quantity the decision commits to, and the
  // decision changes.
  //
  // Exhaustive, not greedy. The floor jumps when a site crosses zero or reaches
  // full height, so a hill-climb in increments cannot see it. 7^6 = 117,649
  // combinations at the default step is microseconds; above the guard it falls
  // back to reporting only the greedy plan rather than silently sampling.
  interface GuaranteedPlan {
    heights: number[]; effect: number; floor: number; spent: number;
    band: PlanBand; active: number;
  }

  const searchGuaranteed = (): GuaranteedPlan | null => {
    const levels = Math.floor(maxH / step) + 1;
    if (levels ** k > 400_000) return null;     // report greedy only, never sample
    let best: GuaranteedPlan | null = null;
    const h = new Array<number>(k).fill(0);
    const walk = (i: number): void => {
      if (i === k) {
        const spent = cost(h);
        if (spent > budget) return;
        const b = bandForPlan(cal, fit.saturation_S || 100, askedConf, h, maxH);
        if (!b.attainable) return;              // refuse, never borrow a band
        const e = surrogate(fit, h).effect;
        const floor = e - b.band;
        if (!best || floor > best.floor) {
          best = { heights: h.slice(), effect: e, floor, spent, band: b,
                   active: h.filter((x) => x > 0).length };
        }
        return;
      }
      for (let v = 0; v <= maxH + 1e-9; v += step) { h[i] = v; walk(i + 1); }
      h[i] = 0;
    };
    walk(0);
    return best;
  };
  const guaranteed = searchGuaranteed();

  // And what the greedy plan can actually promise, under the same rule.
  const greedyBand = bandForPlan(cal, fit.saturation_S || 100, askedConf, heights, maxH);

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
  // The band is recomputed at whatever confidence was asked for, rather than
  // served from the one level it happened to be calibrated at. Asking for more
  // confidence than 30 held-out runs can carry returns a refusal, not a number.
  const recomputed = cal.length
    ? bandAt(cal, fit.saturation_S || 100, askedConf)
    : { band: fit.band_km, attainable: fit.band_attainable !== false,
        needed: runsNeeded(askedConf), have: fit.n_calibration_runs };
  const attainable = (fit.band_attainable !== false) && recomputed.attainable;
  const band = active > 1 && attainable ? recomputed.band : 0;
  const conf = Math.round(askedConf * 10) / 10;
  const ceiling = Math.floor(maxConfidencePct(recomputed.have) * 10) / 10;

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
    // Everything the UI needs to explain a refusal without jargon.
    confidence_ceiling_pct: ceiling,
    runs_needed: recomputed.needed,
    runs_have: recomputed.have,
    refusal: attainable ? null
      : `To promise this ${conf}% of the time we would need about ${recomputed.needed} test floods to check against. We have ${recomputed.have}. Lower the certainty, or run more tests.`,
    // ---- the conditional band, and the plan that maximises it ----------- //
    //
    // `band_pct` above is the POOLED band, kept so nothing downstream breaks.
    // These fields are the honest per-plan figures. For the greedy plan the
    // pooled and conditional bands differ in both directions: narrower for a
    // two-site plan (10.01 against 13.37, so the pooled number understates what
    // the tool can promise) and absent for anything the calibration set does
    // not cover at this level.
    conditional: {
      band_pct: greedyBand.attainable ? greedyBand.band : null,
      attainable: greedyBand.attainable,
      n_active: greedyBand.nActive,
      exact: greedyBand.exact,
      guaranteed_pct: greedyBand.attainable ? effect - greedyBand.band : null,
      runs_in_stratum: greedyBand.have,
      runs_needed: greedyBand.needed,
      why: greedyBand.exact
        ? "One site at full height: the prediction IS the solver run that measured it, so no margin is owed."
        : greedyBand.attainable
          ? `Calibrated on the ${greedyBand.have} runs that activated ${greedyBand.nActive} sites, not on all ${cal.length}. Surrogate error grows with the number of sites, so a pooled band is the wrong width for both.`
          : greedyBand.nActive === 1
            ? "One site at a partial height. The prediction scales a full-height solver run linearly and no run checks that, so no margin can be claimed."
            : `Only ${greedyBand.have} of the stored runs activated ${greedyBand.nActive} sites, and ${greedyBand.needed} are needed at this confidence.`,
    },
    // The plan the FRAMEWORK recommends: best guaranteed floor, not best
    // estimate. Null when no plan in budget has a valid band at this level.
    best_guaranteed: guaranteed && {
      heights: guaranteed.heights,
      cost_aud: guaranteed.spent,
      reduction_pct: guaranteed.effect,
      guaranteed_pct: guaranteed.floor,
      band_pct: guaranteed.band.band,
      n_active: guaranteed.active,
      exact: guaranteed.band.exact,
      // The number that makes the two objectives worth distinguishing.
      floor_gain_vs_greedy: greedyBand.attainable
        ? guaranteed.floor - (effect - greedyBand.band)
        : null,
      estimate_given_up: effect - guaranteed.effect,
      why: "Maximises what the tool can promise rather than what it predicts. A pooled margin is a constant and cannot change this choice; a margin conditioned on the plan can, and does.",
    },
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
