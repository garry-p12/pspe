/**
 * The safety margin, recomputed at whatever confidence the user asks for.
 *
 * The portal used to show one fixed band, calibrated once at 90%. But the whole
 * point of a conformal margin is that the failure rate is *chosen and stated*
 * rather than tuned, so the person spending the money should be able to choose
 * it. These helpers recompute the band from the stored held-out runs.
 *
 * The important behaviour is the REFUSAL. A split-conformal quantile at level
 * `delta` needs at least `1/delta - 1` calibration points; below that no order
 * statistic carries the guarantee, and returning a number anyway would be
 * theatre. With 30 held-out runs the highest honest confidence is about 96.8%.
 * Ask for 99% and the tool must say it cannot, and say how many more runs it
 * would need -- which is the opposite of what a normal-approximation interval
 * does, since that always returns something.
 */

export interface CalibrationPoint {
  n_active: number;
  lead: number;
  rest: number;
  measured: number;
}

/**
 * The surrogate's functional form, in one place.
 *
 * Mirrors `pspe.plan.surrogate.saturating`. Only what is stacked on top of the
 * largest single contribution saturates, which keeps single measures exact and
 * the model continuous where it used to jump. This had three copies in the
 * portal and three in the framework; six copies of a formula behind a displayed
 * guarantee is six chances to disagree, and a parity check can only hold one
 * against one. Checked by `portal/scripts/verify-margin-parity.mjs`.
 */
export function saturating(lead: number, rest: number, S: number): number {
  return lead + S * Math.tanh(rest / S);
}

/** The surrogate's prediction for a stored calibration combination. */
export function predicted(p: CalibrationPoint, S: number): number {
  return saturating(p.lead, p.rest, S);
}

/**
 * Smallest number of held-out runs that supports a given confidence.
 *
 * Searches the rank condition `ceil((n+1)·p) <= n` rather than rearranging it
 * to `n >= 1/delta - 1`, and works from the coverage level `p = conf/100`
 * rather than from `delta = 1 - conf/100`. Both changes are load-bearing, and a
 * parity check against `pspe/plan/margins.py` is what found it: `1 - 90/100`
 * is 0.09999999999999998 in binary floating point, so `ceil(1/delta - 1)`
 * returned 10 where 9 is sufficient, and the rank computed from the same
 * drifted delta clamped at n = 9. The portal therefore declined a 90% band on
 * exactly nine calibration runs — the case the published results sit on.
 *
 * Refusing too often is the benign direction, but it is still wrong, and the
 * framework had the opposite error at the same boundary.
 */
export function runsNeeded(confidencePct: number): number {
  const p = confidencePct / 100;
  if (!(p > 0) || p >= 1) return Infinity;
  for (let n = 2; n < 10000; n++) if (Math.ceil((n + 1) * p) <= n) return n;
  return Infinity;
}

/** Highest confidence the stored runs can honestly support. */
export function maxConfidencePct(n: number): number {
  if (n < 2) return 0;
  return (1 - 1 / (n + 1)) * 100;
}

export interface Band {
  band: number;             // points of reduction, one-sided
  attainable: boolean;
  needed: number;           // runs required for the requested confidence
  have: number;
}

/**
 * One-sided split-conformal band over |measured - predicted| at the requested
 * confidence. Returns attainable:false rather than a number when the stored
 * runs cannot carry the level.
 */
export function bandAt(
  calibration: CalibrationPoint[],
  S: number,
  confidencePct: number,
): Band {
  const have = calibration.length;
  const needed = runsNeeded(confidencePct);
  if (!Number.isFinite(needed) || have < needed) {
    return { band: 0, attainable: false, needed, have };
  }
  const residuals = calibration
    .map((p) => Math.abs(p.measured - predicted(p, S)))
    .sort((a, b) => a - b);
  // The conformal rank, computed from the coverage level directly. The guard
  // above already guarantees rank <= have, so there is no clamp: a clamp here
  // would mean returning the sample maximum, which is what the guarantee is
  // not. Working from `conf/100` rather than `1 - delta` avoids the rounding
  // that made this overshoot by one at 90% (see `runsNeeded`).
  const rank = Math.ceil((have + 1) * (confidencePct / 100));
  return { band: residuals[rank - 1], attainable: true, needed, have };
}
