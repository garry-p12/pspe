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

/** The surrogate's prediction for a stored calibration combination. */
export function predicted(p: CalibrationPoint, S: number): number {
  return p.lead + S * Math.tanh(p.rest / S);
}

/** Smallest number of held-out runs that supports a given confidence. */
export function runsNeeded(confidencePct: number): number {
  const delta = 1 - confidencePct / 100;
  if (delta <= 0) return Infinity;
  return Math.max(2, Math.ceil(1 / delta - 1));
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
  const delta = 1 - confidencePct / 100;
  // ceil((n+1)(1-delta)) is the conformal rank; clamp to the sample.
  const rank = Math.min(have, Math.ceil((have + 1) * (1 - delta)));
  return { band: residuals[rank - 1], attainable: true, needed, have };
}
