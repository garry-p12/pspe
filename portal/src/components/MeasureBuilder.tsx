"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface Estimate {
  reduction_pct: number;
  linear_sum_pct: number;
  band_pct: number;
  best_case_pct: number;
  worst_case_pct: number;
  makes_worse: boolean;
  exact: boolean;
  saturating: boolean;
  method: string;
}

/**
 * Free-form plan builder.
 *
 * Each measure has a height a planner sets. The estimate comes back from the
 * surrogate in milliseconds, as a RANGE: a single measure is a direct solver
 * result and is stated exactly, while combinations carry the interaction the
 * fit cannot pin down. Nothing here is presented with more precision than the
 * model can support, and an adopted plan is queued for full verification.
 */
export function MeasureBuilder({
  siteCount,
  siteElev,
  heights,
  onHeights,
  eventScale,
  maxHeight = 3,
  unitCostPerMPerM = 2600,
  crestLengths,
  budgetM,
}: {
  siteCount: number;
  siteElev: number[];
  heights: number[];
  onHeights: (h: number[]) => void;
  eventScale: number;
  maxHeight?: number;
  unitCostPerMPerM?: number;
  crestLengths: number[];
  budgetM: number;
}) {
  const [est, setEst] = useState<Estimate | null>(null);
  const [busy, setBusy] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const seq = useRef(0);

  useEffect(() => {
    const mine = ++seq.current;
    setBusy(true);
    fetch("/api/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heights, event_scale: eventScale }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => { if (mine === seq.current) { setEst(d); setUnavailable(false); } })
      .catch(() => { if (mine === seq.current) setUnavailable(true); })
      .finally(() => { if (mine === seq.current) setBusy(false); });
  }, [heights, eventScale]);

  const set = useCallback(
    (i: number, v: number) => {
      const next = [...heights];
      next[i] = v;
      onHeights(next);
    },
    [heights, onHeights],
  );

  const cost = heights.reduce(
    (s, h, i) => s + (h > 0 ? (crestLengths[i] ?? 0) * h * unitCostPerMPerM : 0),
    0,
  );
  const overBudget = cost > budgetM * 1e6;
  const active = heights.filter((h) => h > 0).length;

  return (
    <div className="border-b border-line-soft p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <p className="eyebrow">Build a plan</p>
        {busy && <span className="text-[10px] text-ink-faint">estimating…</span>}
      </div>

      <ul className="flex flex-col gap-2">
        {Array.from({ length: siteCount }, (_, i) => (
          <li key={i} className="flex items-center gap-2.5">
            <span className="tnum w-6 shrink-0 text-[11px] text-ink-faint">M{i}</span>
            <input
              type="range" min={0} max={maxHeight} step={0.5}
              value={heights[i] ?? 0}
              onChange={(e) => set(i, Number(e.target.value))}
              className="min-w-0 flex-1 accent-[var(--accent)]"
              aria-label={`Measure ${i} height`}
            />
            <span className="tnum w-12 shrink-0 text-right text-[11px] text-ink-mute">
              {(heights[i] ?? 0).toFixed(1)} m
            </span>
          </li>
        ))}
      </ul>

      <div className="mt-3 flex items-baseline justify-between border-t border-line-soft pt-3">
        <span className="eyebrow">Cost</span>
        <span className={`tnum text-[13px] font-semibold ${overBudget ? "text-bad" : "text-ink"}`}>
          A${(cost / 1e6).toFixed(1)}M
          {overBudget && <span className="ml-1.5 text-[10px] font-normal">over budget</span>}
        </span>
      </div>

      {unavailable && (
        <p className="mt-3 text-[11.5px] leading-relaxed text-ink-faint">
          Estimator not fitted yet — it calibrates against the hydrodynamic runs
          once they complete.
        </p>
      )}

      {est && !unavailable && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          {active === 0 ? (
            <p className="text-[12px] text-ink-mute">
              No measures selected. Choose heights above to see the effect.
            </p>
          ) : est.makes_worse ? (
            <>
              <p className="text-[12.5px] font-semibold text-bad">
                Deepens flooding by {Math.abs(est.reduction_pct).toFixed(0)}%
              </p>
              <p className="mt-1 text-[11.5px] leading-relaxed text-ink-mute">
                This combination holds water in rather than keeping it out.
              </p>
            </>
          ) : (
            <>
              <p className="text-[12.5px] font-semibold text-ok">
                Cuts flooding at the settlement by{" "}
                {est.reduction_pct.toFixed(0)}%
              </p>
              {est.band_pct > 0.5 && (
                <p className="tnum mt-1 text-[11px] text-ink-mute">
                  range {Math.max(0, est.worst_case_pct).toFixed(0)}–
                  {est.best_case_pct.toFixed(0)}%
                </p>
              )}
              {est.saturating && (
                <p className="mt-1.5 text-[11px] leading-relaxed text-ink-faint">
                  Measures overlap: separately they would give{" "}
                  {est.linear_sum_pct.toFixed(0)}%, together they do not add up.
                </p>
              )}
            </>
          )}
          <p className="mt-2 border-t border-line-soft pt-2 text-[10.5px] leading-relaxed text-ink-faint">
            {est.exact
              ? "Direct hydrodynamic result for this measure."
              : "Fast estimate. Verify with a full model run before adopting."}
          </p>
        </div>
      )}
    </div>
  );
}
