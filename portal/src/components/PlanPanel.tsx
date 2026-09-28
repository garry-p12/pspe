"use client";

import { useEffect, useState } from "react";

/**
 * Spend a budget, under a guarantee.
 *
 * The rest of the interface ranks options someone already solved. This asks the
 * question an operator actually has -- "I have this much money, what should I
 * build?" -- and answers it by searching, then says what it is willing to stand
 * behind rather than only its best guess.
 *
 * Two numbers are shown deliberately. The estimate is the surrogate's answer.
 * The guarantee subtracts a conformal margin calibrated on held-out solver
 * runs, so it is the figure that survives the model being wrong as often as it
 * usually is. A business case should be written against the second one.
 */

interface PlanStep { site: number; to_m: number; effect: number; spent: number }
interface Attribution {
  site: number; height_m: number; marginal_pct: number;
  alone_pct: number; overlap_pct: number; cost_aud: number;
}
interface Plan {
  heights: number[];
  cost_aud: number;
  budget_aud: number;
  reduction_pct: number;
  guaranteed_pct: number;
  band_pct: number;
  confidence_pct: number | null;
  band_attainable: boolean;
  band_note: string | null;
  n_calibration_runs: number;
  steps: PlanStep[];
  attribution: Attribution[];
  harmful: { site: number; alone_pct: number }[];
  verify: { required: boolean; why: string };
  method: string;
}

export function PlanPanel({
  budgetM, crestLengths, eventScale, onPlan, trigger = 0, onBusy,
}: {
  budgetM: number;
  crestLengths: number[];
  eventScale: number;
  onPlan?: (heights: number[]) => void;
  /** Bumped by whoever owns the action button; the panel shows results only. */
  trigger?: number;
  onBusy?: (v: boolean) => void;
}) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true); setErr(null);
    try {
      const r = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budget_aud: budgetM * 1e6,
          event_scale: eventScale,
          crest_lengths_m: crestLengths,
        }),
      });
      if (!r.ok) throw new Error(`planner returned ${r.status}`);
      const p: Plan = await r.json();
      setPlan(p);
      onPlan?.(p.heights);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "could not plan");
    } finally {
      setBusy(false);
      onBusy?.(false);
    }
  }

  // The button that used to live here now sits in the section's footer, where
  // it stays reachable without scrolling past twelve options to find it.
  useEffect(() => {
    if (trigger > 0) { onBusy?.(true); run(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trigger]);

  const built = plan?.heights
    .map((h, i) => ({ h, i }))
    .filter((x) => x.h > 0) ?? [];

  return (
    <div className={plan || err || busy ? "mt-5" : "hidden"}>
      <h3 className="eyebrow mb-2">Best plan for this budget</h3>
      {busy && <div className="h-20 animate-pulse rounded-lg bg-bg-inset" />}
      {err && <p className="text-[13px] text-ink-mute">{err}</p>}

      {plan && (
        <div className="mt-3 space-y-3">
          {built.length === 0 ? (
            <p className="text-[13px] text-ink-mute">
              Nothing affordable at this budget improves flooding here. Raising
              the budget is the next thing to try.
            </p>
          ) : (
            <>
              <div>
                <p className="text-[12px] text-ink-faint">Build</p>
                <ul className="mt-1 space-y-0.5">
                  {built.map(({ h, i }) => (
                    <li key={i} className="flex justify-between text-[13px]">
                      <span className="text-ink">Levee at site {i}</span>
                      <span className="tnum text-ink-mute">{h.toFixed(1)} m</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-1 tnum text-[12px] text-ink-faint">
                  A${(plan.cost_aud / 1e6).toFixed(1)}M of A$
                  {(plan.budget_aud / 1e6).toFixed(0)}M
                </p>
              </div>

              <div className="rounded border border-line-soft bg-bg-raised p-3">
                <div className="flex items-baseline justify-between">
                  <span className="text-[12px] text-ink-faint">Estimated</span>
                  <span className="tnum text-[18px] font-semibold text-ink">
                    {plan.reduction_pct.toFixed(0)}%
                  </span>
                </div>
                <div className="mt-1.5 flex items-baseline justify-between">
                  <span className="text-[12px] text-ink-faint">
                    At least, {plan.confidence_pct ?? 90}% of the time
                  </span>
                  <span className="tnum text-[18px] font-semibold text-accent">
                    {plan.guaranteed_pct.toFixed(0)}%
                  </span>
                </div>
                <p className="mt-2 text-[12px] leading-snug text-ink-faint">
                  {plan.band_attainable
                    ? `Margin ±${plan.band_pct.toFixed(0)} points, from ${plan.n_calibration_runs} held-out solver runs. Write the business case against the lower figure.`
                    : plan.band_note}
                </p>
              </div>

              {plan.attribution?.length > 1 && (
                <div>
                  <p className="text-[12px] text-ink-faint">
                    What each measure adds, in this plan
                  </p>
                  <ul className="mt-1 space-y-1">
                    {plan.attribution.map((a) => (
                      <li key={a.site} className="text-[13px]">
                        <div className="flex justify-between">
                          <span className="text-ink">
                            Site {a.site} ({a.height_m.toFixed(1)} m)
                          </span>
                          <span className="tnum font-semibold text-ink">
                            +{a.marginal_pct.toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-[12px] text-ink-faint">
                          {a.alone_pct.toFixed(0)}% on its own; {a.overlap_pct.toFixed(0)}
                          {" "}points of that is already covered by the rest of the plan
                        </p>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-1.5 text-[12px] leading-snug text-ink-faint">
                    Removing a measure from this plan costs what is shown beside
                    it, which is not the same as what it would achieve alone.
                  </p>
                </div>
              )}

              {plan.harmful?.length > 0 && (
                <p className="text-[12px] leading-snug text-ink-faint">
                  Not selected:{" "}
                  {plan.harmful.map((h) => `site ${h.site} (${h.alone_pct.toFixed(0)}%)`).join(", ")}
                  {" "}— the model measures these as deepening flooding at the
                  settlement rather than reducing it.
                </p>
              )}

              {plan.verify.required && (
                <p className="text-[12px] leading-snug text-ink-faint">
                  {plan.verify.why}
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
