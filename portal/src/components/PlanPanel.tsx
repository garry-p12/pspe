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
  confidence_ceiling_pct?: number;
  runs_needed?: number;
  runs_have?: number;
  refusal?: string | null;
  steps: PlanStep[];
  attribution: Attribution[];
  harmful: { site: number; alone_pct: number }[];
  verify: { required: boolean; why: string };
  method: string;
  /** The band conditioned on the stratum this plan lands in, which is the only
   *  one that describes THIS plan. `band_pct` above is pooled over every
   *  calibration run and is wrong in both directions: too wide for a two-site
   *  plan and unsupported for anything larger. */
  conditional?: {
    band_pct: number | null;
    attainable: boolean;
    n_active: number;
    exact: boolean;
    guaranteed_pct: number | null;
    runs_in_stratum: number;
    runs_needed: number;
    why: string;
  };
  /** The plan with the best GUARANTEED floor rather than the best estimate.
   *  Differs from the recommendation whenever conditioning bites. */
  best_guaranteed?: {
    heights: number[];
    cost_aud: number;
    reduction_pct: number;
    guaranteed_pct: number;
    band_pct: number;
    n_active: number;
    exact: boolean;
    floor_gain_vs_greedy: number | null;
    estimate_given_up: number;
    why: string;
  } | null;
}

/** 90 -> "9 times in 10". Percentages mean little to most readers; counts do. */
function timesIn(pct: number): string {
  const nice: Record<number, string> = {
    50: "half the time", 60: "3 times in 5", 70: "7 times in 10",
    75: "3 times in 4", 80: "4 times in 5", 90: "9 times in 10",
    95: "19 times in 20", 99: "99 times in 100",
  };
  if (nice[pct]) return nice[pct];
  if (pct >= 99) return "99 times in 100";
  return `${Math.round(pct)} times in 100`;
}

export function PlanPanel({
  budgetM, crestLengths, eventScale, onPlan, onShowHarm, trigger = 0, onBusy,
}: {
  budgetM: number;
  crestLengths: number[];
  eventScale: number;
  onPlan?: (heights: number[]) => void;
  /** Show on the map where a rejected wall would push the water instead. */
  onShowHarm?: (site: number) => void;
  /** Bumped by whoever owns the action button; the panel shows results only. */
  trigger?: number;
  onBusy?: (v: boolean) => void;
}) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  /** How sure the user has asked to be. Drives the margin, not the plan. */
  const [certainty, setCertainty] = useState(90);

  async function run(conf = certainty) {
    setBusy(true); setErr(null);
    try {
      const r = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budget_aud: budgetM * 1e6,
          event_scale: eventScale,
          crest_lengths_m: crestLengths,
          confidence_pct: conf,
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

  // Moving the certainty slider re-asks for the margin at the new level. The
  // plan itself does not change -- only what we are willing to stand behind --
  // so this is debounced and deliberately does not show the busy spinner.
  useEffect(() => {
    if (!plan) return;
    const t = setTimeout(() => run(certainty), 200);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [certainty]);

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
                  <span className="text-[12px] text-ink-faint">Our best estimate</span>
                  <span className="tnum text-[18px] font-semibold text-ink">
                    {plan.reduction_pct.toFixed(0)}%
                  </span>
                </div>
                <div className="mt-1.5 flex items-baseline justify-between">
                  <span className="text-[12px] text-ink-faint">
                    {plan.band_attainable
                      ? `What we'll stand behind, ${timesIn(certainty)}`
                      : "What we'll stand behind"}
                  </span>
                  <span className="tnum text-[18px] font-semibold text-accent">
                    {plan.band_attainable ? `${plan.guaranteed_pct.toFixed(0)}%` : "—"}
                  </span>
                </div>

                {/* How sure do you need to be? The margin is recomputed at the
                    level the user picks, because a stated failure rate that the
                    user cannot state is just a tuned constant. */}
                <div className="mt-3 border-t border-line-soft pt-3">
                  <div className="flex items-baseline justify-between">
                    <label htmlFor="certainty" className="text-[12px] text-ink">
                      How sure do you need to be?
                    </label>
                    <span className="tnum text-[12px] text-ink-faint">
                      {timesIn(certainty)}
                    </span>
                  </div>
                  <input
                    id="certainty"
                    type="range"
                    min={50}
                    max={99}
                    step={1}
                    value={certainty}
                    onChange={(e) => setCertainty(Number(e.target.value))}
                    className="mt-2 w-full"
                  />
                  <div className="mt-1 flex justify-between text-[11px] text-ink-faint">
                    <span>a coin flip</span>
                    <span>almost always</span>
                  </div>
                </div>

                <p className="mt-2 text-[12px] leading-snug text-ink-faint">
                  {plan.band_attainable
                    ? `Being more sure costs you: the figure we'll stand behind drops as you raise the bar. It sits ${plan.band_pct.toFixed(0)} points below our estimate here, measured against ${plan.n_calibration_runs} past floods we tested on. Write the business case against the lower number.`
                    : (plan.refusal ?? plan.band_note)}
                </p>

                {/* The band for THIS plan, not pooled over all of them.
                    Surrogate error grows with the number of sites a plan
                    activates, so one number for every plan is too wide for a
                    small plan and unsupported for a large one. Showing the
                    pooled figure was not a rounding issue: it let the search
                    pick a plan the evidence could not cover. */}
                {plan.conditional && (
                  <div className="mt-3 border-t border-line-soft pt-3">
                    <div className="flex items-baseline justify-between">
                      <span className="text-[12px] text-ink">
                        Checked against {plan.conditional.n_active}-site floods only
                      </span>
                      <span className="tnum text-[13px] font-semibold text-accent">
                        {plan.conditional.exact
                          ? "exact"
                          : plan.conditional.attainable
                            ? `${plan.conditional.guaranteed_pct!.toFixed(0)}%`
                            : "can't say"}
                      </span>
                    </div>
                    <p className="mt-1 text-[12px] leading-snug text-ink-faint">
                      {plan.conditional.why}
                    </p>
                    {!plan.conditional.attainable && !plan.conditional.exact && (
                      <p className="mt-1 text-[12px] leading-snug text-warn">
                        We have {plan.conditional.runs_in_stratum} past flood
                        {plan.conditional.runs_in_stratum === 1 ? "" : "s"} with
                        this many sites built and need{" "}
                        {plan.conditional.runs_needed}. The figure above it is the
                        pooled one and does not cover this plan.
                      </p>
                    )}
                  </div>
                )}

                {/* The plan that maximises what we can PROMISE, which is not the
                    plan that maximises what we predict. A pooled band is a
                    constant and cannot change this choice; a band conditioned on
                    the plan can, and does. */}
                {plan.best_guaranteed &&
                  plan.best_guaranteed.floor_gain_vs_greedy !== null &&
                  plan.best_guaranteed.floor_gain_vs_greedy > 0.5 && (
                  <div className="mt-3 rounded border border-accent/40 bg-accent/5 p-2.5">
                    <p className="text-[12px] font-semibold text-ink">
                      A plan we can promise more for
                    </p>
                    <p className="mt-1 text-[12px] leading-snug text-ink-faint">
                      {plan.best_guaranteed.n_active === 1
                        ? "One site at full height"
                        : `${plan.best_guaranteed.n_active} sites`}
                      , A${(plan.best_guaranteed.cost_aud / 1e6).toFixed(1)}M:
                      we'd estimate{" "}
                      <span className="tnum">
                        {plan.best_guaranteed.reduction_pct.toFixed(0)}%
                      </span>{" "}
                      — <span className="tnum">
                        {plan.best_guaranteed.estimate_given_up.toFixed(0)} points
                      </span>{" "}
                      less than the plan above — but stand behind{" "}
                      <span className="tnum font-semibold text-accent">
                        {plan.best_guaranteed.guaranteed_pct.toFixed(0)}%
                      </span>
                      , which is{" "}
                      <span className="tnum">
                        {plan.best_guaranteed.floor_gain_vs_greedy.toFixed(0)} points
                      </span>{" "}
                      more.
                      {plan.best_guaranteed.exact &&
                        " Its prediction is a flood we actually ran, not an estimate between two."}
                    </p>
                  </div>
                )}
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
                <div className="rounded border border-line-soft bg-bg-raised p-3">
                  <p className="text-[12px] font-medium text-ink">
                    Some walls would make it worse
                  </p>
                  <p className="mt-1 text-[12px] leading-snug text-ink-faint">
                    A wall does not remove water, it moves it. Build in the wrong
                    place and the water backs up somewhere that matters more.
                    These were tested and rejected for that reason:
                  </p>
                  <ul className="mt-2 space-y-1">
                    {plan.harmful.map((h) => (
                      <li key={h.site}>
                        <button
                          onClick={() => onShowHarm?.(h.site)}
                          className="flex w-full items-baseline justify-between rounded px-1 py-0.5 text-left hover:bg-bg"
                        >
                          <span className="text-[13px] text-ink underline underline-offset-2">
                            Wall at site {h.site}
                          </span>
                          <span className="tnum text-[13px] text-ink">
                            {h.alone_pct > 0 ? "+" : ""}{(-h.alone_pct).toFixed(0)}% worse
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[11px] leading-snug text-ink-faint">
                    Tap one to see on the map where that water would go instead.
                  </p>
                </div>
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
