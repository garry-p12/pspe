"use client";

/**
 * Can we give a guarantee here at all?
 *
 * The margin only means something when the river forecast is good enough that
 * the safety allowance still leaves room to build within. Past roughly 8-12%
 * forecast error the allowance eats the whole benefit -- measured, not assumed.
 *
 * This is worth surfacing because the alternative is a tool that promises a
 * guarantee everywhere and quietly delivers a meaningless one somewhere. Saying
 * "not here, and here is what would have to improve" is the more useful answer,
 * and it is the same discipline as refusing a confidence level we cannot back.
 */

export function ForecastCheck({ spreadPct }: { spreadPct: number | null }) {
  if (spreadPct === null) return null;
  // Below ~8% the allowance fits comfortably; past ~12% it does not. Between
  // them it depends on the site, so say so rather than pretending to a cliff.
  const good = spreadPct <= 8;
  const marginal = spreadPct > 8 && spreadPct <= 12;
  const tone = good ? "text-accent" : marginal ? "text-ink" : "text-ink";

  return (
    <div className="mt-3 rounded border border-line-soft bg-bg-raised p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] text-ink-faint">
          How well we know the river right now
        </span>
        <span className={`tnum text-[13px] font-semibold ${tone}`}>
          ±{spreadPct.toFixed(0)}%
        </span>
      </div>
      <p className="mt-1.5 text-[12px] leading-snug text-ink-faint">
        {good && (
          <>
            Good enough to put a guarantee on a plan. The river forecast is
            accurate to about {spreadPct.toFixed(0)}%, and the safety allowance
            still leaves room to build within.
          </>
        )}
        {marginal && (
          <>
            Borderline. At about {spreadPct.toFixed(0)}% forecast error the
            safety allowance starts to eat the benefit, so treat any guarantee
            here as indicative and verify before committing money.
          </>
        )}
        {!good && !marginal && (
          <>
            Too uncertain to guarantee. At about {spreadPct.toFixed(0)}% the
            safety allowance is larger than what a wall would buy, so we can
            still show you an estimate but not a floor. A river forecast good to
            roughly 10% would change that.
          </>
        )}
      </p>
    </div>
  );
}
