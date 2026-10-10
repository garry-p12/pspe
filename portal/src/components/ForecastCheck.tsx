"use client";

import { Explain } from "./Explain";
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
      <div className="flex items-baseline justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[12px] text-ink-faint">
          River forecast spread
          <Explain title="What this spread means">
            <p>
              It is the day-to-day spread of the seven daily discharge
              maxima in this week&apos;s forecast, as a percentage of the
              peak. It is not a validated forecast error: there is no
              ensemble and no hindcast behind it.
            </p>
            <p>
              It is used as a gate. A conformal safety allowance has to fit
              inside the gain a wall buys; below about 8% it does, past
              about 12% it does not, and in between it depends on the site.
            </p>
          </Explain>
        </span>
        <span className={`tnum text-[13px] font-semibold ${tone}`}>
          ±{spreadPct.toFixed(0)}%
        </span>
      </div>
      {/* One line, and it is the one that changes what the reader does:
          whether a guarantee is on the table at all. The three paragraphs
          of Plan-tab vocabulary that used to argue the case are behind the
          (i) above. */}
      <p className="mt-1 text-[12px] leading-snug text-ink-faint">
        {good ? "Tight enough to guarantee a plan against."
          : marginal ? "Borderline — treat any guarantee as indicative."
            : "Too wide to guarantee a plan; estimates only."}
      </p>
    </div>
  );
}
