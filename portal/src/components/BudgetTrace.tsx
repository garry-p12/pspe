"use client";

/**
 * What more money buys, drawn.
 *
 * The budget slider used to move a number with no sense of what lay either side
 * of it. The interesting shape in this problem is that the curve FLATTENS: the
 * first few million buy most of what is available and the rest buys very
 * little. A table cannot show that and a sentence does not land; a curve does.
 *
 * Everything here comes from options already computed, so dragging the slider
 * costs nothing -- no solver call, no network.
 */

/** Only the two fields this needs, so it accepts the ops Option as-is. */
interface Priced { cost_aud: number; core_reduction_pct?: number }

export function BudgetTrace({
  options, budgetM, maxM = 60,
}: {
  options: Priced[];
  budgetM: number;
  maxM?: number;
}) {
  // Best reduction obtainable at each budget: the upper envelope of the options.
  const pts: { m: number; best: number }[] = [];
  for (let m = 2; m <= maxM; m += 1) {
    const affordable = options.filter((o) => o.cost_aud <= m * 1e6);
    const best = affordable.length
      ? Math.max(0, ...affordable.map((o) => o.core_reduction_pct ?? 0))
      : 0;
    pts.push({ m, best });
  }
  const top = Math.max(1, ...pts.map((p) => p.best));
  const W = 260, H = 54, padL = 2, padR = 2;
  const x = (m: number) => padL + ((m - 2) / (maxM - 2)) * (W - padL - padR);
  const y = (v: number) => H - 6 - (v / top) * (H - 12);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.m).toFixed(1)},${y(p.best).toFixed(1)}`).join(" ");
  const here = pts.find((p) => p.m === Math.round(budgetM)) ?? pts[pts.length - 1];

  // Where the curve stops paying: the first budget beyond which another A$5M
  // adds less than one point of reduction.
  let knee: number | null = null;
  for (const p of pts) {
    const later = pts.find((q) => q.m === p.m + 5);
    if (later && later.best - p.best < 1 && p.best > 0) { knee = p.m; break; }
  }

  return (
    <div className="mt-3">
      <p className="text-[12px] text-ink-faint">What more money buys</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full" role="img"
           aria-label="Flood reduction against budget">
        {knee !== null && (
          <line x1={x(knee)} x2={x(knee)} y1={4} y2={H - 4}
                stroke="var(--line)" strokeDasharray="2 3" />
        )}
        <path d={path} fill="none" stroke="var(--ink-faint)" strokeWidth={1.5} />
        <circle cx={x(here.m)} cy={y(here.best)} r={3.5} fill="var(--accent)" />
      </svg>
      <div className="flex justify-between text-[11px] text-ink-faint">
        <span>A$2M</span>
        <span className="text-ink">
          A${here.m}M buys {here.best.toFixed(0)}%
        </span>
        <span>A${maxM}M</span>
      </div>
      {knee !== null && (
        <p className="mt-1 text-[12px] leading-snug text-ink-faint">
          Past about <span className="text-ink">A${knee}M</span> the line goes
          flat — another A$5M adds less than one point. That is the point where
          extra money stops buying much.
        </p>
      )}
    </div>
  );
}
