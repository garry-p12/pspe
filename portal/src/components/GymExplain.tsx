/**
 * The three panels under the scene: what the decision is, how every approach
 * scored, and how much of it is real.
 *
 * Everything here is rendered from the catalogue. No prose is composed in the
 * browser beyond joining a measured number to a noun, because a page that
 * paraphrases a finding is a second place for it to be wrong.
 *
 * Two choices worth stating, because both were the other way first:
 *
 * - **Every percentage is a share of doing nothing**, which is the rung the
 *   service anchors on too (`EnvSpec.harm_pct`). It used to be a share of the
 *   WORST rung, and on five of the fourteen that is a random policy -- so
 *   "100%" meant "the worst anyone happened to try", a denominator that moves
 *   when somebody adds a row.
 * - **Affordable and unaffordable are separate lists, not a column.** The
 *   approach with the best score is usually one you may not run; when it sits
 *   in the same list as the rest, sorted above them, the page reads as though
 *   it won.
 */

import clsx from "clsx";

export type Rung = {
  key: string; plain: string; return: number; cost: number;
  feasible: number; of: number; pct: number; ok: boolean; note?: string;
};
export type RealRow = { label: string; real: boolean; text: string };
export type PlainFields = {
  title: string; decision: string; score: string; budget: string;
  status: string; caveat: string; dynamics: string;
  when?: string; what?: string; real?: RealRow[];
};
export type Scored = { pct: number; ok: boolean; name: string; spend: string; mine?: boolean };

/** What a row's cost looks like to somebody who is not counting pump-hours. */
export function spendNote(cost: number, limit: number, feasible: number, of: number): string {
  if (cost <= 0) return "spends nothing";
  if (feasible === of) {
    const share = limit > 0 ? cost / limit : 0;
    if (share >= 0.95) return "uses all of it";
    if (share >= 0.6) return "uses most of it";
    if (share >= 0.25) return "uses some of it";
    return "uses very little";
  }
  if (feasible > 0) return `over in ${of - feasible} of ${of} runs`;
  const times = limit > 0 ? cost / limit : 0;
  if (times < 1.05) return "over the allowance";
  // 3.0x reads as a measurement to one decimal place; 3x reads as "three times".
  const n = times >= 10 ? Math.round(times) : Number(times.toFixed(1));
  return `${n}\u00d7 the allowance`;
}

/** The rung every percentage is measured against. Two environments call it
 *  "never" -- there is no barricade to lower, no order to give. */
const BASELINE_KEYS = new Set(["nothing", "never"]);
export const baselineName = (ladder: Rung[]) =>
  ladder.find((r) => BASELINE_KEYS.has(r.key))?.plain ?? null;

export function toScored(ladder: Rung[], limit: number): Scored[] {
  return ladder.map((r) => ({
    pct: r.pct, ok: r.ok, name: r.plain,
    spend: spendNote(r.cost, limit, r.feasible, r.of),
  }));
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3.5 first:mt-0">
      <p className="text-[11px] leading-snug text-ink-faint">{label}</p>
      <p className="mt-0.5 text-[13px] leading-relaxed text-ink">{children}</p>
    </div>
  );
}

/** One approach. The bar is the gist and is capped at the full width; the
 *  number beside it is the truth and is not, because a rung worse than doing
 *  nothing is a real result on two environments. */
function Row({ row }: { row: Scored }) {
  return (
    <li className={clsx(
      "flex items-center gap-3 py-[7px]",
      row.mine && "-mx-2 rounded bg-accent-soft px-2",
    )}>
      <span className="min-w-0 flex-1">
        <span className={clsx("block text-[12.5px] leading-snug",
                              row.mine ? "font-semibold text-ink" : "text-ink")}>
          {row.name}
        </span>
        {row.mine && (
          <span className="mt-0.5 inline-block rounded-sm bg-ink px-1 py-px text-[9.5px]
                           font-semibold uppercase tracking-wide text-bg">
            your run
          </span>
        )}
      </span>
      <span className="h-[7px] w-[88px] shrink-0 rounded-sm bg-bg-inset" aria-hidden>
        <span className={clsx("block h-[7px] rounded-sm", row.ok ? "bg-ink" : "bg-bad/55")}
              style={{ width: `${Math.max(2, Math.min(row.pct, 100))}%` }} />
      </span>
      <span className="w-[42px] shrink-0 text-right font-mono text-[12.5px] tabular-nums text-ink">
        {row.pct}%
      </span>
      <span className={clsx("w-[112px] shrink-0 text-right text-[11px] leading-snug",
                            row.ok ? "text-ink-faint" : "text-bad")}>
        {row.spend}
      </span>
    </li>
  );
}

export function GymExplain({
  plain, ladder, costLimit, result,
}: {
  plain: PlainFields;
  ladder: Rung[];
  costLimit: number;
  result: { damage: number; feasible: boolean } | null;
}) {
  const rows = toScored(ladder, costLimit);
  if (result) {
    rows.push({
      pct: result.damage, ok: result.feasible, name: "your run", mine: true,
      spend: result.feasible ? "within the allowance" : "over the allowance",
    });
  }
  const by = (a: Scored, b: Scored) => a.pct - b.pct;
  const afford = rows.filter((r) => r.ok).sort(by);
  const cannot = rows.filter((r) => !r.ok).sort(by);
  const best = afford[0] ?? null;
  // On FloodEvacuate-v0 the best affordable rung IS doing nothing, and saying
  // it "leaves 100% of the harm of doing nothing" is true and circular. That
  // environment's whole finding is this case, so it gets its own sentence.
  const bestIsNothing = best != null && best.name === baselineName(ladder);
  const real = plain.real ?? [];

  return (
    <div className="mt-6 grid gap-x-8 gap-y-7 border-t border-line pt-6
                    lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.35fr)_minmax(0,0.9fr)]">

      {/* ---- how this decision works -------------------------------- */}
      <section>
        <h2 className="text-[10.5px] font-semibold uppercase tracking-wide text-ink">
          How this decision works
        </h2>
        <div className="mt-3">
          {plain.when && <Field label="When you decide">{plain.when}</Field>}
          {plain.what && <Field label="What you decide">{plain.what}</Field>}
          <Field label="Harm, the score (lower is better)">{plain.score}</Field>
          <Field label="Allowance, for the whole run">{plain.budget}</Field>
        </div>
        <p className="mt-4 rounded-lg bg-bg-inset px-3 py-2.5 text-[11.5px] leading-relaxed text-ink-mute">
          Every run ends with two numbers: the harm done, and how much of the
          allowance was used. An approach only counts if it stays within the
          allowance — otherwise you could look good just by spending without
          limit.
        </p>
      </section>

      {/* ---- results ------------------------------------------------- */}
      <section className="min-w-0">
        <h2 className="text-[10.5px] font-semibold uppercase tracking-wide text-ink">
          Results
        </h2>

        <div className="mt-3 rounded-xl border border-ink px-4 py-3">
          {best ? (
            <>
              <p className="text-[11px] text-ink-faint">Best approach within the allowance</p>
              <p className="mt-0.5 text-[16px] font-semibold leading-snug text-ink">{best.name}</p>
              <p className="mt-1 text-[12.5px] leading-relaxed text-ink-mute">
                {bestIsNothing ? (
                  <>
                    Not acting is the best anyone has managed here. Every approach
                    that does less harm costs more than you are allowed to spend.
                  </>
                ) : (
                  <>
                    It leaves <strong className="font-semibold text-ink">{best.pct}%</strong> of
                    the harm of doing nothing, and {best.spend}.
                  </>
                )}
              </p>
            </>
          ) : (
            <>
              <p className="text-[11px] text-ink-faint">Best approach within the allowance</p>
              <p className="mt-0.5 text-[16px] font-semibold leading-snug text-bad">
                Nothing stayed within it
              </p>
              <p className="mt-1 text-[12.5px] leading-relaxed text-ink-mute">
                Every approach measured here goes over the allowance in at least
                one run — including doing nothing, because the allowance can be
                spent by the world rather than by you.
              </p>
            </>
          )}
        </div>

        <div className="mt-4 flex items-baseline gap-3 border-b border-line pb-1
                        text-[11px] text-ink-faint">
          <span className="min-w-0 flex-1">Harm, as a share of doing nothing</span>
          <span className="w-[112px] shrink-0 text-right">allowance used</span>
        </div>

        {afford.length > 0 && (
          <>
            <p className="mt-3 text-[11.5px] font-semibold text-ink">
              Stayed within the allowance
            </p>
            <ul className="mt-0.5 divide-y divide-line-soft">
              {afford.map((r) => <Row key={`${r.name}-ok`} row={r} />)}
            </ul>
          </>
        )}

        {cannot.length > 0 && (
          <>
            <p className="mt-4 text-[11.5px] font-semibold text-bad">
              Went over the allowance, so they don&rsquo;t count
            </p>
            <ul className="mt-0.5 divide-y divide-line-soft">
              {cannot.map((r) => <Row key={`${r.name}-no`} row={r} />)}
            </ul>
          </>
        )}

        <p className="mt-3 text-[11px] leading-relaxed text-ink-faint">
          100% = all the harm of doing nothing, over the whole run. Lower is
          better, and above 100% means it did worse than not acting at all.
        </p>
        <p className="mt-1.5 text-[11px] leading-relaxed text-ink-faint">
          {result
            ? "Your run is one go. Every other line is twelve, so treat the placing as a sighting shot."
            : "Finish your run and it will be placed in one of these two lists."}
        </p>
      </section>

      {/* ---- what we found, and how real it is ----------------------- */}
      <section>
        <h2 className="text-[10.5px] font-semibold uppercase tracking-wide text-ink">
          What we found
        </h2>
        <p className="mt-3 text-[13px] leading-relaxed text-ink">{plain.status}</p>

        <h3 className="mt-5 text-[10.5px] font-semibold uppercase tracking-wide text-ink">
          How real is this?
        </h3>
        {real.length > 0 ? (
          <ul className="mt-2.5 space-y-2.5">
            {real.map((r) => (
              <li key={r.label}>
                <span className={clsx(
                  "inline-block rounded-sm px-1.5 py-px text-[9.5px] font-semibold uppercase tracking-wide",
                  r.real ? "bg-ink text-bg" : "border border-line text-ink-mute",
                )}>
                  {r.real ? "Real" : "Made up"}
                </span>
                <span className="mt-1 block text-[12px] leading-relaxed text-ink-mute">
                  <strong className="font-semibold text-ink">{r.label}:</strong> {r.text}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-2.5 text-[12px] leading-relaxed text-ink-mute">{plain.caveat}</p>
        )}
      </section>
    </div>
  );
}
