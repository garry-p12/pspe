"use client";

import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";

/**
 * Four decisions, and how well each way of making them did.
 *
 * The first version of this page was written in the register of the paper --
 * "incumbent", "feasible 12/12", "return -4862.70", "episode cost" -- which
 * made an honest page unreadable. The words here come from the `plain` block
 * the service sends, which lives beside the technical wording in
 * `pspe/envs/catalog.py`, so the page is plain without holding a second,
 * driftable copy of any claim.
 *
 * Two translations do the most work:
 *
 * - **Harm is shown against doing nothing**, not as the raw score. "-4862.70"
 *   means nothing to a reader; "as bad as not acting at all" does. Doing
 *   nothing is the worst row on all four environments, so it is a safe 100%.
 * - **Spending is shown against what was allowed**, as "all of it" or "34x
 *   over", not as a number beside a limit. Going over is the single most
 *   important fact about a row and it should not need arithmetic.
 *
 * What is NOT simplified away: every row that spent more than it was allowed
 * says so, in red, including the rows that scored best. On three of these
 * four the best-scoring approach is one nobody could afford, and a page that
 * quietly dropped that column would be advertising rather than reporting.
 */

type Score = { return: number; cost: number; feasible: number; of: number };
type Rung = Score & { key: string; policy: string; plain: string; note: string };
type Plain = {
  title: string; decision: string; score: string;
  budget: string; status: string; caveat: string; dynamics: string;
};
type Env = {
  id: string;
  plain: Plain;
  ladder: Rung[];
  cost_limit: number;
  needs: string;
  available?: boolean;
  reason?: string;
  observation_space?: string;
  action_space?: string;
  caveat: string;
};
type Listing = {
  environments: Env[];
  plain_headline: string;
  plain_how_to_read: string;
  measured_on: string;
  reproduce: string;
};
type StepRow = { step: number; reward: number; cost: number; episode_cost: number };
type Rollout = {
  steps: StepRow[]; total_reward: number; total_cost: number;
  cost_limit: number; feasible: boolean;
};

/**
 * "all of it", "34x over", "over in 5 of 12 runs" — never a number beside a
 * limit, and never flattened to a yes/no.
 *
 * The middle case is the one worth getting right. Two of the evacuation
 * approaches stay inside the budget in most runs and blow past it in the
 * rest, and calling that simply "over the limit" is as misleading as calling
 * it affordable: it is an approach you could usually but not always pay for,
 * which is the actual state of affairs.
 */
function spendWords(r: Rung, limit: number): string {
  if (r.cost <= 0) return "nothing spent";
  const share = r.cost / limit;
  const over = r.of - r.feasible;
  if (over === 0) {
    if (share > 0.95) return "all of it";
    if (share > 0.6) return "most of it";
    if (share > 0.25) return "part of it";
    return "barely any";
  }
  if (over < r.of) return `over in ${over} of ${r.of} runs`;
  return share >= 1.2
    ? `${share < 10 ? share.toFixed(share < 3 ? 1 : 0) : Math.round(share)}x over`
    : "just over";
}

/**
 * One group of approaches, best first, under a heading that says whether you
 * could pay for them.
 *
 * The split is the page. Sorting everything into one list by score buries the
 * finding -- that on three of these four the best-scoring approach is one
 * nobody can afford -- in an ordering the reader has to reverse-engineer.
 * Two headed groups state it.
 */
function Group({
  rows, limit, title, note, best, worst,
}: {
  rows: Rung[]; limit: number; title: string; note: string;
  best: string | null; worst: number;
}) {
  if (!rows.length) return null;
  return (
    <div className="mt-3.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-ink">
          {title}
        </h3>
        <p className="text-[11.5px] text-ink-faint">{note}</p>
      </div>

      <div className="mt-1.5 grid grid-cols-[minmax(0,1fr)_42px_minmax(50px,0.7fr)_112px]
                      items-center gap-x-3 border-b border-line pb-1
                      text-[10.5px] uppercase tracking-wide text-ink-faint">
        <span>what you do</span>
        <span className="text-right">damage</span>
        <span aria-hidden />
        <span className="text-right">what it spent</span>
      </div>

      <ul>
        {rows.map((r) => {
          const pct = Math.round((r.return / worst) * 100);
          const isBest = best === r.key;
          const ok = r.feasible === r.of;
          return (
            <li
              key={r.key}
              className={clsx(
                "grid grid-cols-[minmax(0,1fr)_42px_minmax(50px,0.7fr)_112px] items-center",
                "gap-x-3 border-b border-line-soft py-1.5 last:border-0",
              )}
            >
              <span className={clsx("truncate text-[12.5px]", isBest ? "font-semibold text-ink" : "text-ink-mute")}>
                {isBest && <span className="mr-1 text-ink" aria-hidden>▸</span>}
                {r.plain}
              </span>
              <span className="text-right font-mono text-[12px] tabular-nums text-ink">
                {pct}%
              </span>
              <span className="h-2 rounded-sm bg-bg-inset" aria-hidden>
                <span
                  className={clsx("block h-2 rounded-sm", ok ? "bg-sev-mid" : "bg-bad/40")}
                  style={{ width: `${Math.max(pct, 2)}%` }}
                />
              </span>
              <span
                className={clsx(
                  "text-right text-[11.5px]",
                  ok ? "text-ink-faint" : "font-semibold text-bad",
                )}
              >
                {spendWords(r, limit)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Ladder({ env }: { env: Env }) {
  const worst = Math.min(...env.ladder.map((r) => r.return));
  const byBest = (a: Rung, b: Rung) => b.return - a.return;
  const afford = env.ladder.filter((r) => r.feasible === r.of).sort(byBest);
  const cannot = env.ladder.filter((r) => r.feasible !== r.of).sort(byBest);
  const best = afford.length ? afford[0].key : null;

  // What the budget is costing you, in the page's own unit. Only the BEST of
  // the unaffordable rows is compared: the heading used to read "scores
  // better, but costs more than you have", which was false for two of the
  // evacuation rows -- they cost more AND score exactly what doing nothing
  // scores. The group is defined by its price, so the heading says price and
  // the note says what the best of them would buy.
  const gap = cannot.length && afford.length
    ? Math.round((afford[0].return - cannot[0].return) / worst * 100)
    : 0;

  return (
    <>
      <Group
        rows={afford} limit={env.cost_limit} best={best} worst={worst}
        title="What you can afford"
        note={afford.length > 1 ? "best first" : ""}
      />
      <Group
        rows={cannot} limit={env.cost_limit} best={null} worst={worst}
        title="Costs more than you have"
        note={gap > 0 ? `the best of these would cut ${gap}% more damage` : ""}
      />
    </>
  );
}

function TryIt({ env }: { env: Env }) {
  const [policy, setPolicy] = useState("incumbent");
  const [run, setRun] = useState<Rollout | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const go = useCallback(async () => {
    setBusy(true);
    setErr("");
    try {
      const r = await fetch("/api/gym/rollout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env_id: env.id, policy, seed: 0 }),
      });
      const j = await r.json();
      if (!r.ok) { setErr(j.detail ?? "could not run it"); setRun(null); }
      else setRun(j as Rollout);
    } catch (e) {
      setErr(String((e as Error).message).slice(0, 160));
    } finally {
      setBusy(false);
    }
  }, [env.id, policy]);

  const peak = run ? Math.max(run.cost_limit, run.total_cost) : 1;
  return (
    <div className="mt-3.5 border-t border-line-soft pt-3.5">
      <div className="flex flex-wrap items-center gap-2.5">
        <select
          id={`${env.id}-policy`}
          value={policy}
          onChange={(e) => { setPolicy(e.target.value); setRun(null); }}
          className="rounded border border-line bg-bg px-2 py-1.5 text-[12.5px] text-ink"
        >
          <option value="incumbent">the best you can afford</option>
          <option value="random">deciding at random</option>
          <option value="nothing">doing nothing</option>
        </select>
        <button
          type="button"
          onClick={go}
          disabled={busy}
          className="rounded bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white
                     transition-opacity disabled:opacity-50"
        >
          {busy ? "Running…" : "Watch one run"}
        </button>
        {err && <span className="text-[12px] text-bad">{err}</span>}
      </div>

      {run && (
        <div className="mt-3">
          <div className="relative h-12 w-full overflow-hidden rounded border border-line bg-bg">
            <div
              className="absolute inset-x-0 border-t border-dashed border-ink-faint"
              style={{ bottom: `${(run.cost_limit / peak) * 100}%` }}
              aria-hidden
            />
            <div className="flex h-full items-end gap-[2px] px-1">
              {run.steps.map((s) => (
                <span
                  key={s.step}
                  className={clsx(
                    "min-w-[2px] flex-1 rounded-t",
                    s.episode_cost > run.cost_limit ? "bg-bad/60" : "bg-sev-mid",
                  )}
                  style={{ height: `${Math.max((s.episode_cost / peak) * 100, 1)}%` }}
                />
              ))}
            </div>
          </div>
          <p className="mt-1.5 text-[11.5px] leading-snug text-ink-faint">
            What it had spent after each decision. The dashed line is everything
            it was allowed for the whole run —{" "}
            {run.feasible
              ? "it stayed under."
              : "the bars past it are spending nobody authorised."}
          </p>
        </div>
      )}
    </div>
  );
}

function EnvCard({ env }: { env: Env }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="rounded-xl border border-line bg-bg p-4 md:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="text-[15.5px] font-semibold tracking-tight text-ink">
          {env.plain.title}
        </h2>
        <span className="text-[11.5px] text-ink-faint">{env.plain.dynamics}</span>
      </div>

      <p className="mt-1.5 max-w-[68ch] text-[13px] leading-relaxed text-ink-mute">
        {env.plain.decision}
      </p>

      <dl className="mt-2.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1
                     text-[12.5px] leading-relaxed">
        <dt className="text-ink-faint">Scored on</dt>
        <dd className="text-ink-mute">{env.plain.score}</dd>
        <dt className="text-ink-faint">Allowed</dt>
        <dd className="text-ink-mute">{env.plain.budget}</dd>
      </dl>

      <Ladder env={env} />

      <p className="mt-3 max-w-[70ch] border-l-2 border-ink pl-3 text-[12.5px] leading-relaxed text-ink">
        {env.plain.status}
      </p>

      <TryIt env={env} />

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="mt-3 text-[12px] text-ink-faint underline-offset-2 transition-colors hover:text-ink-mute hover:underline"
      >
        {open ? "Hide" : "What this does not tell you"}
      </button>
      {open && (
        <p className="mt-2 max-w-[70ch] text-[12.5px] leading-relaxed text-ink-mute">
          {env.plain.caveat}
          {env.available === false && (
            <span className="mt-1.5 block text-bad">
              Not available here right now — it needs {env.needs}.
            </span>
          )}
        </p>
      )}
    </section>
  );
}

export function GymBench() {
  const [data, setData] = useState<Listing | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch("/api/gym");
        const j = await r.json();
        if (!alive) return;
        if (!r.ok) setError(j.detail ?? j.error ?? "could not load");
        else setData(j as Listing);
      } catch (e) {
        if (alive) setError(String((e as Error).message).slice(0, 180));
      }
    })();
    return () => { alive = false; };
  }, []);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[760px] px-4 py-7 md:px-6 md:py-9">
        <h1 className="text-[21px] font-semibold tracking-tight text-ink md:text-[24px]">
          Four decisions you can practise on
        </h1>

        {data && (
          <>
            <p className="mt-2 max-w-[66ch] text-[13.5px] leading-relaxed text-ink-mute">
              {data.plain_headline}
            </p>
            <p className="mt-3 max-w-[66ch] text-[12.5px] leading-relaxed text-ink-faint">
              {data.plain_how_to_read}
            </p>
          </>
        )}

        {error && (
          <p className="mt-5 rounded-lg border border-line bg-bg-raised px-4 py-3
                        text-[12.5px] leading-relaxed text-bad">
            {error}
          </p>
        )}
        {!data && !error && (
          <p className="mt-5 text-[12.5px] text-ink-faint">Loading…</p>
        )}

        <div className="mt-6 space-y-4">
          {data?.environments.map((e) => (
            <EnvCard key={e.id} env={e} />
          ))}
        </div>

        {data && (
          <p className="mt-7 text-[11.5px] leading-relaxed text-ink-faint">
            Every figure here is twelve runs of that approach, measured{" "}
            {data.measured_on}. Rerun them with{" "}
            <code className="rounded bg-bg-inset px-1 py-0.5 font-mono text-[11px]">
              {data.reproduce}
            </code>
            .
          </p>
        )}
      </div>
    </div>
  );
}
