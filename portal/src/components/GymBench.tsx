"use client";

import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";

/**
 * The four environments, and every policy that has been run on them.
 *
 * This page is built so it cannot flatter the work, which is the opposite of
 * what a benchmark page usually does. Three decisions carry that:
 *
 * 1. Each environment shows its whole LADDER, not its best row. The best
 *    policy on reward is usually one you are not allowed to run -- on the
 *    polder it spends 3.4 pump-hours against a limit of 2.0 -- so a page
 *    listing only the winner would report numbers nobody may achieve.
 * 2. Every row carries how many of 12 episodes stayed inside the budget,
 *    per episode. Not a mean: the evacuation trigger averages 28.2 against a
 *    limit of 32 while actually spending either nothing or 39-48.
 * 3. The status and caveat sentences are the service's, which are
 *    `pspe/envs/catalog.py`'s, which `scripts/score_gym.py --ladder --check`
 *    tests. There is no second copy here to drift into something kinder.
 *
 * The one piece of interaction is "run one episode", which exists so the cost
 * can be watched accumulating against a budget that is spent over the whole
 * episode rather than per step.
 */

type Score = { return: number; cost: number; feasible: number; of: number };
type Rung = Score & { key: string; policy: string; note: string };
type Env = {
  id: string;
  title: string;
  dynamics: "physics" | "learned";
  summary: string;
  reward: string;
  cost: string;
  budget: string;
  incumbent: string;
  status: string;
  caveat: string;
  needs: string;
  scores: { random: Score; incumbent: Score };
  ladder: Rung[];
  incumbent_method: string;
  observation_space?: string;
  action_space?: string;
  available?: boolean;
  reason?: string;
};
type Listing = {
  environments: Env[];
  headline: string;
  feasibility_note: string;
  ladder_note: string;
  measured_on: string;
  reproduce: string;
};
type StepRow = { step: number; reward: number; cost: number; episode_cost: number };
type Rollout = {
  env_id: string;
  policy: string;
  seed: number;
  steps: StepRow[];
  total_reward: number;
  total_cost: number;
  cost_limit: number;
  feasible: boolean;
  caveat: string;
};

const POLICIES = [
  { key: "incumbent", label: "Best policy that fits the budget" },
  { key: "random", label: "Random" },
  { key: "nothing", label: "Do nothing" },
] as const;

/** Inline code, for the method names the service returns in prose. */
function Prose({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("`") && p.endsWith("`") ? (
          <code key={i} className="rounded bg-bg-inset px-1 py-0.5 font-mono text-[11.5px] text-ink">
            {p.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

function Feasibility({ s }: { s: Score }) {
  const all = s.feasible === s.of;
  const none = s.feasible === 0;
  return (
    <span
      className={clsx(
        "whitespace-nowrap font-mono text-[11.5px]",
        all ? "text-ink" : none ? "text-bad" : "text-warn",
      )}
      title={
        all
          ? `Stayed inside the budget in all ${s.of} episodes.`
          : `Went over the budget in ${s.of - s.feasible} of ${s.of} episodes. ` +
            `A policy that cannot be afforded is not a result.`
      }
    >
      {s.feasible}/{s.of}
      {!all && <span className="ml-1.5 font-sans font-semibold">over</span>}
    </span>
  );
}

/**
 * One ladder, drawn to a shared scale.
 *
 * The bar length is the return, which is negative everywhere here -- these are
 * all losses -- so the scale runs from the worst policy to zero and a longer
 * bar is a worse outcome. Infeasible rows are hatched rather than coloured,
 * because the point is that their number is not available, not that it is bad.
 */
function Ladder({ env }: { env: Env }) {
  const worst = Math.min(...env.ladder.map((r) => r.return), 0);
  const best = env.ladder.find(
    (r) => r.feasible === r.of && r.return === Math.max(
      ...env.ladder.filter((x) => x.feasible === x.of).map((x) => x.return)),
  );
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[12.5px]">
        <thead>
          <tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-ink-faint">
            <th className="py-1.5 pr-3 font-semibold">policy</th>
            <th className="py-1.5 pr-3 text-right font-semibold">harm</th>
            <th className="w-[34%] py-1.5 pr-3 font-semibold">relative</th>
            <th className="py-1.5 pr-3 text-right font-semibold">spent</th>
            <th className="py-1.5 font-semibold">within budget</th>
          </tr>
        </thead>
        <tbody>
          {env.ladder.map((r) => {
            const width = worst === 0 ? 0 : (r.return / worst) * 100;
            const ok = r.feasible === r.of;
            const isBest = best?.key === r.key;
            return (
              <tr key={r.key} className="border-b border-line-soft align-top last:border-0">
                <td className="py-2 pr-3">
                  <span className={clsx("block", isBest ? "font-semibold text-ink" : "text-ink-mute")}>
                    <Prose text={r.policy} />
                    {isBest && (
                      <span className="ml-2 rounded bg-accent-soft px-1.5 py-0.5 text-[10px]
                                       font-semibold uppercase tracking-wide text-ink">
                        the bar
                      </span>
                    )}
                  </span>
                  {r.note && (
                    <span className="mt-0.5 block max-w-[46ch] text-[11.5px] leading-snug text-ink-faint">
                      <Prose text={r.note} />
                    </span>
                  )}
                </td>
                <td className="py-2 pr-3 text-right font-mono tabular-nums text-ink">
                  {r.return.toFixed(Math.abs(r.return) >= 100 ? 0 : 2)}
                </td>
                <td className="py-2 pr-3">
                  <span className="block h-2 w-full rounded-sm bg-bg-inset" aria-hidden>
                    <span
                      className={clsx("block h-2 rounded-sm", ok ? "bg-sev-mid" : "bg-bad/35")}
                      style={{ width: `${Math.max(width, 1.5)}%` }}
                    />
                  </span>
                </td>
                <td className="py-2 pr-3 text-right font-mono tabular-nums text-ink-mute">
                  {r.cost < 1 ? r.cost.toFixed(4) : r.cost.toFixed(2)}
                </td>
                <td className="py-2">
                  <Feasibility s={r} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 max-w-[80ch] text-[11.5px] leading-relaxed text-ink-faint">
        Longer bar is worse: every return here is a loss, scaled against the worst
        policy on this environment. A hatched bar is a policy that went over the
        budget, so its number is not one anybody may use.
      </p>
    </div>
  );
}

function RolloutPanel({ env }: { env: Env }) {
  const [policy, setPolicy] = useState<string>("incumbent");
  const [seed, setSeed] = useState(0);
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
        body: JSON.stringify({ env_id: env.id, policy, seed }),
      });
      const j = await r.json();
      if (!r.ok) {
        setErr(j.detail ?? j.error ?? "the episode could not be run");
        setRun(null);
      } else {
        setRun(j as Rollout);
      }
    } catch (e) {
      setErr(String((e as Error).message).slice(0, 200));
    } finally {
      setBusy(false);
    }
  }, [env.id, policy, seed]);

  const peak = run ? Math.max(run.cost_limit, run.total_cost) : 1;
  return (
    <div className="rounded-lg border border-line bg-bg-raised p-3.5">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-wide text-ink-faint">policy</span>
          <select
            id={`${env.id}-policy`}
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
            className="rounded border border-line bg-bg px-2 py-1.5 text-[12.5px] text-ink"
          >
            {POLICIES.map((p) => (
              <option key={p.key} value={p.key}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-wide text-ink-faint">episode</span>
          <input
            id={`${env.id}-seed`}
            type="number"
            min={0}
            max={999}
            value={seed}
            onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
            className="w-20 rounded border border-line bg-bg px-2 py-1.5 font-mono text-[12.5px] text-ink"
          />
        </label>
        <button
          type="button"
          onClick={go}
          disabled={busy}
          className="rounded bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white
                     transition-opacity disabled:opacity-50"
        >
          {busy ? "Running…" : "Run one episode"}
        </button>
      </div>

      {err && (
        <p className="mt-3 rounded border border-line bg-bg px-3 py-2 text-[12px] leading-relaxed text-bad">
          {err}
        </p>
      )}

      {run && (
        <div className="mt-3.5">
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-[12.5px]">
            <span className="text-ink-mute">
              harm{" "}
              <span className="font-mono tabular-nums text-ink">
                {run.total_reward.toFixed(Math.abs(run.total_reward) >= 100 ? 0 : 2)}
              </span>
            </span>
            <span className="text-ink-mute">
              spent{" "}
              <span
                className={clsx(
                  "font-mono tabular-nums",
                  run.feasible ? "text-ink" : "text-bad",
                )}
              >
                {run.total_cost < 1 ? run.total_cost.toFixed(4) : run.total_cost.toFixed(2)}
              </span>{" "}
              of {run.cost_limit}
            </span>
            <span
              className={clsx(
                "font-semibold",
                run.feasible ? "text-ink" : "text-bad",
              )}
            >
              {run.feasible ? "within budget" : "over budget"}
            </span>
            <span className="text-ink-faint">{run.steps.length} decisions</span>
          </div>

          {/* The budget is spent over the episode, so what matters is the
              running total crossing the limit, not any single step. */}
          <div className="mt-2.5">
            <div className="relative h-14 w-full overflow-hidden rounded border border-line bg-bg">
              <div
                className="absolute inset-x-0 border-t border-dashed border-ink-faint"
                style={{ bottom: `${(run.cost_limit / peak) * 100}%` }}
                aria-hidden
              />
              <div className="flex h-full items-end gap-[2px] px-1 pb-0">
                {run.steps.map((s) => (
                  <span
                    key={s.step}
                    title={`decision ${s.step}: spent ${s.episode_cost.toFixed(4)} so far`}
                    className={clsx(
                      "min-w-[2px] flex-1 rounded-t",
                      s.episode_cost > run.cost_limit ? "bg-bad/60" : "bg-sev-mid",
                    )}
                    style={{ height: `${Math.max((s.episode_cost / peak) * 100, 1)}%` }}
                  />
                ))}
              </div>
            </div>
            <p className="mt-1.5 text-[11.5px] text-ink-faint">
              Running total spent, decision by decision. The dashed line is the
              budget for the whole episode — bars past it are spend nobody
              authorised.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function EnvCard({ env }: { env: Env }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="rounded-xl border border-line bg-bg p-4 md:p-5">
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div>
          <h2 className="text-[16px] font-semibold tracking-tight text-ink">{env.title}</h2>
          <code className="mt-0.5 block font-mono text-[11.5px] text-ink-faint">{env.id}</code>
        </div>
        <span
          className={clsx(
            "rounded px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wide",
            env.dynamics === "physics"
              ? "bg-accent-soft text-ink"
              : "bg-bad/10 text-bad",
          )}
          title={
            env.dynamics === "physics"
              ? "A conservation law is solved at every step."
              : "A neural network predicts the next state and the environment " +
                "feeds its own prediction back in, so error compounds in a way " +
                "a conservation law would forbid."
          }
        >
          {env.dynamics === "physics" ? "solved physics" : "learned model"}
        </span>
      </div>

      <p className="mt-2.5 max-w-[78ch] text-[13px] leading-relaxed text-ink-mute">
        {env.summary}
      </p>

      <dl className="mt-3.5 grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
        {[
          ["What it is scored on", env.reward],
          ["What it spends", env.cost],
          ["What it may spend", env.budget],
          ["The bar to beat", env.incumbent],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-[10.5px] uppercase tracking-wide text-ink-faint">{k}</dt>
            <dd className="mt-0.5 text-[12.5px] leading-snug text-ink-mute">
              <Prose text={v} />
            </dd>
          </div>
        ))}
      </dl>

      <div className="mt-4 rounded-lg border-l-2 border-ink bg-bg-inset px-3.5 py-3">
        <p className="text-[10.5px] uppercase tracking-wide text-ink-faint">
          Where this stands
        </p>
        <p className="mt-1 max-w-[80ch] text-[12.5px] leading-relaxed text-ink">
          <Prose text={env.status} />
        </p>
      </div>

      <div className="mt-4">
        <Ladder env={env} />
      </div>

      <div className="mt-4">
        <RolloutPanel env={env} />
      </div>

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="mt-3.5 text-[12px] text-ink-faint transition-colors hover:text-ink-mute"
      >
        {open ? "Hide" : "Show"} what this environment is not
      </button>
      {open && (
        <div className="mt-2 space-y-2 text-[12.5px] leading-relaxed text-ink-mute">
          <p className="max-w-[80ch]">{env.caveat}</p>
          {env.needs && (
            <p className="max-w-[80ch] text-ink-faint">
              Needs on disk: {env.needs}.
            </p>
          )}
          {env.available === false && (
            <p className="max-w-[80ch] text-bad">
              Not constructible here right now: {env.reason}
            </p>
          )}
          {env.observation_space && (
            <p className="font-mono text-[11.5px] text-ink-faint">
              observes {env.observation_space} · acts {env.action_space}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export function GymBench() {
  const [data, setData] = useState<Listing | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch("/api/gym?probe=1");
        const j = await r.json();
        if (!alive) return;
        if (!r.ok) setErr(j.detail ?? j.error ?? "the environments could not be listed");
        else setData(j as Listing);
      } catch (e) {
        if (alive) setErr(String((e as Error).message).slice(0, 200));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1100px] px-4 py-6 md:px-6 md:py-8">
        <header>
          <h1 className="text-[22px] font-semibold tracking-tight text-ink md:text-[26px]">
            Four decisions you can train against
          </h1>
          <p className="mt-2 max-w-[74ch] text-[13.5px] leading-relaxed text-ink-mute">
            Each of these is a hazard the rest of this tool simulates, turned into
            something an algorithm can practise on: a decision to make every few
            hours, a budget it may not exceed, and a score. They are installable
            with <code className="rounded bg-bg-inset px-1 py-0.5 font-mono text-[11.5px]">
            gymnasium.make</code>.
          </p>
          {data && (
            <>
              <p className="mt-4 max-w-[80ch] rounded-lg border border-line bg-bg-raised px-4 py-3
                            text-[13px] leading-relaxed text-ink">
                {data.headline}
              </p>
              <p className="mt-2 max-w-[80ch] text-[12px] leading-relaxed text-ink-faint">
                {data.feasibility_note} {data.ladder_note}
              </p>
              <p className="mt-2 text-[11.5px] text-ink-faint">
                Measured {data.measured_on} · regenerate with{" "}
                <code className="rounded bg-bg-inset px-1 py-0.5 font-mono text-[11px]">
                  {data.reproduce}
                </code>
              </p>
            </>
          )}
        </header>

        {err && (
          <p className="mt-6 rounded-lg border border-line bg-bg-raised px-4 py-3
                        text-[12.5px] leading-relaxed text-bad">
            {err}
          </p>
        )}

        {!data && !err && (
          <p className="mt-6 text-[12.5px] text-ink-faint">
            Constructing all four environments to read their shapes off the code…
          </p>
        )}

        <div className="mt-6 space-y-5">
          {data?.environments.map((e) => (
            <EnvCard key={e.id} env={e} />
          ))}
        </div>
      </div>
    </div>
  );
}
