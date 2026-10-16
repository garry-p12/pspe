"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

/**
 * Four decisions, one at a time, with the budget as the thing you move.
 *
 * Earlier versions of this page showed the budget as a fact about each
 * approach -- affordable or not -- which left the reader no way to ask the
 * question the page exists to answer: *what is the limit costing us?* Here
 * the limit is a line you drag, every point re-sorts around it, and the panel
 * says what the next notch up would buy. That is the finding made operable
 * rather than described.
 *
 * Honesty notes, because dragging a budget makes it easy to imply more than
 * was measured:
 *
 * - Points sit at each approach's MEAN spend over twelve runs. At the real
 *   budget the list also shows what was measured per run ("over in 10 of 12
 *   runs"), which is the stricter and more useful fact.
 * - Drag away from the real budget and those per-run counts are withdrawn,
 *   because they were measured against the real limit and do not transfer.
 *   The page says so rather than rescaling a number it did not measure.
 * - Damage is relative to doing nothing, which is the worst row on all four
 *   environments, so 100% is a real measurement and not a chosen anchor.
 */

type Score = { return: number; cost: number; feasible: number; of: number };
type Rung = Score & { key: string; policy: string; plain: string; note: string };
type Plain = {
  title: string; decision: string; score: string;
  budget: string; status: string; caveat: string; dynamics: string;
};
type Env = {
  id: string; plain: Plain; ladder: Rung[]; cost_limit: number;
  needs: string; available?: boolean; reason?: string;
};
type Listing = {
  environments: Env[]; plain_headline: string; measured_on: string;
  reproduce: string;
};
type StepRow = { step: number; reward: number; cost: number; episode_cost: number };
type Rollout = {
  steps: StepRow[]; total_reward: number; total_cost: number;
  cost_limit: number; feasible: boolean;
};

/** A row with everything the view needs, derived once. */
type Point = Rung & { damage: number; spend: number };

const VB = { w: 820, h: 440 };
const PAD = { l: 26, r: 26, t: 30, b: 46 };
const PLOT = { w: VB.w - PAD.l - PAD.r, h: VB.h - PAD.t - PAD.b };

/** Short tab labels. The card keeps the full title. */
const TAB: Record<string, string> = {
  "FloodOperate-v0": "Run the pumps",
  "FloodOperateFine-v0": "Pumps, finer detail",
  "FloodEvacuate-v0": "Call the evacuation",
  "Firebreak-v0": "Cut firebreaks",
};

function fmtMultiple(m: number): string {
  if (m >= 10) return `${Math.round(m)}x`;
  if (m >= 2) return `${m.toFixed(1).replace(/\.0$/, "")}x`;
  return `${m.toFixed(2).replace(/0$/, "").replace(/\.$/, "")}x`;
}

/**
 * What a row spent, in words.
 *
 * `atRealBudget` is the distinction that matters: the per-run counts were
 * measured against the environment's own limit, so once the reader drags the
 * line somewhere else they no longer describe anything and are withdrawn.
 */
function spendWords(p: Point, multiple: number, atRealBudget: boolean): string {
  if (p.spend <= 0) return "nothing spent";
  const share = p.spend / multiple;
  const over = p.of - p.feasible;
  if (share <= 1) {
    if (atRealBudget && over > 0 && over < p.of) return `over in ${over} of ${p.of} runs`;
    if (share > 0.95) return "all of it";
    if (share > 0.6) return "most of it";
    if (share > 0.25) return "part of it";
    return "barely any";
  }
  if (atRealBudget && over > 0 && over < p.of) return `over in ${over} of ${p.of} runs`;
  return share >= 1.2 ? `${fmtMultiple(share)} over` : "just over";
}

function Chart({
  points, multiple, setMultiple, bestKey, xMax,
}: {
  points: Point[]; multiple: number; setMultiple: (m: number) => void;
  bestKey: string | null; xMax: number;
}) {
  const svg = useRef<SVGSVGElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const x = useCallback((v: number) => PAD.l + (v / xMax) * PLOT.w, [xMax]);
  const y = (damage: number) => PAD.t + (damage / 100) * PLOT.h;

  const fromClient = useCallback((clientX: number) => {
    const el = svg.current;
    if (!el) return multiple;
    const r = el.getBoundingClientRect();
    const vx = ((clientX - r.left) / r.width) * VB.w;
    return Math.min(Math.max(((vx - PAD.l) / PLOT.w) * xMax, 0.05), xMax);
  }, [multiple, xMax]);

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => setMultiple(fromClient(e.clientX));
    const up = () => setDragging(false);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging, fromClient, setMultiple]);

  const ticks = Array.from({ length: Math.floor(xMax) }, (_, i) => i + 1)
    .filter((t) => t <= xMax - 0.15);

  return (
    <figure className="mt-4">
      <svg
        ref={svg}
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className={clsx("w-full select-none", dragging ? "cursor-grabbing" : "cursor-grab")}
        role="img"
        aria-label="Damage against what each approach spends, with a draggable budget line"
        onPointerDown={(e) => { setDragging(true); setMultiple(fromClient(e.clientX)); }}
      >
        {/* everything you can pay for */}
        <rect x={PAD.l} y={PAD.t} width={Math.max(x(multiple) - PAD.l, 0)} height={PLOT.h}
              className="fill-bg-inset" />

        {[0, 25, 50, 75, 100].map((d) => (
          <line key={d} x1={PAD.l} x2={PAD.l + PLOT.w} y1={y(d)} y2={y(d)}
                className="stroke-line-soft" strokeWidth={1} />
        ))}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={PAD.t} y2={PAD.t + PLOT.h}
                  className="stroke-line-soft" strokeWidth={1} />
            <text x={x(t)} y={PAD.t + PLOT.h + 16} textAnchor="middle"
                  className="fill-ink-faint text-[11px]">{t}x</text>
          </g>
        ))}

        <text x={PAD.l + 8} y={PAD.t + 18} className="fill-ink-faint text-[11px]">
          less damage ↑
        </text>

        {/* a dashed rule out to each point, so a value can be read off the row */}
        {points.map((p) => (
          <line key={`g-${p.key}`} x1={PAD.l} x2={PAD.l + PLOT.w} y1={y(p.damage)} y2={y(p.damage)}
                className={clsx(p.spend <= multiple ? "stroke-ink-faint" : "stroke-bad")}
                strokeWidth={1} strokeDasharray="3 5" opacity={0.42} />
        ))}

        {points.map((p) => {
          const ok = p.spend <= multiple;
          const isBest = p.key === bestKey;
          return (
            <g key={p.key}>
              <circle
                cx={x(p.spend)} cy={y(p.damage)} r={isBest ? 11 : 8}
                className={clsx(
                  isBest ? "fill-ink"
                         : ok ? "fill-sev-mid" : "fill-bg",
                  !ok && "stroke-bad",
                )}
                strokeWidth={!ok ? 2 : 0}
              />
              {isBest && (
                <circle cx={x(p.spend)} cy={y(p.damage)} r={16}
                        className="fill-none stroke-ink" strokeWidth={1.5} opacity={0.35} />
              )}
              <title>{`${p.plain} — ${p.damage}% damage, spends ${fmtMultiple(p.spend)} of the real budget`}</title>
            </g>
          );
        })}

        {/* the budget: the one thing on this chart you can move */}
        <g
          tabIndex={0}
          role="slider"
          aria-label="What you have to spend, as a multiple of the real budget"
          aria-valuemin={0.05}
          aria-valuemax={Math.round(xMax * 100) / 100}
          aria-valuenow={Math.round(multiple * 100) / 100}
          aria-valuetext={`${fmtMultiple(multiple)} of the real budget`}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 0.5 : 0.05;
            if (e.key === "ArrowLeft") { e.preventDefault(); setMultiple(Math.max(0.05, multiple - step)); }
            if (e.key === "ArrowRight") { e.preventDefault(); setMultiple(Math.min(xMax, multiple + step)); }
          }}
          className="cursor-ew-resize outline-none [&:focus-visible>rect]:stroke-accent"
        >
          <line x1={x(multiple)} x2={x(multiple)} y1={PAD.t - 6} y2={PAD.t + PLOT.h}
                className="stroke-ink" strokeWidth={2.5} />
          <rect x={x(multiple) - 5} y={PAD.t + PLOT.h / 2 - 22} width={10} height={44} rx={5}
                className="fill-ink stroke-bg" strokeWidth={2} />
          <text x={x(multiple)} y={PAD.t - 13} textAnchor="middle"
                className="fill-ink text-[11px] font-semibold">
            what you have
          </text>
        </g>
      </svg>

      <figcaption className="mt-1 flex flex-wrap items-baseline justify-between gap-x-4
                             text-[11.5px] text-ink-faint">
        <span>← spends nothing</span>
        <span>what it spends, as a multiple of your allowance (drag the black line)</span>
      </figcaption>
    </figure>
  );
}

function TryIt({ env }: { env: Env }) {
  const [policy, setPolicy] = useState("incumbent");
  const [run, setRun] = useState<Rollout | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => { setRun(null); setErr(""); }, [env.id]);

  const go = useCallback(async () => {
    setBusy(true); setErr("");
    try {
      const r = await fetch("/api/gym/rollout", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env_id: env.id, policy, seed: 0 }),
      });
      const j = await r.json();
      if (!r.ok) { setErr(j.detail ?? "could not run it"); setRun(null); }
      else setRun(j as Rollout);
    } catch (e) {
      setErr(String((e as Error).message).slice(0, 160));
    } finally { setBusy(false); }
  }, [env.id, policy]);

  const peak = run ? Math.max(run.cost_limit, run.total_cost) : 1;
  return (
    <div className="mt-5 border-t border-line-soft pt-4">
      <div className="flex flex-wrap items-center gap-2.5">
        <select
          id={`${env.id}-policy`} value={policy}
          onChange={(e) => { setPolicy(e.target.value); setRun(null); }}
          className="rounded border border-line bg-bg px-2 py-1.5 text-[12.5px] text-ink"
        >
          <option value="incumbent">the best you can afford</option>
          <option value="random">deciding at random</option>
          <option value="nothing">doing nothing</option>
        </select>
        <button type="button" onClick={go} disabled={busy}
                className="rounded bg-accent px-3 py-1.5 text-[12.5px] font-medium text-white
                           transition-opacity disabled:opacity-50">
          {busy ? "Running…" : "Watch one run"}
        </button>
        {err && <span className="text-[12px] text-bad">{err}</span>}
      </div>
      {run && (
        <div className="mt-3">
          <div className="relative h-12 w-full overflow-hidden rounded border border-line bg-bg">
            <div className="absolute inset-x-0 border-t border-dashed border-ink-faint"
                 style={{ bottom: `${(run.cost_limit / peak) * 100}%` }} aria-hidden />
            <div className="flex h-full items-end gap-[2px] px-1">
              {run.steps.map((s) => (
                <span key={s.step}
                      className={clsx("min-w-[2px] flex-1 rounded-t",
                        s.episode_cost > run.cost_limit ? "bg-bad/60" : "bg-sev-mid")}
                      style={{ height: `${Math.max((s.episode_cost / peak) * 100, 1)}%` }} />
              ))}
            </div>
          </div>
          <p className="mt-1.5 text-[11.5px] leading-snug text-ink-faint">
            What it had spent after each decision, against everything it was
            allowed for the whole run —{" "}
            {run.feasible ? "it stayed under." : "the bars past the line are spending nobody authorised."}
          </p>
        </div>
      )}
    </div>
  );
}

function EnvView({ env }: { env: Env }) {
  const [multiple, setMultiple] = useState(1);
  const [open, setOpen] = useState(false);
  useEffect(() => { setMultiple(1); setOpen(false); }, [env.id]);

  const points = useMemo<Point[]>(() => {
    const worst = Math.min(...env.ladder.map((r) => r.return));
    return env.ladder
      .map((r) => ({ ...r, damage: Math.round((r.return / worst) * 100),
                     spend: r.cost / env.cost_limit }))
      .sort((a, b) => a.damage - b.damage);
  }, [env]);

  const xMax = useMemo(
    () => Math.max(2.2, ...points.map((p) => p.spend * 1.12), multiple * 1.12),
    [points, multiple],
  );

  const afford = points.filter((p) => p.spend <= multiple);
  const best = afford.length ? afford[0] : null;           // already damage-sorted
  const atReal = Math.abs(multiple - 1) < 0.005;

  // What the next notch up would buy: the cheapest approach that beats what
  // you can currently afford. Naming the price is the point -- "better exists"
  // is not useful without "and this is what it would take".
  const upgrade = useMemo(() => {
    const better = points.filter(
      (p) => p.spend > multiple && (!best || p.damage < best.damage));
    if (!better.length) return null;
    return better.reduce((a, b) => (b.spend < a.spend ? b : a));
  }, [points, multiple, best]);

  return (
    <div className="mt-5 grid gap-6 lg:grid-cols-[minmax(0,1.55fr)_minmax(260px,1fr)]">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <h2 className="text-[19px] font-semibold tracking-tight text-ink">
            {env.plain.title}
          </h2>
          <span className="rounded-full border border-line px-2.5 py-0.5 text-[11.5px] text-ink-mute">
            {env.plain.dynamics}
          </span>
        </div>

        <p className="mt-2 max-w-[72ch] text-[13.5px] leading-relaxed text-ink">
          {env.plain.decision}{" "}
          <span className="text-ink-faint">
            Scored on {env.plain.score}. Allowed {env.plain.budget}.
          </span>
        </p>

        <Chart points={points} multiple={multiple} setMultiple={setMultiple}
               bestKey={best?.key ?? null} xMax={xMax} />

        {!atReal && (
          <p className="mt-2 text-[11.5px] leading-relaxed text-ink-faint">
            You have moved the line off the real allowance, so each approach is
            placed by what it spent on average.{" "}
            <button type="button" onClick={() => setMultiple(1)}
                    className="underline underline-offset-2 hover:text-ink-mute">
              Put it back
            </button>
          </p>
        )}
      </div>

      <div className="min-w-0">
        <div className="rounded-xl bg-ink p-4 text-bg">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[12.5px] opacity-70">Budget</span>
            <span className="font-mono text-[19px] font-semibold tabular-nums">
              {fmtMultiple(multiple)}
            </span>
          </div>
          <p className="mt-2.5 text-[12.5px] opacity-70">Best you can afford</p>
          {best ? (
            <div className="mt-0.5 flex items-baseline justify-between gap-3">
              <span className="text-[16px] font-semibold leading-tight">{best.plain}</span>
              <span className="font-mono text-[19px] font-semibold tabular-nums">
                {best.damage}%
              </span>
            </div>
          ) : (
            <p className="mt-0.5 text-[15px] font-semibold leading-tight">
              Nothing fits this budget.
            </p>
          )}
          <p className="mt-3 text-[12.5px] leading-relaxed opacity-80">
            {upgrade ? (
              <>
                With {fmtMultiple(upgrade.spend)} the allowance,{" "}
                <span className="font-semibold opacity-100">{upgrade.plain}</span>{" "}
                would get you to {upgrade.damage}%.
              </>
            ) : (
              "Nothing measured here does better, at any price."
            )}
          </p>
        </div>

        {/* The anchor has to be stated somewhere or the percentages are
            uninterpretable. Doing nothing is the worst row on all four
            environments, so 100% is a measurement rather than a chosen zero. */}
        <p className="mt-4 flex items-baseline justify-between gap-3 border-b border-line
                      pb-1 text-[10.5px] uppercase tracking-wide text-ink-faint">
          <span>every approach, best first</span>
          <span className="normal-case tracking-normal">
            damage, next to doing nothing
          </span>
        </p>

        <ol>
          {points.map((p, i) => {
            const ok = p.spend <= multiple;
            const isBest = p.key === best?.key;
            return (
              <li key={p.key}
                  className="grid grid-cols-[22px_minmax(0,1fr)_auto] items-start gap-x-3
                             border-b border-line-soft py-2 last:border-0">
                <span
                  className={clsx(
                    "mt-[1px] flex h-[22px] w-[22px] items-center justify-center rounded-full",
                    "text-[11px] font-semibold",
                    isBest ? "bg-ink text-bg"
                      : ok ? "bg-bg-inset text-ink-mute"
                           : "border-[1.5px] border-bad text-bad",
                  )}
                >
                  {i + 1}
                </span>
                <span className="min-w-0">
                  <span className={clsx("block text-[12.5px] leading-snug",
                                        isBest ? "font-semibold text-ink" : "text-ink-mute")}>
                    {p.plain}
                  </span>
                  <span className={clsx("block text-[11.5px] leading-snug",
                                        ok ? "text-ink-faint" : "text-bad")}>
                    {spendWords(p, multiple, atReal)}
                  </span>
                </span>
                <span className="font-mono text-[12.5px] tabular-nums text-ink">
                  {p.damage}%
                </span>
              </li>
            );
          })}
        </ol>
      </div>

      <div className="lg:col-span-2">
        <p className="max-w-[80ch] border-l-2 border-ink pl-3.5 text-[13.5px] leading-relaxed text-ink">
          {env.plain.status}
        </p>
        <TryIt env={env} />
        <button type="button" onClick={() => setOpen((o) => !o)}
                className="mt-3.5 text-[12px] text-ink-faint underline-offset-2
                           transition-colors hover:text-ink-mute hover:underline">
          {open ? "Hide" : "What this does not tell you"}
        </button>
        {open && (
          <p className="mt-2 max-w-[78ch] text-[12.5px] leading-relaxed text-ink-mute">
            {env.plain.caveat}
            {env.available === false && (
              <span className="mt-1.5 block text-bad">
                Not available here right now — it needs {env.needs}.
              </span>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

export function GymBench() {
  const [data, setData] = useState<Listing | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState(0);

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

  const env = data?.environments[tab];

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1180px] px-4 py-6 md:px-8 md:py-8">
        <div className="flex flex-wrap items-start justify-between gap-x-8 gap-y-4">
          <div className="max-w-[46ch]">
            <h1 className="text-[26px] font-semibold leading-[1.15] tracking-tight text-ink md:text-[30px]">
              Four decisions you can practise on
            </h1>
            {data && (
              <p className="mt-2.5 text-[13.5px] leading-relaxed text-ink-mute">
                {data.plain_headline}
              </p>
            )}
          </div>

          {data && (
            <nav className="flex flex-wrap gap-1 rounded-lg border border-line bg-bg-raised p-1"
                 aria-label="Environments">
              {data.environments.map((e, i) => (
                <button
                  key={e.id} type="button" onClick={() => setTab(i)}
                  aria-current={i === tab ? "page" : undefined}
                  className={clsx(
                    "rounded-md px-3 py-1.5 text-[12.5px] transition-colors",
                    i === tab
                      ? "bg-bg font-semibold text-ink shadow-[0_1px_2px_rgba(16,20,28,0.08)]"
                      : "text-ink-mute hover:text-ink",
                  )}
                >
                  {TAB[e.id] ?? e.plain.title}
                </button>
              ))}
            </nav>
          )}
        </div>

        <hr className="mt-5 border-line" />

        {error && (
          <p className="mt-5 rounded-lg border border-line bg-bg-raised px-4 py-3
                        text-[12.5px] leading-relaxed text-bad">{error}</p>
        )}
        {!data && !error && <p className="mt-5 text-[12.5px] text-ink-faint">Loading…</p>}

        {env && <EnvView key={env.id} env={env} />}

        {data && (
          <p className="mt-8 text-[11.5px] leading-relaxed text-ink-faint">
            Every figure here is twelve runs of that approach, measured{" "}
            {data.measured_on}. Rerun them with{" "}
            <code className="rounded bg-bg-inset px-1 py-0.5 font-mono text-[11px]">
              {data.reproduce}
            </code>.
          </p>
        )}
      </div>
    </div>
  );
}
