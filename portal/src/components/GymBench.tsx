"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { IsoScene, type Scene } from "./IsoScene";

/**
 * Make the decision yourself, against the environment a benchmark is scored in.
 *
 * The page runs the real thing. Every step posts to the service, which advances
 * the actual environment -- the same local-inertial solver, the same budget
 * arithmetic, the same reward -- and sends back its state. Nothing here is a
 * browser-side approximation, so none of these numbers needs a disclaimer
 * about differing from the real ones.
 *
 * Two honesty constraints shape the layout:
 *
 * - **The finished run is placed in the ladder, and marked as one episode.**
 *   The ladder rows are twelve episodes each. Dropping a single run into that
 *   list without saying so would invite a comparison the sample size does not
 *   support, so the row says it.
 * - **The controls are coarser than the action space.** Four pump settings,
 *   not a continuous dial. A policy may use the whole interval; these are
 *   what an operator would be offered, and the page says the budget is spent
 *   over the whole run rather than per decision.
 */

type Opt = { label: string; action: number[] | null; policy?: string };
type Group = { name: string; options: Opt[]; default: number };
type Stat = { label: string; value: string; fraction?: number; over?: boolean };
type Plain = {
  title: string; decision: string; score: string;
  budget: string; status: string; caveat: string; dynamics: string;
};
type Rung = {
  key: string; plain: string; return: number; cost: number;
  feasible: number; of: number;
};
type EnvRow = { id: string; plain: Plain; ladder: Rung[]; cost_limit: number };
type Listing = { environments: EnvRow[]; measured_on: string; reproduce: string };
type Result = {
  damage: number; return: number; spent: number; feasible: boolean; note: string;
};
type Play = {
  session: string; env_id: string; t: number; horizon: number; done: boolean;
  scene: Scene; stats: Stat[]; cost_limit: number; spent: number;
  over_budget: boolean; controls?: Group[]; plain?: Plain; result?: Result;
};

const post = async (url: string, body: unknown) => {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail ?? j.error ?? "the service refused that");
  return j;
};

const START_YAW = -0.42;

function Stats({ stats }: { stats: Stat[] }) {
  return (
    <div className="flex flex-wrap gap-2.5">
      {stats.map((s) => (
        <div key={s.label}
             className="min-w-[128px] rounded-lg border border-line bg-bg px-3 py-2">
          <p className="text-[10.5px] uppercase tracking-wide text-ink-faint">{s.label}</p>
          <p className={clsx("mt-0.5 text-[13.5px] font-semibold",
                             s.over ? "text-bad" : "text-ink")}>
            {s.value}
          </p>
          {s.fraction !== undefined && (
            <span className="mt-1.5 block h-[3px] w-full rounded-sm bg-bg-inset" aria-hidden>
              <span className={clsx("block h-[3px] rounded-sm",
                                    s.over ? "bg-bad" : "bg-sev-mid")}
                    style={{ width: `${Math.min(s.fraction, 1) * 100}%` }} />
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function Segmented({
  group, value, onChange, disabled,
}: {
  group: Group; value: string; onChange: (v: string) => void; disabled: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[12px] text-ink-mute">{group.name}</span>
      <div className="flex rounded-md border border-line bg-bg p-0.5">
        {group.options.map((o) => (
          <button
            key={o.label} type="button" disabled={disabled}
            onClick={() => onChange(o.label)}
            aria-pressed={o.label === value}
            className={clsx(
              "rounded px-2.5 py-1 text-[12px] transition-colors disabled:opacity-40",
              o.label === value
                ? "bg-accent font-medium text-white"
                : "text-ink-mute hover:text-ink",
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Ladder({ env, result }: { env: EnvRow; result: Result | null }) {
  const worst = Math.min(...env.ladder.map((r) => r.return));
  const rows = env.ladder.map((r) => ({
    key: r.key,
    name: r.plain,
    damage: Math.round((r.return / worst) * 100),
    ok: r.feasible === r.of,
    note: r.cost <= 0 ? "nothing spent"
      : r.feasible === r.of ? (r.cost / env.cost_limit > 0.95 ? "all of it" : "part of it")
      : r.feasible > 0 ? `over in ${r.of - r.feasible} of ${r.of} runs`
      : `${(r.cost / env.cost_limit).toFixed(1)}x over`,
    mine: false,
  }));
  if (result) {
    rows.push({
      key: "__yours", name: "your run", damage: result.damage,
      ok: result.feasible,
      note: result.feasible ? "within budget" : "over budget",
      mine: true,
    });
  }
  rows.sort((a, b) => a.damage - b.damage);
  return (
    <>
      <div className="mt-5 flex items-baseline justify-between border-b border-line pb-1">
        <h3 className="text-[10.5px] font-semibold uppercase tracking-wide text-ink">
          Where you&rsquo;d rank
        </h3>
        <span className="text-[10.5px] text-ink-faint">damage</span>
      </div>
      <ul>
        {rows.map((r) => (
          <li key={r.key}
              className={clsx("flex items-start justify-between gap-3 border-b border-line-soft py-2 last:border-0",
                              r.mine && "-mx-2 rounded bg-accent-soft px-2")}>
            <span className="min-w-0">
              <span className={clsx("block text-[12.5px] leading-snug",
                                    r.mine ? "font-semibold text-ink" : "text-ink-mute")}>
                {r.name}
              </span>
              <span className={clsx("block text-[11px] leading-snug",
                                    r.ok ? "text-ink-faint" : "text-bad")}>
                {r.note}
              </span>
            </span>
            <span className="font-mono text-[12.5px] tabular-nums text-ink">{r.damage}%</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
        {result
          ? "Your run is one episode. Every other row is twelve, so treat the placing as a sighting shot."
          : "When you finish, your run is placed in this list."}
      </p>
    </>
  );
}

export function GymBench() {
  const [list, setList] = useState<Listing | null>(null);
  const [envId, setEnvId] = useState<string>("FloodOperate-v0");
  const [play, setPlay] = useState<Play | null>(null);
  const [controls, setControls] = useState<Group[]>([]);
  const [choice, setChoice] = useState<string[]>([]);
  const [yaw, setYaw] = useState(START_YAW);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const seq = useRef(0);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch("/api/gym");
        const j = await r.json();
        if (!alive) return;
        if (!r.ok) setError(j.detail ?? j.error ?? "could not load");
        else setList(j as Listing);
      } catch (e) {
        if (alive) setError(String((e as Error).message).slice(0, 180));
      }
    })();
    return () => { alive = false; };
  }, []);

  const begin = useCallback(async (id: string) => {
    const mine = ++seq.current;
    setBusy(true); setError(""); setPlay(null);
    try {
      const j: Play = await post("/api/gym/play/start", { env_id: id, seed: 0 });
      if (mine !== seq.current) return;
      setPlay(j);
      setControls(j.controls ?? []);
      setChoice((j.controls ?? []).map((g) => g.options[g.default]?.label ?? g.options[0].label));
      setYaw(START_YAW);
    } catch (e) {
      if (mine === seq.current) setError(String((e as Error).message).slice(0, 220));
    } finally {
      if (mine === seq.current) setBusy(false);
    }
  }, []);

  useEffect(() => { begin(envId); }, [envId, begin]);

  const advance = useCallback(async (toEnd: boolean) => {
    if (!play || play.done) return;
    setBusy(true); setError("");
    try {
      const j: Play = await post("/api/gym/play/step", {
        session: play.session, choices: choice, to_end: toEnd,
      });
      setPlay((prev) => ({ ...j, controls: prev?.controls, plain: prev?.plain }));
    } catch (e) {
      setError(String((e as Error).message).slice(0, 220));
    } finally {
      setBusy(false);
    }
  }, [play, choice]);

  const env = useMemo(
    () => list?.environments.find((e) => e.id === envId) ?? null, [list, envId]);
  const plain = play?.plain ?? env?.plain;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid max-w-[1200px] gap-6 px-4 py-6 md:px-6
                      lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
            <div className="min-w-0">
              <h1 className="text-[20px] font-semibold tracking-tight text-ink">
                {plain?.title ?? "Loading…"}
              </h1>
              {plain && (
                <p className="mt-1 max-w-[62ch] text-[13px] leading-relaxed text-ink-mute">
                  {plain.decision}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11.5px] text-ink-faint">Drag to turn</span>
              <button type="button" onClick={() => setYaw(START_YAW)}
                      className="rounded border border-line bg-bg px-2.5 py-1 text-[12px] text-ink-mute
                                 transition-colors hover:text-ink">
                Reset view
              </button>
            </div>
          </div>

          {play && <div className="mt-3.5"><Stats stats={play.stats} /></div>}

          {error && (
            <p className="mt-4 rounded-lg border border-line bg-bg-raised px-4 py-3
                          text-[12.5px] leading-relaxed text-bad">{error}</p>
          )}

          {play ? (
            <>
              <div className="mt-2 rounded-xl border border-line bg-bg-raised">
                <IsoScene scene={play.scene} yaw={yaw} onYaw={setYaw} />
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1
                              text-[11.5px] text-ink-faint">
                {([["#88a86a", play.scene.legend.ground],
                   ["#5b9bd5", play.scene.legend.water],
                   ["#a89878", play.scene.legend.marker]] as const).map(([c, t]) => (
                  <span key={t} className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: c }} />
                    {t}
                  </span>
                ))}
                {play.scene.water_scale && play.scene.water_unit && (
                  <span className="text-ink-faint">
                    full water column = {play.scene.water_scale}{" "}
                    {play.scene.water_unit}
                  </span>
                )}
                <span className="ml-auto">
                  Height is the simulator&rsquo;s own terrain and water. Nothing here is drawn for effect.
                </span>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-3
                              rounded-xl border border-line bg-bg p-3">
                {controls.map((g, i) => (
                  <Segmented
                    key={g.name} group={g} value={choice[i] ?? ""}
                    disabled={busy || play.done}
                    onChange={(v) => setChoice((c) => c.map((x, j) => (j === i ? v : x)))}
                  />
                ))}
                <div className="ml-auto flex items-center gap-2">
                  <button type="button" onClick={() => begin(envId)} disabled={busy}
                          className="text-[12.5px] text-ink-faint transition-colors
                                     hover:text-ink-mute disabled:opacity-40">
                    Start over
                  </button>
                  <button type="button" onClick={() => advance(true)}
                          disabled={busy || play.done}
                          className="rounded border border-line bg-bg px-3 py-1.5 text-[12.5px]
                                     text-ink transition-colors hover:bg-bg-inset disabled:opacity-40">
                    Skip to end
                  </button>
                  <button type="button" onClick={() => advance(false)}
                          disabled={busy || play.done}
                          className="rounded bg-accent px-3 py-1.5 text-[12.5px] font-medium
                                     text-white disabled:opacity-40">
                    {busy ? "Running…" : play.done ? "Finished" : "Next step →"}
                  </button>
                </div>
              </div>
            </>
          ) : (
            !error && <p className="mt-5 text-[12.5px] text-ink-faint">Starting a run…</p>
          )}

          {plain && (
            <p className="mt-4 max-w-[78ch] border-l-2 border-ink pl-3.5 text-[12.5px]
                          leading-relaxed text-ink">
              {plain.status}
            </p>
          )}
        </div>

        <aside className="min-w-0">
          <h2 className="text-[10.5px] font-semibold uppercase tracking-wide text-ink">
            Decision
          </h2>
          <ul className="mt-1.5 space-y-1.5">
            {list?.environments.map((e) => (
              <li key={e.id}>
                <button
                  type="button" onClick={() => setEnvId(e.id)}
                  aria-current={e.id === envId ? "true" : undefined}
                  className={clsx(
                    "flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-2.5 text-left",
                    e.id === envId
                      ? "border-ink bg-ink text-bg"
                      : "border-line bg-bg hover:bg-bg-raised",
                  )}
                >
                  <span className={clsx("text-[13px] font-medium",
                                        e.id === envId ? "text-bg" : "text-ink")}>
                    {e.plain.title}
                  </span>
                  <span className={clsx("shrink-0 text-[10.5px]",
                                        e.id === envId ? "text-bg/70" : "text-ink-faint")}>
                    {e.plain.dynamics}
                  </span>
                </button>
              </li>
            ))}
          </ul>

          {env && <Ladder env={env} result={play?.result ?? null} />}

          {plain && (
            <p className="mt-4 text-[11px] leading-relaxed text-ink-faint">
              {plain.caveat}
            </p>
          )}
          {list && (
            <p className="mt-3 text-[11px] leading-relaxed text-ink-faint">
              This is the real environment, not a practice copy — each step runs
              the same simulator the measurements came from. Ladder figures are
              twelve runs each, measured {list.measured_on}.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
