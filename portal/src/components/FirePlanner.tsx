"use client";

/**
 * The wildfire half of the portal.
 *
 * It answers the same three questions the flood side answers, in the same
 * order, because they are the questions a hazard decision actually has:
 * what is happening, what happens next, and what should we do about it.
 *
 *   Observe   live NASA FIRMS detections over a box, anywhere on Earth
 *   Forecast  tomorrow's fire from a trained spread model, SCORED against
 *             the day it predicted
 *   Plan      where crews cut, under a daily budget, against two references
 *
 * The evidence split is the thing to preserve. Observe is live and works
 * anywhere. Forecast and Plan run on held-out records of real US fires,
 * because the spread model needs eleven remote-sensing drivers and those are
 * a prepared dataset rather than a feed. Every panel says which it is; a
 * forecast for a live box would mean inventing drivers, and the service
 * refuses rather than doing that.
 */

import { useCallback, useEffect, useState } from "react";

import { Explain } from "./Explain";

type Evidence = "observed" | "record" | "projection";

interface RecordRow {
  id: number;
  burned_next_pct: number;
  burning_now_pct: number;
  grew: boolean;
}

interface RecordReply {
  ok: boolean;
  reason?: string;
  id?: number;
  grid?: number;
  burning_now_png?: string;
  burns_next_png?: string;
  burning_now_pct?: number;
  burns_next_pct?: number;
  drivers?: Record<string, { label: string; png: string }>;
  evidence?: Evidence;
  note?: string;
}

interface ForecastReply {
  ok: boolean;
  reason?: string;
  forecast_png?: string;
  truth_png?: string;
  csi?: number;
  average_precision?: number | null;
  hit?: number;
  false_alarm?: number;
  missed?: number;
  burned_truth_pct?: number;
  burned_forecast_pct?: number;
  note?: string;
}

interface PlanReply {
  ok: boolean;
  reason?: string;
  treatment_png?: string;
  spent_fraction?: number;
  within_budget?: boolean;
  burned?: { none: number; uniform: number; planned: number };
  reduction_vs_none_pct?: number;
  reduction_vs_uniform_pct?: number;
  uniform_reduction_pct?: number;
  seconds?: number;
  note?: string;
}

interface SkillReply {
  ok: boolean;
  ours?: number;
  ours_sd?: number;
  originating_baseline?: { name: string; value: number };
  state_of_the_art?: { name: string; value: number }[];
  all_zeros_floor?: number;
  note?: string;
}

const BADGE: Record<Evidence, { label: string; cls: string }> = {
  observed: { label: "OBSERVED", cls: "bg-[var(--badge-observed)] text-bg" },
  record: { label: "RECORD", cls: "bg-[var(--badge-validated)] text-bg" },
  projection: { label: "MODEL", cls: "bg-[var(--badge-projection)] text-bg" },
};

function Badge({ kind }: { kind: Evidence }) {
  const b = BADGE[kind];
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${b.cls}`}>
      {b.label}
    </span>
  );
}

/** A 64x64 raster, drawn big and crisp. Fire is a few cells; smoothing hides it. */
function Raster({ png, alt, dim }: { png?: string; alt: string; dim?: boolean }) {
  if (!png) return <div className="aspect-square w-full rounded-lg bg-bg-inset" />;
  return (
    <img
      src={`data:image/png;base64,${png}`}
      alt={alt}
      className={`aspect-square w-full rounded-lg bg-bg-inset ${dim ? "opacity-70" : ""}`}
      style={{ imageRendering: "pixelated" }}
    />
  );
}

/** Two rasters on the same footprint, the second over the first. */
function Stack({ base, over, alt }: { base?: string; over?: string; alt: string }) {
  return (
    <div className="relative">
      <Raster png={base} alt={alt} dim />
      {over && (
        <img
          src={`data:image/png;base64,${over}`}
          alt=""
          className="absolute inset-0 aspect-square w-full rounded-lg"
          style={{ imageRendering: "pixelated" }}
        />
      )}
    </div>
  );
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-line bg-bg-inset p-3">
      <p className="anno">{label}</p>
      <p className="mt-0.5 text-[20px] font-semibold leading-tight tracking-tight text-ink tnum">
        {value}
      </p>
      {sub ? <p className="anno mt-0.5">{sub}</p> : null}
    </div>
  );
}

export function FirePlanner() {
  const [tab, setTab] = useState<"forecast" | "plan" | "observe">("forecast");
  const [rows, setRows] = useState<RecordRow[]>([]);
  const [picked, setPicked] = useState<number | null>(null);
  const [rec, setRec] = useState<RecordReply | null>(null);
  const [fc, setFc] = useState<ForecastReply | null>(null);
  const [plan, setPlan] = useState<PlanReply | null>(null);
  const [skill, setSkill] = useState<SkillReply | null>(null);
  const [budget, setBudget] = useState(3);
  const [horizon, setHorizon] = useState(3);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/fire/records")
      .then((r) => r.json())
      .then((d) => {
        if (d?.ok) {
          setRows(d.records ?? []);
          if (d.records?.length) setPicked(d.records[0].id);
        } else setFailed(d?.reason ?? d?.detail ?? "no records");
      })
      .catch(() => setFailed("the hazard service is not reachable"));
    fetch("/api/fire/skill").then((r) => r.json()).then(setSkill).catch(() => {});
  }, []);

  useEffect(() => {
    if (picked === null) return;
    setRec(null); setFc(null); setPlan(null);
    setBusy("record");
    fetch("/api/fire/record", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: picked }),
    })
      .then((r) => r.json()).then(setRec)
      .finally(() => setBusy(null));
  }, [picked]);

  const runForecast = useCallback(() => {
    if (picked === null) return;
    setBusy("forecast");
    fetch("/api/fire/forecast", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: picked }),
    })
      .then((r) => r.json()).then(setFc)
      .finally(() => setBusy(null));
  }, [picked]);

  const runPlan = useCallback(() => {
    if (picked === null) return;
    setBusy("plan");
    fetch("/api/fire/plan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: picked, budget: budget / 100, horizon, steps: 60 }),
    })
      .then((r) => r.json()).then(setPlan)
      .finally(() => setBusy(null));
  }, [picked, budget, horizon]);

  if (failed) {
    return (
      <main className="mx-auto max-w-2xl px-5 py-16">
        <h1 className="text-[22px] font-semibold text-ink">Wildfire planner</h1>
        <div className="mt-4 rounded-lg border border-line bg-bg-inset p-4">
          <p className="text-[13px] font-medium text-ink">Not available here</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{failed}</p>
        </div>
      </main>
    );
  }

  return (
    <div className="flex h-dvh flex-col lg:flex-row">
      {/* ---- left rail -------------------------------------------------- */}
      <aside className="flex w-full shrink-0 flex-col overflow-y-auto border-r border-line
                        bg-bg-raised lg:w-[380px]">
        <div className="border-b border-line px-5 py-4">
          <h1 className="text-[15px] font-semibold text-ink">Wildfire planner</h1>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
            What is burning, what burns next, and where to put the crews.{" "}
            <span className="text-ink-mute">
              Observe is live and works anywhere; Forecast and Plan run on real
              held-out fire days.
            </span>{" "}
            <Explain title="What is live here">
              <p>
                <strong>Observe</strong> is live and works anywhere: NASA FIRMS
                serves satellite fire detections for any box on Earth.
              </p>
              <p>
                <strong>Forecast</strong> and <strong>Plan</strong> run on
                held-out records — real US fire days with real drivers and a
                real trained model, on a day that has already happened. The
                spread model needs eleven remote-sensing drivers, and those are
                a prepared dataset rather than a live feed. Forecasting a live
                box would mean inventing them.
              </p>
            </Explain>
          </p>
          <div className="mt-3 flex gap-1">
            {(["forecast", "plan", "observe"] as const).map((t) => (
              <button
                key={t}
                data-on={tab === t}
                onClick={() => setTab(t)}
                className="rounded-md px-3 py-1.5 text-[13px] capitalize
                           data-[on=true]:bg-ink data-[on=true]:text-bg"
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* record picker -------------------------------------------------- */}
        {tab !== "observe" && (
          <div className="border-b border-line px-5 py-4">
            <div className="flex items-baseline justify-between">
              <h3 className="eyebrow">Pick a fire</h3>
              <Badge kind="record" />
            </div>
            <p className="anno mt-1">
              held-out days, biggest next-day growth first
            </p>
            {/* Six seconds cold: the service parses the held-out split and
                enough of the training split to recover the driver statistics.
                An empty list for six seconds reads as "no fires here", which
                is a different claim from "still reading". */}
            {!rows.length && (
              <p className="anno mt-2">reading the fire records…</p>
            )}
            <ul className="mt-2 max-h-[220px] overflow-y-auto">
              {rows.map((r) => (
                <li key={r.id}>
                  <button
                    aria-pressed={picked === r.id}
                    onClick={() => setPicked(r.id)}
                    className="flex w-full items-baseline justify-between gap-2 rounded-md
                               px-2 py-1.5 text-left text-[13px]
                               aria-pressed:bg-ink aria-pressed:text-bg"
                  >
                    <span>fire {r.id}</span>
                    <span className="tnum">
                      {r.burning_now_pct}% → {r.burned_next_pct}%
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* forecast ------------------------------------------------------- */}
        {tab === "forecast" && (
          <div className="border-b border-line px-5 py-4">
            <h3 className="text-[15px] font-semibold text-ink">
              Where it goes tomorrow
            </h3>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
              The model sees today and the drivers, never tomorrow. Then we
              score it against tomorrow.
            </p>
            <button
              className="mt-3 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                         text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
              disabled={busy !== null || picked === null}
              onClick={runForecast}
            >
              {busy === "forecast" ? "Forecasting…" : "Forecast this fire"}
            </button>

            {fc?.ok && (
              <>
                <div className="mt-4 grid grid-cols-2 gap-2">
                  <Tile label="Found" value={`${fc.hit}`} sub="burning cells" />
                  <Tile label="Missed" value={`${fc.missed}`} sub="burning cells" />
                  <Tile label="False alarms" value={`${fc.false_alarm}`} sub="cells" />
                  <Tile
                    label="Average precision"
                    value={fc.average_precision == null ? "—" : fc.average_precision.toFixed(3)}
                    sub="on this fire"
                  />
                </div>
                <p className="anno mt-2 leading-relaxed">
                  Burned for real {fc.burned_truth_pct}% · model said{" "}
                  {fc.burned_forecast_pct}%
                </p>
                <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
                  {fc.note}
                </p>
              </>
            )}
            {fc && !fc.ok && (
              <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">{fc.reason}</p>
            )}

            {skill?.ok && (
              <div className="mt-4 rounded-lg border border-line bg-bg-inset p-3">
                <p className="text-[13px] font-medium text-ink">
                  How good is this forecaster?
                </p>
                <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
                  {skill.ours?.toFixed(4)} AUC-PR on the held-out split, against{" "}
                  {skill.originating_baseline?.value} for the dataset&apos;s own
                  baseline and {skill.state_of_the_art?.[1]?.value} for the
                  current state of the art. <strong>We are behind the field.</strong>{" "}
                  Forecast skill is an input to the plan, not the contribution.
                </p>
              </div>
            )}
          </div>
        )}

        {/* plan ----------------------------------------------------------- */}
        {tab === "plan" && (
          <div className="border-b border-line px-5 py-4">
            <h3 className="text-[15px] font-semibold text-ink">Where the crews go</h3>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
              A fixed share of the patch can be treated each day. The decision
              is where — and the comparison that matters is against spreading
              the same crew-days evenly.
            </p>

            <label htmlFor="fire-budget" className="mt-3 flex items-baseline justify-between">
              <span className="eyebrow">Daily crew budget</span>
              <span className="tnum text-[13px] text-ink">{budget}% of the patch</span>
            </label>
            <input id="fire-budget" type="range" min={1} max={10} step={1}
                   value={budget} onChange={(e) => setBudget(Number(e.target.value))}
                   className="mt-1 w-full" />

            <label htmlFor="fire-h" className="mt-3 flex items-baseline justify-between">
              <span className="eyebrow">Days ahead</span>
              <span className="tnum text-[13px] text-ink">{horizon}</span>
            </label>
            <input id="fire-h" type="range" min={1} max={5} step={1}
                   value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}
                   className="mt-1 w-full" />
            <p className="anno mt-0.5">
              at one day a greedy allocation is already optimal
            </p>

            <button
              className="mt-4 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                         text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
              disabled={busy !== null || picked === null}
              onClick={runPlan}
            >
              {busy === "plan" ? "Planning…" : "Plan the breaks"}
            </button>

            {plan?.ok && plan.burned && (
              <>
                <div className="mt-4 grid grid-cols-2 gap-2">
                  <Tile
                    label="Against doing nothing"
                    value={`−${plan.reduction_vs_none_pct}%`}
                    sub="burned, population-weighted"
                  />
                  <Tile
                    label="Against spreading evenly"
                    value={`−${plan.reduction_vs_uniform_pct}%`}
                    sub="the comparison that counts"
                  />
                </div>
                <ul className="mt-3">
                  {([["do nothing", plan.burned.none],
                     ["spread evenly", plan.burned.uniform],
                     ["planned", plan.burned.planned]] as const).map(([k, v], i) => (
                    <li key={k}
                        className="flex items-baseline justify-between border-b border-line-soft
                                   py-1.5 text-[13px] last:border-0">
                      <span className={i === 2 ? "font-medium text-ink" : "text-ink-mute"}>{k}</span>
                      <span className="tnum text-ink">{v.toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
                <p className="anno mt-2">
                  spent {(plan.spent_fraction! * 100).toFixed(1)}% of the patch ·{" "}
                  {plan.within_budget ? "within budget" : "OVER BUDGET"} ·{" "}
                  {plan.seconds}s
                </p>
                <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">{plan.note}</p>
              </>
            )}
            {plan && !plan.ok && (
              <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">{plan.reason}</p>
            )}
          </div>
        )}

        {tab === "observe" && <ObservePanel />}
      </aside>

      {/* ---- the rasters ------------------------------------------------- */}
      <main className="flex-1 overflow-y-auto bg-bg p-5">
        {tab !== "observe" && (
          <>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
              <figure>
                <div className="mb-1.5 flex items-baseline justify-between">
                  <figcaption className="text-[13px] font-medium text-ink">
                    Burning today
                  </figcaption>
                  <Badge kind="observed" />
                </div>
                <Raster png={rec?.burning_now_png} alt="fire today" />
                <p className="anno mt-1">{rec?.burning_now_pct ?? "—"}% of the patch</p>
              </figure>

              <figure>
                <div className="mb-1.5 flex items-baseline justify-between">
                  <figcaption className="text-[13px] font-medium text-ink">
                    Burned tomorrow
                  </figcaption>
                  <Badge kind="observed" />
                </div>
                <Raster png={rec?.burns_next_png} alt="fire tomorrow" />
                <p className="anno mt-1">{rec?.burns_next_pct ?? "—"}% — what happened</p>
              </figure>

              {fc?.ok && (
                <figure>
                  <div className="mb-1.5 flex items-baseline justify-between">
                    <figcaption className="text-[13px] font-medium text-ink">
                      The model&apos;s guess
                    </figcaption>
                    <Badge kind="projection" />
                  </div>
                  <Stack base={fc.truth_png} over={fc.forecast_png} alt="forecast over truth" />
                  <p className="anno mt-1">
                    bright = forecast · faint = what burned
                  </p>
                </figure>
              )}

              {plan?.ok && (
                <figure>
                  <div className="mb-1.5 flex items-baseline justify-between">
                    <figcaption className="text-[13px] font-medium text-ink">
                      Where the crews cut
                    </figcaption>
                    <Badge kind="projection" />
                  </div>
                  <Stack base={rec?.burning_now_png} over={plan.treatment_png}
                         alt="treatment over fire" />
                  <p className="anno mt-1">blue = treated · orange = burning now</p>
                </figure>
              )}
            </div>

            {rec?.drivers && (
              <>
                <h3 className="eyebrow mt-6">What the model is reading</h3>
                <p className="anno mt-1">
                  four of the eleven drivers behind the forecast
                </p>
                <div className="mt-2 grid grid-cols-2 gap-4 sm:grid-cols-4">
                  {Object.entries(rec.drivers).map(([k, v]) => (
                    <figure key={k}>
                      <Raster png={v.png} alt={v.label} />
                      <figcaption className="anno mt-1 capitalize">{v.label}</figcaption>
                    </figure>
                  ))}
                </div>
              </>
            )}
          </>
        )}
        {tab === "observe" && <ObserveMap />}
      </main>
    </div>
  );
}

/* --------------------------------------------------------------------------
   Observe: the live half.
   -------------------------------------------------------------------------- */
interface ObserveDay {
  date: string;
  observed: boolean;
  cells: number;
  png: string;
}
interface ObserveReply {
  ok: boolean;
  reason?: string;
  days?: ObserveDay[];
  any_fire?: boolean;
  blind_days?: number;
  note?: string;
}

/** Shared between the rail and the map, so both see the same fetch. */
let OBSERVE_CACHE: ObserveReply | null = null;
const OBSERVE_SUBS = new Set<(r: ObserveReply | null) => void>();
function setObserve(r: ObserveReply | null) {
  OBSERVE_CACHE = r;
  OBSERVE_SUBS.forEach((f) => f(r));
}

function useObserve() {
  const [r, setR] = useState<ObserveReply | null>(OBSERVE_CACHE);
  useEffect(() => {
    OBSERVE_SUBS.add(setR);
    return () => { OBSERVE_SUBS.delete(setR); };
  }, []);
  return r;
}

/** A few places that are usually burning somewhere in fire season. */
const BOXES: { name: string; box: number[] }[] = [
  { name: "Northern California", box: [-124.0, 38.5, -120.0, 41.5] },
  { name: "Pacific Northwest", box: [-124.0, 42.0, -118.0, 46.5] },
  { name: "Southern California", box: [-120.0, 33.0, -116.0, 35.5] },
  { name: "Interior Alaska", box: [-152.0, 63.0, -144.0, 66.0] },
];

function ObservePanel() {
  const r = useObserve();
  const [busy, setBusy] = useState(false);
  const [which, setWhich] = useState(0);

  const run = useCallback((i: number) => {
    const b = BOXES[i].box;
    setWhich(i); setBusy(true); setObserve(null);
    fetch("/api/fire/observe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ west: b[0], south: b[1], east: b[2], north: b[3], days: 7 }),
    })
      .then((x) => x.json()).then(setObserve)
      .catch(() => setObserve({ ok: false, reason: "the service is not reachable" }))
      .finally(() => setBusy(false));
  }, []);

  return (
    <div className="border-b border-line px-5 py-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-[15px] font-semibold text-ink">What is burning now</h3>
        <Badge kind="observed" />
      </div>
      <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
        NASA FIRMS satellite detections, last seven days. This half is live and
        works anywhere on Earth.
      </p>

      <ul className="mt-3">
        {BOXES.map((b, i) => (
          <li key={b.name}>
            <button
              aria-pressed={which === i}
              onClick={() => run(i)}
              disabled={busy}
              className="w-full rounded-md px-2 py-1.5 text-left text-[13px]
                         aria-pressed:bg-ink aria-pressed:text-bg disabled:opacity-40"
            >
              {b.name}
            </button>
          </li>
        ))}
      </ul>

      {busy && <p className="anno mt-3">asking the satellite record…</p>}

      {r && !r.ok && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">No live feed here</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{r.reason}</p>
        </div>
      )}

      {r?.ok && (
        <>
          <ul className="mt-3">
            {r.days?.map((d) => (
              <li key={d.date}
                  className="flex items-baseline justify-between border-b border-line-soft
                             py-1.5 text-[13px] last:border-0">
                <span className="text-ink">{d.date}</span>
                <span className={d.observed ? "tnum text-ink-mute" : "text-bad"}>
                  {d.observed ? `${d.cells} cells` : "not observed"}
                </span>
              </li>
            ))}
          </ul>
          {!!r.blind_days && (
            <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
              {r.blind_days} of {r.days?.length} days had no satellite coverage.
              Those are marked unobserved rather than drawn as an empty fire —
              &ldquo;we did not look&rdquo; and &ldquo;nothing burned&rdquo; are
              different facts.
            </p>
          )}
          {!r.any_fire && (
            <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
              Nothing detected in this box over the window. That is a real
              answer, not an error.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function ObserveMap() {
  const r = useObserve();
  if (!r?.ok || !r.days?.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-[13px] text-ink-mute">
          Pick a region to pull the last seven days of detections.
        </p>
      </div>
    );
  }
  return (
    <>
      <h3 className="eyebrow">Seven days of detections</h3>
      <p className="anno mt-1">
        each frame is the active front at satellite overpass, not the burn scar
      </p>
      <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
        {r.days.map((d) => (
          <figure key={d.date}>
            <Raster png={d.png} alt={d.date} dim={!d.observed} />
            <figcaption className="anno mt-1">
              {d.date.slice(5)} ·{" "}
              {d.observed ? `${d.cells}` : <span className="text-bad">no pass</span>}
            </figcaption>
          </figure>
        ))}
      </div>
    </>
  );
}
