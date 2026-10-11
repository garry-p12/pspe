"use client";

/**
 * Operating the structures a plan placed: a schedule, not a design.
 *
 * `Build a plan` answers what to build. This answers how to run it, which has
 * a different kind of answer -- not one number but an hour-by-hour schedule,
 * and not one schedule but a comparison, because the result worth knowing is
 * usually that the obvious controller is close to the best one and far
 * cheaper than pumping flat out.
 *
 * Nothing here invents a polder. The gate and the pump are the ones on the
 * map, and with none placed the panel says what is missing rather than
 * showing a demonstration.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Explain } from "./Explain";

export interface OperateFrame {
  t_h: number;
  gate: number;
  pump: number;
  depth_png: string;
}

export interface OperateResult {
  policy: string;
  label: string;
  return: number;
  pump_hours: number;
  within_budget: boolean;
  wet_km2: number;
  deepest_m: number;
  frames: OperateFrame[];
}

export interface OperateReply {
  ok: boolean;
  reason?: string;
  stage?: string;
  pending?: boolean;
  elapsed_s?: number;
  name?: string;
  bounds?: number[];
  protected_basis?: string;
  protected_km2?: number;
  budget_pump_hours?: number;
  rated_pump_m3s?: number;
  depth_vmax_m?: number;
  results?: OperateResult[];
  best?: string;
  seconds?: number;
  note?: string;
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

/** The schedule itself: one column per control interval, gate over pump. */
function Schedule({ frames }: { frames: OperateFrame[] }) {
  if (!frames.length) return null;
  return (
    <div className="mt-2">
      <div className="flex items-end gap-[2px]" aria-hidden>
        {frames.map((f, i) => (
          <div key={i} className="flex-1">
            <div className="h-5 w-full rounded-[2px] bg-[var(--gate)]"
                 style={{ opacity: 0.18 + 0.82 * f.gate }} />
            <div className="mt-[2px] h-5 w-full rounded-[2px] bg-[var(--pump)]"
                 style={{ opacity: 0.18 + 0.82 * f.pump }} />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between">
        <span className="anno">0 h</span>
        <span className="anno">{frames[frames.length - 1].t_h.toFixed(1)} h</span>
      </div>
      <p className="anno mt-0.5">
        top: gate open · bottom: pump running · darker is more
      </p>
    </div>
  );
}

export function OperatePanel({
  canRun,
  busy,
  elapsed,
  reply,
  onRun,
  onPick,
  picked,
  frame,
  onFrame,
}: {
  canRun: boolean;
  busy: boolean;
  elapsed: number;
  reply: OperateReply | null;
  onRun: (budget: number) => void;
  onPick: (policy: string) => void;
  picked: string | null;
  frame: number | null;
  onFrame: (i: number | null) => void;
}) {
  const [budget, setBudget] = useState(3);
  const chosen = reply?.results?.find((r) => r.policy === picked) ?? reply?.results?.[0];

  return (
    <div className="border-b border-line px-5 py-4">
      <h3 className="text-[15px] font-semibold text-ink">Run the structures</h3>
      <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
        A gate drains for nothing when the river is low and seals when it is
        high. A pump works whenever you like and costs fuel. The whole decision
        is when to spend it.{" "}
        <Explain title="What this compares">
          <p>
            Every row below runs the <em>same</em> structures on the same
            terrain through the same event. Only the schedule differs, so the
            difference between two rows is the operating decision and nothing
            else.
          </p>
          <p>
            No observational record says what this ground would have done under
            a schedule other than the one that was run, so these are model runs
            compared with each other, not with an outcome.
          </p>
        </Explain>
      </p>

      {!canRun && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">Nothing to operate yet</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
            Place a <strong>Floodgate</strong> or a <strong>Pump station</strong>{" "}
            on a bank first. A levee has no setting — it is either there or it
            is not — so there is no schedule to choose until one of those is on
            the map.
          </p>
        </div>
      )}

      {canRun && (
        <>
          <label htmlFor="op-budget" className="mt-3 flex items-baseline justify-between">
            <span className="eyebrow">Fuel for the pumps</span>
            <span className="tnum text-[13px] text-ink">{budget} pump-hours</span>
          </label>
          <input
            id="op-budget"
            type="range"
            min={0}
            max={12}
            step={0.5}
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
            className="mt-1 w-full"
          />
          <p className="anno mt-0.5">at the rated capacity of what you placed</p>

          <button
            /* The same primary-action styling the street view uses. The first
               version reached for a `btn-primary` class that does not exist
               in this stylesheet, so the control rendered as bare centred
               text and did not read as a button at all. */
            className="mt-4 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium text-bg
                       transition-opacity hover:opacity-90 disabled:opacity-40"
            disabled={busy}
            onClick={() => onRun(budget)}
          >
            {busy ? `Running the event… ${elapsed}s` : "Run the event"}
          </button>
        </>
      )}

      {reply && !reply.ok && reply.reason && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">
            {reply.stage === "separation"
              ? "No operating decision here"
              : "Could not run this"}
          </p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{reply.reason}</p>
          {reply.stage === "separation" && (
            <p className="anno mt-2">
              This is a result, not a failure: four identical rows presented
              as a choice would be the misleading answer.
            </p>
          )}
        </div>
      )}

      {reply?.ok && reply.results && (
        <>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <Tile
              label="Best schedule"
              value={chosen ? `${chosen.wet_km2.toFixed(2)} km²` : "—"}
              sub="left wet behind the levee"
            />
            <Tile
              label="Fuel it used"
              value={chosen ? `${chosen.pump_hours.toFixed(1)} h` : "—"}
              sub={`of ${reply.budget_pump_hours} allowed`}
            />
          </div>

          <h4 className="eyebrow mt-4">Ways of running it</h4>
          <ul className="mt-1">
            {reply.results.map((r) => {
              const on = (picked ?? reply.results![0].policy) === r.policy;
              return (
                <li key={r.policy}>
                  <button
                    aria-pressed={on}
                    onClick={() => onPick(r.policy)}
                    className="flex w-full items-baseline justify-between gap-2
                               rounded-md px-2 py-2 text-left
                               aria-pressed:bg-ink aria-pressed:text-bg"
                  >
                    <span className="text-[13px]">{r.label}</span>
                    <span className="tnum shrink-0 text-[13px]">
                      {r.wet_km2.toFixed(2)} km²
                      <span className={on ? "opacity-70" : "text-ink-mute"}>
                        {" · "}{r.pump_hours.toFixed(1)} h
                      </span>
                      {!r.within_budget && (
                        <span className="ml-1 text-warn">over</span>
                      )}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>

          {chosen && <Schedule frames={chosen.frames} />}

          {chosen && chosen.frames.length > 1 && (
            <div className="mt-3">
              <label htmlFor="op-t" className="flex items-baseline justify-between">
                <span className="eyebrow">When</span>
                <span className="tnum text-[13px] text-ink">
                  {frame === null ? "peak" : `${chosen.frames[frame].t_h.toFixed(1)} h`}
                </span>
              </label>
              <input
                id="op-t"
                type="range"
                min={-1}
                max={chosen.frames.length - 1}
                step={1}
                value={frame ?? -1}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  onFrame(v < 0 ? null : v);
                }}
                className="mt-1 w-full"
              />
            </div>
          )}

          <p className="anno mt-3">
            {reply.protected_km2?.toFixed(2)} km² scored ·{" "}
            {reply.protected_basis}
          </p>
        </>
      )}
    </div>
  );
}
