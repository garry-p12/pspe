"use client";

/**
 * How much notice this reach's upstream gauge actually gives.
 *
 * The card above this one says which roads close. It does not say whether
 * anyone would know in time, and that is a different question with a
 * different answer: it depends on how far upstream the nearest instrument is
 * and whether it was recording when it mattered.
 *
 * Measured from the record, not modelled — the lag between the upstream
 * gauge's peak and the outlet's on real floods. What makes it worth a panel
 * rather than a number is the failures: on Onion Creek the gauge has never
 * recorded a flood above 800 m³/s, and the two largest since it was built
 * left no trace at all. A median over the days it worked would describe the
 * warning by the floods nobody needed warning for.
 */

import { useEffect, useState } from "react";

import { Explain } from "./Explain";

export interface WarningEvent {
  date: string;
  peak_m3s: number;
  lag_min: number | null;
  why: string | null;
}

export interface WarningReply {
  ok: boolean;
  reason?: string;
  upstream?: { site: string; name: string };
  downstream?: { site: string; name: string };
  events?: WarningEvent[];
  measured?: number;
  tested?: number;
  median_lag_min?: number | null;
  min_lag_min?: number | null;
  blind_on?: string[];
  before_gauge_existed?: string[];
  upstream_record_from?: string | null;
}

/** Under two hours is not a warning anyone can act on at city scale: it is
 * the lead time below which the evacuation study stops finding a route
 * out for everyone, measured in `pspe/envs/evacuate.py`. */
const SHORT_MIN = 120;

function hhmm(min: number): string {
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  return h ? `${h} h ${m ? `${m} min` : ""}`.trim() : `${m} min`;
}

export function WarningTime({ box, name }: { box: object | null; name: string | null }) {
  const [reply, setReply] = useState<WarningReply | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!box) return;
    let live = true;
    setBusy(true);
    fetch("/api/warning", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...box, name: name ?? "", events: 14 }),
    })
      .then((r) => r.json())
      .then((d) => { if (live) setReply(d); })
      .catch(() => { if (live) setReply(null); })
      .finally(() => { if (live) setBusy(false); });
    return () => { live = false; };
  }, [box, name]);

  if (!box) return null;

  if (busy && !reply) {
    return (
      <div className="border-b border-line-soft px-5 py-3">
        <h3 className="eyebrow">Warning time</h3>
        <p className="anno mt-1">reading the gauge record…</p>
      </div>
    );
  }
  if (!reply?.ok) {
    if (!reply?.reason) return null;
    return (
      <div className="border-b border-line-soft px-5 py-3">
        <h3 className="eyebrow">Warning time</h3>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{reply.reason}</p>
      </div>
    );
  }

  const seen = (reply.events ?? []).filter(
    (e) => !(reply.before_gauge_existed ?? []).includes(e.date));
  const blind = reply.blind_on ?? [];
  const short = seen.filter((e) => e.lag_min !== null && e.lag_min < SHORT_MIN);

  return (
    <div className="border-b border-line-soft px-5 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="eyebrow shrink-0">Warning time</h3>
        <span className="anno truncate">
          from {reply.upstream?.name.split(",")[0]}
        </span>
      </div>

      <p data-warning-lag
         className="mt-1 text-[22px] font-semibold leading-tight tracking-tight text-ink tnum">
        {reply.median_lag_min ? hhmm(reply.median_lag_min) : "—"}
      </p>
      <p className="anno mt-0.5">
        median on the {reply.measured} flood{reply.measured === 1 ? "" : "s"} it
        recorded, of {seen.length} since {reply.upstream_record_from?.slice(0, 4)}
      </p>

      {blind.length > 0 && (
        <div className="mt-2 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">
            No warning on the {blind.length === 1 ? "largest flood" : `${blind.length} largest floods`}
          </p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
            The upstream gauge left no usable trace on{" "}
            {blind.map((d) => d).join(" or ")} — the biggest{" "}
            {blind.length === 1 ? "event" : "events"} since it was built. Gauges
            fail in the floods worth warning for.
          </p>
        </div>
      )}

      {short.length > 0 && (
        <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
          And it is not always this much: {short.map((e) => e.date).join(", ")}{" "}
          gave {short.map((e) => `${e.lag_min} min`).join(", ")}.
        </p>
      )}

      <ul className="mt-2">
        {seen.map((e) => (
          <li key={e.date}
              className="flex items-baseline gap-2 border-b border-line-soft py-1.5 last:border-0">
            <span className="flex-1 truncate text-[13px] text-ink">{e.date}</span>
            <span className="tnum text-[13px] text-ink-mute">
              {e.peak_m3s.toFixed(0)} m³/s
            </span>
            <span className={`w-[64px] text-right text-[13px] tnum ${
              e.lag_min === null ? "font-medium text-bad"
                : e.lag_min < SHORT_MIN ? "font-medium text-ink"
                : "text-ink-mute"}`}>
              {e.lag_min === null ? "none" : `${e.lag_min} min`}
            </span>
          </li>
        ))}
      </ul>

      <p className="anno mt-1.5 flex items-center gap-1.5 leading-relaxed">
        measured from the gauge record
        <Explain title="What this is and is not">
          <p>
            The lag between the upstream gauge&apos;s peak and the
            outlet&apos;s, on floods that actually happened. Not a model.
          </p>
          <p>
            Floods before the upstream gauge was built are excluded rather
            than counted as failures — {reply.before_gauge_existed?.length ?? 0}{" "}
            of the peaks on this reach predate it. &ldquo;Not yet
            instrumented&rdquo; and &ldquo;instrumented and silent&rdquo; are
            different facts, and only the second says anything about a warning
            system.
          </p>
          <p>
            A lag is only meaningful when the wave routed from one gauge to the
            other. A storm centred below the upstream gauge peaks downstream
            first, which shows here as a negative or near-zero lag.
          </p>
        </Explain>
      </p>
    </div>
  );
}
