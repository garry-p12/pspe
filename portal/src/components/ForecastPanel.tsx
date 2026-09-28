"use client";

import { useEffect, useState } from "react";

interface Band { level: "quiet" | "watch" | "act"; headline: string; detail: string }
interface Forecast {
  issued: string;
  total_mm: number;
  max_48h_mm: number;
  reference_event_48h_mm: number;
  fraction_of_reference: number;
  peak_discharge_m3s: number;
  daily: { date: string; mm: number }[];
  band: Band;
  sources: string[];
}

const TONE = {
  quiet: { dot: "bg-ok", text: "text-ok", ring: "border-ok/30", bg: "bg-ok/5" },
  watch: { dot: "bg-warn", text: "text-warn", ring: "border-warn/40", bg: "bg-warn/5" },
  act:   { dot: "bg-bad", text: "text-bad", ring: "border-bad/50", bg: "bg-bad/8" },
} as const;

export function ForecastPanel() {
  const [f, setF] = useState<Forecast | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch("/api/forecast")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setF)
      .catch(() => setErr(true));
  }, []);

  if (err) {
    return (
      <div className="border-b border-line-soft p-4">
        <p className="eyebrow mb-1">Live forecast</p>
        <p className="text-[13px] text-ink-faint">
          Forecast feed unavailable. Design-event planning below is unaffected.
        </p>
      </div>
    );
  }
  if (!f) {
    return (
      <div className="border-b border-line-soft p-4">
        <div className="h-16 animate-pulse rounded-lg bg-bg-inset" />
      </div>
    );
  }

  const t = TONE[f.band.level];
  const maxDay = Math.max(1, ...f.daily.map((d) => d.mm));
  const issued = new Date(f.issued);

  return (
    <div className={`border-b border-line-soft p-4`}>
      <div className="mb-2 flex items-center justify-between">
        <p className="eyebrow">Next 7 days</p>
        <span className="text-[11px] text-ink-faint">
          {issued.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>

      <div className={`rounded-lg border ${t.ring} ${t.bg} px-3 py-2.5`}>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${t.dot}`} />
          <p className={`text-[15px] font-semibold ${t.text}`}>{f.band.headline}</p>
        </div>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
          {f.band.detail}
        </p>
      </div>

      <div className="mt-3 flex items-end gap-[3px]" aria-hidden>
        {f.daily.map((d) => (
          <div key={d.date} className="flex-1" title={`${d.date}: ${d.mm.toFixed(1)} mm`}>
            <div
              className="w-full rounded-sm bg-accent/60"
              style={{ height: `${Math.max(2, (d.mm / maxDay) * 34)}px` }}
            />
            <p className="mt-1 text-center text-[11px] text-ink-faint">
              {new Date(d.date).toLocaleDateString(undefined, { weekday: "narrow" })}
            </p>
          </div>
        ))}
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 border-t border-line-soft pt-3">
        <div>
          <dt className="eyebrow mb-0.5">Worst 48 h</dt>
          <dd className="tnum text-[15px] font-semibold text-ink">
            {f.max_48h_mm.toFixed(0)} mm
          </dd>
          <dd className="text-[11px] text-ink-faint">
            {(f.fraction_of_reference * 100).toFixed(0)}% of the 2022 event
          </dd>
        </div>
        <div>
          <dt className="eyebrow mb-0.5">Peak river flow</dt>
          <dd className="tnum text-[15px] font-semibold text-ink">
            {f.peak_discharge_m3s.toFixed(1)} m³/s
          </dd>
          <dd className="text-[11px] text-ink-faint">forecast maximum</dd>
        </div>
      </dl>
    </div>
  );
}
