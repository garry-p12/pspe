"use client";

import { useEffect, useState } from "react";

interface Band { level: "quiet" | "watch" | "act"; headline: string; detail: string }
interface Forecast {
  issued: string;
  reference_event_48h_mm: number | null;
  total_mm: number;
  max_48h_mm: number;
  fraction_of_reference: number | null;
  peak_discharge_m3s: number;
  daily: { date: string; mm: number }[];
  band: Band;
  sources: string[];
}

/* The rainfall a levee is designed against, so the chart has somewhere to put
 * the forecast. Without it the bars are decoration: 13 mm looks alarming drawn
 * at full height and means nothing until it is set beside the number that
 * matters. */
const FLOOD_THRESHOLD_MM = 60;

export function ForecastPanel({ lat, lon, place }: {
  lat?: number; lon?: number; place?: string;
} = {}) {
  const [f, setF] = useState<Forecast | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    setF(null); setErr(false);
    const q = lat != null && lon != null ? `?lat=${lat}&lon=${lon}` : "";
    fetch(`/api/forecast${q}`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setF)
      .catch(() => setErr(true));
  }, [lat, lon]);

  if (err) {
    return (
      <section className="px-5 py-4">
        <h2 className="eyebrow mb-1">Live forecast</h2>
        <p className="text-[13px] text-ink-faint">
          Forecast feed unavailable. Design-event planning is unaffected.
        </p>
      </section>
    );
  }
  if (!f) {
    return (
      <section className="px-5 py-4">
        <div className="h-20 animate-pulse rounded-lg bg-bg-inset" />
      </section>
    );
  }

  // The chart is scaled to the flood threshold, not to the tallest bar. A week
  // of nothing then reads as a week of nothing, instead of being stretched to
  // fill the axis and looking like weather.
  const scale = Math.max(FLOOD_THRESHOLD_MM, ...f.daily.map((d) => d.mm));
  const issued = new Date(f.issued);
  const CHART_H = 46;

  return (
    <section className="px-5 py-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="eyebrow">Next 7 days{place ? ` · ${place}` : ""}</h2>
        <span className="anno">
          Updated{" "}
          {issued.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}
        </span>
      </div>

      <div className="rounded-lg border border-line bg-bg-raised px-3.5 py-3">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-ink" />
          <p className="text-[14px] font-semibold text-ink">{f.band.headline}</p>
        </div>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{f.band.detail}</p>
      </div>

      <div className="relative mt-6 mb-1">
        <div
          className="absolute inset-x-0 border-t border-dashed border-line"
          style={{ bottom: `${CHART_H + 20}px` }}
        />
        <p
          className="anno absolute right-0 pb-1"
          style={{ bottom: `${CHART_H + 20}px` }}
        >
          flood threshold {FLOOD_THRESHOLD_MM} mm
        </p>
        <div className="flex items-end gap-2" style={{ height: `${CHART_H}px` }}>
          {f.daily.map((d) => (
            <div key={d.date} className="flex flex-1 flex-col justify-end">
              <p className="anno tnum mb-1 text-center">{d.mm.toFixed(0)}</p>
              <div
                className="w-full rounded-[2px] bg-ink"
                style={{
                  height: `${Math.max(2, (d.mm / scale) * CHART_H)}px`,
                  opacity: d.mm > 0 ? 1 : 0.18,
                }}
              />
            </div>
          ))}
        </div>
        <div className="mt-1.5 flex gap-2">
          {f.daily.map((d) => (
            <p key={d.date} className="anno flex-1 text-center">
              {new Date(d.date).toLocaleDateString(undefined, { weekday: "narrow" })}
            </p>
          ))}
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-line px-3.5 py-3">
          <dt className="text-[13px] text-ink-mute">Worst 48 h</dt>
          <dd className="tnum mt-0.5 text-[24px] font-semibold leading-none text-ink">
            {f.max_48h_mm.toFixed(0)}
            <span className="ml-1 text-[13px] font-normal text-ink-mute">mm</span>
          </dd>
          <dd className="anno mt-1.5">
            {f.fraction_of_reference != null
              ? `${(f.fraction_of_reference * 100).toFixed(0)}% of the 2022 event`
              : "no local benchmark event"}
          </dd>
        </div>
        <div className="rounded-lg border border-line px-3.5 py-3">
          <dt className="text-[13px] text-ink-mute">Peak river flow</dt>
          <dd className="tnum mt-0.5 text-[24px] font-semibold leading-none text-ink">
            {f.peak_discharge_m3s.toFixed(1)}
            <span className="ml-1 text-[13px] font-normal text-ink-mute">m³/s</span>
          </dd>
          <dd className="anno mt-1.5">Forecast maximum</dd>
        </div>
      </dl>
    </section>
  );
}
