"use client";

import type { GaugeFeed } from "./LayersPanel";

/**
 * What the river is doing right now, from the instrument rather than the model.
 *
 * The portal has opened on a solved field for its whole life: a storm the user
 * picked, run through a solver, with nothing on screen that the world could
 * contradict. This is the one panel that reports an observation, so it leads
 * the forecast column.
 *
 * It deliberately does NOT say "minor flood" or "major flood". Those are NWS
 * flood categories, they are set per gauge, and we do not have them — so the
 * comparison offered is the one the data supports: this reading against the
 * highest this gauge has seen in the last three days. Naming a flood category
 * we had not fetched would be the most quotable sentence on the page and the
 * least defensible.
 */
function Spark({ series }: { series: { t: string; m: number }[] }) {
  if (series.length < 3) return null;
  const v = series.map((p) => p.m);
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = Math.max(hi - lo, 1e-6);
  const W = 108;
  const H = 26;
  const d = v
    .map((y, i) => `${(i / (v.length - 1)) * W},${H - ((y - lo) / span) * (H - 3) - 1.5}`)
    .join(" L");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img"
         aria-label={`Last three days, ${lo.toFixed(2)} to ${hi.toFixed(2)} metres`}
         className="shrink-0">
      <path d={`M${d}`} fill="none" stroke="var(--ink)" strokeWidth="1.4"
            strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function GaugeHeadline({ feed }: { feed: GaugeFeed | null }) {
  if (!feed) return null;
  if (!feed.supported) {
    return (
      <div className="border-b border-line-soft px-5 py-3">
        <p className="eyebrow">Nothing live here</p>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{feed.note}</p>
      </div>
    );
  }
  const g = feed.gauges.find((x) => x.stage_m !== null);
  if (!g) {
    return (
      <div className="border-b border-line-soft px-5 py-3">
        <p className="eyebrow">No gauged stream in this box</p>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
          Everything on this map is a model. Move to a reach with a gauge on it
          and this panel will report the instrument instead.
        </p>
      </div>
    );
  }
  const t = g.trend_cm_h ?? 0;
  const verdict = t > 1 ? "River rising" : t < -1 ? "River falling" : "River steady";
  const peak = g.series.length ? Math.max(...g.series.map((p) => p.m)) : null;
  const when = g.at ? new Date(g.at) : null;

  return (
    <div className="border-b border-line-soft px-5 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <p className="anno">now · observed</p>
        <p className="anno">
          {when ? when.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : ""}
        </p>
      </div>
      <h2 className="mt-0.5 text-[22px] font-semibold leading-tight tracking-tight text-ink">
        {verdict}
      </h2>
      <div className="mt-1.5 flex items-end justify-between gap-3">
        <p className="text-[13px] leading-relaxed text-ink-mute">
          {g.name.split(",")[0]} reads{" "}
          <span className="tnum font-medium text-ink">{g.stage_m?.toFixed(2)} m</span>
          {g.trend_cm_h !== null && (
            <>, {t >= 0 ? "up" : "down"}{" "}
              <span className="tnum font-medium text-ink">
                {Math.abs(t).toFixed(0)} cm
              </span>{" "}
              an hour</>
          )}
          {g.q_m3s !== null && (
            <> · <span className="tnum">{g.q_m3s.toFixed(1)}</span> m³/s</>
          )}
        </p>
        <Spark series={g.series} />
      </div>
      {peak !== null && (
        <p className="anno mt-1.5 leading-relaxed">
          Highest in the last three days {peak.toFixed(2)} m. Flood categories are
          set per gauge by the weather service and are not fetched here, so this
          compares the reading with its own recent range rather than naming one.
        </p>
      )}
      {feed.gauges.length > 1 && (
        <p className="anno mt-1 leading-relaxed">
          {feed.gauges.length} gauges in this box · {feed.source}
        </p>
      )}
    </div>
  );
}
