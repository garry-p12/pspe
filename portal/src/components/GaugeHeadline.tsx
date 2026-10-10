"use client";

import type { GaugeFeed } from "./LayersPanel";
import { Explain } from "./Explain";

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

/** Where this reading sits against the weather service's own thresholds. */
function band(stage: number, L: NonNullable<GaugeFeed["gauges"][number]["flood_levels_m"]>): string {
  const steps: [number | undefined, string][] = [
    [L.major, "above the MAJOR flood level"],
    [L.moderate, "above the moderate flood level"],
    [L.minor, "above the minor flood level"],
    [L.action, "above the action level, below minor flood"],
  ];
  for (const [v, text] of steps) {
    if (v !== undefined && stage >= v) return `${text}.`;
  }
  const next = L.action ?? L.minor;
  return next === undefined
    ? "Below every level the weather service has set here."
    : `${(next - stage).toFixed(2)} m below the action level`;
}

/**
 * The gauge this panel speaks for, and the one the chart under it plots.
 *
 * These were two different predicates -- the headline wanted a stage plus
 * flood levels, the chart wanted a series plus flood levels -- so on a feed
 * where one gauge had a short series the sentence and the trace beneath it
 * described two different creeks, identified only by a nine-pixel station
 * code. The feed now arrives ranked with the study's own reach first, so
 * taking the first usable gauge is both correct and the same answer in both
 * places.
 */
export function pickGauge(feed: GaugeFeed | null) {
  if (!feed) return null;
  const gs = feed.gauges;
  return gs.find((x) => x.stage_m !== null && x.series.length > 2 && x.flood_levels_m)
    ?? gs.find((x) => x.stage_m !== null && x.flood_levels_m)
    ?? gs.find((x) => x.stage_m !== null)
    ?? null;
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
  // Prefer a gauge the weather service has set flood levels for. Any gauge
  // can say how high the water is; only one with categories can say whether
  // that is a flood, which is the thing the headline is for.
  const g = pickGauge(feed);
  if (!g) {
    return (
      <div className="border-b border-line-soft px-5 py-3">
        <p className="eyebrow">No gauged stream in this box</p>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
          No gauge here — everything on this map is a model. Move to a gauged
          reach for a real reading.
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
        <p className="anno">observed</p>
        <p className="anno">
          {when ? when.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : ""}
        </p>
      </div>
      <h2 className="mt-0.5 text-[22px] font-semibold leading-tight tracking-tight text-ink">
        {verdict}
      </h2>
      <div className="mt-1.5 flex items-end justify-between gap-3">
        {/* Three numbers with their units, not a sentence wrapped round
            them. The connective prose ("reads", "an hour") was the longest
            part of the line and carried nothing the units do not. */}
        <p className="text-[13px] leading-relaxed text-ink-mute">
          {g.name.split(",")[0]}{" · "}
          <span className="tnum font-medium text-ink">{g.stage_m?.toFixed(2)} m</span>
          {g.trend_cm_h !== null && (
            <> · <span className="tnum font-medium text-ink">
              {t >= 0 ? "+" : "−"}{Math.abs(t).toFixed(0)} cm/h
            </span></>
          )}
          {g.q_m3s !== null && (
            <> · <span className="tnum">{g.q_m3s.toFixed(1)}</span> m³/s</>
          )}
        </p>
        <Spark series={g.series} />
      </div>
      {g.flood_levels_m ? (
        <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
          {band(g.stage_m ?? 0, g.flood_levels_m)}
          {peak !== null && (
            <span className="anno"> · 3-day high {peak.toFixed(2)} m</span>
          )}
        </p>
      ) : peak !== null && (
        <p className="anno mt-1.5 flex items-center gap-1.5 leading-relaxed">
          3-day high {peak.toFixed(2)} m · no published flood level
          <Explain title="No flood level here">
            <p>
              The weather service has not set flood categories for this
              gauge, so the reading is compared with its own recent range
              rather than named as a category it has not been given.
            </p>
          </Explain>
        </p>
      )}
      {g.forecast && (
        <p className="anno mt-1 leading-relaxed">
          Forecast {g.forecast.stage_m.toFixed(2)} m
          {g.forecast.at ? ` by ${new Date(g.forecast.at).toLocaleString([], {
            weekday: "short", hour: "numeric" })}` : ""} · NWS
        </p>
      )}

    </div>
  );
}
