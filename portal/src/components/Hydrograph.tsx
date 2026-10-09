"use client";

import { useMemo } from "react";
import type { Gauge } from "./LayersPanel";

/**
 * Three days of river, under the map, against the levels that define a flood.
 *
 * The strip the wireframe puts here has a play button and a forecast band
 * running out to Sunday. This has neither, and the reason is worth stating on
 * the page rather than in a commit: the weather service publishes a forecast
 * for a gauge only when one is current, and a play control that scrubs through
 * a model while nothing on the map moves is a control over nothing. What is
 * drawn is what exists — the observed record at fifteen-minute cadence, and the
 * NWS thresholds that say whether it is a flood.
 *
 * Those thresholds are the point. A stage of 1.36 m means nothing on its own;
 * 1.36 m against an action level of 4.57 m is a quiet creek, and the same
 * reading on a gauge whose minor flood level is 1.5 m is an evacuation. Four
 * of the twenty-five gauged streams in the Austin box have published levels,
 * and the strip says which gauge it is showing so the reader can tell whether
 * this is the one they care about.
 */

const H = 150;
const PAD = { l: 52, r: 96, t: 16, b: 26 };
const W = 1000;

const BANDS: [keyof NonNullable<Gauge["flood_levels_m"]>, string][] = [
  ["action", "action"],
  ["minor", "minor flood"],
  ["moderate", "moderate flood"],
  ["major", "major flood"],
];

export function Hydrograph({ gauge }: { gauge: Gauge }) {
  const pts = gauge.series;

  const { yMin, yMax } = useMemo(() => {
    const vals = pts.map((p) => p.m);
    const levels = Object.values(gauge.flood_levels_m ?? {}).filter(
      (v): v is number => typeof v === "number");
    const lo = Math.min(...vals, gauge.stage_m ?? Infinity) - 0.15;
    // Include the action level so the reader can see how far off it is, but
    // not the major level, which on a quiet creek is ten metres up and would
    // flatten three days of river into one pixel.
    const nearest = levels.filter((v) => v <= (Math.max(...vals) + 3));
    const hi = Math.max(...vals, ...(nearest.length ? [nearest[0]] : []),
                        lo + 0.6) + 0.15;
    return { yMin: lo, yMax: hi };
  }, [pts, gauge]);

  const t0 = pts.length ? new Date(pts[0].t).getTime() : 0;
  const t1 = pts.length ? new Date(pts[pts.length - 1].t).getTime() : 1;
  const x = (t: number) =>
    PAD.l + ((t - t0) / Math.max(t1 - t0, 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) =>
    PAD.t + ((yMax - v) / Math.max(yMax - yMin, 1e-6)) * (H - PAD.t - PAD.b);

  const path = useMemo(
    () => pts.map((p, i) =>
      `${i ? "L" : "M"}${x(new Date(p.t).getTime()).toFixed(1)},${y(p.m).toFixed(1)}`
    ).join(" "),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pts, yMin, yMax, t0, t1],
  );

  const days = useMemo(() => {
    const out: { t: number; label: string }[] = [];
    if (!pts.length) return out;
    const d = new Date(t0);
    d.setHours(0, 0, 0, 0);
    for (let k = 0; k < 6; k++) {
      const t = d.getTime() + k * 86400000;
      if (t >= t0 && t <= t1) {
        out.push({ t, label: new Date(t).toLocaleDateString([], { weekday: "short" }) });
      }
    }
    return out;
  }, [pts, t0, t1]);

  // A level above the window is deliberately not drawn -- the Colorado's
  // action level is 7.62 m against three days spanning 3.7 to 4.5, and
  // including it flattens the river into a straight line. But then the strip
  // would carry the words "NWS flood levels" over a chart with none on it, so
  // it says how far away the nearest one is instead.
  const offScale = useMemo(() => {
    const L = gauge.flood_levels_m;
    if (!L) return null;
    const shown = BANDS.some(([k]) => {
      const v = L[k];
      return v !== undefined && v <= yMax && v >= yMin;
    });
    if (shown) return null;
    const above = BANDS.map(([k, label]) => [L[k], label] as const)
      .filter((e): e is [number, string] => typeof e[0] === "number" && e[0] > yMax)
      .sort((a, b) => a[0] - b[0])[0];
    return above
      ? `${above[1]} ${above[0].toFixed(2)} m, ${(above[0] - (gauge.stage_m ?? 0)).toFixed(2)} m above now — off this scale`
      : null;
  }, [gauge, yMin, yMax]);

  if (pts.length < 3) return null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" data-chart="hydrograph"
         aria-label={
           `Three days of stage at ${gauge.name}, now ${gauge.stage_m?.toFixed(2)} metres`
           + (gauge.flood_levels_m?.minor !== undefined
             ? `, minor flood level ${gauge.flood_levels_m.minor.toFixed(2)} metres` : "")
         }
         className="block">
      <rect x="0" y="0" width={W} height={H} fill="var(--bg)" />

      {days.map((d) => (
        <g key={d.t}>
          <line x1={x(d.t)} x2={x(d.t)} y1={PAD.t} y2={H - PAD.b}
                stroke="var(--line-soft)" strokeWidth="1" />
          <text x={x(d.t) + 4} y={H - 10} fontSize="10.5" fill="var(--ink-faint)"
                fontFamily="var(--font-mono-stack)">{d.label}</text>
        </g>
      ))}

      {BANDS.map(([k, label]) => {
        const v = gauge.flood_levels_m?.[k];
        if (v === undefined || v > yMax || v < yMin) return null;
        return (
          <g key={k}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)}
                  stroke="var(--ink)" strokeWidth="1.1" strokeDasharray="5 4" />
            <text x={W - PAD.r + 6} y={y(v) + 4} fontSize="10.5" fill="var(--ink-mute)"
                  fontFamily="var(--font-mono-stack)">
              {label} {v.toFixed(2)} m
            </text>
          </g>
        );
      })}

      <path d={path} fill="none" stroke="var(--ink)" strokeWidth="1.8"
            strokeLinejoin="round" strokeLinecap="round" />

      {/* Now: the right-hand end of the record, which is all there is. */}
      <circle cx={x(t1)} cy={y(pts[pts.length - 1].m)} r="4"
              fill="var(--bg)" stroke="var(--ink)" strokeWidth="2.5" />
      <text x={x(t1) + 8} y={y(pts[pts.length - 1].m) - 8} fontSize="11"
            fill="var(--ink)" fontWeight="600" fontFamily="var(--font-mono-stack)">
        now {pts[pts.length - 1].m.toFixed(2)} m
      </text>

      <text x={PAD.l} y={11} fontSize="10.5" fill="var(--ink-mute)"
            fontFamily="var(--font-mono-stack)">
        {gauge.name.split(",")[0]} · {gauge.lid ?? gauge.site} · every 15 min
      </text>
      <text x={W - PAD.r} y={11} textAnchor="end" fontSize="10.5"
            fill="var(--ink-faint)" fontFamily="var(--font-mono-stack)">
        {offScale
          ? `observed · ${offScale}`
          : gauge.flood_levels_m
            ? "observed · NWS flood levels"
            : "observed · no flood levels published for this gauge"}
      </text>
      {[yMin, yMax].map((v, i) => (
        <text key={i} x={PAD.l - 6} y={y(v) + (i ? 9 : 0)} textAnchor="end"
              fontSize="10.5" fill="var(--ink-faint)"
              fontFamily="var(--font-mono-stack)">{v.toFixed(1)}</text>
      ))}
    </svg>
  );
}
