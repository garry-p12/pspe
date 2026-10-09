"use client";

import { useCallback, useMemo, useRef } from "react";
import type { Assessment, Belt, SectionResult, Span } from "@/lib/section";

/**
 * The section itself: ground, the solved water on it, and the wall being drawn.
 *
 * Monochrome and hatched rather than coloured, for the reason the rest of this
 * panel is: the only colour on this screen belongs to the map, where it is
 * carrying depth. Water is a diagonal hatch, fill is a solid dark body, and
 * the crest is a single black rule -- a plot that survives a projector, a
 * greyscale print and a colour-blind reader without a legend.
 *
 * ## Vertical window
 *
 * The y axis is NOT the line's elevation range. The measured Lismore section
 * spans 3.2 m to 67.5 m because a hill stands in the middle of it, and a plot
 * drawn to that range renders the entire floodplain -- the thing being
 * designed for -- as four pixels of flat. The window is cut just above the
 * water and the crest, and ground that leaves the top is said so in words
 * rather than quietly flattened.
 */

const W = 340;
const H = 196;
// The top margin carries the clipping note, so it is wider than the plot
// alone needs: drawn inside the plot the note landed on whatever ground
// happened to be high at that end of the line.
const PAD = { l: 30, r: 8, t: 20, b: 22 };

export function SectionChart({
  res, span, crestM, a, belts,
  onCrest, onSpan,
}: {
  res: SectionResult;
  span: Span;
  crestM: number;
  a: Assessment;
  /** The channel. Marked because an alignment over it is refused, and a
   *  refusal the reader cannot see the reason for is just an error message. */
  belts: Belt[];
  onCrest: (m: number) => void;
  onSpan: (s: Span) => void;
}) {
  const p = res.profile;
  const svg = useRef<SVGSVGElement>(null);
  const grabbed = useRef<"crest" | "west" | "east" | null>(null);

  const { yMin, yMax, clipped } = useMemo(() => {
    const grounds = p.map((q) => q.ground_m);
    const lo = Math.min(...grounds) - 1;
    const waters = p.map((q) => q.water_m ?? -Infinity);
    const top = Math.max(crestM, ...waters, lo + 6) + 4;
    const hi = Math.min(Math.max(...grounds) + 2, top);
    return { yMin: lo, yMax: hi, clipped: Math.max(...grounds) > hi };
  }, [p, crestM]);

  const x = useCallback(
    (m: number) => PAD.l + (m / Math.max(res.length_m, 1)) * (W - PAD.l - PAD.r),
    [res.length_m],
  );
  const y = useCallback(
    (e: number) =>
      PAD.t + ((yMax - e) / Math.max(yMax - yMin, 1e-6)) * (H - PAD.t - PAD.b),
    [yMin, yMax],
  );
  const yc = useCallback((e: number) => Math.min(Math.max(y(e), PAD.t - 6), H - PAD.b), [y]);

  const groundPath = useMemo(() => {
    const pts = p.map((q) => `${x(q.m).toFixed(1)},${yc(q.ground_m).toFixed(1)}`);
    return `M${PAD.l},${H - PAD.b} L${pts.join(" L")} L${W - PAD.r},${H - PAD.b} Z`;
  }, [p, x, yc]);

  // One polygon per wet run: the surface across, then the bed back under it.
  const waterPaths = useMemo(() => {
    const out: string[] = [];
    let run: number[] = [];
    const flush = () => {
      if (run.length < 1) { run = []; return; }
      const top = run.map((i) => `${x(p[i].m).toFixed(1)},${yc(p[i].water_m as number).toFixed(1)}`);
      const bed = [...run].reverse().map((i) => `${x(p[i].m).toFixed(1)},${yc(p[i].ground_m).toFixed(1)}`);
      out.push(`M${top.join(" L")} L${bed.join(" L")} Z`);
      run = [];
    };
    for (let i = 0; i < p.length; i++) {
      if (p[i].water_m !== null) run.push(i);
      else flush();
    }
    flush();
    return out;
  }, [p, x, yc]);

  // The embankment body, where the crest stands above the ground it sits on.
  const fillPath = useMemo(() => {
    const seg = p.slice(span.i0, span.i1 + 1).filter((q) => crestM > q.ground_m);
    if (seg.length < 2) return null;
    const under = [...seg].reverse().map((q) => `${x(q.m).toFixed(1)},${yc(q.ground_m).toFixed(1)}`);
    const top = seg.map((q) => `${x(q.m).toFixed(1)},${yc(crestM).toFixed(1)}`);
    return `M${top.join(" L")} L${under.join(" L")} Z`;
  }, [p, span, crestM, x, yc]);

  const ticks = useMemo(() => {
    const steps = [1, 2, 5, 10, 20, 50];
    const want = (yMax - yMin) / 4;
    const step = steps.find((s) => s >= want) ?? 100;
    const out: number[] = [];
    for (let e = Math.ceil(yMin / step) * step; e <= yMax; e += step) out.push(e);
    return out;
  }, [yMin, yMax]);

  const nearestIdx = useCallback(
    (px: number) => {
      const m = ((px - PAD.l) / (W - PAD.l - PAD.r)) * res.length_m;
      let best = 0, bd = Infinity;
      for (let i = 0; i < p.length; i++) {
        const d = Math.abs(p[i].m - m);
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    },
    [p, res.length_m],
  );

  const onMove = useCallback(
    (ev: React.PointerEvent<SVGSVGElement>) => {
      if (!grabbed.current || !svg.current) return;
      const r = svg.current.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const py = ((ev.clientY - r.top) / r.height) * H;
      if (grabbed.current === "crest") {
        const e = yMax - ((py - PAD.t) / (H - PAD.t - PAD.b)) * (yMax - yMin);
        onCrest(Math.round(Math.min(Math.max(e, yMin), yMax) * 10) / 10);
        return;
      }
      const i = nearestIdx(px);
      if (grabbed.current === "west") onSpan({ i0: Math.min(i, span.i1 - 1), i1: span.i1 });
      else onSpan({ i0: span.i0, i1: Math.max(i, span.i0 + 1) });
    },
    [yMin, yMax, nearestIdx, onCrest, onSpan, span.i0, span.i1],
  );

  // The stage the wall actually stands against: the water on ITS alignment,
  // not the reach's, which the channel can put on the far floodplain.
  const spanStage = useMemo(() => {
    let m: number | null = null;
    for (let i = span.i0; i <= span.i1; i++) {
      const w = p[i].water_m;
      if (w !== null && (m === null || w > m)) m = w;
    }
    return m;
  }, [p, span]);

  const crestY = yc(crestM);
  const over = a.verdict === "overtops";

  return (
    <svg
      ref={svg}
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      data-chart="section"
      aria-label={
        `Elevation section ${(res.length_m / 1000).toFixed(1)} km long. ` +
        `Ground from ${yMin.toFixed(0)} to ${Math.max(...p.map((q) => q.ground_m)).toFixed(0)} metres. ` +
        `The water this wall holds back peaks at ${spanStage?.toFixed(2) ?? "nothing"} metres. ` +
        `Crest at ${crestM.toFixed(1)} metres over ${(a.lengthM / 1000).toFixed(2)} km: ${a.verdict}.`
      }
      className="block touch-none select-none"
      onPointerMove={onMove}
      onPointerUp={() => { grabbed.current = null; }}
      onPointerLeave={() => { grabbed.current = null; }}
    >
      <defs>
        <pattern id="sec-water" width="5" height="5" patternTransform="rotate(45)"
                 patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-faint)" strokeWidth="1.1" />
        </pattern>
      </defs>

      {ticks.map((e) => (
        <g key={e}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(e)} y2={y(e)}
                stroke="var(--line)" strokeWidth="0.8" />
          <text x={PAD.l - 4} y={y(e) + 3} textAnchor="end"
                fontSize="8.5" fill="var(--ink-faint)" fontFamily="var(--font-mono-stack)">
            {e.toFixed(0)}
          </text>
        </g>
      ))}

      {belts.map((b) => (
        <g key={`belt-${b.i0}`}>
          <rect x={x(p[b.i0].m)} y={PAD.t} height={H - PAD.t - PAD.b}
                width={Math.max(x(p[b.i1].m) - x(p[b.i0].m), 1)}
                fill="var(--ink)" fillOpacity="0.07" />
          <line x1={x(p[b.i0].m)} x2={x(p[b.i1].m)} y1={H - PAD.b + 3} y2={H - PAD.b + 3}
                stroke="var(--ink)" strokeWidth="2.5" />
          <text x={(x(p[b.i0].m) + x(p[b.i1].m)) / 2} y={H - PAD.b + 13}
                textAnchor="middle" fontSize="8.5" fill="var(--ink-mute)"
                fontFamily="var(--font-mono-stack)">channel</text>
        </g>
      ))}

      {waterPaths.map((d, i) => (
        <path key={i} d={d} fill="url(#sec-water)" stroke="var(--ink-mute)" strokeWidth="1" />
      ))}

      <path d={groundPath} fill="var(--bg-inset)" stroke="var(--ink)" strokeWidth="1.2" />

      {fillPath && (
        <path d={fillPath} fill="var(--ink)" fillOpacity="0.82"
              stroke="var(--ink)" strokeWidth="1" />
      )}

      {/* The crest, and the two ends that have to tie into the bank. */}
      <line x1={x(p[span.i0].m)} x2={x(p[span.i1].m)} y1={crestY} y2={crestY}
            stroke="var(--ink)" strokeWidth={over ? 2 : 2.6}
            strokeDasharray={over ? "4 3" : undefined} />
      <rect x={PAD.l} y={crestY - 7} width={W - PAD.l - PAD.r} height="14"
            fill="transparent" style={{ cursor: "ns-resize" }}
            onPointerDown={(e) => { grabbed.current = "crest"; e.currentTarget.releasePointerCapture?.(e.pointerId); }} />
      {([["west", span.i0], ["east", span.i1]] as const).map(([side, i]) => (
        <g key={side} style={{ cursor: "ew-resize" }}
           onPointerDown={() => { grabbed.current = side; }}>
          <line x1={x(p[i].m)} x2={x(p[i].m)} y1={crestY} y2={yc(p[i].ground_m)}
                stroke="var(--ink)" strokeWidth="1.4" />
          <circle cx={x(p[i].m)} cy={crestY} r="5.5"
                  fill="var(--bg)" stroke="var(--ink)" strokeWidth="2" />
          <circle cx={x(p[i].m)} cy={crestY} r="11" fill="transparent" />
        </g>
      ))}

      {over && a.overtopAtM !== null && (
        <g>
          <circle cx={x(a.overtopAtM)} cy={crestY} r="3" fill="var(--ink)" />
          <text x={x(a.overtopAtM)} y={crestY - 9} textAnchor="middle"
                fontSize="9" fontWeight="700" fill="var(--ink)">
            +{a.overtopM.toFixed(2)} m over
          </text>
        </g>
      )}

      <text x={PAD.l} y={H - 6} fontSize="8.5" fill="var(--ink-faint)"
            fontFamily="var(--font-mono-stack)">A</text>
      <text x={(PAD.l + W - PAD.r) / 2} y={H - 6} textAnchor="middle" fontSize="8.5"
            fill="var(--ink-faint)" fontFamily="var(--font-mono-stack)">
        {(res.length_m / 1000).toFixed(2)} km
      </text>
      <text x={W - PAD.r} y={H - 6} textAnchor="end" fontSize="8.5" fill="var(--ink-faint)"
            fontFamily="var(--font-mono-stack)">B</text>
      {clipped && (
        <text x={W - PAD.r} y={10} textAnchor="end" fontSize="8.5"
              fill="var(--ink-faint)" fontFamily="var(--font-mono-stack)">
          ground continues above {yMax.toFixed(0)} m
        </text>
      )}
    </svg>
  );
}
