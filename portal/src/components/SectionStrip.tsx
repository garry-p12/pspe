"use client";

import { useCallback, useMemo, useRef } from "react";
import type { SectionResult, WallVerdict } from "@/lib/section";

/**
 * The elevation section, full width under the map, with the wall on it.
 *
 * This is the one drawing in the portal that is not achromatic, and the
 * departure is deliberate: a cross-section is read as a picture of ground and
 * water, and those two have colours everyone already knows. The rest of the
 * interface stays monochrome because its colour is carrying data on the map;
 * here the colour IS the ground and the water, and a greyscale section asks
 * the reader to consult a legend to find out which half is the river.
 *
 * The levee is drawn where it actually is: a trapezoid at one station, not a
 * crest running the length of the drawing. A section is cut ACROSS a bank, so
 * the wall is a point in it and its length runs out of the plane — which is
 * why the cost quoted beside this comes from the bank's length and not from
 * anything measurable on screen.
 *
 * Drag the handle: sideways moves the wall along the section, upwards raises
 * the crest.
 */

const H = 300;
const PAD = { l: 54, r: 18, t: 34, b: 30 };

export function SectionStrip({
  res, stationIdx, crestM, wall, landward, title, onStation, onCrest,
}: {
  res: SectionResult;
  stationIdx: number;
  crestM: number;
  wall: WallVerdict;
  landward: "a" | "b";
  title: string;
  onStation: (i: number) => void;
  onCrest: (m: number) => void;
}) {
  const p = res.profile;
  const svg = useRef<SVGSVGElement>(null);
  const grabbed = useRef(false);
  const W = 1000;

  const { yMin, yMax } = useMemo(() => {
    const g = p.map((q) => q.ground_m);
    const w = p.map((q) => q.water_m ?? -Infinity);
    const lo = Math.min(...g) - 0.5;
    const hi = Math.max(crestM + 1.2, ...w, lo + 5,
                        // enough ground to read, but a hill behind the town
                        // must not flatten the floodplain into a line
                        Math.min(Math.max(...g), Math.max(...w, lo) + 10));
    return { yMin: lo, yMax: hi };
  }, [p, crestM]);

  const x = useCallback(
    (m: number) => PAD.l + (m / Math.max(res.length_m, 1)) * (W - PAD.l - PAD.r),
    [res.length_m],
  );
  const y = useCallback(
    (e: number) => PAD.t + ((yMax - e) / Math.max(yMax - yMin, 1e-6)) * (H - PAD.t - PAD.b),
    [yMin, yMax],
  );
  const yc = useCallback((e: number) => Math.min(Math.max(y(e), PAD.t - 10), H - PAD.b), [y]);

  const ground = useMemo(() => {
    const pts = p.map((q) => `${x(q.m).toFixed(1)},${yc(q.ground_m).toFixed(1)}`);
    return `M${PAD.l},${H - PAD.b} L${pts.join(" L")} L${W - PAD.r},${H - PAD.b} Z`;
  }, [p, x, yc]);

  const water = useMemo(() => {
    const out: string[] = [];
    let run: number[] = [];
    const flush = () => {
      if (run.length > 1) {
        const top = run.map((i) => `${x(p[i].m).toFixed(1)},${yc(p[i].water_m as number).toFixed(1)}`);
        const bed = [...run].reverse().map((i) => `${x(p[i].m).toFixed(1)},${yc(p[i].ground_m).toFixed(1)}`);
        out.push(`M${top.join(" L")} L${bed.join(" L")} Z`);
      }
      run = [];
    };
    for (let i = 0; i < p.length; i++) {
      if (p[i].water_m !== null) run.push(i); else flush();
    }
    flush();
    return out;
  }, [p, x, yc]);

  const ticks = useMemo(() => {
    const steps = [1, 2, 5, 10, 20, 50];
    const step = steps.find((s) => s >= (yMax - yMin) / 4) ?? 100;
    const out: number[] = [];
    for (let e = Math.ceil(yMin / step) * step; e <= yMax; e += step) out.push(e);
    return out;
  }, [yMin, yMax]);

  const st = p[Math.min(Math.max(stationIdx, 0), p.length - 1)];
  const gx = x(st.m);
  const gy = yc(crestM);
  const gg = yc(st.ground_m);
  const half = Math.max((W - PAD.l - PAD.r) / p.length, 5);

  const move = useCallback((ev: React.PointerEvent<SVGSVGElement>) => {
    if (!grabbed.current || !svg.current) return;
    const r = svg.current.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const py = ((ev.clientY - r.top) / r.height) * H;
    const m = ((px - PAD.l) / (W - PAD.l - PAD.r)) * res.length_m;
    let best = 0, bd = Infinity;
    for (let i = 0; i < p.length; i++) {
      const d = Math.abs(p[i].m - m);
      if (d < bd) { bd = d; best = i; }
    }
    onStation(best);
    const e = yMax - ((py - PAD.t) / (H - PAD.t - PAD.b)) * (yMax - yMin);
    onCrest(Math.round(Math.min(Math.max(e, yMin), yMax) * 100) / 100);
  }, [p, res.length_m, yMin, yMax, onStation, onCrest]);

  const peak = useMemo(() => {
    let m: number | null = null;
    for (const q of p) if (q.water_m !== null && (m === null || q.water_m > m)) m = q.water_m;
    return m;
  }, [p]);

  const over = wall.verdict === "overtops";

  return (
    <svg
      ref={svg}
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      data-chart="strip"
      aria-label={
        `Elevation section ${(res.length_m / 1000).toFixed(2)} km across ${title}. ` +
        `A is the ${landward === "a" ? "landward" : "river"} end. ` +
        `Wall at ${(st.m / 1000).toFixed(2)} km, crest ${crestM.toFixed(2)} metres, ` +
        `${wall.heightM.toFixed(2)} metres above the ground: ${wall.verdict}.`
      }
      className="block touch-none select-none"
      onPointerMove={move}
      onPointerUp={() => { grabbed.current = false; }}
      onPointerLeave={() => { grabbed.current = false; }}
    >
      <rect x="0" y="0" width={W} height={H} fill="#fbfaf8" />
      {ticks.map((e) => (
        <g key={e}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(e)} y2={y(e)}
                stroke="#e6e1d8" strokeWidth="1" />
          <text x={PAD.l - 8} y={y(e) + 4} textAnchor="end" fontSize="11"
                fill="#8b8b93" fontFamily="var(--font-mono-stack)">{e.toFixed(0)}</text>
        </g>
      ))}

      {water.map((d, i) => (
        <path key={i} d={d} fill="#8fb8d8" fillOpacity="0.92" />
      ))}
      <path d={ground} fill="#ded3c0" stroke="#a89880" strokeWidth="1.4" />

      {peak !== null && (
        <g>
          <line x1={PAD.l} x2={W - PAD.r} y1={yc(peak)} y2={yc(peak)}
                stroke="#3f6f9c" strokeWidth="1.4" strokeDasharray="6 4" />
          <text x={W - PAD.r} y={yc(peak) - 6} textAnchor="end" fontSize="11"
                fill="#3f6f9c" fontFamily="var(--font-mono-stack)">
            solved peak {peak.toFixed(2)} m
          </text>
        </g>
      )}

      {/* The wall: a trapezoid on the ground it stands on. */}
      {wall.heightM > 0.01 && (
        <path
          d={`M${(gx - half * 1.9).toFixed(1)},${gg.toFixed(1)} `
             + `L${(gx - half * 0.7).toFixed(1)},${gy.toFixed(1)} `
             + `L${(gx + half * 0.7).toFixed(1)},${gy.toFixed(1)} `
             + `L${(gx + half * 1.9).toFixed(1)},${gg.toFixed(1)} Z`}
          fill={over ? "#d8a27a" : "#c87f3e"}
          stroke="#8a5529"
          strokeWidth="1.2"
        />
      )}
      <line x1={gx} x2={gx} y1={gy} y2={H - PAD.b}
            stroke="#8a5529" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />

      <g style={{ cursor: "grab" }}
         onPointerDown={(e) => { grabbed.current = true;
                                 (e.target as Element).releasePointerCapture?.(e.pointerId); }}>
        <circle cx={gx} cy={gy} r="9" fill="#ffffff" stroke="#8a5529" strokeWidth="2.5" />
        <circle cx={gx} cy={gy} r="20" fill="transparent" />
      </g>
      <g>
        <rect x={gx - 62} y={gy - 34} width="124" height="21" rx="4"
              fill="#18181b" opacity="0.92" />
        <text x={gx} y={gy - 19} textAnchor="middle" fontSize="11.5" fill="#ffffff"
              fontFamily="var(--font-mono-stack)">
          crest {crestM.toFixed(2)} m
        </text>
      </g>

      <text x={PAD.l} y={18} fontSize="11.5" fill="#52525b"
            fontFamily="var(--font-mono-stack)">
        A · {landward === "a" ? "town side" : "river side"}
      </text>
      <text x={W / 2} y={18} textAnchor="middle" fontSize="11.5" fill="#52525b"
            fontFamily="var(--font-mono-stack)">
        {title} · elevation section
      </text>
      <text x={W - PAD.r} y={18} textAnchor="end" fontSize="11.5" fill="#52525b"
            fontFamily="var(--font-mono-stack)">
        B · {landward === "a" ? "river side" : "town side"}
      </text>
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="#8b8b93"
            fontFamily="var(--font-mono-stack)">
        {(res.length_m / 1000).toFixed(2)} km · sampled every{" "}
        {(res.length_m / Math.max(res.samples - 1, 1)).toFixed(0)} m from a{" "}
        {res.dx_m.toFixed(0)} m model
      </text>
    </svg>
  );
}
