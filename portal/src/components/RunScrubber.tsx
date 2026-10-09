"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Play the flood arriving, on a card floating over the map.
 *
 * The design puts a scrubber here with a play button, running out to Sunday
 * under a forecast band. This is that control, and it is wired to the one
 * thing in this system that actually moves: the solved run. Nine instantaneous
 * depth fields across eighteen hours, which the solver computed on the way to
 * the peak and used to throw away.
 *
 * **Why the gauge record is not what this plays.** The strip in the sidebar
 * shows the river at a gauge on a wall clock — Tuesday to now, at fifteen
 * minutes. The model runs on its own clock: hour zero is when the design storm
 * starts, and there is no Tuesday in it. Putting a play button against the
 * gauge chart would have been a control over nothing, which is the failure
 * mode this component exists to avoid, so the two clocks stay apart: gauge
 * data in the sidebar beside the gauge reading, model time here on the map
 * where the thing it moves is.
 *
 * The curve behind the handle is the flooded area through the run, measured
 * from the same frames the handle selects — so the shape of the hydrograph and
 * the water on the map cannot disagree.
 */

export interface Frame {
  t_h: number;
  flooded_km2: number;
  peak_depth_m: number;
  depth_png: string;
}

const W = 1000;
const H = 54;
const PAD = { l: 8, r: 54, t: 10, b: 16 };

export function RunScrubber({
  frames, index, onIndex, stormHours, peakKm2,
}: {
  frames: Frame[];
  /** null means the peak field — where the map sits when nothing is playing. */
  index: number | null;
  onIndex: (i: number | null) => void;
  stormHours: number;
  /** The peak's own flooded area. NOT the last frame's: the peak is the
   *  envelope over the whole run and is larger than any instant in it —
   *  18.7 km² against 5.3 at hour eighteen — so reading the last frame while
   *  the map draws the envelope put two different floods on one card. */
  peakKm2: number;
}) {
  const [playing, setPlaying] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const n = frames.length;
  const stop = useCallback(() => {
    setPlaying(false);
    if (timer.current) { clearInterval(timer.current); timer.current = null; }
  }, []);

  // The current frame is read through a ref, not closed over, so the interval
  // survives the frame changing. Depending on `index` instead tore the timer
  // down and rebuilt it on every tick, which drifts the cadence. The ref is
  // written in an effect rather than during render, because a render can be
  // discarded and a ref written in one cannot be taken back.
  const idxRef = useRef(index);
  useEffect(() => { idxRef.current = index; }, [index]);
  useEffect(() => {
    if (!playing || n < 2) return;
    timer.current = setInterval(() => {
      const cur = idxRef.current;
      const next = cur === null ? 0 : cur + 1;
      if (next >= n) { setPlaying(false); onIndex(null); return; }
      onIndex(next);
    }, 650);
    const t = timer.current;
    return () => clearInterval(t);
  }, [playing, n, onIndex]);

  useEffect(() => stop, [stop]);

  if (n < 2) return null;

  const runH = frames[n - 1].t_h || 1;
  const maxA = Math.max(...frames.map((f) => f.flooded_km2), 1e-6);
  const x = (t: number) => PAD.l + (t / runH) * (W - PAD.l - PAD.r);
  const y = (a: number) =>
    PAD.t + (1 - a / maxA) * (H - PAD.t - PAD.b);

  const path = frames
    .map((f, i) => `${i ? "L" : "M"}${x(f.t_h).toFixed(1)},${y(f.flooded_km2).toFixed(1)}`)
    .join(" ");
  const area = `${path} L${x(runH).toFixed(1)},${H - PAD.b} L${x(0).toFixed(1)},${H - PAD.b} Z`;

  const atPeak = index === null;
  const cur = atPeak ? n - 1 : index;
  const f = frames[cur];
  const shownKm2 = atPeak ? peakKm2 : f.flooded_km2;

  return (
    <div
      className="pointer-events-auto flex items-stretch gap-3 rounded-[12px] border
                 border-line bg-bg/95 p-2.5 pr-3 backdrop-blur
                 shadow-[0_2px_6px_rgba(16,20,28,0.10),0_14px_34px_rgba(16,20,28,0.13)]"
      data-scrubber
    >
      <button
        onClick={() => (playing ? stop()
          : (onIndex(index === null || index >= n - 1 ? 0 : index), setPlaying(true)))}
        aria-label={playing ? "Pause the run" : "Play the run"}
        className="my-auto grid h-11 w-11 shrink-0 place-items-center rounded-full
                   bg-ink text-bg transition-opacity hover:opacity-90"
      >
        {playing ? (
          <svg width="13" height="14" viewBox="0 0 13 14" aria-hidden>
            <rect x="1" y="1" width="4" height="12" fill="currentColor" />
            <rect x="8" y="1" width="4" height="12" fill="currentColor" />
          </svg>
        ) : (
          <svg width="13" height="14" viewBox="0 0 13 14" aria-hidden>
            <path d="M2 1 L12 7 L2 13 Z" fill="currentColor" />
          </svg>
        )}
      </button>

      <div className="my-auto w-[118px] shrink-0">
        <p className="anno leading-tight">
          {atPeak ? `peak of ${runH.toFixed(0)} h`
            : `hour ${f.t_h.toFixed(1)} of ${runH.toFixed(0)}`}
        </p>
        <p className="tnum text-[16px] font-semibold leading-tight text-ink">
          {shownKm2.toFixed(1)} km²
        </p>
        <p className="anno leading-tight">
          {atPeak ? "wet at any time" : "wet at this hour"}
        </p>
      </div>

      <div className="min-w-0 flex-1">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img"
             preserveAspectRatio="none"
             aria-label={`Flooded area through the run, now ${f.flooded_km2.toFixed(1)} square kilometres at hour ${f.t_h.toFixed(1)}`}>
          {/* The storm window: forcing on, then the recession. */}
          <rect x={x(0)} y={PAD.t - 4} width={Math.max(x(Math.min(stormHours, runH)) - x(0), 0)}
                height={H - PAD.t - PAD.b + 8} fill="var(--bg-inset)" />
          <path d={area} fill="var(--ink)" fillOpacity="0.09" />
          <path d={path} fill="none" stroke="var(--ink)" strokeWidth="2"
                strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
          {!atPeak && (
            <>
              <line x1={x(f.t_h)} x2={x(f.t_h)} y1={PAD.t - 4} y2={H - PAD.b + 2}
                    stroke="var(--ink)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
              <circle cx={x(f.t_h)} cy={y(f.flooded_km2)} r="4" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
            </>
          )}
          <text x={x(Math.min(stormHours, runH)) + 5} y={H - 4} fontSize="11"
                fill="var(--ink-faint)" fontFamily="var(--font-mono-stack)">
            rain stops
          </text>
          <text x={W - PAD.r - 4} y={y(maxA) - 3} textAnchor="end" fontSize="11"
                fill="var(--ink-faint)" fontFamily="var(--font-mono-stack)">
            {maxA.toFixed(1)} km² at hour{" "}
            {frames.reduce((a, q) => (q.flooded_km2 > a.flooded_km2 ? q : a),
                           frames[0]).t_h.toFixed(1)}
          </text>
        </svg>
        <input
          type="range" min={0} max={n - 1} step={1} value={cur}
          aria-label="Hour of the run"
          onChange={(e) => { stop(); onIndex(Number(e.target.value)); }}
          className="-mt-1 h-[3px] w-full cursor-pointer appearance-none rounded-full
                     bg-line accent-ink"
        />
      </div>

      <button
        onClick={() => { stop(); onIndex(null); }}
        disabled={index === null}
        className="my-auto shrink-0 self-center rounded-md border border-line px-2.5 py-1.5
                   text-[12px] text-ink hover:bg-bg-inset disabled:opacity-40"
      >
        Peak
      </button>
    </div>
  );
}
