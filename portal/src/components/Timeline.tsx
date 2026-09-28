"use client";

import { useEffect, useRef } from "react";
import type { FrameMeta } from "@/lib/types";
import { fmtHours } from "@/lib/data";

export function Timeline({
  frames,
  value,
  onChange,
  playing,
  onPlayToggle,
}: {
  frames: FrameMeta[];
  value: number;
  onChange: (i: number) => void;
  playing: boolean;
  onPlayToggle: () => void;
}) {
  const raf = useRef<number | null>(null);
  const last = useRef(0);

  useEffect(() => {
    if (!playing) return;
    const tick = (t: number) => {
      if (t - last.current > 110) {
        last.current = t;
        onChange(value >= frames.length - 1 ? 0 : value + 1);
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, value, frames.length, onChange]);

  const f = frames[value];
  const peak = frames.reduce((a, b) => (b.wet_cells > a.wet_cells ? b : a), frames[0]);

  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        onClick={onPlayToggle}
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line bg-bg-inset text-ink transition-colors hover:border-accent/50 hover:text-accent"
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? (
          <svg width="11" height="12" viewBox="0 0 11 12" fill="currentColor">
            <rect x="0" y="0" width="3.5" height="12" rx="1" />
            <rect x="7" y="0" width="3.5" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="11" height="12" viewBox="0 0 11 12" fill="currentColor">
            <path d="M0 1.2v9.6a1 1 0 0 0 1.53.85l7.7-4.8a1 1 0 0 0 0-1.7L1.53.35A1 1 0 0 0 0 1.2Z" />
          </svg>
        )}
      </button>

      <div className="relative min-w-[220px] flex-1">
        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full accent-[var(--accent)]"
          aria-label="Time through the flood"
        />
        {/* Peak-inundation marker: the moment the event is actually about. */}
        <div
          className="pointer-events-none absolute top-[-3px] h-[7px] w-[2px] rounded bg-warn"
          style={{ left: `${(peak.i / (frames.length - 1)) * 100}%` }}
          title={`peak inundation — ${fmtHours(peak.t_hours)}`}
        />
      </div>

      <div className="tnum shrink-0 text-right">
        <p className="text-[13px] font-medium text-ink">{fmtHours(f.t_hours)}</p>
        <p className="text-[12px] text-ink-faint">
          t+{f.t_hours.toFixed(1)} h · frame {f.i + 1}/{frames.length}
        </p>
      </div>
    </div>
  );
}
