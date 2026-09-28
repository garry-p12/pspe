"use client";

import { useState } from "react";

/**
 * The Perceive stage, in the interface.
 *
 * Everything else here is the model's account of the world. This is the only
 * panel that reports a measurement, and it exists so an operator can ask the
 * question a twin has to be able to answer: is the thing I am looking at
 * actually happening?
 *
 * Three honesty constraints are surfaced rather than hidden. The acquisition
 * date is shown because Sentinel-1 repeats every twelve days and a week-old
 * pass is not "now". Permanent water is reported separately from new water,
 * because a change detector answers "what is newly wet" and painting the river
 * the same colour as a flood is how a viewer concludes the model over-predicts.
 * And an empty result is labelled as a statement about the overpass, since
 * C-band cannot see standing water under canopy or among buildings.
 */

export interface Observation {
  ok: boolean;
  reason?: string;
  acquired?: string;
  days_old?: number;
  baselines?: number;
  relative_orbit?: number | null;
  orbit_state?: string;
  flooded_km2?: number;
  permanent_km2?: number;
  observed_png?: string;
  bounds?: number[];
  seconds?: number;
  note?: string;
}

export function ObservePanel({
  bounds, onObservation,
}: {
  bounds: { west: number; south: number; east: number; north: number } | null;
  onObservation?: (o: Observation | null) => void;
}) {
  const [obs, setObs] = useState<Observation | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (!bounds) return;
    setBusy(true); setErr(null);
    try {
      const r = await fetch("/api/observe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...bounds, name: "This district", days_back: 60, size: 512 }),
      });
      const o: Observation = await r.json();
      if (!r.ok) throw new Error(o.reason ?? `service returned ${r.status}`);
      setObs(o);
      onObservation?.(o.ok ? o : null);
      if (!o.ok) setErr(o.reason ?? "no usable overpass");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "could not reach the satellite archive");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border-b border-line-soft p-4">
      <p className="eyebrow mb-2">What the satellite saw</p>
      <button
        onClick={run}
        disabled={busy || !bounds}
        className="w-full rounded border border-line bg-bg-raised px-3 py-2
                   text-[12px] text-ink transition-colors hover:border-accent/50
                   disabled:cursor-not-allowed disabled:opacity-50"
      >
        {busy ? "Checking the last overpass…" : "Check the latest Sentinel-1 pass"}
      </button>

      {err && <p className="mt-2 text-[11px] leading-snug text-ink-mute">{err}</p>}

      {obs?.ok && (
        <div className="mt-3 space-y-2">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] text-ink-faint">Observed flooding</span>
            <span className="tnum text-[15px] font-semibold text-ink">
              {obs.flooded_km2?.toFixed(1)} km²
            </span>
          </div>
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] text-ink-faint">Permanent water, excluded</span>
            <span className="tnum text-[12px] text-ink-mute">
              {obs.permanent_km2?.toFixed(0)} km²
            </span>
          </div>
          <p className="text-[11px] text-ink-mute">
            {obs.acquired?.slice(0, 10)}
            {obs.days_old !== undefined && (
              <> · {obs.days_old === 0 ? "today" : `${obs.days_old} days ago`}</>
            )}
            {obs.relative_orbit != null && <> · orbit {obs.relative_orbit}</>}
          </p>
          <p className="text-[10.5px] leading-snug text-ink-faint">
            Differenced against {obs.baselines} earlier passes on the same orbit.{" "}
            {obs.note}
          </p>
        </div>
      )}
    </div>
  );
}
