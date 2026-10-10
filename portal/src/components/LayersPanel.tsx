"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * What is drawn on the map, and where each layer comes from.
 *
 * Every row carries its source and its cadence under the name, because the
 * four things on this map are not the same kind of thing and the difference
 * matters more than the toggle does. Water depth is a model run at a moment.
 * Gauges are instruments reporting every fifteen minutes. Radar is an
 * observation six minutes old. Road status is derived from the first by a
 * threshold the user chose. A map that stacks them without saying which is
 * which invites the reader to treat the model as an instrument.
 *
 * A layer with no feed here is shown and disabled rather than hidden, with
 * the reason in the same place the cadence would be. Hiding it would let the
 * absence pass as "not applicable"; the Richmond study genuinely has no live
 * gauge feed, and that is a fact about the data, not about the map.
 */

export type LayerKey = "depth" | "gauges" | "radar" | "roads";

export interface LayerState {
  depth: boolean;
  gauges: boolean;
  radar: boolean;
  roads: boolean;
}

export const DEFAULT_LAYERS: LayerState = {
  depth: true, gauges: true, radar: false, roads: true,
};

export interface Gauge {
  site: string;
  name: string;
  lat: number;
  lon: number;
  stage_m: number | null;
  q_m3s: number | null;
  trend_cm_h: number | null;
  at: string | null;
  series: { t: string; m: number }[];
  /** NWS flood categories for this gauge, where it has them. Most small urban
   *  creeks do not: of 25 gauged streams in the Austin box, 18 match an NWS
   *  forecast point and 4 have published levels. */
  flood_levels_m: { action?: number; minor?: number;
                    moderate?: number; major?: number } | null;
  flood_category: string | null;
  forecast: { stage_m: number; at: string; category: string } | null;
  /** The forecast CURVE, from the NWPS stageflow endpoint: the hourly trace
   *  the forecaster issued, typically five days of it. `forecast` above is
   *  only its peak, which is enough for a headline and not enough to draw. */
  forecast_series: { issued: string | null;
                     points: { t: string; m: number }[] } | null;
  lid: string | null;
}

export interface GaugeFeed {
  supported: boolean;
  gauges: Gauge[];
  cadence_min: number | null;
  note?: string;
  source?: string;
}

function Row({
  label, sub, on, onToggle, disabled, reason,
}: {
  label: string; sub: string; on: boolean;
  onToggle: () => void; disabled?: boolean; reason?: string;
}) {
  return (
    <div className="flex items-start justify-between gap-3 py-2">
      <div className="min-w-0">
        <p className={`text-[13px] ${disabled ? "text-ink-faint" : "text-ink"}`}>{label}</p>
        <p className="anno mt-0.5 leading-relaxed">{disabled ? reason : sub}</p>
      </div>
      <button
        role="switch"
        aria-checked={on && !disabled}
        aria-label={label}
        disabled={disabled}
        onClick={onToggle}
        className={`mt-0.5 h-[22px] w-[38px] shrink-0 rounded-full border transition-colors
          ${disabled ? "cursor-not-allowed border-line bg-bg-inset"
            : on ? "border-ink bg-ink" : "border-line bg-bg-inset"}`}
      >
        <span
          className={`block h-[16px] w-[16px] rounded-full bg-bg shadow-sm transition-transform
            ${on && !disabled ? "translate-x-[19px]" : "translate-x-[2px]"}
            ${disabled ? "bg-line" : ""}`}
        />
      </button>
    </div>
  );
}

export function LayersPanel({
  bounds, layers, onLayers, onGauges, modelRunNote, reach,
}: {
  bounds: { west: number; south: number; east: number; north: number } | null;
  layers: LayerState;
  onLayers: (l: LayerState) => void;
  onGauges: (f: GaugeFeed | null) => void;
  /** When the depth field on screen was solved. */
  modelRunNote?: string;
  /** The watercourse this study is about. Sent with the gauge query so the
   *  reach's own gauge is ranked first -- without it the feed comes back
   *  biggest-discharge-first and the headline reports the Colorado River on
   *  a study about Onion Creek. */
  reach?: string;
}) {
  // The answer is stored WITH the box it answers for, so "still loading" is a
  // comparison rather than a second piece of state set at the top of an
  // effect. Setting a busy flag synchronously in an effect body is the
  // cascading-render pattern React asks you not to write, and here it was not
  // buying anything: the box itself already says whether the reply is stale.
  const key = bounds
    ? `${bounds.west.toFixed(4)},${bounds.south.toFixed(4)},${bounds.east.toFixed(4)},${bounds.north.toFixed(4)}`
    : "";
  const [answer, setAnswer] = useState<{ key: string; feed: GaugeFeed | null } | null>(null);
  const feed = answer?.key === key ? answer.feed : null;
  // Only the FIRST fetch for a box is "loading". A scheduled re-poll keeps
  // the last reading on screen until the new one lands, so the panel does
  // not blink empty every five minutes.
  const busy = Boolean(bounds) && answer === null;

  /* Poll, rather than claim to.
     The header has said "LIVE · 27 gauges every 15 min" since this panel was
     written, and nothing ever re-fetched: the badge, the reading and the
     hydrograph were all frozen at page load and went stale in silence. USGS
     publishes on a 15-minute cadence, so asking every five minutes is at
     most one cycle behind and costs one cached request. */
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 5 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!bounds) return;
    let live = true;
    fetch("/api/gauges", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...bounds, reach: reach ?? "" }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d: GaugeFeed) => { if (live) { setAnswer({ key, feed: d }); onGauges(d); } })
      .catch(() => { if (live) { setAnswer({ key, feed: null }); onGauges(null); } });
    return () => { live = false; };
  }, [bounds, key, onGauges, reach, tick]);

  const set = useCallback(
    (k: LayerKey) => onLayers({ ...layers, [k]: !layers[k] }),
    [layers, onLayers],
  );

  const n = feed?.gauges.length ?? 0;
  const gaugeSub = busy ? "loading…"
    : feed?.supported === false ? "no open feed for this area"
      : n ? `${n} station${n === 1 ? "" : "s"}`
        : "no gauged stream in this box";

  return (
    <div className="border-b border-line-soft px-5 py-3">
      <h3 className="eyebrow mb-1">Layers</h3>
      <Row
        label="Water depth"
        sub={modelRunNote ?? ""}
        on={layers.depth}
        onToggle={() => set("depth")}
      />
      <Row
        label="River gauges"
        sub={gaugeSub}
        on={layers.gauges}
        onToggle={() => set("gauges")}
        disabled={feed?.supported === false || (!busy && n === 0)}
        reason={feed?.supported === false
          ? "USGS covers the United States; no equivalent feed here"
          : "no gauged stream in this box"}
      />
      <Row
        label="Rain radar"
        sub=""
        on={layers.radar}
        onToggle={() => set("radar")}
        disabled={feed?.supported === false}
        reason="NOAA radar covers the United States"
      />
      <Row
        label="Road status"
        sub=""
        on={layers.roads}
        onToggle={() => set("roads")}
      />
      <Row
        label="Cameras"
        sub=""
        on={false}
        onToggle={() => {}}
        disabled
        reason="no verified public feed wired up yet"
      />
    </div>
  );
}
