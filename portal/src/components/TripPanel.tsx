"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { DEPTH_PRESETS } from "@/lib/roadquery";

/**
 * Get me from here to there, through a flood that is still moving.
 *
 * The other two views answer questions about a place: how deep, what a wall
 * changes, what gets cut. This one answers the question a person actually has
 * on the night, which is about a trip: can I get from home to the hospital, by
 * which road, and how long have I got.
 *
 * Two things make it more than a map with a line on it. The flood is routed
 * against a SCHEDULE, so the water on each road is the water that will be
 * there when the driver reaches it rather than the worst it ever gets -- which
 * is what makes a deadline possible. And the answer comes with a written brief
 * that is generated from the same record the line is drawn from, so the words
 * and the map cannot disagree.
 */

export interface Place { name: string; short: string; lat: number; lon: number }

export interface TripRoute {
  ok: boolean;
  minutes: number;
  km: number;
  max_depth_m: number;
  via: string[];
  coords: [number, number][];
  depart_min: number;
  water_on_the_way: {
    name: string | null; class: string; depth_m: number;
    km_along: number; minutes_after_leaving: number; passable: boolean;
  }[];
}

export interface TripResult {
  ok: boolean;
  reason?: string;
  origin: Place; destination: Place;
  span_km: number;
  bounds: { west: number; south: number; east: number; north: number };
  grid: [number, number];
  dx_m: number;
  vehicle_depth_m: number;
  network_km: number;
  edges: number;
  snap_km: { origin: number; destination: number };
  route: TripRoute | null;
  dry_route: TripRoute | null;
  static_route: TripRoute | null;
  leave_now: boolean;
  deadline_min: number | null;
  deadline_route: TripRoute | null;
  direct_road_cuts: { name: string | null; depth_m: number; km_along: number }[];
  direct_road_cuts_at_deadline: { name: string | null; depth_m: number; km_along: number }[];
  cut_km_now: number;
  cut_km_peak: number;
  cut_timeline: { minutes: number; cut_km: number }[];
  snapshot_minutes: number[];
  driver: string; driver_note: string;
  storm: { rain_mm_h: number; storm_hours: number; run_hours: number };
  depth_png: string; terrain_png: string;
  roads: GeoJSON.FeatureCollection;
  inspect?: unknown;
  /** Null when the written summary did not match the plan and was withheld. */
  brief: string | null;
  brief_check?: { faithfulness: number; mismatched: string[]; withheld: boolean };
  timing: { seconds: number; steps: number; solve_reused?: boolean };
  cached?: boolean;
}

const STORMS = [
  { mm_h: 20, hours: 6, label: "Heavy rain", note: "120 mm — a wet day" },
  { mm_h: 50, hours: 6, label: "Severe storm", note: "300 mm — major flooding" },
  { mm_h: 80, hours: 6, label: "Extreme", note: "480 mm — near the 2022 event" },
] as const;

function dur(min: number): string {
  const m = Math.round(min);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")}`;
}

/** One place search box. Two of these are the whole input to this view. */
function PlaceInput({
  label, value, onPick, placeholder,
}: {
  label: string;
  value: Place | null;
  onPick: (p: Place | null) => void;
  placeholder: string;
}) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Place[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (q.trim().length < 3 || (value && q === value.short)) { setHits([]); return; }
    const id = setTimeout(() => {
      fetch(`/api/geocode?q=${encodeURIComponent(q)}`)
        .then((r) => r.json())
        .then((d) => { setHits(d.results ?? []); setOpen(true); })
        .catch(() => setHits([]));
    }, 350);
    return () => clearTimeout(id);
  }, [q, value]);

  return (
    <div>
      <label className="eyebrow">{label}</label>
      <input
        value={q}
        onChange={(e) => { setQ(e.target.value); onPick(null); }}
        placeholder={placeholder}
        className="mt-1 w-full rounded-md border border-line bg-bg-inset px-2.5 py-1.5
                   text-[13px] text-ink placeholder:text-ink-faint
                   focus:border-accent/60 focus:outline-none"
      />
      {open && hits.length > 0 && (
        <ul className="mt-1 flex flex-col gap-0.5">
          {hits.map((p) => (
            <li key={`${p.lat},${p.lon}`}>
              <button
                onClick={() => { onPick(p); setQ(p.short); setHits([]); setOpen(false); }}
                title={p.name}
                className="w-full truncate rounded px-2 py-1 text-left text-[13px]
                           text-ink-mute hover:bg-bg-inset hover:text-ink"
              >
                {p.short}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function TripPanel({
  onResult, onRoutes,
}: {
  onResult: (r: TripResult | null) => void;
  onRoutes: (r: {
    quickest: { coords: [number, number][]; minutes: number } | null;
    robust: { coords: [number, number][]; minutes: number } | null;
    dest: { lat: number; lon: number } | null;
  } | null) => void;
}) {
  const [from, setFrom] = useState<Place | null>(null);
  const [to, setTo] = useState<Place | null>(null);
  const [vehicle, setVehicle] = useState(0.25);
  const [storm, setStorm] = useState(1);
  const [driver, setDriver] = useState<"river" | "rain" | "coastal">("river");
  /** River discharge in m3/s. Empty means "let the model estimate it".
   *
   *  This control exists because the estimate is not always usable. Bankfull is
   *  derived from DEM geometry, and on the Wilsons River at Lismore it comes out
   *  at 234 m3/s -- so even the Extreme preset, at 4x bankfull, reaches under a
   *  thousand, while the 2022 flood of record was an order of magnitude larger.
   *  Without a way to say which event you are planning against, the trip view
   *  could only ever model the flood the terrain implies, which is not the one
   *  an operator is asked about. */
  const [qText, setQText] = useState("");
  const [res, setRes] = useState<TripResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const find = useCallback(async () => {
    if (!from || !to) return;
    setBusy(true); setErr(null); setElapsed(0); setRes(null);
    onResult(null); onRoutes(null);
    timer.current = setInterval(() => setElapsed((t) => t + 1), 1000);
    const s = STORMS[storm];
    try {
      const r = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin: { lat: from.lat, lon: from.lon, name: from.short },
          destination: { lat: to.lat, lon: to.lon, name: to.short },
          name: `${from.short} to ${to.short}`,
          vehicle_depth_m: vehicle,
          rain_mm_h: s.mm_h, storm_hours: s.hours, run_hours: 12,
          driver, snapshots: 13, depart_step_min: 30,
          ...(driver === "river" && Number(qText) > 0
            ? { river_q_m3s: Number(qText) } : {}),
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      // A refusal that still carries the flood is not the same as a failure:
      // show the reason, and keep the map useful.
      if (!d.ok) {
        if (d.solved) onResult(d);
        throw new Error(d.reason ?? "no route could be computed");
      }
      setRes(d); onResult(d);
      onRoutes({
        quickest: d.route ? { coords: d.route.coords, minutes: d.route.minutes } : null,
        robust: null,
        dest: { lat: to.lat, lon: to.lon },
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "could not plan this trip");
    } finally {
      setBusy(false);
      if (timer.current) clearInterval(timer.current);
    }
  }, [from, to, vehicle, storm, driver, qText, onResult, onRoutes]);

  const r = res?.route ?? null;

  return (
    <div className="px-5 pb-5 pt-4">
      <h2 className="eyebrow mb-2">Plan a trip</h2>
      <p className="mb-3 text-[13px] leading-relaxed text-ink-mute">
        Where someone has to get from, and to. The route avoids roads that are
        under water when they would reach them.
      </p>

      <div className="flex flex-col gap-2.5">
        <PlaceInput label="From" value={from} onPick={setFrom}
                    placeholder="Town, suburb or address…" />
        <PlaceInput label="To" value={to} onPick={setTo}
                    placeholder="Hospital, evacuation centre, town…" />
      </div>

      <div className="mt-4">
        <h3 className="eyebrow">What is driving the water</h3>
        <div className="mt-1.5 flex gap-1">
          {(["river", "coastal", "rain"] as const).map((k) => (
            <button key={k} className="pill" data-on={driver === k}
                    onClick={() => setDriver(k)}>
              {k === "river" ? "River" : k === "coastal" ? "Coastal" : "Rainfall"}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-3">
        <h3 className="eyebrow">How big a storm</h3>
        <div className="mt-1.5 flex gap-1">
          {STORMS.map((s, i) => (
            <button key={s.label} className="pill" data-on={storm === i}
                    onClick={() => setStorm(i)} title={s.note}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {driver === "river" && (
        <div className="mt-3">
          <label htmlFor="trip-q" className="eyebrow">
            How much water is coming down the river
          </label>
          <div className="mt-1.5 flex items-center gap-2">
            <input
              id="trip-q"
              value={qText}
              onChange={(e) => setQText(e.target.value.replace(/[^0-9.]/g, ""))}
              placeholder="estimate from the terrain"
              inputMode="decimal"
              className="w-36 rounded-md border border-line bg-bg-inset px-2.5 py-1.5
                         text-[13px] text-ink placeholder:text-ink-faint
                         focus:border-accent/60 focus:outline-none"
            />
            <span className="text-[13px] text-ink-mute">m³/s</span>
            {qText && (
              <button onClick={() => setQText("")}
                      className="text-[13px] text-ink-mute hover:text-ink">
                use the estimate
              </button>
            )}
          </div>
          <p className="anno mt-1.5 leading-relaxed">
            Leave this empty and the model estimates bankfull from the terrain,
            then applies the storm above as a multiple of it. That estimate comes
            from a 30 m elevation model, not a gauged rating, and on some reaches
            it is far smaller than a flood of record — so if you are planning
            against a known event, type its peak here.
          </p>
        </div>
      )}

      <div className="mt-3">
        <h3 className="eyebrow">What is being driven</h3>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {DEPTH_PRESETS.map((p) => (
            <button key={p.m} className="pill" data-on={Math.abs(vehicle - p.m) < 1e-6}
                    onClick={() => setVehicle(p.m)} title={p.note}>
              {p.label}
            </button>
          ))}
        </div>
        <p className="anno mt-1.5">
          Water deeper than {(vehicle * 100).toFixed(0)} cm stops this vehicle
        </p>
      </div>

      <button
        onClick={find}
        disabled={busy || !from || !to}
        className="mt-4 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                   text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
      >
        {busy ? `Modelling the flood and the roads… ${elapsed}s`
          : !from || !to ? "Choose both ends of the trip"
            : "Find a way through"}
      </button>
      {busy && (
        <p className="anno mt-2 leading-relaxed">
          Terrain for a box containing both ends, a shallow-water solve across
          it, and the road network from OpenStreetMap. A minute or two is normal.
        </p>
      )}
      {err && <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">{err}</p>}

      {/* ---- the answer ------------------------------------------------- */}
      {res && (
        <>
          {/* The brief first, because it is the answer. Everything below it is
              the working. */}
          {res.brief ? (
            <div className="mt-5 rounded-xl border border-ink/40 bg-bg-inset px-4 py-3.5">
              <h3 className="eyebrow">What to do</h3>
              <p className="mt-1.5 text-[14px] leading-relaxed text-ink">{res.brief}</p>
            </div>
          ) : (
            /* The written summary is checked against the route record before it
               is shown, and withheld if it disagrees. Better a missing
               paragraph than a confident one that names the wrong road. */
            <div className="mt-5 rounded-xl border border-line px-4 py-3.5">
              <h3 className="eyebrow">No written summary</h3>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
                The summary did not match the route it was describing
                {res.brief_check?.mismatched?.length
                  ? ` (${res.brief_check.mismatched.join(", ")})`
                  : ""}
                , so it has been withheld. The figures below are the route
                record itself and are unaffected.
              </p>
            </div>
          )}

          {r ? (
            <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
              <div className="flex gap-7">
                <div>
                  <p className="tnum text-[26px] font-semibold leading-none text-ink">
                    {dur(r.minutes)}
                  </p>
                  <p className="anno mt-1.5">driving</p>
                </div>
                <div>
                  <p className="tnum text-[26px] font-semibold leading-none text-ink">
                    {r.km.toFixed(1)} km
                  </p>
                  <p className="anno mt-1.5">by road</p>
                </div>
                {res.dry_route && res.dry_route.minutes < r.minutes - 0.5 && (
                  <div>
                    <p className="tnum text-[26px] font-semibold leading-none text-ink">
                      +{dur(r.minutes - res.dry_route.minutes)}
                    </p>
                    <p className="anno mt-1.5">vs a dry day</p>
                  </div>
                )}
              </div>
              {r.via.length > 0 && (
                <p className="mt-3 text-[13px] leading-relaxed text-ink">
                  Via {r.via.slice(0, 5).join(", ")}
                </p>
              )}
              {!res.leave_now && (
                <p className="mt-2 text-[13px] leading-relaxed text-ink">
                  Leaving right now does not get through. This is the trip
                  leaving in {dur(r.depart_min)}.
                </p>
              )}
            </div>
          ) : (
            <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
              <span className="chip-harm text-[13px]">No way through</span>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
                Every road between these two points is under more water than
                this vehicle can cross, at every departure time in the next{" "}
                {res.storm.run_hours} hours.
              </p>
            </div>
          )}

          {res.cut_km_peak <= 0.05 && (
            <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
              <h3 className="eyebrow">No flooding on this trip</h3>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
                In this scenario no road between these two points goes under
                more than {(res.vehicle_depth_m * 100).toFixed(0)} cm at any
                point in the next {res.storm.run_hours} hours, so there is
                nothing to route around and no deadline. The drive is the
                ordinary drive.
              </p>
              <p className="anno mt-2 leading-relaxed">
                {res.driver === "rain"
                  ? `The storm is falling as rain across the whole box at `
                    + `${res.dx_m.toFixed(0)} m cells, which spreads it thin. A `
                    + `river or coastal driver concentrates the water where a `
                    + `channel or a coast actually puts it.`
                  : `Try a larger storm, or a driver that matches what floods `
                    + `this place.`}
              </p>
            </div>
          )}

          {/* The deadline. This is the thing a peak-depth map cannot tell you,
              and the reason the flood is carried as a schedule. */}
          {res.deadline_min !== null && (
            <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
              <div className="flex items-baseline justify-between">
                <h3 className="eyebrow">How long this stays possible</h3>
                <span className="tnum text-[18px] font-semibold text-ink">
                  {dur(res.deadline_min)}
                </span>
              </div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
                Leave within {dur(res.deadline_min)} and there is still a way
                through. After that, every route between these points is cut
                for the rest of the modelled window.
              </p>
            </div>
          )}

          {/* The later departure, which is where the detour appears: by then
              the quick road is under water and the route goes round it. This is
              the answer a peak-depth map cannot produce at all. */}
          {res.deadline_route && r &&
           (res.deadline_route.km !== r.km || res.deadline_route.minutes !== r.minutes) && (
            <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
              <h3 className="eyebrow">If they leave at the last moment</h3>
              <div className="mt-1.5 flex items-baseline justify-between">
                <span className="text-[13px] text-ink">
                  A different way round, leaving in {dur(res.deadline_route.depart_min)}
                </span>
                <span className="tnum text-[15px] font-semibold text-ink">
                  {dur(res.deadline_route.minutes)}
                </span>
              </div>
              <p className="anno mt-1">
                {res.deadline_route.km.toFixed(1)} km
                {res.deadline_route.minutes > r.minutes &&
                  ` · ${dur(res.deadline_route.minutes - r.minutes)} longer than leaving now`}
                {res.deadline_route.via.length > 0 &&
                  ` · via ${res.deadline_route.via.slice(0, 4).join(", ")}`}
              </p>
              {res.direct_road_cuts_at_deadline?.length > 0 && (
                <>
                  <p className="mt-2.5 text-[13px] leading-relaxed text-ink">
                    By then the quick road is under water:
                  </p>
                  <ul className="mt-1 space-y-1">
                    {res.direct_road_cuts_at_deadline.slice(0, 3).map((c, i) => (
                      <li key={i} className="flex justify-between text-[13px]">
                        <span className="truncate text-ink-mute">
                          {c.name ?? "Unnamed road"}
                        </span>
                        <span className="tnum text-[var(--warn,#dc2626)]">
                          {(c.depth_m * 100).toFixed(0)} cm
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}

          {/* Water on the chosen route, in order. */}
          {r && r.water_on_the_way.length > 0 && (
            <div className="mt-4">
              <h3 className="eyebrow">Water on the way</h3>
              <ul className="mt-2 overflow-hidden rounded-xl border border-line">
                {r.water_on_the_way.slice(0, 8).map((c, i) => (
                  <li key={`${c.name}-${i}`}
                      className={`flex items-baseline gap-3 px-3.5 py-2.5
                                  ${i > 0 ? "border-t border-line-soft" : ""}`}>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] text-ink">
                        {c.name ?? "Unnamed road"}
                      </span>
                      <span className="anno">
                        {c.km_along.toFixed(1)} km in · {dur(c.minutes_after_leaving)} after leaving
                      </span>
                    </span>
                    <span className={`tnum shrink-0 text-[13px] ${
                      c.passable ? "text-ink" : "text-[var(--warn,#dc2626)]"}`}>
                      {(c.depth_m * 100).toFixed(0)} cm
                    </span>
                  </li>
                ))}
              </ul>
              <p className="anno mt-2">
                Depth where the road is deepest, at the time the driver reaches it
              </p>
            </div>
          )}

          {/* What the direct road does — "avoids flooding" means something
              concrete only when you can see what it avoided. */}
          {res.direct_road_cuts.length > 0 && (
            <div className="mt-4 rounded-xl border border-line px-4 py-3.5">
              <h3 className="eyebrow">Why not the direct road</h3>
              <ul className="mt-1.5 space-y-1">
                {res.direct_road_cuts.slice(0, 4).map((c, i) => (
                  <li key={i} className="flex justify-between text-[13px]">
                    <span className="truncate text-ink">{c.name ?? "Unnamed road"}</span>
                    <span className="tnum text-[var(--warn,#dc2626)]">
                      {(c.depth_m * 100).toFixed(0)} cm
                    </span>
                  </li>
                ))}
              </ul>
              <p className="anno mt-2 leading-relaxed">
                The quickest route ignoring water goes through these, which is
                why the one above is longer.
              </p>
            </div>
          )}

          {/* Why the schedule matters, stated as the difference it made. */}
          {res.static_route && r && (
            <p className="mt-4 text-[13px] leading-relaxed text-ink-mute">
              {res.static_route.minutes > r.minutes + 0.5
                ? `Routing against the worst the flood ever gets would have sent
                   this trip the long way round — ${dur(res.static_route.minutes)}
                   instead of ${dur(r.minutes)} — because it refuses roads that
                   are dry when the driver passes them.`
                : `Here the worst-case and the timed answer agree, so this route
                   is open whenever the driver leaves within the deadline.`}
            </p>
          )}

          {res.cut_timeline.length > 2 && res.cut_km_peak > 0.05 && (
            <div className="mt-4">
              <h3 className="eyebrow">Roads lost as the flood arrives</h3>
              <div className="mt-2 flex items-end gap-[3px]">
                {res.cut_timeline.map((t) => {
                  const peak = Math.max(...res.cut_timeline.map((x) => x.cut_km), 1);
                  const h = Math.max(2, Math.round((t.cut_km / peak) * 54));
                  const past = res.deadline_min !== null && t.minutes > res.deadline_min;
                  return (
                    <div key={t.minutes} className="flex flex-1 flex-col items-center gap-1">
                      <div
                        className={`w-full rounded-sm ${past ? "bg-[var(--warn,#dc2626)]/70" : "bg-ink/70"}`}
                        style={{ height: `${h}px` }}
                        title={`${dur(t.minutes)}: ${t.cut_km.toFixed(0)} km impassable`}
                      />
                    </div>
                  );
                })}
              </div>
              <p className="anno mt-1.5">
                0 to {dur(res.cut_timeline[res.cut_timeline.length - 1].minutes)} ·
                up to {Math.max(...res.cut_timeline.map((x) => x.cut_km)).toFixed(0)} km
                of {res.network_km.toFixed(0)} km impassable
                {res.deadline_min !== null && " · red is after the deadline"}
              </p>
            </div>
          )}

          <div className="mt-4 border-t border-line-soft pt-3">
            <p className="anno leading-relaxed">
              {/* Not "impassable now": the run starts dry, so the t=0 figure is
                  always about zero and says nothing about the flood. */}
              Up to {res.cut_km_peak.toFixed(0)} km of the{" "}
              {res.network_km.toFixed(0)} km network goes impassable at the worst
              of this flood. {res.edges} road segments over a{" "}
              {res.span_km.toFixed(0)} km trip at {res.dx_m.toFixed(0)} m cells
              {res.timing.solve_reused
                ? `, re-routed in ${res.timing.seconds.toFixed(0)}s on the flood already solved for this storm`
                : `, solved in ${res.timing.seconds.toFixed(0)}s`}.
            </p>
            <p className="anno mt-1.5 leading-relaxed">{res.driver_note}</p>
            {(res.snap_km.origin > 0.5 || res.snap_km.destination > 0.5) && (
              <p className="anno mt-1.5 leading-relaxed">
                Nearest mapped road is {res.snap_km.origin.toFixed(1)} km from
                the start and {res.snap_km.destination.toFixed(1)} km from the
                destination; the trip is routed between those points.
              </p>
            )}
            <p className="anno mt-1.5 leading-relaxed">
              A design storm, not a forecast of today. Speeds are free-flow by
              road class, with no traffic model.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
