"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SectionChart } from "./SectionChart";
import { STORM_PRESETS } from "@/lib/storms";
import {
  assess, channelBelts, dist, lowestHoldingCrest, money,
  panels, reaches, sampleSpacing, spanFor, stageOf,
  type Reach, type SectionResult, type Span,
} from "@/lib/section";

/**
 * Cut the floodplain open along a line, and design a wall on what you see.
 *
 * The Plan tab answers "which of six surveyed sites should we build", which is
 * the question a funded programme has. This answers the one that comes before
 * it: along THIS line, how high does a crest have to be, how far does it have
 * to run, and what does it cost -- with the water on the section being the
 * water the solver produced, not an assumed design stage.
 *
 * Three things here are not cosmetic.
 *
 * It works per REACH, not per line. A section across a floodplain crosses
 * several disconnected bodies of water at different stages; the measured
 * Lismore line crosses nine, from 8.80 m to 20.67 m. Designing to the highest
 * of them cost AUD 88.5 M against AUD 29.7 M for the crest that defends the
 * town, and bought nothing, because the water that set the figure is on the
 * far side of a 67 m hill.
 *
 * It states the height rather than asking for it. The cheapest crest that
 * holds is a search, not an addition: raising the crest lengthens the span
 * needed to tie into the bank, so clearing the water and tying in do not
 * happen at the same elevation.
 *
 * And it distinguishes overtopping from an open end. The second is the failure
 * a section cannot see -- an embankment that stops on ground below its own
 * crest has a gap at that elevation, whatever the plot shows -- so it is
 * reported as an incomplete design, with the shortfall in metres, instead of
 * being folded into a pass or a fail.
 */

export interface SectionEnds {
  a: { lat: number; lon: number } | null;
  b: { lat: number; lon: number } | null;
}

export interface SectionOverlay {
  a: { lat: number; lon: number };
  b: { lat: number; lon: number };
  /** The stations the crest runs between, as fractions along the line, so the
   *  map can draw the embankment where it actually is. */
  crest: { from: number; to: number } | null;
  verdict: string;
}

function Row({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-[13px] text-ink-mute">{label}</span>
      <span className="text-right">
        <span className="tnum text-[13px] font-medium text-ink">{value}</span>
        {sub && <span className="anno ml-1.5">{sub}</span>}
      </span>
    </div>
  );
}

export function SectionPanel({
  ends, picking, onPicking, onClearEnds, onOverlay,
}: {
  ends: SectionEnds;
  picking: boolean;
  onPicking: (v: boolean) => void;
  onClearEnds: () => void;
  onOverlay: (o: SectionOverlay | null) => void;
}) {
  const [storm, setStorm] = useState(1);
  const [driver, setDriver] = useState<"river" | "rain" | "coastal">("river");
  /** Which driver was ASKED for, kept alongside the one that ran. A river
   *  driver that finds no through-channel in the derived box falls back to
   *  rainfall, and a dry section then means "this rain did not reach here"
   *  rather than "this river does not reach here" -- two different answers
   *  that look identical in a profile. */
  const [asked, setAsked] = useState<"river" | "rain" | "coastal">("river");
  const [qText, setQText] = useState("");
  const [res, setRes] = useState<SectionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  /** Which body of water is being defended. */
  const [reachIdx, setReachIdx] = useState(0);
  /** Counts solves, so the design below can be keyed to one. */
  const [solveN, setSolveN] = useState(0);
  /** The crest the user moved to, if they moved it.
   *
   *  Held as an override against a key rather than as state seeded by an
   *  effect: the design shown is DERIVED from the solve and the chosen reach,
   *  and seeding it in an effect means a render where the chart draws the old
   *  reach's wall on the new reach's water.
   */
  const [ov, setOv] = useState<{ key: string; crestM: number; span: Span } | null>(null);

  /** Which floodplain panel of that reach the embankment runs along. */
  const [panelIdx, setPanelIdx] = useState(0);

  const rs = useMemo(() => (res?.ok ? reaches(res.profile) : []), [res]);
  const reach: Reach | null = rs[reachIdx] ?? null;
  const belts = useMemo(
    () => (res?.ok && reach ? channelBelts(res.profile, reach) : []),
    [res, reach]);
  const pns = useMemo(
    () => (res?.ok && reach ? panels(res.profile, reach, belts) : []),
    [res, reach, belts]);
  const panel: Span | null = pns[panelIdx] ?? pns[0] ?? null;
  const spacing = useMemo(() => (res ? sampleSpacing(res) : 1), [res]);
  const rate = res?.cost_per_m_per_m_aud ?? 2600;

  // The cheapest crest that holds this reach, recomputed when the reach
  // changes. This is the panel's answer; the controls below it exist to walk
  // away from that answer deliberately, not to arrive at it.
  const cheapest = useMemo(
    () => (res?.ok && panel
      ? lowestHoldingCrest(res.profile, spacing, panel, rate, belts)
      : null),
    [res, panel, spacing, rate, belts],
  );

  const key = `${solveN}:${reachIdx}:${panelIdx}`;
  /** The stage over the panel being defended, which is NOT the reach's stage
   *  when the channel splits it: the water standing on the far floodplain is
   *  not what this wall holds back. */
  const stage = useMemo(
    () => (res?.ok && panel ? stageOf(res.profile, panel) : null),
    [res, panel]);
  const dflt = useMemo(() => {
    if (!res?.ok || !panel || stage === null) return null;
    if (cheapest) return { crestM: cheapest.crestM, span: cheapest.span };
    // No crest within 8 m of the stage ties in: the line is too short to find
    // the bank. Show the stage itself and say so.
    const c = Math.round(stage * 10) / 10;
    return { crestM: c, span: spanFor(res.profile, c, panel, belts) };
  }, [res, panel, stage, cheapest, belts]);

  const mine = ov?.key === key ? ov : null;
  const crestM = mine ? mine.crestM : dflt?.crestM ?? null;
  const span = mine ? mine.span : dflt?.span ?? null;
  const setCrestM = useCallback(
    (m: number) => setOv((o) => {
      const base = o?.key === key ? o : null;
      const sp = base?.span ?? dflt?.span;
      return sp ? { key, crestM: m, span: sp } : o;
    }),
    [key, dflt],
  );
  const setSpan = useCallback(
    (sp: Span) => setOv((o) => {
      const base = o?.key === key ? o : null;
      const c = base?.crestM ?? dflt?.crestM;
      return c !== undefined ? { key, crestM: c, span: sp } : o;
    }),
    [key, dflt],
  );

  const a = useMemo(
    () => (res?.ok && span && crestM !== null
      ? assess(res.profile, spacing, span, crestM, rate, belts)
      : null),
    [res, span, crestM, spacing, rate, belts],
  );

  // What the line and the wall are, for the map.
  useEffect(() => {
    if (!ends.a || !ends.b) { onOverlay(null); return; }
    const L = res?.ok ? res.length_m : 0;
    onOverlay({
      a: ends.a, b: ends.b,
      crest: res?.ok && span && L > 0
        ? { from: res.profile[span.i0].m / L, to: res.profile[span.i1].m / L }
        : null,
      verdict: a?.verdict ?? "",
    });
  }, [ends.a, ends.b, res, span, a?.verdict, onOverlay]);

  const solve = useCallback(async () => {
    if (!ends.a || !ends.b) return;
    setBusy(true); setErr(null); setElapsed(0); setRes(null); setReachIdx(0);
    setOv(null); setSolveN((n) => n + 1); setAsked(driver); setPanelIdx(0);
    timer.current = setInterval(() => setElapsed((t) => t + 1), 1000);
    const s = STORM_PRESETS[storm];
    try {
      const r = await fetch("/api/section", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          a: ends.a, b: ends.b, name: "Cross-section", samples: 140,
          rain_mm_h: s.mm_h, storm_hours: s.hours, run_hours: 12, driver,
          ...(driver === "river" && Number(qText) > 0
            ? { river_q_m3s: Number(qText) } : {}),
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? d.error ?? `HTTP ${r.status}`);
      if (!d.ok) throw new Error(d.reason ?? "this line could not be solved");
      setRes(d);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "could not solve this line");
    } finally {
      setBusy(false);
      if (timer.current) clearInterval(timer.current);
    }
  }, [ends.a, ends.b, storm, driver, qText]);

  /** The glanceable half of `driver_note`: what forced the water, in a line. */
  const forcing = res?.ok ? res.driver_note.split(". ")[0].replace(/\.$/, "") : null;

  // The teaching comparison: what the line's own highest water would cost if
  // it were mistaken for the design stage of this reach.
  const naive = useMemo(() => {
    if (!res?.ok || !panel || stage === null || res.peak_water_m === null) return null;
    if (res.peak_water_m <= stage + 0.5) return null;
    const c = res.peak_water_m;
    const sp = spanFor(res.profile, c, panel, belts);
    return { crestM: c, ...assess(res.profile, spacing, sp, c, rate, belts) };
  }, [res, panel, stage, spacing, rate, belts]);

  const p = res?.ok ? res.profile : [];

  // Answer first. Once a section has been cut, the verdict and the wall open
  // the panel and the inputs that produced them fold away -- the same move the
  // Plan tab makes, and for the same reason: the controls are read once and
  // the answer is read every time.
  const inputs = (
    <>
        {/* ---- the line ------------------------------------------------- */}
        <div className="rounded-lg border border-line bg-bg-raised p-3">
          <div className="flex items-start justify-between gap-2">
            <p className="eyebrow">The line</p>
            {(ends.a || ends.b) && (
              <button onClick={() => { onClearEnds(); setRes(null); onPicking(true); }}
                      className="text-[12px] text-ink-faint hover:text-ink">
                clear
              </button>
            )}
          </div>
          {!ends.a && (
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
              {picking
                ? "Click the map where the section should start."
                : "Turn on picking, then click the map twice."}
            </p>
          )}
          {ends.a && !ends.b && (
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
              Now click the far side of the floodplain.
            </p>
          )}
          {ends.a && ends.b && (
            <p className="anno mt-1.5 leading-relaxed">
              A {ends.a.lat.toFixed(4)}, {ends.a.lon.toFixed(4)}<br />
              B {ends.b.lat.toFixed(4)}, {ends.b.lon.toFixed(4)}
            </p>
          )}
          {!picking && !(ends.a && ends.b) && (
            <button onClick={() => onPicking(true)}
                    className="mt-2 rounded-md border border-line px-2.5 py-1 text-[13px]
                               text-ink hover:bg-bg-inset">
              Pick on the map
            </button>
          )}
        </div>

        {/* ---- forcing --------------------------------------------------- */}
        <div className="mt-3">
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
            {STORM_PRESETS.map((s, i) => (
              <button key={s.label} className="pill" data-on={storm === i}
                      onClick={() => setStorm(i)} title={s.note}>
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {driver === "river" && (
          <div className="mt-3">
            <label htmlFor="sec-q" className="eyebrow">
              How much water is coming down the river
            </label>
            <div className="mt-1.5 flex items-center gap-2">
              <input
                id="sec-q"
                value={qText}
                onChange={(e) => setQText(e.target.value.replace(/[^0-9.]/g, ""))}
                placeholder="estimate from the terrain"
                inputMode="decimal"
                className="w-36 rounded-md border border-line bg-bg-inset px-2.5 py-1.5
                           text-[13px] text-ink placeholder:text-ink-faint
                           focus:border-accent/60 focus:outline-none"
              />
              <span className="text-[13px] text-ink-mute">m³/s</span>
            </div>
            <p className="anno mt-1.5 leading-relaxed">
              Bankfull is derived from the 30 m elevation model, and on the Wilsons
              River it comes out an order of magnitude under the 2022 flood of
              record — so if you are designing against a known event, type its peak.
            </p>
          </div>
        )}

        <button
          onClick={solve}
          disabled={busy || !ends.a || !ends.b}
          className="mt-4 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                     text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {busy ? `Solving the flood on this line… ${elapsed}s`
            : !ends.a || !ends.b ? "Draw a line first"
              : res?.ok ? "Cut it again"
                : "Cut the section"}
        </button>
        {busy && (
          <p className="anno mt-1.5 leading-relaxed">
            Terrain is fetched for a box containing the line and the flood solved
            across it. Around 30 s for a 7 km line; a line over ground already
            solved for a trip reuses that solve.
          </p>
        )}
        {err && (
          <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
            <p className="text-[13px] font-medium text-ink">Could not cut this section</p>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{err}</p>
          </div>
        )}
    </>
  );

  return (
    <div className="px-5 pb-5 pt-4">
      {!res?.ok && (
        <>
          <h2 className="eyebrow mb-2">Cut a cross-section</h2>
          <p className="mb-3 text-[13px] leading-relaxed text-ink-mute">
            Draw a line across the floodplain and see the ground under it, the
            water the model puts on it, and what a levee along it would have to
            be.
          </p>
        </>
      )}
      {!res?.ok && inputs}

      {/* ---- the answer ------------------------------------------------ */}
      {res?.ok && rs.length === 0 && (
        <div className="mt-4 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[14px] font-semibold text-ink">
            This line stayed dry
          </p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
            The solve put no standing water on it, so there is nothing here to
            build against. That is a statement about this storm on this line,
            not about the valley.
          </p>
          {res.driver !== asked && (
            <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
              <span className="chip-harm">read this before concluding anything</span>{" "}
              You asked for {asked === "river" ? "a river" : asked}, and none was found in
              the box this line derives — so the storm ran as{" "}
              {res.driver === "rain" ? "rainfall" : res.driver} instead. Spread over the
              whole box it floods very little. A dry line here is not evidence the{" "}
              {asked} does not reach it; draw the line so it crosses the channel, or
              widen it until the box contains one.
            </p>
          )}
          <p className="anno mt-2 leading-relaxed">{res.driver_note}</p>
        </div>
      )}

      {res?.ok && reach && a && crestM !== null && span && stage !== null && (
        <>
          <div className="mt-4 rounded-lg border border-line bg-bg-raised p-3">
            <p className="eyebrow">
              {a.verdict === "holds" ? "This embankment holds"
                : a.verdict === "overtops" ? `Overtopped by ${a.overtopM.toFixed(2)} m`
                  : a.verdict === "crosses-channel" ? "That is a dam, not a levee"
                    : a.verdict === "dry" ? "Nothing to hold back here"
                      : "It holds, but it has an open end"}
            </p>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
              {a.verdict === "holds" && (
                <>A crest at <span className="tnum text-ink">{crestM.toFixed(2)} m</span> runs{" "}
                  {dist(a.lengthM)} along this line and clears the{" "}
                  <span className="tnum text-ink">{stage.toFixed(2)} m</span> peak standing
                  on it. Its west end {a.westEnd === "river"
                    ? "meets the channel bank"
                    : `ties into ground at ${a.westTieM.toFixed(1)} m`}; its east end{" "}
                  {a.eastEnd === "river"
                    ? "meets the channel bank"
                    : `ties into ground at ${a.eastTieM.toFixed(1)} m`}.</>
              )}
              {a.verdict === "crosses-channel" && (
                <>This alignment runs over the channel between{" "}
                  <span className="tnum text-ink">
                    {dist(res.profile[belts[0].i0].m)}</span> and{" "}
                  <span className="tnum text-ink">
                    {dist(res.profile[belts[0].i1].m)}</span>, where the bed sits{" "}
                  {(stage - res.profile[belts[0].i0].ground_m).toFixed(1)} m under water. A
                  continuous embankment across that is a dam. Pull the ends back onto the
                  floodplain either side.</>
              )}
              {a.verdict === "overtops" && (
                <>The water reaches <span className="tnum text-ink">
                  {(crestM + a.overtopM).toFixed(2)} m</span> at{" "}
                  {dist(a.overtopAtM ?? 0)} along the line, above this crest. Raise it
                  to at least that, or use the lowest crest that holds below.</>
              )}
              {a.verdict === "open-end" && (
                <>It clears the water, but the{" "}
                  {a.westEnd === "open" ? "west" : "east"} end stops on ground at{" "}
                  <span className="tnum text-ink">
                    {(a.westEnd === "open" ? a.westTieM : a.eastTieM).toFixed(2)} m
                  </span>,{" "}
                  {(a.westEnd === "open" ? a.westShortM : a.eastShortM).toFixed(2)} m below
                  the crest.
                  Water standing at the design stage goes around an opening at that
                  elevation. A section cannot see around its own ends, so this is an
                  incomplete design rather than a failure.</>
              )}
            </p>
            {res.driver !== asked ? (
              <p className="mt-2 border-t border-line-soft pt-2 text-[13px] leading-relaxed text-ink-mute">
                <span className="chip-harm">not the driver you asked for</span>{" "}
                You asked for {asked === "river" ? "a river" : asked}; the model ran{" "}
                {res.driver === "rain" ? "rainfall" : res.driver} instead, because no{" "}
                {asked === "river" ? "through-channel" : "sea edge"} was found in the box
                it derived. A dry stretch here means this {res.driver} did not reach it,
                not that the {asked} does not.
              </p>
            ) : (
              <p className="anno mt-2 leading-relaxed">Forced by {res.driver}: {forcing}</p>
            )}
            <div className="mt-2.5 border-t border-line-soft pt-2">
              <Row label="Cost, indicative" value={money(a.costAud)}
                   sub={`${rate.toFixed(0)}/m/m`} />
              <Row label="Crest length" value={dist(a.lengthM)} />
              <Row label="Height, max" value={`${a.maxHeightM.toFixed(2)} m`}
                   sub={`mean ${a.meanHeightM.toFixed(2)} m`} />
              <Row label="Fill cross-section" value={`${a.fillM2.toFixed(0)} m²`} />
            </div>
          </div>

          <div className="mt-3">
            <SectionChart res={res} span={span} crestM={crestM}
                          a={a} belts={belts} onCrest={setCrestM} onSpan={setSpan} />
          </div>

          <p className="anno mt-1.5 leading-relaxed">
            The water drawn is the flood solved WITHOUT this wall. A levee changes the
            flood; this sizes a crest against the unmitigated peak and does not re-solve.
            The Plan tab does re-solve, for its six surveyed sites.
          </p>
          <p className="anno mt-1.5 leading-relaxed">
            The wall runs ALONG this line — a tie-back embankment from high ground
            out across the floodplain, not a levee following the river.
            {belts.length > 0 && " The channel is shaded; an alignment over it is refused."}
          </p>

          {/* ---- the controls -------------------------------------------- */}
          <div className="mt-2">
            <label htmlFor="sec-crest" className="flex items-baseline justify-between">
              <span className="eyebrow">Crest elevation</span>
              <span className="tnum text-[13px] text-ink">{crestM.toFixed(2)} m</span>
            </label>
            <input
              id="sec-crest"
              type="range"
              min={Math.floor(reach.minGroundM)}
              max={Math.ceil(stage + 6)}
              step={0.1}
              value={crestM}
              onChange={(e) => setCrestM(Number(e.target.value))}
              className="mt-1 w-full accent-ink"
            />
            <p className="anno leading-relaxed">
              {crestM >= stage
                ? `${(crestM - stage).toFixed(2)} m of freeboard over the solved peak`
                : `${(stage - crestM).toFixed(2)} m below the solved peak`}
            </p>
            {crestM - stage < 0.3 && crestM >= stage && (
              <p className="anno leading-relaxed">
                That is the cheapest crest that holds, not a design standard. Levee
                design normally carries 0.3 to 0.6 m of freeboard for wave run-up and
                for being wrong about the stage.
              </p>
            )}
          </div>

          <div className="mt-2 grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="sec-w" className="eyebrow">West end</label>
              <input id="sec-w" type="range" min={0} max={p.length - 2} step={1}
                     value={span.i0}
                     onChange={(e) => setSpan({
                       i0: Math.min(Number(e.target.value), span.i1 - 1), i1: span.i1 })}
                     className="mt-1 w-full accent-ink" />
              <p className="anno">{dist(p[span.i0].m)} · {p[span.i0].ground_m.toFixed(1)} m</p>
            </div>
            <div>
              <label htmlFor="sec-e" className="eyebrow">East end</label>
              <input id="sec-e" type="range" min={1} max={p.length - 1} step={1}
                     value={span.i1}
                     onChange={(e) => setSpan({
                       i0: span.i0, i1: Math.max(Number(e.target.value), span.i0 + 1) })}
                     className="mt-1 w-full accent-ink" />
              <p className="anno">{dist(p[span.i1].m)} · {p[span.i1].ground_m.toFixed(1)} m</p>
            </div>
          </div>

          {cheapest && (crestM !== cheapest.crestM
            || span.i0 !== cheapest.span.i0 || span.i1 !== cheapest.span.i1) && (
            <button
              onClick={() => { setCrestM(cheapest.crestM); setSpan(cheapest.span); }}
              className="mt-2.5 w-full rounded-md border border-line px-3 py-2 text-left
                         text-[13px] text-ink hover:bg-bg-inset"
            >
              Back to the cheapest crest that holds —{" "}
              <span className="tnum">{cheapest.crestM.toFixed(2)} m</span>,{" "}
              {money(assess(res.profile, spacing, cheapest.span, cheapest.crestM, rate).costAud)}
            </button>
          )}
          {!cheapest && (
            <p className="anno mt-2 leading-relaxed">
              No crest within 8 m of the {stage.toFixed(2)} m stage ties into ground at
              its landward end, so this line is too short to find the bank. Extend it
              outward and cut it again.
            </p>
          )}

          {pns.length > 1 && (
            <div className="mt-4">
              <h3 className="eyebrow">Which side of the channel</h3>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {pns.map((sp, i) => (
                  <button key={`${sp.i0}-${sp.i1}`} className="pill whitespace-nowrap"
                          data-on={i === (pns[panelIdx] ? panelIdx : 0)}
                          onClick={() => setPanelIdx(i)}>
                    {dist(res.profile[sp.i0].m)} – {dist(res.profile[sp.i1].m)}
                  </button>
                ))}
              </div>
              <p className="anno mt-1.5 leading-relaxed">
                The channel splits this reach into {pns.length} floodplain panels. Each
                has its own stage and its own wall; water on the far side of the river
                is not what this one holds back.
              </p>
            </div>
          )}

          {/* ---- which body of water ----------------------------------- */}
          {rs.length > 1 && (
            <div className="mt-4">
              <h3 className="eyebrow">
                This line crosses {rs.length} separate bodies of water
              </h3>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
                A levee holds against one of them. Each has its own stage, so each
                has its own crest.
              </p>
              <ul className="mt-2 flex flex-col gap-1">
                {rs.map((r, i) => (
                  <li key={`${r.i0}-${r.i1}`}>
                    <button
                      onClick={() => setReachIdx(i)}
                      aria-pressed={i === reachIdx}
                      className={`flex w-full items-baseline gap-2 rounded-md px-2 py-1.5
                                  text-left text-[13px] ${
                        i === reachIdx ? "bg-ink text-bg" : "text-ink-mute hover:bg-bg-inset"}`}
                    >
                      <span className="flex-1 truncate">
                        {dist(r.m0)} – {dist(r.m1)}
                      </span>
                      <span className="tnum">{r.maxWaterM.toFixed(2)} m</span>
                      <span className={`tnum ${i === reachIdx ? "opacity-70" : "text-ink-faint"}`}>
                        {r.maxDepthM.toFixed(1)} deep
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              {naive && (
                <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
                  The highest water anywhere on this line is{" "}
                  <span className="tnum text-ink">{res.peak_water_m?.toFixed(2)} m</span>.
                  Taking that as the design stage for this reach would cost{" "}
                  <span className="tnum text-ink">{money(naive.costAud)}</span> against{" "}
                  <span className="tnum text-ink">{money(a.costAud)}</span>, and protect
                  nothing further — that water is{" "}
                  {dist(Math.abs((naive.overtopAtM ?? 0) - reach.m0))} away and{" "}
                  {(res.peak_water_m! - stage).toFixed(1)} m higher.
                </p>
              )}
            </div>
          )}

          {/* ---- provenance -------------------------------------------- */}
          <details className="fold mt-3 border-t border-line-soft">
            <summary>Where these numbers come from</summary>
            <div className="pb-2">
              <Row label="Line length" value={dist(res.length_m)} />
              <Row label="Solved at" value={`${res.dx_m.toFixed(0)} m cells`} />
              <Row label="Profile sampled every" value={`${spacing.toFixed(0)} m`} />
              <Row label="Wet along the line" value={`${(res.wet_fraction * 100).toFixed(0)}%`} />
              <Row label="Highest ground on the line"
                   value={`${res.highest_ground.ground_m.toFixed(1)} m`}
                   sub={`at ${dist(res.highest_ground.m)}`} />
              <p className="anno mt-2 leading-relaxed">{res.note}</p>
              {spacing < res.dx_m && (
                <p className="anno mt-1.5 leading-relaxed">
                  The profile is sampled finer than the model it is read from, so
                  neighbouring samples can land in the same cell and carry no new
                  information. Nearest cell, not interpolated: the elevation model is
                  already the limiting resolution.
                </p>
              )}
              <p className="anno mt-1.5 leading-relaxed">{res.driver_note}</p>
              <p className="anno mt-1.5 leading-relaxed">
                A section fixes a wall&apos;s height, not its plan. This one is read as
                the alignment: the wall follows the line, so its length is distance
                along the line and its height at each station is the crest above the
                ground there. The channel is found as the belt where the water is at
                least 2 m deeper than over the floodplain either side, and an alignment
                that crosses one is refused rather than priced.
              </p>
              <p className="anno mt-1.5 leading-relaxed">
                Cost is mean height × crest length × {rate.toFixed(0)} AUD per metre
                per metre — the same indicative rate the Plan tab&apos;s six surveyed
                sites are costed with, so the two figures are comparable.
                Order-of-magnitude only: real cost depends on foundation, land
                acquisition and access.
              </p>
            </div>
          </details>
        </>
      )}

      {res?.ok && (
        <details className="fold mt-3 border-t border-line-soft">
          <summary>The line and the storm</summary>
          <div className="pb-1">{inputs}</div>
        </details>
      )}
    </div>
  );
}
