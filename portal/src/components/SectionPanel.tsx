"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { STORM_PRESETS } from "@/lib/storms";
import { money, type SectionResult, type WallVerdict } from "@/lib/section";
import type { SitesFeed } from "./LeveeKit";

/**
 * Cut the floodplain open across a bank, and size the wall that defends it.
 *
 * Two changes from the first version of this view, both because a section is
 * read differently from the way it was being drawn.
 *
 * **The lines are offered, not drawn.** Asking the user to click two points
 * hands back the one question the data can answer: which bank matters is
 * something the terrain and the solved flood settle together, and `/sites`
 * already settles it for the levee kit. Each candidate bank now carries a
 * section ACROSS it, oriented so A is the landward end -- decided by counting
 * solved water on each side, because the dry side of a bank is not reliably
 * the higher one and on a levee it is deliberately not. Drawing a line by
 * hand stays, for when the interesting bank is not one of the six.
 *
 * **The wall is a point in the section, not a crest along it.** A section is
 * cut across a bank and the levee follows the bank, so the wall appears in
 * the drawing as one trapezoid and its length runs out of the plane. That
 * length is the bank's, which is why the cost here is crest height times the
 * BANK length and not times anything measurable on the chart.
 *
 * And the test it applies is the one a wall is bought for: not "does the
 * crest clear the water" but "does the landward side stay dry". Water that
 * reaches round behind a wall drowns the town just the same, and on a section
 * that shows up as wet ground on the A side standing above the crest.
 */

export interface SectionEnds {
  a: { lat: number; lon: number } | null;
  b: { lat: number; lon: number } | null;
}

export interface SectionOverlay {
  a: { lat: number; lon: number };
  b: { lat: number; lon: number };
  /** Where the wall sits, as a fraction along the line. */
  wallAt: number | null;
  verdict: string;
}

function Tile({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: "good" | "bad";
}) {
  return (
    <div className="rounded-lg border border-line bg-bg-raised px-3 py-2.5">
      <p className="anno">{label}</p>
      <p className={`tnum mt-0.5 text-[20px] font-semibold leading-none ${
        tone === "good" ? "text-ok" : tone === "bad" ? "text-bad" : "text-ink"}`}>
        {value}
      </p>
      {sub && <p className="anno mt-1">{sub}</p>}
    </div>
  );
}

export function SectionPanel({
  ends, picking, onPicking, onClearEnds, onOverlay,
  sites, sitesBusy, onFindSites, canFind, chosenSite, onChooseSite,
  result, busy, elapsed, err, onSolve,
  stationIdx, crestM, wall, onCrest, onSnap, homesKeptDry,
}: {
  ends: SectionEnds;
  picking: boolean;
  onPicking: (v: boolean) => void;
  onClearEnds: () => void;
  onOverlay: (o: SectionOverlay | null) => void;
  sites: SitesFeed | null;
  sitesBusy: boolean;
  onFindSites: () => void;
  canFind: boolean;
  chosenSite: number | null;
  onChooseSite: (i: number | null) => void;
  result: SectionResult | null;
  busy: boolean;
  elapsed: number;
  err: string | null;
  onSolve: (p: { driver: string; storm: number; q: number | null }) => void;
  stationIdx: number;
  crestM: number;
  wall: WallVerdict | null;
  onCrest: (m: number) => void;
  onSnap: () => void;
  homesKeptDry: number | null;
}) {
  const [driver, setDriver] = useState<"river" | "rain" | "coastal">("rain");
  const [storm, setStorm] = useState(1);
  const [qText, setQText] = useState("");
  const reported = useRef("");

  const bank = chosenSite !== null ? sites?.sites[chosenSite] ?? null : null;
  const bankLen = bank?.length_m ?? null;

  // What the map draws: the line, and where the wall sits on it. Keyed, so a
  // parent re-render does not report the same overlay over and over.
  useEffect(() => {
    if (!ends.a || !ends.b) {
      if (reported.current !== "") { reported.current = ""; onOverlay(null); }
      return;
    }
    const f = result?.ok && result.length_m > 0
      ? result.profile[Math.min(stationIdx, result.profile.length - 1)].m / result.length_m
      : null;
    const key = `${ends.a.lat},${ends.b.lat},${f},${wall?.verdict ?? ""}`;
    if (reported.current === key) return;
    reported.current = key;
    onOverlay({ a: ends.a, b: ends.b, wallAt: f, verdict: wall?.verdict ?? "" });
  }, [ends.a, ends.b, result, stationIdx, wall?.verdict, onOverlay]);

  const solve = useCallback(() => {
    onSolve({ driver, storm, q: Number(qText) > 0 ? Number(qText) : null });
  }, [driver, storm, qText, onSolve]);

  const freeboard = wall?.freeboardM ?? null;

  return (
    <div className="px-5 pb-5 pt-4">
      <h2 className="eyebrow">1 · Choose a line to protect</h2>
      {!sites && (
        <>
          <p className="mt-1 mb-2 text-[13px] leading-relaxed text-ink-mute">
            The banks worth defending come out of the terrain and the solved
            flood, with a section drawn across each.
          </p>
          <button
            onClick={onFindSites}
            disabled={sitesBusy || !canFind}
            className="w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium text-bg
                       transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {sitesBusy ? "Looking for banks that almost hold…"
              : canFind ? "Find the lines" : "Waiting for a solved place"}
          </button>
        </>
      )}
      {sites?.ok && (
        <ul className="mt-1.5 flex flex-col gap-1">
          {sites.sites.map((s, i) => (
            <li key={`${s.lat},${s.lon}`}>
              <button
                onClick={() => onChooseSite(chosenSite === i ? null : i)}
                aria-pressed={chosenSite === i}
                className={`flex w-full items-baseline gap-2 rounded-md px-2.5 py-2 text-left
                            text-[13px] ${chosenSite === i
                              ? "bg-ink text-bg" : "text-ink-mute hover:bg-bg-inset"}`}
              >
                <span className="flex-1 truncate">{s.name ?? "unnamed bank"}</span>
                <span className="tnum">{s.length_m} m</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <details className="fold mt-1">
        <summary>Or draw a line yourself</summary>
        <div className="pb-2">
          {!ends.a && (
            <p className="text-[13px] leading-relaxed text-ink-mute">
              {picking ? "Click the map where the section should start."
                : "Turn on picking, then click the map twice."}
            </p>
          )}
          {ends.a && !ends.b && (
            <p className="text-[13px] leading-relaxed text-ink-mute">
              Now click the far side of the floodplain.
            </p>
          )}
          {ends.a && ends.b && (
            <p className="anno leading-relaxed">
              A {ends.a.lat.toFixed(4)}, {ends.a.lon.toFixed(4)}<br />
              B {ends.b.lat.toFixed(4)}, {ends.b.lon.toFixed(4)}
            </p>
          )}
          <div className="mt-2 flex items-center gap-3">
            {!picking && (
              <button onClick={() => onPicking(true)}
                      className="rounded-md border border-line px-2.5 py-1 text-[13px]
                                 text-ink hover:bg-bg-inset">Pick on the map</button>
            )}
            {(ends.a || ends.b) && (
              <button onClick={() => { onClearEnds(); onChooseSite(null); onPicking(true); }}
                      className="text-[12px] text-ink-faint hover:text-ink">clear</button>
            )}
          </div>
        </div>
      </details>

      {ends.a && ends.b && !result && (
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
          <div className="mt-3 flex gap-1">
            {STORM_PRESETS.map((s, i) => (
              <button key={s.label} className="pill" data-on={storm === i}
                      onClick={() => setStorm(i)} title={s.note}>{s.label}</button>
            ))}
          </div>
          {driver === "river" && (
            <div className="mt-2.5">
              <label htmlFor="sec-q" className="eyebrow">Discharge</label>
              <div className="mt-1 flex items-center gap-2">
                <input id="sec-q" value={qText} inputMode="decimal"
                       onChange={(e) => setQText(e.target.value.replace(/[^0-9.]/g, ""))}
                       placeholder="estimate from the terrain"
                       className="w-36 rounded-md border border-line bg-bg-inset px-2.5 py-1.5
                                  text-[13px] text-ink placeholder:text-ink-faint
                                  focus:border-accent/60 focus:outline-none" />
                <span className="text-[13px] text-ink-mute">m³/s</span>
              </div>
            </div>
          )}
          <button
            onClick={solve}
            disabled={busy}
            className="mt-3 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                       text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {busy ? `Cutting the section… ${elapsed}s` : "Cut the section"}
          </button>
        </div>
      )}
      {err && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">Could not cut this section</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{err}</p>
        </div>
      )}

      {result?.ok && wall && (
        <>
          <h2 className="eyebrow mt-4">2 · Drag the levee in the section below</h2>

          <div className="mt-2 grid grid-cols-2 gap-2">
            <Tile label="Crest height" value={`${wall.heightM.toFixed(2)} m`}
                  sub="above the ground it stands on" />
            <Tile label="Cost"
                  value={bankLen ? money(wall.costAud) : "—"}
                  sub={bankLen ? `over ${bankLen} m of bank` : "needs a chosen bank"} />
            <Tile label="Above the solved peak"
                  value={freeboard === null ? "—"
                    : `${freeboard >= 0 ? "+" : ""}${freeboard.toFixed(2)} m`}
                  tone={freeboard === null ? undefined : freeboard >= 0 ? "good" : "bad"} />
            <Tile label="Homes kept dry"
                  value={homesKeptDry === null ? "—" : String(homesKeptDry)}
                  sub={homesKeptDry === null ? "not counted yet" : "on the landward side"} />
          </div>

          <div className="mt-3 rounded-lg border border-line bg-bg-raised p-3">
            <span className={wall.verdict === "holds" ? "chip-good" : "chip-harm"}>
              {wall.verdict === "holds" ? "Holds"
                : wall.verdict === "overtops" ? "Overtopped"
                  : wall.verdict === "wet-behind" ? "Water behind it"
                    : wall.verdict === "in-channel" ? "That is in the river"
                      : "Nothing to hold"}
            </span>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-mute">
              {wall.verdict === "holds" && freeboard !== null && (
                <>The landward side of the wall stays dry, with{" "}
                  <span className="tnum text-ink">{freeboard.toFixed(2)} m</span> to spare
                  over the water it faces.</>
              )}
              {wall.verdict === "overtops" && freeboard !== null && (
                <>The water it faces reaches{" "}
                  <span className="tnum text-ink">{wall.faceStageM?.toFixed(2)} m</span>,{" "}
                  {Math.abs(freeboard).toFixed(2)} m over this crest. Drag the handle up,
                  or take the crest that holds.</>
              )}
              {wall.verdict === "wet-behind" && (
                <>The crest clears the water it faces, and there is already water{" "}
                  <span className="tnum text-ink">
                    {((wall.behindStageM ?? 0) - wall.groundM).toFixed(2)} m
                  </span>{" "}
                  deep behind it. A wall does not drain what is already inside, and on
                  this line the flood reaches the landward side by another route.</>
              )}
              {wall.verdict === "in-channel" && (
                <>This station sits in the channel. The section cannot tell a bank
                  from a riverbed by elevation — the bed is simply the lowest ground
                  on the line, which is also where a crest looks cheapest — so it is
                  found from the depth instead. A wall here is a weir. Drag it onto
                  the bank.</>
              )}
              {wall.verdict === "dry" && (
                <>There is no water on the river side of this station, so there is
                  nothing here for a wall to hold back.</>
              )}
            </p>
            <button onClick={onSnap}
                    className="mt-2 w-full rounded-md border border-line px-3 py-1.5
                               text-[13px] text-ink hover:bg-bg-inset">
              Put it at the cheapest crest that holds
            </button>
          </div>

          <div className="mt-3">
            <label htmlFor="sec-crest" className="flex items-baseline justify-between">
              <span className="eyebrow">Crest elevation</span>
              <span className="tnum text-[13px] text-ink">{crestM.toFixed(2)} m</span>
            </label>
            <input id="sec-crest" type="range"
                   min={Math.floor(Math.min(...result.profile.map((q) => q.ground_m)))}
                   max={Math.ceil((result.peak_water_m ?? crestM) + 4)}
                   step={0.05} value={crestM}
                   onChange={(e) => onCrest(Number(e.target.value))}
                   className="mt-1 w-full accent-ink" />
            <p className="anno leading-relaxed">
              Or drag the handle on the section. Sideways moves the wall along the
              line; upwards raises the crest.
            </p>
          </div>

          <details className="fold mt-3 border-t border-line-soft">
            <summary>What this costing assumes</summary>
            <p className="anno pb-2 leading-relaxed">
              Cost is crest height above the ground it stands on, times the
              BANK&apos;s length, times A$2,600 per metre of crest per metre of
              height. The section is cut across the bank, so the wall&apos;s length
              runs out of the plane of the drawing and is not measurable on it.
            </p>
            <p className="anno pb-2 leading-relaxed">{result.note}</p>
            <p className="anno pb-2 leading-relaxed">{result.driver_note}</p>
          </details>
        </>
      )}
    </div>
  );
}
