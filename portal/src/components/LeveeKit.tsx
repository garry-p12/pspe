"use client";

import { useCallback, useMemo } from "react";
import { money } from "@/lib/section";

/**
 * Build a plan out of parts, and watch the budget and the damage move.
 *
 * The district tab offers a library someone solved offline and the Anywhere
 * tab asks the user to click candidate sites on a map, which puts the one
 * question the terrain can answer back onto the planner. This asks instead:
 * here are the places a wall would pay, here are the parts you can buy, you
 * have this much money.
 *
 * **Where the sites come from.** Not clicked and not surveyed by hand: a levee
 * pays where the bank is ALMOST high enough. Ground four metres under water is
 * not worth raising and ground already above the flood does not need it. The
 * service solves the baseline, finds the dry ground touching the flood that
 * sits within a couple of metres of being overtopped, and returns the
 * contiguous runs of it — each carrying the height its LOWEST point would
 * need, because that is the height the whole wall has to be.
 *
 * **What the impact row is not.** The numbers on the right of each row come
 * from the framework's surrogate, which predicts what a combination of
 * heights does from the single-site effects. That is a prediction, and the
 * conformal margin beside it is how wrong it is allowed to be — not
 * decoration, and not a confidence interval someone typed in. Where the
 * margin cannot be calibrated, because fewer than four sites are in play, the
 * panel says so instead of showing a number it cannot stand behind.
 */

export interface Site {
  lat: number;
  lon: number;
  name: string | null;
  length_m: number;
  extent_m: number;
  needs_m: number;
  width_m: number;
  cells: number;
  outline: [number, number][];
}

export interface SitesFeed {
  ok: boolean;
  reason?: string;
  sites: Site[];
  dx_m: number;
  shortfall_m: number;
  note: string;
  driver_note: string;
}

/** The parts, and what each costs per metre of crest per metre of height. */
export const PIECES = [
  { id: "earth1", label: "Earth levee", h: 1.0, note: "1.0 m high", unit: 2600 },
  { id: "earth2", label: "Raised levee", h: 2.0, note: "2.0 m high", unit: 2600 },
  { id: "wall", label: "Flood wall", h: 1.5, note: "1.5 m · concrete", unit: 8600 },
] as const;

export type PieceId = (typeof PIECES)[number]["id"];

export interface Placed {
  site: number;
  piece: PieceId;
}

export function costOf(p: Placed, sites: Site[]): number {
  const piece = PIECES.find((x) => x.id === p.piece);
  const s = sites[p.site];
  if (!piece || !s) return 0;
  return piece.h * s.length_m * piece.unit;
}

function Row({
  label, before, after, unit, better,
}: {
  label: string; before: number | null; after: number | null;
  unit: string; better: "down" | "up";
}) {
  const known = before !== null && after !== null;
  const d = known ? after - before : 0;
  // Enough decimals to SHOW the change. Two small walls on a 130 km2 box move
  // the flooded area by a fraction of a square kilometre, and rounding to whole
  // numbers printed "21 km2 -> 21 km2, no change" over a solve that had in fact
  // changed something. A row whose precision hides its own result is worse than
  // no row.
  const dp = known && Math.abs(d) > 0 && Math.abs(d) < 0.5 ? 2
    : (before ?? 0) < 10 ? 1 : 0;
  const good = better === "down" ? d < -1e-9 : d > 1e-9;
  const bad = better === "down" ? d > 1e-9 : d < -1e-9;
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-line-soft py-1.5">
      <span className="text-[13px] text-ink-mute">{label}</span>
      <span className="flex items-baseline gap-2">
        <span className="anno">
          {before === null ? "—" : `${before.toFixed(dp)}${unit}`}
        </span>
        <span className="anno">→</span>
        <span className="tnum text-[14px] font-semibold text-ink">
          {after === null ? "—" : `${after.toFixed(dp)}${unit}`}
        </span>
        <span className={`tnum w-[52px] text-right text-[12px] ${
          good ? "text-ink" : bad ? "text-bad" : "text-ink-faint"}`}>
          {!known ? "–"
            : Math.abs(d) < 1e-9 ? "no change"
              : `${d > 0 ? "+" : ""}${d.toFixed(dp)}`}
        </span>
      </span>
    </div>
  );
}

export function LeveeKit({
  feed, busy, elapsed, err, onFindSites, canFind,
  placed, onPlaced, budgetM, onBudget,
  armed, onArm, impact, planBusy, onPlan, showPlan, onShowPlan,
}: {
  feed: SitesFeed | null;
  busy: boolean;
  elapsed: number;
  err: string | null;
  onFindSites: () => void;
  /** False until a place is solved. Without this the button looked ready and
   *  did nothing: `findSites` returned early on a null box and left the panel
   *  sitting on its own call to action. */
  canFind: boolean;
  placed: Placed[];
  onPlaced: (p: Placed[]) => void;
  budgetM: number;
  onBudget: (m: number) => void;
  /** The piece waiting to be dropped, if any. */
  armed: PieceId | null;
  onArm: (p: PieceId | null) => void;
  impact: {
    base: { area_km2: number; cut_km: number | null; homes: number | null };
    with: { area_km2: number; cut_km: number | null; homes: number | null };
    band_pct: number | null;
    reduction_pct: number;
    band_reason?: string;
  } | null;
  planBusy: boolean;
  onPlan: () => void;
  showPlan: boolean;
  onShowPlan: (v: boolean) => void;
}) {
  const sites = useMemo(() => feed?.sites ?? [], [feed]);
  const spent = useMemo(
    () => placed.reduce((s, p) => s + costOf(p, sites), 0),
    [placed, sites],
  );
  const budget = budgetM * 1e6;
  const left = budget - spent;

  const place = useCallback((siteIdx: number) => {
    if (!armed) return;
    onPlaced([...placed.filter((p) => p.site !== siteIdx), { site: siteIdx, piece: armed }]);
    onArm(null);
  }, [armed, placed, onPlaced, onArm]);

  return (
    <div className="px-5 pb-5 pt-4">
      <h2 className="eyebrow mb-2">Build a plan</h2>

      {!feed && (
        <>
          <p className="mb-3 text-[13px] leading-relaxed text-ink-mute">
            A levee pays where the bank is almost high enough. Nothing is
            clicked here — the places worth raising come out of the terrain and
            the solved flood.
          </p>
          <button
            onClick={onFindSites}
            disabled={busy || !canFind}
            className="w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium text-bg
                       transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {busy ? `Looking for banks that almost hold… ${elapsed}s`
              : canFind ? "Find the sites"
                : "Waiting for a solved place"}
          </button>
        </>
      )}
      {err && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">No sites here</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{err}</p>
        </div>
      )}

      {feed?.ok && (
        <>
          {/* ---- the money ------------------------------------------- */}
          <div className="flex items-baseline justify-between">
            <p className="tnum text-[24px] font-semibold leading-none text-ink">
              {money(Math.max(left, 0))} left
            </p>
            <p className="anno">{placed.length} piece{placed.length === 1 ? "" : "s"}</p>
          </div>
          <p className="anno mt-1">spent {money(spent)} of {money(budget)}</p>
          <div className="mt-2 flex h-[10px] gap-[2px] overflow-hidden rounded-[3px]">
            {Array.from({ length: 24 }, (_, i) => (
              <span key={i} className={`flex-1 ${
                i < Math.round((Math.min(spent, budget) / budget) * 24)
                  ? "bg-ink" : "bg-bg-inset"}`} />
            ))}
          </div>
          {left < 0 && (
            <p className="mt-1.5 text-[13px] font-medium text-bad">
              Over budget by {money(-left)}.
            </p>
          )}
          <label htmlFor="kit-budget" className="anno mt-2 block">
            budget A${budgetM} M
          </label>
          <input
            id="kit-budget" type="range" min={2} max={80} step={1} value={budgetM}
            onChange={(e) => onBudget(Number(e.target.value))}
            className="w-full accent-ink"
          />

          {/* ---- the tray --------------------------------------------- */}
          <div className="mt-4">
            <div className="flex items-baseline justify-between gap-2">
              <h3 className="eyebrow">
                {armed ? "Now click a dashed site on the map" : "Pick a piece"}
              </h3>
              {armed && (
                <button onClick={() => onArm(null)}
                        className="text-[12px] text-ink-faint hover:text-ink">cancel</button>
              )}
            </div>
            <div className="mt-1.5 grid grid-cols-2 gap-2">
              {PIECES.map((p) => (
                <button
                  key={p.id}
                  onClick={() => onArm(armed === p.id ? null : p.id)}
                  aria-pressed={armed === p.id}
                  className={`rounded-lg border p-2.5 text-left transition-colors ${
                    armed === p.id ? "border-ink bg-bg-inset" : "border-line hover:bg-bg-inset"}`}
                >
                  <span className="block h-[6px] w-full rounded-[2px]"
                        style={{ background: p.id === "wall" ? "#9ca3af" : "#b08968",
                                 opacity: p.id === "earth1" ? 0.75 : 1 }} />
                  <span className="mt-2 block text-[13px] font-medium text-ink">{p.label}</span>
                  <span className="anno block">{p.note}</span>
                  <span className="anno block">A${p.unit.toLocaleString()}/m/m</span>
                </button>
              ))}
            </div>
            <p className="anno mt-1.5 leading-relaxed">
              Cost is the piece&apos;s height times the site&apos;s own length times
              its unit rate. Concrete is dearer than earth and takes less room.
            </p>
          </div>

          {/* ---- the sites -------------------------------------------- */}
          <div className="mt-4">
            <h3 className="eyebrow">
              {sites.length} place{sites.length === 1 ? "" : "s"} a wall would pay
            </h3>
            <ul className="mt-1.5 flex flex-col gap-1">
              {sites.map((s, i) => {
                const on = placed.find((p) => p.site === i);
                const piece = on && PIECES.find((x) => x.id === on.piece);
                return (
                  <li key={`${s.lat},${s.lon}`}>
                    <button
                      onClick={() => (on
                        ? onPlaced(placed.filter((p) => p.site !== i))
                        : place(i))}
                      aria-pressed={Boolean(on)}
                      className={`flex w-full items-baseline gap-2 rounded-md px-2 py-1.5
                                  text-left text-[13px] ${on
                                    ? "bg-ink text-bg" : "text-ink-mute hover:bg-bg-inset"}`}
                    >
                      <span className="flex-1 truncate">
                        {s.name ?? "unnamed bank"}
                      </span>
                      <span className="tnum">{(s.length_m / 1000).toFixed(2)} km</span>
                      <span className={`tnum ${on ? "opacity-75" : "text-ink-faint"}`}>
                        {piece ? money(costOf(on!, sites)) : `needs ${s.needs_m.toFixed(2)} m`}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
            <p className="anno mt-1.5 leading-relaxed">{feed.note}</p>
          </div>

          {/* ---- what it buys ----------------------------------------- */}
          <div className="mt-4">
            <div className="flex items-baseline justify-between gap-2">
              <h3 className="eyebrow">What it buys</h3>
              {impact && (
                <div className="flex gap-1">
                  {([false, true] as const).map((v) => (
                    <button key={String(v)} className="pill" data-on={showPlan === v}
                            onClick={() => onShowPlan(v)}>
                      {v ? "With plan" : "Before"}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {!impact && (
              <button
                onClick={onPlan}
                disabled={planBusy || placed.length < 2}
                className="mt-2 w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                           text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
              >
                {planBusy ? "Solving every combination…"
                  : placed.length < 2 ? "Place at least two pieces"
                    : "Solve this plan"}
              </button>
            )}
            {impact && (
              <div className="mt-2">
                <Row label="Land flooded" before={impact.base.area_km2}
                     after={impact.with.area_km2} unit=" km²" better="down" />
                <Row label="Homes over floor" before={impact.base.homes}
                     after={impact.with.homes} unit="" better="down" />
                <Row label="Road cut" before={impact.base.cut_km}
                     after={impact.with.cut_km} unit=" km" better="down" />
                {(impact.with.homes === null || impact.with.cut_km === null) && (
                  <p className="anno mt-1.5 leading-relaxed">
                    Homes and road cut are not counted for a plan yet. Both need
                    the planned field re-scored against the buildings and the
                    road network, which is another pass over every candidate —
                    the dash is &quot;not computed&quot;, not &quot;unchanged&quot;.
                  </p>
                )}
                <div className="mt-2.5 rounded-lg border border-line bg-bg-raised p-3">
                  <p className="eyebrow">
                    {impact.band_pct === null
                      ? "No guarantee on this one"
                      : `At least ${Math.max(impact.reduction_pct - impact.band_pct, 0).toFixed(1)}% less flooding`}
                  </p>
                  <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">
                    {impact.band_pct === null
                      ? <>{placed.length < 4
                            ? <>Place {4 - placed.length} more piece
                                {4 - placed.length === 1 ? "" : "s"} and the margin can be
                                calibrated. Split conformal needs nine combinations of two
                                or more sites before any finite order statistic carries 90%,
                                and {placed.length} site{placed.length === 1 ? "" : "s"} yields{" "}
                                {placed.length < 2 ? 0 : placed.length === 2 ? 2 : 7}. Four
                                yields seventeen.</>
                            : (impact.band_reason ?? "The margin could not be calibrated.")}</>
                      : <>The surrogate predicts {impact.reduction_pct.toFixed(1)}% and the
                          conformal margin is ±{impact.band_pct.toFixed(1)} points, so the
                          floor is what is promised. The margin is calibrated on the
                          solved combinations, not assumed.</>}
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="mt-3 flex gap-2">
            <button onClick={() => onPlaced(placed.slice(0, -1))}
                    disabled={!placed.length}
                    className="flex-1 rounded-md border border-line px-3 py-2 text-[13px]
                               text-ink hover:bg-bg-inset disabled:opacity-40">
              Undo
            </button>
            <button onClick={() => onPlaced([])}
                    disabled={!placed.length}
                    className="flex-1 rounded-md border border-line px-3 py-2 text-[13px]
                               text-ink hover:bg-bg-inset disabled:opacity-40">
              Clear
            </button>
          </div>
        </>
      )}
    </div>
  );
}
