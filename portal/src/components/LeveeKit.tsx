"use client";

import { useCallback, useMemo } from "react";
import { money } from "@/lib/section";
import { EARTH_FILL, T_WALL, GATED_OUTFALL, PUMP_STATION, ALL_IN,
         CREST_WIDTH_M, SIDE_SLOPE, earthLeveeUsd, wallUsd } from "@/lib/rates";

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
  /** A section ACROSS this bank, A at the landward end. */
  line: {
    a: { lat: number; lon: number };
    b: { lat: number; lon: number };
    length_m: number;
  };
}

export interface SitesFeed {
  ok: boolean;
  reason?: string;
  sites: Site[];
  dx_m: number;
  shortfall_m: number;
  note: string;
  driver_note: string;
  /** The flood these sites were found in, so the map can draw the scenario
   *  the panel names rather than the study's own design storm. */
  bounds?: { west: number; south: number; east: number; north: number };
  depth_png?: string;
  flooded_km2?: number;
  peak_depth_m?: number;
  river_q_m3s?: number | null;
}

/**
 * The parts, and what each of them IS to the solver.
 *
 * Five pieces, and three different things. A levee and a wall are bed:
 * `site_masks` sweeps their section along the site's own alignment and the
 * flood runs over the raised ground. A floodgate is a one-way exchange
 * between the cell behind the bank and the cell in front of it — `max(dH, 0)`
 * is the flap, and it seals with no branch. A pump lifts water that is
 * already inside the perimeter, which is the case no levee placement can
 * answer: in the pluvial result a quarter of the rain fell inside the area
 * being protected, and nothing that only keeps water out could touch it.
 *
 * None of the three is a graphic. Each one is a term in `pspe.simulate.flood`
 * and each is solved for every candidate plan in the same batched timestep
 * loop, which is why a gate and a pump can carry a conformal margin on the
 * same footing a levee does.
 *
 * `intensity` is what the planner enumerates over — metres of crest for the
 * two bed pieces, barrels for the gate, cubic metres a second for the pump —
 * and it is also what each is priced by, so the number solved and the number
 * costed cannot drift apart.
 */
export const PIECES = [
  { id: "earth1", kind: "levee", cost_model: "earth", label: "Earth levee",
    intensity: 1.0, note: "1.0 m high · 1V:3H batters",
    unit: EARTH_FILL.usd, rate: "/m³ fill", swatch: "#b08968" },
  { id: "earth2", kind: "levee", cost_model: "earth", label: "Raised levee",
    intensity: 2.0, note: "2.0 m high · 1V:3H batters",
    unit: EARTH_FILL.usd, rate: "/m³ fill", swatch: "#8a5529" },
  { id: "wall", kind: "levee", cost_model: "wall", label: "Flood wall",
    intensity: 1.5, note: "1.5 m · concrete T-wall",
    unit: T_WALL.usd, rate: "/m/m", swatch: "#9ca3af" },
  { id: "gate", kind: "gate", cost_model: "gate", label: "Floodgate",
    intensity: 1.0, note: "one 1.22 m barrel · drains one way",
    unit: GATED_OUTFALL.usd, rate: "/outfall", swatch: "#3f6f9c" },
  { id: "pump", kind: "pump", cost_model: "pump", label: "Pump station",
    intensity: 1.0, note: "1 m³/s firm · lifts water from inside",
    unit: PUMP_STATION.usd, rate: "/m³/s", swatch: "#2f7d52" },
] as const;

export type PieceId = (typeof PIECES)[number]["id"];
export type Piece = (typeof PIECES)[number];

export interface Placed {
  site: number;
  piece: PieceId;
}

export function pieceOf(id: PieceId): Piece | undefined {
  return PIECES.find((x) => x.id === id);
}

/**
 * What one placed piece costs, from its own cross-section.
 *
 * An embankment is NOT a flat rate per metre of height. Its volume per metre
 * of crest is `crest·H + slope·H²`, so a 2 m levee is 2.0 times a 1 m one and
 * not 2.0 — it is 2.97 times. The portal quoted a flat A$2,600/m/m, which
 * understated every tall wall it ever priced.
 */
export function costOf(p: Placed, sites: Site[]): number {
  const piece = pieceOf(p.piece);
  const s = sites[p.site];
  if (!piece || !s) return 0;
  if (piece.cost_model === "earth") return earthLeveeUsd(s.length_m, piece.intensity);
  if (piece.cost_model === "wall") return wallUsd(s.length_m, piece.intensity);
  return piece.intensity * piece.unit;        // a gate or a pump, per unit
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
  armed, onArm, impact, planBusy, onPlan, noWayOut, onSave,
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
    protectBasis: string | null;
    protectKm2: number | null;
    dxM: number | null;
    sitesDxM: number | null;
  } | null;
  planBusy: boolean;
  onPlan: () => void;
  /** True when every named road out is cut under this scenario. */
  noWayOut: boolean;
  onSave: () => void;
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

      {/* The tab finds its own sites when it opens, so this is a progress
          report rather than a call to action. A button here was a turnstile in
          front of the one thing the tab exists to show, and the sites are not
          a choice anyone makes: they come out of the terrain. */}
      {!feed && (
        <>
          <p className="mb-3 text-[13px] leading-relaxed text-ink-mute">
            A levee pays where the bank is almost high enough. Nothing is
            clicked here — the places worth raising come out of the terrain and
            the solved flood.
          </p>
          {busy ? (
            <div className="rounded-lg border border-line bg-bg-inset p-3">
              <p className="text-[13px] font-medium text-ink">
                Looking for banks that almost hold
              </p>
              <p className="anno mt-1 leading-relaxed">
                Solving the baseline at 40 m to get the water surface, then
                tracing the dry ground against it · {elapsed}s
              </p>
              <div className="mt-2 h-[3px] overflow-hidden rounded-full bg-line-soft">
                <span className="block h-full w-1/3 animate-pulse rounded-full bg-ink" />
              </div>
            </div>
          ) : !err && (
            // No button until the flood is settled.
            //
            // The scenario list is a separate fetch, so for a second or two
            // the panel knows the place but not which flood it is judging
            // against. A "Find the sites" button in that window searches the
            // DEFAULT scenario -- the design rainstorm -- and then the
            // heading flips to a recorded flood over somebody else's answer.
            // That is the bug the scenario control was added to fix, and a
            // button is all it takes to reintroduce it by hand.
            <div className="rounded-lg border border-line bg-bg-inset p-3">
              <p className="text-[13px] font-medium text-ink">
                {canFind ? "Reading this creek's flood record"
                  : "Waiting for a solved place"}
              </p>
              <p className="anno mt-1 leading-relaxed">
                {canFind
                  ? "A levee answers a question about a named flood, so the "
                    + "gauge record comes first and the search follows it."
                  : "Nothing to plan against until a place is solved."}
              </p>
            </div>
          )}
        </>
      )}
      {err && (
        <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
          <p className="text-[13px] font-medium text-ink">No sites here</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-mute">{err}</p>
          <button onClick={onFindSites} disabled={busy || !canFind}
                  className="mt-2 rounded-md border border-line px-3 py-1.5 text-[13px]
                             text-ink hover:bg-bg-inset disabled:opacity-40">
            Look again
          </button>
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
            budget US${budgetM} M
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
                {armed ? "Now click a dashed site on the map" : "Place on the map"}
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
                  {/* A bar for the two pieces that are a line of bed, a dot
                      for the two that are a point structure. The shape is
                      the difference, not decoration. */}
                  {p.kind === "levee" ? (
                    <span className="block h-[6px] w-full rounded-[2px]"
                          style={{ background: p.swatch }} />
                  ) : (
                    <span className="block h-[10px] w-[10px] rounded-full"
                          style={{ background: p.swatch }} />
                  )}
                  <span className="mt-2 block text-[13px] font-medium text-ink">{p.label}</span>
                  <span className="anno block leading-snug">{p.note}</span>
                  <span className="anno block">
                    {p.unit >= 1000
                      ? `${money(p.unit)}${p.rate}`
                      : `US$${p.unit.toLocaleString()}${p.rate}`}
                  </span>
                  {/* What this site would actually cost, if one is armed and
                      the reader can see the length it would run. */}
                </button>
              ))}
            </div>
            <p className="anno mt-1.5 leading-relaxed">
              An embankment is priced from its own section — crest{" "}
              {CREST_WIDTH_M.toFixed(2)} m, batters 1V:{SIDE_SLOPE}H — so a 2 m
              levee is 2.97 times a 1 m one, not twice. A wall is linear in
              height. A gate and a pump are priced per structure.
            </p>
            <p className="anno mt-1 leading-relaxed">
              USACE construction rates, escalated to 2025 and × 0.88 for Texas.
              Construction only: design, land and contingency roughly double
              the bill ({ALL_IN.toFixed(1)}×).
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
                const piece = on && pieceOf(on.piece);
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
            <h3 className="eyebrow">What it buys</h3>
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
                {impact.dxM !== null && impact.sitesDxM !== null
                  && Math.abs(impact.dxM - impact.sitesDxM) > 1 && (
                  <p className="anno mt-1.5 leading-relaxed">
                    Solved at {impact.dxM.toFixed(0)} m, where the search for
                    banks ran at {impact.sitesDxM.toFixed(0)} m — one bed per
                    candidate combination against one for the search. These
                    areas and the one under the map are two grids of the same
                    flood and will not agree to the decimal; the before and
                    after here are both from this grid, so the CHANGE is
                    comparable even where the totals are not.
                  </p>
                )}
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
                  {impact.protectBasis && (
                    <p className="anno mt-1.5 leading-relaxed">
                      Per cent of mean depth over {impact.protectBasis}
                      {impact.protectKm2 !== null
                        ? ` — ${impact.protectKm2.toFixed(1)} km²` : ""}. Not
                      over the whole box: a wall that halves the water behind
                      it moves a box-wide average by almost nothing.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Every road out is cut.
              Not a warning about the plan -- a fact about the flood it is
              being judged against, and the one that decides whether a levee
              is the right instrument at all. A town nobody can leave needs
              a route, not a wall. */}
          {noWayOut && (
            <div className="mt-3 flex items-baseline gap-2">
              <span className="shrink-0 rounded-[4px] bg-bad px-1.5 py-0.5 text-[11px]
                               font-semibold text-bg">No way out</span>
              <span className="text-[12px] leading-snug text-ink-mute">
                Every named road out of here is cut for cars under this flood.
              </span>
            </div>
          )}

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
            {/* Save writes the plan to a file rather than to a server. There
                is no account here and no database, and a Save that quietly
                dropped the plan into browser storage would lose it on the
                next machine. A JSON file carries the scenario, the measures,
                the costs and the solved result, which is what anyone would
                need to reproduce it. */}
            <button onClick={onSave} disabled={!placed.length}
                    className="flex-[1.4] rounded-md bg-ink px-3 py-2 text-[13px]
                               font-medium text-bg hover:opacity-90 disabled:opacity-40">
              Save plan
            </button>
          </div>
        </>
      )}
    </div>
  );
}
