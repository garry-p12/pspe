"use client";

import { useMemo, useState } from "react";
import { eventLabel, km, type OpsPayload } from "@/lib/ops";
import { OptionList } from "./OptionList";

/**
 * What to build, and what doing nothing costs.
 *
 * The section opens with the baseline rather than with the options, because
 * every number below is a change against it and a percentage with no
 * denominator on screen is a number someone has to take on trust.
 *
 * Options are then one line each: what it is, what it costs, what it moves.
 * The previous version gave each option a card with its own prose, which made
 * twelve options into a page of reading when the decision is a comparison.
 */

interface Option {
  id: string;
  name: string;
  heights: number[];
  cost_aud: number;
  road_cut_km: number;
  area_flooded_km2: number;
  core_reduction_pct?: number;
  named_roads_cut?: { name: string; km: number }[];
}

/** "Levee at site 3" -> title "Levee · site 3", detail "Single levee". */
function describe(o: Option): { title: string; detail: string } {
  const active = o.heights
    .map((h, i) => ({ h, i }))
    .filter((x) => x.h > 0);
  if (active.length === 0) return { title: "Do nothing", detail: "No works" };
  if (active.length === 1) {
    return { title: `Levee · site ${active[0].i}`, detail: "Single levee" };
  }
  return {
    title: `Sites ${active.map((a) => a.i).join(" + ")}`,
    detail: active.map((a) => `${a.h.toFixed(1)} m`).join(", "),
  };
}

/** Short road name: "Tatham Road" -> "Tatham Rd". */
const shortRoad = (n: string) => n.replace(/\bRoad\b/, "Rd").replace(/\bStreet\b/, "St");

export function PlanSection({
  ops, budgetM, onBudget, scale, onScale, chosen, onChoose,
  onBuildYourOwn, onFindPlan, planning, children,
}: {
  /** The planner's own result, rendered inside the scroll area. */
  children?: React.ReactNode;
  ops: OpsPayload;
  budgetM: number;
  onBudget: (m: number) => void;
  scale: number;
  onScale: (s: number) => void;
  chosen: string;
  onChoose: (id: string) => void;
  onBuildYourOwn: () => void;
  onFindPlan: () => void;
  planning: boolean;
}) {
  const [filter, setFilter] = useState<"all" | "helps" | "worsens">("all");
  const [expanded, setExpanded] = useState(false);

  const forScale = useMemo(
    () => (ops.options as Option[]).filter((o) => (o as { event_scale?: number }).event_scale === scale),
    [ops, scale],
  );
  const base = forScale.find((o) => o.id === "base");
  const measures = forScale.filter((o) => o.id !== "base");
  const affordable = measures.filter((o) => o.cost_aud <= budgetM * 1e6);

  const filtered = useMemo(() => {
    const rows = affordable.filter((o) => {
      const d = o.core_reduction_pct ?? 0;
      if (filter === "helps") return d > 0.5;
      if (filter === "worsens") return d < -0.5;
      return true;
    });
    // Best first. A list someone scans top-down should put the answer at the
    // top, not leave it to be found.
    return [...rows].sort(
      (a, b) => (b.core_reduction_pct ?? 0) - (a.core_reduction_pct ?? 0),
    );
  }, [affordable, filter]);

  const best = filtered.find((o) => (o.core_reduction_pct ?? 0) > 0.5);
  const shown = expanded ? filtered : filtered.slice(0, 5);
  const cutFrac = base ? base.road_cut_km / ops.road_network_km : 0;
  const worst = (base?.named_roads_cut ?? []).slice(0, 3);
  const moreRoads = Math.max(0, (base?.named_roads_cut?.length ?? 0) - 3);

  return (
    // No scroll container of its own: the panel around it already scrolls, and
    // nesting a second one collapsed this one's height and clipped everything
    // below the budget. The action sticks to the bottom of that scroller
    // instead, which is what kept it reachable in the first place.
    <div>
      <div className="px-5 pb-5 pt-4">
        {/* ---- the baseline every number below is measured against ------- */}
        {base && (
          <div className="rounded-xl border border-line px-4 py-3.5">
            <div className="flex items-baseline justify-between">
              <label className="flex items-baseline gap-1">
                <select
                  id="design-event"
                  value={scale}
                  onChange={(e) => onScale(Number(e.target.value))}
                  className="cursor-pointer appearance-none bg-transparent text-[13px]
                             font-medium text-ink outline-none"
                >
                  {(ops.event_scales ?? [1]).map((s) => (
                    <option key={s} value={s}>{eventLabel(s)}</option>
                  ))}
                </select>
                <span aria-hidden className="text-[10px] text-ink-faint">▾</span>
              </label>
              <span className="text-[13px] text-ink-mute">Do nothing</span>
            </div>

            <div className="mt-2.5 flex gap-7">
              <div>
                <p className="tnum text-[26px] font-semibold leading-none text-ink">
                  {km(base.road_cut_km)}
                </p>
                <p className="anno mt-1.5">road cut of {km(ops.road_network_km)}</p>
              </div>
              <div>
                <p className="tnum text-[26px] font-semibold leading-none text-ink">
                  {base.area_flooded_km2.toFixed(0)} km²
                </p>
                <p className="anno mt-1.5">land flooded</p>
              </div>
            </div>

            {/* Proportion as counted blocks: a share of a network is a count of
                segments, and blocks are read as a count where a bar is read as
                a magnitude. */}
            <div className="mt-3 flex gap-[3px]" aria-hidden>
              {Array.from({ length: 28 }, (_, i) => (
                <span
                  key={i}
                  className="h-2 flex-1 rounded-[1px]"
                  style={{
                    background: i < Math.round(cutFrac * 28)
                      ? "var(--ink)" : "var(--bg-inset)",
                  }}
                />
              ))}
            </div>
            <p className="sr-only">
              {(cutFrac * 100).toFixed(0)}% of the mapped network is cut.
            </p>

            {worst.length > 0 && (
              <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">
                Worst:{" "}
                {worst.map((r, i) => (
                  <span key={r.name}>
                    {i > 0 && ", "}
                    {shortRoad(r.name)} {r.km.toFixed(1)} km
                  </span>
                ))}
                {moreRoads > 0 && (
                  <span className="ml-1 underline decoration-dotted">+{moreRoads}</span>
                )}
              </p>
            )}
          </div>
        )}

        {/* ---- budget ---------------------------------------------------- */}
        <div className="mt-5">
          <div className="flex items-baseline justify-between">
            <label htmlFor="budget" className="eyebrow">Budget</label>
            <span className="tnum text-[15px] font-semibold text-ink">A${budgetM}M</span>
          </div>
          <input
            id="budget"
            type="range" min={2} max={60} step={1} value={budgetM}
            onChange={(e) => onBudget(Number(e.target.value))}
            className="mt-2 w-full accent-[var(--ink)]"
          />
        </div>

        <OptionList
          options={measures}
          budgetM={budgetM}
          chosen={chosen}
          onChoose={onChoose}
        />

        <div className="mt-1 flex justify-end">
          <button
            onClick={onBuildYourOwn}
            className="text-[13px] text-ink underline underline-offset-2 hover:no-underline"
          >
            Build your own
          </button>
        </div>

        {children}

        <p className="anno mt-4 leading-relaxed">
          Appraised against the design event; road cut at{" "}
          {(ops.cut_depth_m * 100).toFixed(0)} cm. Costs indicative at A$2,600 per
          metre of crest per metre of height.
        </p>
      </div>

      {/* ---- the action, always reachable ------------------------------- */}
      <div className="sticky bottom-0 z-10 border-t border-line bg-bg px-5 py-3.5">
        <button
          onClick={onFindPlan}
          disabled={planning}
          className="w-full rounded-lg bg-ink px-4 py-3 text-[14px] font-medium
                     text-bg transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {planning ? "Searching…" : `Find the best plan for A$${budgetM}M`}
        </button>
      </div>
    </div>
  );
}
