"use client";

import { useMemo, useState } from "react";
import { money } from "@/lib/ops";

/**
 * A list of mitigation options, filtered and ranked.
 *
 * Shared by the district and by an arbitrary location deliberately. The two
 * used to render their results with different components and they drifted:
 * the district got a ranked, filterable list and Anywhere got one paragraph
 * of the winner, even though both had solved the same number of scenarios.
 * One component means a place the tool has never seen is presented exactly as
 * well as the one it was built around.
 */

export interface OptionRow {
  id: string;
  heights: number[];
  cost_aud: number;
  core_reduction_pct?: number;
}

/** "site 3 at 3 m" -> title "Levee · site 3", detail "Single levee". */
export function describe(o: OptionRow): { title: string; detail: string } {
  const active = o.heights.map((h, i) => ({ h, i })).filter((x) => x.h > 0);
  if (active.length === 0) return { title: "Do nothing", detail: "No works" };
  if (active.length === 1) {
    return { title: `Levee · site ${active[0].i}`, detail: "Single levee" };
  }
  return {
    title: `Sites ${active.map((a) => a.i).join(" + ")}`,
    detail: active.map((a) => `${a.h.toFixed(1)} m`).join(", "),
  };
}

export function OptionList({
  options, budgetM, chosen, onChoose, unit = "flood depth",
}: {
  options: OptionRow[];
  budgetM: number;
  chosen?: string;
  onChoose?: (id: string) => void;
  unit?: string;
}) {
  const [filter, setFilter] = useState<"all" | "helps" | "worsens">("all");
  const [expanded, setExpanded] = useState(false);

  const measures = options.filter((o) => o.heights.some((h) => h > 0));
  const affordable = measures.filter((o) => o.cost_aud <= budgetM * 1e6);

  const filtered = useMemo(() => {
    const rows = affordable.filter((o) => {
      const d = o.core_reduction_pct ?? 0;
      if (filter === "helps") return d > 0.5;
      if (filter === "worsens") return d < -0.5;
      return true;
    });
    // Best first: a list scanned top-down should put the answer at the top.
    return [...rows].sort(
      (a, b) => (b.core_reduction_pct ?? 0) - (a.core_reduction_pct ?? 0),
    );
  }, [affordable, filter]);

  const best = filtered.find((o) => (o.core_reduction_pct ?? 0) > 0.5);
  const shown = expanded ? filtered : filtered.slice(0, 5);

  return (
    <>
      <p className="anno mt-2">
        {affordable.length} of {measures.length} options fit this budget
      </p>

      <div className="mt-4 flex items-center justify-between">
        <h3 className="eyebrow">Options</h3>
        <div className="flex gap-1">
          {(["all", "helps", "worsens"] as const).map((f) => (
            <button
              key={f}
              className="pill"
              data-on={filter === f}
              onClick={() => { setFilter(f); setExpanded(false); }}
            >
              {f === "all" ? "All" : f === "helps" ? "Helps" : "Worsens"}
            </button>
          ))}
        </div>
      </div>

      <ul className="mt-2.5 overflow-hidden rounded-xl border border-line">
        {shown.map((o, i) => {
          const d = o.core_reduction_pct ?? 0;
          const { title, detail } = describe(o);
          const sel = o.id === chosen;
          return (
            <li key={o.id} className={i > 0 ? "border-t border-line-soft" : ""}>
              <button
                onClick={() => onChoose?.(o.id)}
                aria-pressed={sel}
                disabled={!onChoose}
                className={`flex w-full items-center gap-3 px-3.5 py-3 text-left transition-colors
                            ${sel ? "bg-bg-inset" : onChoose ? "hover:bg-bg-raised" : ""}`}
              >
                <span
                  aria-hidden
                  className={`h-[15px] w-[15px] shrink-0 rounded-full border transition-colors ${
                    sel ? "border-[5px] border-ink" : "border-line"
                  }`}
                />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="truncate text-[13px] font-medium text-ink">{title}</span>
                    {best?.id === o.id && (
                      <span className="chip-good shrink-0 text-[11px]">Best value</span>
                    )}
                  </span>
                  <span className="anno mt-0.5 block truncate">
                    {detail} · {money(o.cost_aud)}
                  </span>
                </span>
                {d < -0.5 ? (
                  <span className="chip-harm tnum shrink-0 text-[13px]">
                    +{Math.abs(d).toFixed(0)}%
                  </span>
                ) : (
                  <span className="tnum shrink-0 text-[15px] font-semibold text-ink">
                    −{d.toFixed(0)}%
                  </span>
                )}
              </button>
            </li>
          );
        })}

        {filtered.length === 0 && (
          <li className="px-3.5 py-4 text-[13px] text-ink-mute">
            Nothing in this budget {filter === "helps" ? "reduces" : "changes"} flooding here.
          </li>
        )}

        {filtered.length > 5 && (
          <li className="border-t border-line-soft">
            <button
              onClick={() => setExpanded((v) => !v)}
              className="w-full py-2.5 text-center text-[13px] text-ink-mute hover:text-ink"
            >
              {expanded ? "Show fewer" : `Show all ${filtered.length}`}
            </button>
          </li>
        )}
      </ul>

      <p className="anno mt-2.5">Change in {unit} vs. doing nothing</p>
    </>
  );
}
