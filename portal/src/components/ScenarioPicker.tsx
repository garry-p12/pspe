"use client";

import { Explain } from "./Explain";
import type { Scenario, PeaksGauge } from "@/lib/scenarios";

/**
 * Which flood the plan is being judged against.
 *
 * The design puts a dropdown at the top of the Plan panel reading "2022 flood
 * (recorded)". This is that control, with the floods this reach has actually
 * carried in it. They come from the USGS annual peak streamflow record for the
 * gauge on the reach, and the fitted ones from a log-Pearson III fit to the
 * same record.
 *
 * The distinction between the two is the point of the line under the select.
 * "31 Oct 2013" is a thing that happened and was measured at a gauge house;
 * "1% annual chance" is a thing a distribution fitted to 58 years says, and on
 * a 58-year record the 1% quantile is an extrapolation. Both belong in the
 * list. Neither should be able to pass for the other.
 */
export function ScenarioPicker({
  list, chosen, gauge, onChoose, disabled,
}: {
  list: Scenario[];
  chosen: Scenario;
  gauge: PeaksGauge | null;
  onChoose: (s: Scenario) => void;
  disabled?: boolean;
}) {
  const grouped = {
    recorded: list.filter((s) => s.kind === "recorded"),
    fitted: list.filter((s) => s.kind === "fitted"),
    study: list.filter((s) => s.kind === "study"),
  };

  return (
    <div className="border-b border-line px-5 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor="scenario" className="eyebrow">Judged against</label>
        {gauge && (
          <span className="anno truncate" title={gauge.name}>
            {gauge.years} y at {gauge.name.split(",")[0]}
          </span>
        )}
      </div>
      <select
        id="scenario"
        value={chosen.id}
        disabled={disabled}
        onChange={(e) => {
          const s = list.find((x) => x.id === e.target.value);
          if (s) onChoose(s);
        }}
        className="mt-1.5 w-full rounded-md border border-line bg-bg px-2.5 py-2
                   text-[13px] text-ink disabled:opacity-50"
      >
        {grouped.recorded.length > 0 && (
          <optgroup label="Floods this reach has carried">
            {grouped.recorded.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </optgroup>
        )}
        {grouped.fitted.length > 0 && (
          <optgroup label="Fitted from the record">
            {grouped.fitted.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </optgroup>
        )}
        <optgroup label="The study's own storm">
          {grouped.study.map((s) => (
            <option key={s.id} value={s.id}>{s.label}</option>
          ))}
        </optgroup>
      </select>
      {/* One visible line, and it is the one that stops a reader quoting
          this as a hindcast: a flood of this SIZE, not that day replayed.
          The 344-character argument for why -- the single largest block in
          the portal, paid for on every Plan and Cross-section screen -- is
          behind the (i). */}
      <p className="anno mt-1.5 flex items-center gap-1.5 leading-relaxed">
        {chosen.kind === "recorded"
          ? "A flood of this size, not that day replayed"
          : chosen.kind === "fitted" ? "Fitted from the record, not observed"
            : "The study's own design storm"}
        <Explain title="What this scenario is">
          <p>{chosen.note}</p>
          {chosen.kind === "recorded" && (
            <p>
              The discharge is the recorded one; what the map draws is this
              solver&apos;s response to it. The flow enters at the edge of
              the box rather than down eighty kilometres of catchment, and
              the model recovers about half of FEMA&apos;s mapped
              floodplain.
            </p>
          )}
        </Explain>
      </p>
      {/* Stays visible, and in the warning colour: it changes the choice
          the reader is making in this very control. */}
      {chosen.driver === "rain" && (
        <p className="mt-1.5 text-[12px] leading-relaxed text-bad">
          Rain lands on both sides of a wall — pick a flood that arrives
          down the channel to see what a levee buys.
        </p>
      )}
    </div>
  );
}
