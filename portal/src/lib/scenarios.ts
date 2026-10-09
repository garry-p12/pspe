/**
 * The floods a plan is judged against.
 *
 * The Plan tab used to solve whatever the Forecast tab happened to be showing,
 * which for the shipped Austin study is a 50 mm/h design rainstorm falling
 * evenly over the whole box. That is a real question and it is the wrong one
 * to put a levee in front of: rain lands on both sides of a wall, so the wall
 * holds water in. Measured on Onion Creek, two 2 m levees under that storm
 * made the flooded area 0.05% WORSE. Driving it from the live gauge instead
 * gave 0.13 km² of water, because Onion Creek is not in flood today.
 *
 * A levee answers a question about a named flood. Onion Creek has an
 * instrumented record of those going back to 1921 — 3,823 m³/s on 31 October
 * 2013, 3,455 on 30 October 2015, 3,908 in September 1921 — published by the
 * USGS as annual peak streamflow. Those are the recorded scenarios. The fitted
 * ones come from a log-Pearson III fit to the same record, which is how the
 * regulatory discharges in an American flood study are derived.
 *
 * A recorded flood and a fitted quantile are different kinds of claim and the
 * list says which is which, because "the 2013 flood" is a thing that happened
 * and "the 1% flood" is a thing a distribution says.
 */

export interface PeakRow {
  date: string;
  q_m3s: number;
  q_cfs: number;
  stage_m: number | null;
}

export interface QuantileRow {
  aep_pct: number;
  return_period_y: number;
  q_m3s: number;
}

export interface PeaksGauge {
  site: string;
  name: string;
  years: number;
  first: string;
  last: string;
  peaks: PeakRow[];
  quantiles: QuantileRow[];
  fit: { n: number; skew?: number; note: string };
}

export interface PeaksFeed {
  ok: boolean;
  reason?: string;
  gauges: PeaksGauge[];
  source: string;
}

export interface Scenario {
  id: string;
  /** What the picker shows. */
  label: string;
  /** The line under it: where the number came from. */
  note: string;
  kind: "study" | "recorded" | "fitted";
  driver: "rain" | "river";
  river_q_m3s: number | null;
}

/** The design storm the shipped study was solved with. Always available,
 *  because it needs no gauge — and always last, because a levee is the wrong
 *  instrument for it. */
export function studyScenario(rainMmH: number, stormH: number): Scenario {
  return {
    id: "study",
    label: `Design storm · ${rainMmH} mm/h for ${stormH} h`,
    note: "Rain falling evenly over the whole box. A levee has nothing to "
        + "block under this: the water lands on both sides of it.",
    kind: "study",
    driver: "rain",
    river_q_m3s: null,
  };
}

function fmtDate(iso: string): string {
  const [y, m, d] = iso.split("-");
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return d && m ? `${Number(d)} ${MON[Number(m) - 1]} ${y}` : iso;
}

export function fmtQ(q: number): string {
  return q >= 1000 ? `${(q / 1000).toFixed(2)} × 10³ m³/s` : `${Math.round(q)} m³/s`;
}

/**
 * The scenario list for a reach, from its longest-recorded gauge.
 *
 * Recorded floods first, biggest first, capped at four: a list of every peak
 * since 1921 is a record, not a menu. Then the fitted quantiles a planner
 * actually designs to. Then the study's own storm, last.
 */
export function scenariosFrom(
  feed: PeaksFeed | null,
  rainMmH: number,
  stormH: number,
): { list: Scenario[]; gauge: PeaksGauge | null } {
  const study = studyScenario(rainMmH, stormH);
  const g = feed?.ok ? (feed.gauges[0] ?? null) : null;
  if (!g) return { list: [study], gauge: null };

  // Biggest first, but the DEFAULT is the biggest within living memory of the
  // built environment. Onion Creek's largest recorded peak is 9 September
  // 1921, which predates the suburb underneath it by half a century and is
  // not the flood anyone here remembers; 31 October 2013 is, and it is 2% off
  // the record. So the recent ones are listed first and the older record
  // stays in the list below them rather than opening the panel.
  const RECENT_Y = 35;
  const cutoff = new Date().getFullYear() - RECENT_Y;
  const top = g.peaks.slice(0, 6);
  const recent = top.filter((p) => Number(p.date.slice(0, 4)) >= cutoff);
  const older = top.filter((p) => Number(p.date.slice(0, 4)) < cutoff);
  const ordered = [...recent, ...older].slice(0, 4);

  const recorded: Scenario[] = ordered.map((p) => ({
    id: `rec:${p.date}`,
    label: `${fmtDate(p.date)} flood · ${fmtQ(p.q_m3s)}`,
    note: `Recorded at ${g.name}${p.stage_m !== null
      ? `, stage ${p.stage_m.toFixed(2)} m` : ""}. USGS annual peak streamflow`
      + `, ${g.years} years of record.`,
    kind: "recorded" as const,
    driver: "river" as const,
    river_q_m3s: p.q_m3s,
  }));

  // 1% and 4% only. The 2-year flood is not something anyone builds against
  // and the 0.2% is far outside what 58 years of record can support.
  const fitted: Scenario[] = g.quantiles
    .filter((q) => q.aep_pct === 1 || q.aep_pct === 4)
    .map((q) => ({
      id: `aep:${q.aep_pct}`,
      label: `${q.aep_pct}% annual chance · ${fmtQ(q.q_m3s)}`,
      note: `Fitted, not observed: log-Pearson III on ${g.years} years at `
          + `${g.name}. A ${q.return_period_y}-year flood.`,
      kind: "fitted" as const,
      driver: "river" as const,
      river_q_m3s: q.q_m3s,
    }));

  return { list: [...recorded, ...fitted, study], gauge: g };
}
