/**
 * What flood works cost in the United States, and where each figure comes from.
 *
 * Every price the portal shows used to trace back to one bare float —
 * `2600`, Australian dollars per metre of crest per metre of height, no
 * source, no base year, no index — copied into four files. For a study in
 * Austin, Texas that is wrong twice: wrong currency, and wrong country's
 * construction market.
 *
 * ## Where these come from
 *
 * The spine is the **USACE National Levee Safety Program**, *Summary of Costs
 * Associated with Levee-related Activities*, FINAL November 2023, price level
 * March 2023 — a national-average, riverine, Class 5 (order-of-magnitude)
 * guide built from USACE and levee-sponsor bid data, and the only publication
 * that gives embankment, floodwall and gated-outfall costs on one consistent
 * basis with an explicit statement of what is and is not in them.
 *
 * Each figure is then escalated from 2Q23 to 4Q25 with the Civil Works
 * Construction Cost Index System (CWCCIS Feature Code 11, Levees &
 * Floodwalls: 1144.19 → 1214.81, a factor of 1.0617; Feature Code 13 for
 * pumping plant) and multiplied by **0.88**, the CWCCIS Table 3 Texas state
 * adjustment factor — the same factor USACE Fort Worth applied to the Dallas
 * Floodway estimate.
 *
 * Cross-checks that agree: USACE Galveston's Hunting Bayou study used $7/CY
 * of embankment fill from Harris County Flood Control District bid tabs in
 * 2009, which escalates to $15.5/m³ in 2025 — within 2% of the figure here.
 *
 * ## What these are not
 *
 * **Construction cost only.** Labour, material and plant. They exclude
 * design, land acquisition, utility relocation, environmental mitigation and
 * contingency, which together roughly double the bill: the brochure's own
 * all-in build-up runs 2.1× construction for both embankment and T-wall, and
 * that is what `ALL_IN` is.
 *
 * They are also Class 5 planning numbers. They are the right order of
 * magnitude for deciding whether a wall is worth studying and the wrong thing
 * to put in a tender.
 */

export interface Rate {
  /** The figure, in 2025 US dollars, Austin-adjusted. */
  usd: number;
  /** Low and high of the published range, same basis. */
  range: [number, number];
  unit: string;
  source: string;
}

/** Compacted embankment fill in place, per cubic metre. */
export const EARTH_FILL: Rate = {
  usd: 15.2,
  range: [4.64, 49.9],
  unit: "USD per m³ of compacted fill",
  source: "USACE NLSP, Summary of Costs Associated with Levee-related "
        + "Activities, Nov 2023 ($3.80–$40.80/CY at Mar 2023), escalated "
        + "CWCCIS FC11 to 4Q25 and × 0.88 Texas factor",
};

/** The embankment's own cross-section, which is what turns a height into a
 *  volume. Crest 10 ft, batters 1V:3H — the brochure's own section, and the
 *  one `site_masks` sweeps in the solver. */
export const CREST_WIDTH_M = 3.048;
export const SIDE_SLOPE = 3.0;

/** Concrete T-wall, per metre of wall per metre of wall height. Flat with
 *  height, unlike an embankment, because the structure is a stem and a
 *  footing rather than a prism of dirt. */
export const T_WALL: Rate = {
  usd: 10800,
  range: [5000, 15000],
  unit: "USD per m of wall per m of height",
  source: "USACE NLSP Nov 2023 T-wall rates, escalated to 4Q25 × 0.88 Texas; "
        + "cross-checked against Aerts (2018) Water 10:1646 Table 4, which "
        + "gives roughly half this from HSDRRS-scale contracts",
};

/** Sheet pile with a concrete cap. A third the price of a T-wall and limited
 *  to about 2.7 m of exposed height. */
export const I_WALL: Rate = {
  usd: 3600,
  range: [2900, 5000],
  unit: "USD per m of wall per m of height",
  source: "West Pittston Borough PA Flood Mitigation Feasibility Study, "
        + "Appendix G (Borton-Lawson / Reilly, Oct 2019), escalated to 2025",
};

/** A complete gated outfall through a levee: gatewell, sluice gate, flap
 *  gate, inlet and outlet structures, 45 m of pipe, earthwork and demolition
 *  of what was there. Not the gate casting on its own, which is a few tens of
 *  thousands and is not what gets built. */
export const GATED_OUTFALL: Rate = {
  usd: 725000,
  range: [650000, 805000],
  unit: "USD per 1.22 m outfall structure",
  source: "USACE NLSP Nov 2023, Interior Drainage System Construction Costs "
        + "Per Unit, 48-inch diameter, escalated to 4Q25 × 0.88 Texas",
};

/** Interior-drainage pump station, per cubic metre per second of firm
 *  capacity. The best-corroborated figure in the set: eight of twelve
 *  independent observations across four unrelated sources and four price
 *  levels land between $0.64 M and $0.95 M per m³/s after escalation. */
export const PUMP_STATION: Rate = {
  usd: 800000,
  range: [250000, 2000000],
  unit: "USD per m³/s of firm capacity",
  source: "Synthesis of 12 US stations — Aerts (2018) Water 10:1646 Table A5 "
        + "(Gulf Coast built stations), a USACE MCACES/MII estimate, and "
        + "West Pittston PA — escalated CWCCIS FC13 to 4Q25 × 0.88 Texas",
};

/**
 * Construction cost to total project cost.
 *
 * The brochure's adders, as a share of construction: utility relocation 1%
 * (rural) to 5% (urban), real estate 3% to 8%, environmental 4% to 13%,
 * planning / engineering / design / construction management, and
 * contingency at 35–55%. They come to 2.1× for embankment and 2.13× for
 * T-wall, which is close enough to carry one number.
 */
export const ALL_IN = 2.1;

/**
 * What a run of embankment costs, from its real trapezoidal section.
 *
 * NOT a rate per metre per metre. The volume of a trapezoid is
 * `(crest · H + slope · H²)` per metre of crest, so the cost per metre of
 * height RISES with height — a 3 m levee is not one and a half times a 2 m
 * one, it is 2.0 times. Quoting a flat per-metre-per-metre rate, which is
 * what the portal did, understates a tall wall and overstates a short one.
 */
export function earthLeveeUsd(lengthM: number, heightM: number): number {
  const m3PerM = CREST_WIDTH_M * heightM + SIDE_SLOPE * heightM * heightM;
  return lengthM * m3PerM * EARTH_FILL.usd;
}

/** A concrete wall, which really is linear in height. */
export function wallUsd(lengthM: number, heightM: number, rate: Rate = T_WALL): number {
  return lengthM * heightM * rate.usd;
}

/** The embankment rate expressed the way a planner quotes it, for the one
 *  place that still needs a scalar: cost per metre of crest per metre of
 *  height, at a stated design height. */
export function earthLeveeRateAt(heightM: number): number {
  return heightM > 0 ? earthLeveeUsd(1, heightM) / heightM : 0;
}
