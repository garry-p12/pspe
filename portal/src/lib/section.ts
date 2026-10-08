/**
 * A levee read off an elevation section, and whether it holds.
 *
 * The service's `/section` samples ground elevation and the SOLVED peak water
 * surface along a line the user draws. This turns that profile into the answer
 * an engineer asks for first -- how high does the crest have to be, how far
 * does it have to run, what does that cost -- and, more usefully, into the two
 * ways such a wall fails.
 *
 * ## The whole-line peak is the wrong design stage
 *
 * A line drawn across a floodplain does not cross one body of water. The
 * Lismore section measured here, 7.1 km at 80 m cells, crosses NINE separate
 * wet runs whose peak surfaces range from 8.80 m to 20.67 m: the main channel
 * reach, a backwater, three single-cell puddles, and a gorge at the far end
 * that is 4 km from the town and 8.5 m higher. `peak_water_m` in the response
 * is the maximum over all of them.
 *
 * Designing to that figure costs AUD 88.5 M against AUD 29.7 M for the crest
 * that actually defends the town reach -- three times the money, for no extra
 * protection of the thing being protected, because the water that set the
 * number is on the other side of a 67 m hill. So everything here works per
 * REACH: a contiguous run of wet samples, with its own peak surface.
 *
 * ## What is being priced
 *
 * The line is read as the levee's ALIGNMENT: a tie-back embankment running
 * from high ground out across the floodplain, which is what a line drawn
 * across a floodplain describes and what the ends of a real levee system do.
 * It is not a levee following the river.
 *
 * That reading has a limit, and ignoring it was the first version's mistake.
 * The cheapest crest that held the Lismore reach came back as a 2.59 km wall
 * costing AUD 29.7 M -- and 760 m of it ran across the river, over a bed 760 m
 * wide sitting 3.2 to 4.5 m above sea level under 8.8 m of water. That is a
 * dam. So the channel is found first, as the belt where the water is at least
 * 2 m deeper than it is over the floodplain either side of it, and an
 * alignment that crosses one is refused rather than priced.
 *
 * ## Two ways it fails, and only one of them is overtopping
 *
 * A crest below the water in the reach it defends is overtopped, which is the
 * failure everyone draws. The other one is an END: an embankment that stops on
 * ground lower than its own crest has an opening at that elevation, and water
 * standing at the design stage goes around it. A section cannot see around its
 * own ends, so this is reported as an incomplete design rather than as a
 * verdict -- the honest statement is "this end is 4.4 m below the crest",
 * not "it fails".
 */

export interface SectionSample {
  /** Distance from end A along the line, metres. */
  m: number;
  ground_m: number;
  depth_m: number;
  /** Water SURFACE elevation, or null where the solve left this sample dry. */
  water_m: number | null;
}

export interface SectionResult {
  ok: boolean;
  reason?: string;
  a: { lat: number; lon: number };
  b: { lat: number; lon: number };
  name: string;
  length_m: number;
  dx_m: number;
  samples: number;
  profile: SectionSample[];
  /** Maximum water surface anywhere on the line. Not a design stage -- see
   *  the note above. Null when the line came back entirely dry. */
  peak_water_m: number | null;
  wet_fraction: number;
  highest_ground: { m: number; ground_m: number };
  cost_per_m_per_m_aud: number;
  driver: string;
  driver_note: string;
  river_q_m3s: number | null;
  note: string;
}

/** A contiguous run of wet samples: one body of water, with its own stage. */
export interface Reach {
  i0: number;
  i1: number;
  m0: number;
  m1: number;
  samples: number;
  /** The design stage for a levee on this reach. */
  maxWaterM: number;
  maxDepthM: number;
  minGroundM: number;
}

export interface Span {
  i0: number;
  i1: number;
}

/** A stretch of the reach that is channel rather than floodplain. */
export type Belt = Span;

export type Verdict =
  | "holds" | "overtops" | "open-end" | "crosses-channel" | "dry";

/** What each end of the embankment meets. A river end is not a defect: a
 *  tie-back runs from high ground to the bank, and the bank is where it is
 *  meant to stop. An open end is a wall that stopped in the middle of the
 *  floodplain at an elevation water can walk around. */
export type End = "tied" | "open" | "river";

export interface Assessment {
  verdict: Verdict;
  /** How far the water rises above the crest inside the span, metres. */
  overtopM: number;
  /** Where that happens, metres along the line. */
  overtopAtM: number | null;
  /** Ground elevation each end of the embankment ties into. */
  westTieM: number;
  eastTieM: number;
  westEnd: End;
  eastEnd: End;
  /** How far that end falls short of the crest; 0 when it ties in. */
  westShortM: number;
  eastShortM: number;
  /** The span ran out of line before finding ground at crest level. The tie-in
   *  is then the last sample, which is a limit of the line, not of the bank. */
  westAtLineEnd: boolean;
  eastAtLineEnd: boolean;
  /** Cross-sectional area of fill, m^2 -- mean height times length. */
  fillM2: number;
  lengthM: number;
  maxHeightM: number;
  meanHeightM: number;
  costAud: number;
}

/** Spacing between profile samples. Finer than `dx_m`: the service samples the
 *  line at `samples` points whatever the DEM's resolution, so neighbouring
 *  samples can land in the same cell and carry no new information. */
export function sampleSpacing(r: SectionResult): number {
  return r.samples > 1 ? r.length_m / (r.samples - 1) : r.length_m;
}

/**
 * Contiguous wet runs, longest first.
 *
 * `minSamples` filters the selector list, not the physics: a single wet sample
 * is a cell-sized puddle in a 30 m DEM, and offering it as something to build
 * a levee along would be inviting a design against numerical noise. Three of
 * the nine runs on the measured Lismore line are single samples.
 */
export function reaches(profile: SectionSample[], minSamples = 2): Reach[] {
  const out: Reach[] = [];
  let i0: number | null = null;
  const close = (i1: number) => {
    if (i0 === null) return;
    const seg = profile.slice(i0, i1 + 1);
    const ws = seg.map((q) => q.water_m ?? -Infinity);
    out.push({
      i0,
      i1,
      m0: profile[i0].m,
      m1: profile[i1].m,
      samples: i1 - i0 + 1,
      maxWaterM: Math.max(...ws),
      maxDepthM: Math.max(...seg.map((q) => q.depth_m)),
      minGroundM: Math.min(...seg.map((q) => q.ground_m)),
    });
    i0 = null;
  };
  for (let i = 0; i < profile.length; i++) {
    if (profile[i].water_m !== null) {
      if (i0 === null) i0 = i;
    } else {
      close(i - 1);
    }
  }
  close(profile.length - 1);
  return out
    .filter((r) => r.samples >= minSamples)
    .sort((a, b) => b.samples - a.samples);
}

/**
 * The channel inside a reach: where the water is much deeper than it is over
 * the floodplain either side.
 *
 * Measured against the reach's OWN median depth rather than an absolute. A
 * threshold like "deeper than 5 m" calls a 2.5 m backwater floodplain on one
 * section and calls half a shallow estuary a channel on the next; the median
 * is what makes "much deeper" mean the same thing on both. Measured on the
 * Lismore reach the threshold lands at 5.0 m and picks the 760 m of river bed
 * between 1,676 m and 2,386 m; on the 200 m backwater further along the line
 * it lands at 3.5 m, above that reach's 2.49 m deepest point, and finds no
 * channel at all -- which is right, a backwater is not something an
 * embankment cannot cross.
 *
 * `gapSamples` closes the short shallow bars between the threads of a braided
 * channel. Without it the Lismore river comes back as four separate belts
 * with 100 m gaps that nothing could be built in anyway.
 */
export function channelBelts(
  profile: SectionSample[],
  reach: Reach,
  deeperByM = 2,
  gapSamples = 3,
): Belt[] {
  const depths = [];
  for (let i = reach.i0; i <= reach.i1; i++) depths.push(profile[i].depth_m);
  const sorted = [...depths].sort((x, z) => x - z);
  const median = sorted[Math.floor(sorted.length / 2)];
  const threshold = Math.max(median + deeperByM, reach.maxDepthM * 0.5);
  const belts: Belt[] = [];
  for (let i = reach.i0; i <= reach.i1; i++) {
    if (profile[i].depth_m <= threshold) continue;
    const last = belts[belts.length - 1];
    if (last && i - last.i1 <= gapSamples) last.i1 = i;
    else belts.push({ i0: i, i1: i });
  }
  return belts;
}

/**
 * The floodplain panels of a reach: its wet stretches either side of the
 * channel, which is where a tie-back embankment can actually go.
 *
 * Longest first, so the default alignment is the one with the most floodplain
 * behind it to protect.
 */
export function panels(
  profile: SectionSample[],
  reach: Reach,
  belts: Belt[],
): Span[] {
  const out: Span[] = [];
  let at = reach.i0;
  for (const b of belts) {
    if (b.i0 - 1 >= at) out.push({ i0: at, i1: b.i0 - 1 });
    at = b.i1 + 1;
  }
  if (reach.i1 >= at) out.push({ i0: at, i1: reach.i1 });
  return out
    .filter((sp) => sp.i1 > sp.i0)
    .sort((x, z) => (z.i1 - z.i0) - (x.i1 - x.i0));
}

/** Whether an alignment runs over any part of the channel. */
export function crossesChannel(span: Span, belts: Belt[]): Belt | null {
  return belts.find((b) => span.i0 <= b.i1 && span.i1 >= b.i0) ?? null;
}

/**
 * How far the embankment has to run to tie into ground at crest level.
 *
 * Outward from the reach in both directions, stopping at the first sample
 * whose ground reaches the crest. This is the span a levee at that crest
 * actually needs; a shorter one has an open end at a lower elevation.
 */
export function spanFor(
  profile: SectionSample[],
  crestM: number,
  base: Span,
  belts: Belt[] = [],
): Span {
  const blocked = (i: number) => belts.some((b) => i >= b.i0 && i <= b.i1);
  let i0 = base.i0;
  while (i0 > 0 && profile[i0].ground_m < crestM && !blocked(i0 - 1)) i0 -= 1;
  let i1 = base.i1;
  while (i1 < profile.length - 1 && profile[i1].ground_m < crestM && !blocked(i1 + 1)) i1 += 1;
  return { i0, i1 };
}

/** What a crest at this elevation, running over this alignment, does and costs. */
export function assess(
  profile: SectionSample[],
  spacingM: number,
  span: Span,
  crestM: number,
  ratePerMPerM: number,
  belts: Belt[] = [],
): Assessment {
  const seg = profile.slice(span.i0, span.i1 + 1);
  const wet = seg.filter((q) => q.water_m !== null);
  let overtopM = 0;
  let overtopAtM: number | null = null;
  for (const q of wet) {
    const over = (q.water_m as number) - crestM;
    if (over > overtopM) {
      overtopM = over;
      overtopAtM = q.m;
    }
  }
  const westTieM = profile[span.i0].ground_m;
  const eastTieM = profile[span.i1].ground_m;
  const westShortM = Math.max(crestM - westTieM, 0);
  const eastShortM = Math.max(crestM - eastTieM, 0);
  const onRiver = (i: number) => belts.some((b) => i + 1 >= b.i0 && i - 1 <= b.i1);
  const end = (i: number, short: number): End =>
    short <= 1e-6 ? "tied" : onRiver(i) ? "river" : "open";
  const westEnd = end(span.i0, westShortM);
  const eastEnd = end(span.i1, eastShortM);

  const heights = seg.map((q) => Math.max(crestM - q.ground_m, 0));
  // Mean height times length, the same model the district's six surveyed
  // sites are costed with (eval/build_scenario_library.py), so a figure from
  // this panel and one from the Plan tab are comparable.
  const fillM2 = heights.reduce((s, h) => s + h, 0) * spacingM;
  const lengthM = (span.i1 - span.i0) * spacingM;

  const verdict: Verdict =
    crossesChannel(span, belts) ? "crosses-channel"
      : wet.length === 0 ? "dry"
        : overtopM > 1e-6 ? "overtops"
          : westEnd === "open" || eastEnd === "open" ? "open-end"
            : "holds";

  return {
    verdict,
    overtopM: Math.round(overtopM * 100) / 100,
    overtopAtM,
    westTieM,
    eastTieM,
    westEnd,
    eastEnd,
    westShortM: Math.round(westShortM * 100) / 100,
    eastShortM: Math.round(eastShortM * 100) / 100,
    westAtLineEnd: span.i0 === 0 && westEnd === "tied",
    eastAtLineEnd: span.i1 === profile.length - 1 && eastEnd === "tied",
    fillM2,
    lengthM,
    maxHeightM: heights.length ? Math.max(...heights) : 0,
    meanHeightM: lengthM > 0 ? fillM2 / lengthM : 0,
    costAud: fillM2 * ratePerMPerM,
  };
}

/**
 * The lowest crest that holds this floodplain panel, from its own stage upward.
 *
 * Not "the stage plus a freeboard": raising the crest also lengthens the run
 * needed to tie into the bank, so a crest that clears the water can still have
 * an open end, and the cheapest holding design is the lowest elevation at
 * which both conditions land together. Returns null when no crest within
 * `maxRiseM` of the stage ties in -- the line is too short, not the site
 * hopeless.
 */
export function lowestHoldingCrest(
  profile: SectionSample[],
  spacingM: number,
  panel: Span,
  ratePerMPerM: number,
  belts: Belt[] = [],
  maxRiseM = 8,
  stepM = 0.1,
): { crestM: number; span: Span } | null {
  let stage = -Infinity;
  for (let i = panel.i0; i <= panel.i1; i++) {
    const w = profile[i].water_m;
    if (w !== null && w > stage) stage = w;
  }
  if (!Number.isFinite(stage)) return null;
  const steps = Math.round(maxRiseM / stepM);
  for (let k = 0; k <= steps; k++) {
    const crestM = Math.round((stage + k * stepM) * 100) / 100;
    const span = spanFor(profile, crestM, panel, belts);
    if (assess(profile, spacingM, span, crestM, ratePerMPerM, belts).verdict === "holds") {
      return { crestM, span };
    }
  }
  return null;
}

/** The highest water standing on a stretch of the line. */
export function stageOf(profile: SectionSample[], span: Span): number | null {
  let stage = -Infinity;
  for (let i = span.i0; i <= span.i1; i++) {
    const w = profile[i].water_m;
    if (w !== null && w > stage) stage = w;
  }
  return Number.isFinite(stage) ? stage : null;
}

/** Interpolate a lon/lat on the line at a distance along it. */
export function pointAt(
  r: Pick<SectionResult, "a" | "b" | "length_m">,
  m: number,
): [number, number] {
  const f = r.length_m > 0 ? Math.min(Math.max(m / r.length_m, 0), 1) : 0;
  return [
    r.a.lon + (r.b.lon - r.a.lon) * f,
    r.a.lat + (r.b.lat - r.a.lat) * f,
  ];
}

export function money(aud: number): string {
  if (aud >= 1e9) return `A$${(aud / 1e9).toFixed(2)} bn`;
  if (aud >= 1e6) return `A$${(aud / 1e6).toFixed(1)} M`;
  if (aud >= 1e3) return `A$${(aud / 1e3).toFixed(0)} k`;
  return `A$${aud.toFixed(0)}`;
}

export function dist(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`;
}
