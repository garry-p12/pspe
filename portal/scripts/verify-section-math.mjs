// The cross-section's levee arithmetic, checked against a real solved section.
//
// `src/lib/section.ts` is the only place the portal decides how high a crest
// has to be, how far it runs and what it costs, and none of it is checked by
// the service: /section returns a profile and nothing more. This is where it
// is checked — and it is the harness that would have caught §5.14d, where the
// panel priced a 2.59 km "levee" whose middle 760 m ran across the river.
//
// Two kinds of assertion here, and the difference matters.
//
// **Against an independent baseline.** The structural quantities — where the
// water is, which run is the main one, where the channel is, what floodplain
// lies either side of it — come from `scripts/section_baseline.py`, a second
// implementation of the same written spec in a second language. Asserting
// TypeScript against numbers copied out of the TypeScript's own output proves
// only that it still does what it did.
//
// **As properties.** The crest search and the costing are checked by what
// must be TRUE of them — the crest it returns holds, nothing 0.1 m lower
// does, raising a crest cannot start overtopping, an alignment over the
// channel is refused, the trapezoid is dearer than the flat rate it replaced
// — rather than by a remembered figure. A property survives a change of
// fixture; a literal does not, which is how this harness came to be checking
// a river in New South Wales long after the portal stopped shipping one.
//
//   node scripts/make-section-fixture.mjs                  # needs the service
//   python3 scripts/section_baseline.py scripts/fixtures/section_onion_creek.json
//   npx tsc src/lib/section.ts src/lib/rates.ts \
//     --outDir /tmp/secjs --module commonjs --target es2020 --skipLibCheck
//   node scripts/verify-section-math.mjs /tmp/secjs/section.js
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";

const [, , jsPath, jsonArg] = process.argv;
const jsonPath = jsonArg
  ?? new URL("./fixtures/section_onion_creek.json", import.meta.url).pathname;
if (!jsPath) {
  console.error("usage: node verify-section-math.mjs <section.js> [fixture.json]");
  console.error("  compile src/lib/section.ts first; see the header.");
  process.exit(2);
}
const expPath = jsonPath.replace(/\.json$/, ".expected.json");
if (!existsSync(jsonPath) || !existsSync(expPath)) {
  console.error(`missing ${existsSync(jsonPath) ? expPath : jsonPath}`);
  console.error("  run make-section-fixture.mjs then section_baseline.py.");
  process.exit(2);
}

const S = createRequire(import.meta.url)(jsPath);
const res = JSON.parse(readFileSync(jsonPath, "utf8"));
const E = JSON.parse(readFileSync(expPath, "utf8"));
const spacing = S.sampleSpacing(res);
// USACE compacted fill in place, 2025 USD per cubic metre, Texas-adjusted.
// See src/lib/rates.ts, which is the one place this figure is written down.
const rate = 15.2;

let fails = 0;
const eq = (name, got, want, tol = 0) => {
  const ok = typeof want === "number"
    ? Math.abs(got - want) <= tol
    : String(got) === String(want);
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}${ok ? "" : `  (expected ${want})`}`);
};
const is = (name, cond, extra = "") => {
  if (!cond) fails++;
  console.log(`${cond ? "ok  " : "FAIL"} ${name}${extra ? `  ${extra}` : ""}`);
};

console.log(`line ${res.length_m} m, ${res.samples} samples, ${res.dx_m} m cells`);
console.log(`checked against ${expPath.split("/").pop()}\n`);

// --- against the independent baseline -------------------------------------
eq("sample spacing, m", Number(spacing.toFixed(3)), E.spacing_m, 0.005);

const all = S.reaches(res.profile, 1);
const kept = S.reaches(res.profile);
eq("wet runs, unfiltered", all.length, E.runs_unfiltered);
eq("wet runs of 2+ samples", kept.length, E.runs_2plus);
const main = kept[0];
eq("main reach start, m", main.m0, E.main.m0, 0.1);
eq("main reach end, m", main.m1, E.main.m1, 0.1);
eq("main reach samples", main.samples, E.main.samples);
eq("main reach stage, m", Number(main.maxWaterM.toFixed(3)), E.main.maxWaterM, 0.005);
eq("main reach deepest, m", Number(main.maxDepthM.toFixed(3)), E.main.maxDepthM, 0.005);
is("longest run is first",
   kept.every((r, i) => i === 0 || r.samples <= kept[i - 1].samples));

// The channel has to be found before anything is priced, and an alignment
// over it refused. This is the §5.14d defect's own check.
const belts = S.channelBelts(res.profile, main);
eq("channel belts on the main reach", belts.length, E.belts.length);
E.belts.forEach((b, i) => {
  if (!belts[i]) { fails++; console.log(`FAIL belt ${i} is missing`); return; }
  eq(`belt ${i} starts, m`, res.profile[belts[i].i0].m, b.m0, 0.1);
  eq(`belt ${i} ends, m`, res.profile[belts[i].i1].m, b.m1, 0.1);
  eq(`belt ${i} merges its threads into one`,
     belts[i].i1 - belts[i].i0 + 1, b.samples);
});

const pans = S.panels(res.profile, main, belts);
eq("floodplain panels either side of the channel", pans.length, E.panels.length);
E.panels.forEach((p, i) => {
  if (!pans[i]) { fails++; console.log(`FAIL panel ${i} is missing`); return; }
  eq(`panel ${i} starts, m`, res.profile[pans[i].i0].m, p.m0, 0.1);
  eq(`panel ${i} ends, m`, res.profile[pans[i].i1].m, p.m1, 0.1);
  eq(`panel ${i} stage, m`,
     Number(S.stageOf(res.profile, pans[i]).toFixed(3)), p.stage_m, 0.005);
});
is("widest panel is first",
   pans.every((p, i) => i === 0 || (p.i1 - p.i0) <= (pans[i - 1].i1 - pans[i - 1].i0)));

// --- properties of the crest search ---------------------------------------
const panel = pans[0];
const stage = S.stageOf(res.profile, panel);
const best = S.lowestHoldingCrest(res.profile, spacing, panel, rate, belts);
if (!best) {
  console.log("FAIL no holding crest found on the widest panel");
  process.exit(1);
}
is("the crest it returns is at or above the panel's stage",
   best.crestM >= stage - 1e-6, `${best.crestM} vs ${stage.toFixed(2)}`);
const a = S.assess(res.profile, spacing, best.span, best.crestM, rate, belts);
eq("and it holds", a.verdict, "holds");
is("its alignment does not cross the channel",
   S.crossesChannel(best.span, belts) === null);
is("nothing 0.1 m lower holds",
   (() => {
     const lower = best.crestM - 0.1;
     const sp = S.spanFor(res.profile, lower, panel, belts);
     return S.assess(res.profile, spacing, sp, lower, rate, belts).verdict !== "holds";
   })(), `crest ${best.crestM} m`);

// --- properties of the costing --------------------------------------------
eq("cost is the earthwork at the published rate",
   Number((a.costUsd / 1e6).toFixed(3)),
   Number(((a.fillM3 * rate) / 1e6).toFixed(3)), 0.001);
is("the trapezoidal section is dearer than the flat-rate model it replaced",
   a.fillM3 > a.fillM2,
   `${a.fillM3.toFixed(0)} m3 vs ${a.fillM2.toFixed(0)} m2 of profile`);
// Volume goes as crest*H + slope*H^2, so doubling the height is more than
// doubling the cost. A flat rate per metre of height says it is exactly
// double, which under-prices every tall wall.
is("doubling the height more than doubles the cost",
   (() => {
     const g = res.profile[panel.i0].ground_m;
     const one = S.wallAt(res.profile, panel.i0, g + 1, 1000, rate, "a", belts);
     const two = S.wallAt(res.profile, panel.i0, g + 2, 1000, rate, "a", belts);
     return two.costUsd > one.costUsd * 2.5;
   })(),
   "a 2 m embankment is 2.97x a 1 m one, not 2x");
is("at the same crest, a shorter wall costs less",
   (() => {
     const cut = { i0: Math.min(best.span.i0 + 4, best.span.i1), i1: best.span.i1 };
     return S.assess(res.profile, spacing, cut, best.crestM, rate, belts).costUsd
          < a.costUsd;
   })());

// --- the refusal, and the two failure modes -------------------------------
if (belts.length) {
  const over = { i0: Math.max(belts[0].i0 - 2, 0),
                 i1: Math.min(belts[0].i1 + 2, res.profile.length - 1) };
  eq("an alignment over the channel is refused",
     S.assess(res.profile, spacing, over, main.maxWaterM, rate, belts).verdict,
     "crosses-channel");
  eq("and crossesChannel names the belt it hits",
     S.crossesChannel(over, belts)?.i0, belts[0].i0);
  // spanFor must not walk an alignment into the channel looking for a tie-in.
  const far = S.spanFor(res.profile, main.maxWaterM + 10, panel, belts);
  is("a crest above every bank still stops at the channel",
     far.i0 > belts[0].i1 || far.i1 < belts[0].i0,
     `span ${far.i0}-${far.i1}, belt ${belts[0].i0}-${belts[0].i1}`);
}

const low = S.assess(res.profile, spacing, best.span, stage - 1.0, rate, belts);
eq("1 m under the stage overtops", low.verdict, "overtops");
eq("by 1 m", Number(low.overtopM.toFixed(3)), 1.0, 0.005);
is("and the overtopping is inside the alignment",
   low.overtopAtM >= res.profile[best.span.i0].m
   && low.overtopAtM <= res.profile[best.span.i1].m);

// Raising the crest cannot start overtopping. This is the monotonicity the
// crest search relies on, and the reason the search is a search rather than
// "stage plus a freeboard".
let mono = true;
for (let c = stage; c <= stage + 5; c += 0.25) {
  const sp = S.spanFor(res.profile, c, panel, belts);
  if (S.assess(res.profile, spacing, sp, c, rate, belts).verdict === "overtops") mono = false;
}
is("no crest at or above the stage overtops it", mono);

// A dry line offers no design at all.
const dry = res.profile.map((q) => ({ ...q, water_m: null, depth_m: 0 }));
eq("a dry line offers no reach", S.reaches(dry).length, 0);

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
process.exit(fails ? 1 : 0);
