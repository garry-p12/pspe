// The cross-section's levee arithmetic, checked against a REAL solved section.
//
// `src/lib/section.ts` is the only place the portal decides how high a crest
// has to be, how far it runs and what it costs, and none of that is checked by
// the service: /section returns a profile and nothing more. So it is checked
// here, on the response from an actual 7.1 km solve across the Wilsons River
// floodplain at Lismore (80 m cells, river driver, 18,750 m3/s), against
// figures computed independently in Python from the same JSON.
//
// Compile first:
//   npx tsc src/lib/section.ts --outDir <dir> --module commonjs --target es2020
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const [, , jsPath, jsonPath] = process.argv;
if (!jsPath || !jsonPath) {
  console.error("usage: node verify-section-math.mjs <section.js> <section_response.json>");
  process.exit(2);
}
const S = createRequire(import.meta.url)(jsPath);
const res = JSON.parse(readFileSync(jsonPath, "utf8"));
const spacing = S.sampleSpacing(res);
const rate = res.cost_per_m_per_m_aud;

let fails = 0;
const eq = (name, got, want, tol = 0) => {
  const ok = typeof want === "number"
    ? Math.abs(got - want) <= tol
    : String(got) === String(want);
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}${ok ? "" : `  (expected ${want})`}`);
};

console.log(`line ${res.length_m} m, ${res.samples} samples, ${res.dx_m} m cells`);
eq("sample spacing", Number(spacing.toFixed(1)), 50.8, 0.05);

// --- reaches ---------------------------------------------------------------
const all = S.reaches(res.profile, 1);
const kept = S.reaches(res.profile);
eq("wet runs, unfiltered", all.length, 9);
eq("wet runs of 2+ samples", kept.length, 6);
const main = kept[0];
eq("main reach start, m", main.m0, 50.8, 0.1);
eq("main reach end, m", main.m1, 2538.8, 0.1);
eq("main reach samples", main.samples, 50);
eq("main reach stage, m", main.maxWaterM, 12.14, 0.005);
eq("main reach deepest, m", main.maxDepthM, 8.82, 0.005);
eq("longest run is first", kept.every((r, i) => i === 0 || r.samples <= kept[i - 1].samples), true);

// --- the channel, and the floodplain either side of it --------------------
// The first version of this panel priced a 2.59 km wall at AUD 29.7 M whose
// middle 760 m ran across the Wilsons River, over a bed at 3.2 to 4.5 m under
// 8.8 m of water. That is a dam. The channel has to be found before anything
// is priced, and an alignment over it refused.
const belts = S.channelBelts(res.profile, main);
eq("one channel belt on the main reach", belts.length, 1);
eq("channel starts, m", res.profile[belts[0].i0].m, 1675.6, 0.1);
eq("channel ends, m", res.profile[belts[0].i1].m, 2386.4, 0.1);
eq("the braided threads are merged into one belt",
   belts[0].i1 - belts[0].i0 + 1, 15);
// The shallow reaches are not channels. A fixed depth threshold would have
// called the 2.49 m backwater one.
const back = kept.find((r) => Math.abs(r.maxWaterM - 8.80) < 0.01);
eq("a 2.49 m backwater has no channel", S.channelBelts(res.profile, back).length, 0);

const pans = S.panels(res.profile, main, belts);
eq("the channel splits the reach into two floodplain panels", pans.length, 2);
eq("widest panel starts, m", res.profile[pans[0].i0].m, 50.8, 0.1);
eq("widest panel ends at the channel bank, m", res.profile[pans[0].i1].m, 1624.8, 0.1);
eq("stage on that panel, m", S.stageOf(res.profile, pans[0]), 11.57, 0.005);
eq("which is below the whole reach's stage", S.stageOf(res.profile, pans[0]) < main.maxWaterM, true);

// --- the cheapest crest that holds that panel -----------------------------
const best = S.lowestHoldingCrest(res.profile, spacing, pans[0], rate, belts);
if (!best) { console.log("FAIL no holding crest found"); process.exit(1); }
eq("cheapest holding crest, m", best.crestM, 11.57, 0.005);
eq("alignment runs to the line's west end", best.span.i0, 0);
eq("and stops at the channel bank", best.span.i1, 32);
const a = S.assess(res.profile, spacing, best.span, best.crestM, rate, belts);
eq("verdict", a.verdict, "holds");
eq("crest length, m", Math.round(a.lengthM), 1625, 1);
eq("max height, m", Number(a.maxHeightM.toFixed(2)), 3.91, 0.005);
eq("mean height, m", Number(a.meanHeightM.toFixed(2)), 2.65, 0.005);
eq("cost, AUD M", Number((a.costAud / 1e6).toFixed(1)), 11.2, 0.05);
eq("landward end ties in", a.westEnd, "tied");
eq("the other end is the river, not a defect", a.eastEnd, "river");
eq("no overtopping", a.overtopM, 0);
// The west end ties in ON the line's own last sample, which is a limit of the
// line and not evidence that the bank keeps rising. The panel has to say so.
eq("west tie-in is at the line's end", a.westAtLineEnd, true);

// --- the refusal ----------------------------------------------------------
// The alignment the first version chose, scored under the rules that replaced
// it.
const dam = S.assess(res.profile, spacing, { i0: 0, i1: 51 }, 12.14, rate, belts);
eq("the old 2.59 km alignment is refused", dam.verdict, "crosses-channel");
eq("crossesChannel names the belt it hits",
   S.crossesChannel({ i0: 0, i1: 51 }, belts)?.i0, belts[0].i0);
eq("an alignment inside one panel is not refused",
   S.assess(res.profile, spacing, best.span, best.crestM, rate, belts).verdict !== "crosses-channel", true);
// spanFor must not walk an alignment into the channel looking for a tie-in.
const far = S.spanFor(res.profile, 20.0, pans[0], belts);
eq("a crest above every bank still stops at the channel", far.i1 <= belts[0].i0, true,
   `i1=${far.i1}, channel starts at ${belts[0].i0}`);

// --- the mistake this panel exists to prevent -----------------------------
// Designing to the highest water ANYWHERE on the line, which is a different
// body of water 4 km away and 9 m higher.
const naiveSpan = S.spanFor(res.profile, res.peak_water_m, pans[0], belts);
const naive = S.assess(res.profile, spacing, naiveSpan, res.peak_water_m, rate, belts);
eq("whole-line peak, m", res.peak_water_m, 20.67, 0.005);
eq("the same alignment at the whole-line peak costs more",
   naive.costAud > a.costAud * 2, true, `A$${(naive.costAud / 1e6).toFixed(1)} M vs A$${(a.costAud / 1e6).toFixed(1)} M`);
eq("and it no longer ties in anywhere on the line", naive.verdict, "open-end");

// --- the two failure modes ------------------------------------------------
const stage = S.stageOf(res.profile, pans[0]);
const low = S.assess(res.profile, spacing, best.span, stage - 1.0, rate, belts);
eq("1 m under the stage overtops", low.verdict, "overtops");
eq("overtopping depth, m", low.overtopM, 1.0, 0.005);
eq("overtopping is inside the alignment", low.overtopAtM >= res.profile[best.span.i0].m
   && low.overtopAtM <= res.profile[best.span.i1].m, true);
// Clearing the water but stopping short of the bank: an open end, not a fail.
const cut = { i0: best.span.i0 + 6, i1: best.span.i1 };
const open = S.assess(res.profile, spacing, cut, best.crestM, rate, belts);
eq("an alignment cut short of the bank has an open end", open.verdict, "open-end");
eq("the open end is the landward one", open.westEnd, "open");
eq("the shortfall is reported, m", open.westShortM, 3.44, 0.005);
eq("at the same crest, a shorter wall costs less", open.costAud < a.costAud, true);
// But not across crests, and this is the whole reason the cheapest holding
// crest is SEARCHED rather than set to the stage plus a freeboard: raising the
// crest lengthens the run needed to tie in, so a wall that is shorter and
// higher can cost more than one that is longer and lower.
const higher = S.assess(res.profile, spacing, cut, 13.5, rate, belts);
eq("shorter but 1.9 m higher costs more than the holding design",
   higher.costAud > a.costAud, true,
   `A$${(higher.costAud / 1e6).toFixed(1)} M vs A$${(a.costAud / 1e6).toFixed(1)} M`);
// Raising the crest cannot start overtopping.
let mono = true;
for (let c = stage; c <= stage + 5; c += 0.25) {
  const sp = S.spanFor(res.profile, c, pans[0], belts);
  if (S.assess(res.profile, spacing, sp, c, rate, belts).verdict === "overtops") mono = false;
}
eq("no crest at or above the stage overtops it", mono, true);

// A dry reach list means no design is offered at all.
const dry = res.profile.map((q) => ({ ...q, water_m: null, depth_m: 0 }));
eq("a dry line offers no reach", S.reaches(dry).length, 0);

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
process.exit(fails ? 1 : 0);
