// The browser's safety margin and the framework's must be the same margin.
//
// The district tab's headline -- "estimated 52%, at least 39%" -- is computed by
// `src/lib/margin.ts` and `src/app/api/plan/route.ts`, in TypeScript, from a
// precomputed surrogate fit. The Anywhere tab's band comes from the service,
// which reaches `pspe/plan/margins.py` through `surrogate.fit`. Two
// implementations of one guarantee, and until this script existed only the
// router had been held against its Python original.
//
// The project's own rule 13 is that a guarantee meaning one thing in Richmond
// and another elsewhere is worse than no guarantee. This checks three things
// across a sweep of delta and sample size:
//
//   1. the conformal rank -- does each side pick the same order statistic
//   2. the refusal threshold -- does each side decline at the same n
//   3. the saturating surrogate -- lead + S*tanh(rest/S), same value
//
//     node scripts/verify-margin-parity.mjs
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const REPO = "/Users/guruprasadparasnis/pspe";
const PYTHON = process.env.PYTHON ?? "/usr/bin/python3";
const TSC = `${REPO}/portal/node_modules/.bin/tsc`;

const outDir = mkdtempSync(join(tmpdir(), "margin-parity-"));
execFileSync(TSC, [
  "src/lib/margin.ts", "src/lib/routemargin.ts", "src/lib/routes.ts",
  "src/lib/inspect.ts",
  "--outDir", outDir, "--module", "es2022", "--target", "es2022",
  "--moduleResolution", "bundler", "--skipLibCheck", "--types", "geojson",
], { cwd: `${REPO}/portal`, stdio: "inherit" });
for (const f of ["margin.js", "routemargin.js", "routes.js", "inspect.js"]) {
  const at = join(outDir, f);
  writeFileSync(at, readFileSync(at, "utf8").replace(/from "\.\/([a-z]+)"/g, 'from "./$1.js"'));
}
const { bandAt, bandForPlan, runsNeeded, maxConfidencePct, saturating } = await import(join(outDir, "margin.js"));
const { predictedReduction } = await import(join(outDir, "routemargin.js"));

// One sample, generated once here and HANDED to the Python side.
//
// The first version regenerated it on both sides from the same LCG. That looked
// careful and was wrong: 1103515245 * x exceeds 2^53, so JavaScript loses
// integer precision where Python keeps it, the two sequences diverge, and the
// script reported 130 parity failures that were entirely its own. Comparing two
// implementations means feeding them the same bytes, not the same recipe.
let seed = 12345;
const next = () => {
  seed = (Math.imul(1103515, seed) + Math.imul(245, seed) + 12345) % 2147483648;
  return Math.abs(seed) / 2147483648;
};
const SAMPLE = Array.from({ length: 200 }, () => next() * 4);
const residuals = (n) => SAMPLE.slice(0, n);
const DELTAS = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.02];
const NS = [2, 3, 5, 7, 8, 9, 11, 12, 19, 30, 49, 120];

// TS side. bandAt wants CalibrationPoint[]; constructing lead/rest/measured so
// that |measured - predicted| equals the intended residual exactly.
const tsRows = [];
for (const d of DELTAS) {
  const conf = (1 - d) * 100;
  for (const n of NS) {
    const r = residuals(n);
    // lead and rest are NON-ZERO on purpose. The first version of this test set
    // both to zero, so `predicted` evaluated to zero and the residual was just
    // the sample: it proved the rank selection agreed and never touched the
    // saturating formula, which is the part that produces the displayed
    // estimate. A parity check that cannot fail on the thing it is named for is
    // not a parity check.
    const pts = r.map((v, i) => ({
      n_active: 2, lead: 1 + (i % 5) * 0.7, rest: 0.3 + (i % 3) * 1.1, measured: v,
    }));
    const b = bandAt(pts, 100, conf);
    tsRows.push({ delta: d, n, band: b.attainable ? +b.band.toFixed(10) : null,
                  needed: Number.isFinite(b.needed) ? b.needed : null });
  }
}
// The saturating form itself, including the regimes that matter: rest = 0
// (a single measure, which must be exact), rest >> S (saturated), and a small S.
const SAT = [
  { lead: 12, rest: 0, S: 100 },
  { lead: 12, rest: 7, S: 100 },
  { lead: 12, rest: 40, S: 12 },
  { lead: 0.5, rest: 0.25, S: 0.3 },
  { lead: 30, rest: 300, S: 20 },
  { lead: 1e-6, rest: 1e-6, S: 1e-3 },
];
const satTs = SAT.map((c) => +saturating(c.lead, c.rest, c.S).toFixed(10));

// Conditional-band cases. Strata of 21 / 8 / 1 reproduce the portal's own
// calibration shape: attainable at 90%, too thin, far too thin. The last two
// exercise the exactness rule for a single site, which has no calibration
// points at all and must not borrow another stratum's band.
const mkPts = (n, m) => Array.from({ length: n }, (_, i) => ({
  n_active: m, lead: 2 + (i % 4) * 1.3, rest: 0.4 + (i % 3) * 0.9,
  measured: 2 + (i % 4) * 1.3 + ((i % 7) - 3) * 1.1,
}));
const POOL = [...mkPts(21, 2), ...mkPts(8, 3), ...mkPts(1, 6)];
const COND = [
  { label: "2 (21 runs)",      points: POOL, S: 6, conf: 90, heights: [3, 1, 0, 0, 0, 0], maxH: 3 },
  { label: "3 (8 runs, thin)", points: POOL, S: 6, conf: 90, heights: [3, 1, 0.5, 0, 0, 0], maxH: 3 },
  { label: "6 (1 run)",        points: POOL, S: 6, conf: 90, heights: [3, 1, 1, 1, 1, 1], maxH: 3 },
  { label: "1 at full height", points: POOL, S: 6, conf: 90, heights: [3, 0, 0, 0, 0, 0], maxH: 3 },
  { label: "1 partial",        points: POOL, S: 6, conf: 90, heights: [2, 0, 0, 0, 0, 0], maxH: 3 },
  { label: "none active",      points: POOL, S: 6, conf: 90, heights: [0, 0, 0, 0, 0, 0], maxH: 3 },
];

const surr = [
  { alpha: [12, 7, 3], h: [3, 0, 0], max: 3, S: 100 },
  { alpha: [12, 7, 3], h: [3, 3, 0], max: 3, S: 100 },
  { alpha: [12, 7, 3], h: [3, 3, 3], max: 3, S: 12 },
  { alpha: [12, 7, 3], h: [1.5, 2, 0.5], max: 3, S: 40 },
  { alpha: [12, 7, 3], h: [0, 0, 0], max: 3, S: 100 },
].map((c) => ({ ...c, ts: +predictedReduction(c.alpha, c.h, c.max, c.S).toFixed(10) }));

const py = `
import json, sys, math
sys.path.insert(0, ${JSON.stringify(REPO)})
from pspe.plan import margins, surrogate
SAMPLE = ${JSON.stringify(SAMPLE)}
def resid(n):
    return SAMPLE[:n]
rows=[]
for d in ${JSON.stringify(DELTAS)}:
    for n in ${JSON.stringify(NS)}:
        rr = resid(n)
        pred = [surrogate.saturating(1 + (i % 5) * 0.7, 0.3 + (i % 3) * 1.1, 100.0)
                for i in range(n)]
        q = margins.conformal_quantile([abs(rr[i] - pred[i]) for i in range(n)], d)
        rows.append({"delta": d, "n": n,
                     "band": None if q is None else round(q, 10),
                     "needed": margins.min_calibration(d)})
cond=[]
for c in ${JSON.stringify(COND)}:
    act=[x for x in c["heights"] if x>0]; m=len(act); mh=c["maxH"]
    if m==0 or (m==1 and abs(act[0]-mh)<1e-9):
        cond.append({"band": 0.0})
    elif m==1:
        cond.append({"band": None})
    else:
        resid=[abs(p["measured"]-surrogate.saturating(p["lead"],p["rest"],c["S"]))
               for p in c["points"] if p["n_active"]==m]
        q=margins.conditional_quantile({m: resid}, 1.0-c["conf"]/100.0)[m]
        cond.append({"band": None if q is None else round(q,10)})
sat=[]
for c in ${JSON.stringify(SAT)}:
    sat.append(round(surrogate.saturating(c["lead"], c["rest"], c["S"]), 10))
surr=[]
for c in ${JSON.stringify(surr.map(({ alpha, h, max, S }) => ({ alpha, h, max, S })))}:
    surr.append(round(surrogate.effect(c["alpha"], c["h"], c["max"], c["S"]), 10))
print(json.dumps({"rows": rows, "surr": surr, "sat": sat, "cond": cond}))
`;
const pf = join(outDir, "p.py");
writeFileSync(pf, py);
const ref = JSON.parse(execFileSync(PYTHON, [pf], { cwd: REPO, encoding: "utf8",
  stdio: ["ignore", "pipe", "pipe"] }).trim().split("\n").pop());

let bandBad = 0, needBad = 0, surrBad = 0;
const disagree = [];
ref.rows.forEach((p, i) => {
  const t = tsRows[i];
  const bothNull = p.band === null && t.band === null;
  const bandSame = bothNull || (p.band !== null && t.band !== null && Math.abs(p.band - t.band) < 1e-9);
  const needSame = p.needed === t.needed;
  if (!bandSame) { bandBad++; disagree.push(`band  delta=${p.delta} n=${p.n}: py=${p.band} ts=${t.band}`); }
  if (!needSame) { needBad++; disagree.push(`needed delta=${p.delta}: py=${p.needed} ts=${t.needed}`); }
});
// The CONDITIONAL band, stratum by stratum, against
// margins.conditional_quantile. This is the band the planner now optimises, so
// a drift here changes which plan the portal recommends -- not just the number
// printed beside it.
let condBad = 0;
ref.cond.forEach((row, i) => {
  const c = COND[i];
  const got = bandForPlan(c.points, c.S, c.conf, c.heights, c.maxH);
  const pyAtt = row.band !== null;
  if (got.attainable !== pyAtt) {
    condBad++;
    disagree.push(`conditional n_active=${c.label}: attainable py=${pyAtt} ts=${got.attainable}`);
  } else if (pyAtt && Math.abs(row.band - got.band) > 1e-9) {
    condBad++;
    disagree.push(`conditional n_active=${c.label}: py=${row.band} ts=${got.band}`);
  }
});

let satBad = 0;
ref.sat.forEach((v, i) => {
  if (Math.abs(v - satTs[i]) > 1e-9) {
    satBad++;
    disagree.push(`saturating lead=${SAT[i].lead} rest=${SAT[i].rest} S=${SAT[i].S}: py=${v} ts=${satTs[i]}`);
  }
});
ref.surr.forEach((v, i) => {
  if (Math.abs(v - surr[i].ts) > 1e-9) {
    surrBad++;
    disagree.push(`surrogate case ${i}: py=${v} ts=${surr[i].ts}`);
  }
});

console.log(`conformal band:      ${ref.rows.length - bandBad}/${ref.rows.length} agree`);
console.log(`refusal threshold:   ${ref.rows.length - needBad}/${ref.rows.length} agree`);
console.log(`conditional band:    ${COND.length - condBad}/${COND.length} agree  (per stratum, exactness, refusal)`);
console.log(`saturating form:     ${SAT.length - satBad}/${SAT.length} agree  (rest=0, saturated, tiny S)`);
console.log(`effect(alpha,h,max,S): ${surr.length - surrBad}/${surr.length} agree`);
console.log(`maxConfidencePct(30) = ${maxConfidencePct(30).toFixed(2)}%  (the portal's stated ceiling)`);
console.log(`runsNeeded(90) = ${runsNeeded(90)} (TS) vs min_calibration(0.1) = ${ref.rows.find(r => r.delta === 0.10).needed} (python)`);
if (disagree.length) {
  console.log("\n--- disagreements ---");
  [...new Set(disagree)].slice(0, 20).forEach((d) => console.log("  " + d));
}
const ok = bandBad === 0 && needBad === 0 && surrBad === 0 && satBad === 0 && condBad === 0;
console.log(ok ? "\nPASS — one guarantee, two implementations" : `\nFAIL — ${disagree.length} disagreements`);
process.exit(ok ? 0 : 1);
