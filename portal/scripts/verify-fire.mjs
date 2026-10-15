// The wildfire planner: does each tab answer its own question, honestly?
//
// The thing most worth asserting is the evidence split. Observe is live and
// works anywhere; Forecast and Plan are held-out records because the spread
// model needs eleven drivers that are not a feed. A portal that blurred those
// would be claiming a live forecast it cannot make.
import { chromium } from "playwright";

let bad = 0;
const ok = (name, cond, note = "") => {
  console.log(`${cond ? "ok  " : "FAIL"} ${name}${note ? "  " + note : ""}`);
  if (!cond) bad += 1;
};

// Absent data is not a broken portal, and a harness that cannot tell them
// apart is worth less than no harness. The forecast and plan tabs need the
// NDWS records; the observe tab does not. If the records are not on this
// machine, those checks are SKIPPED with a reason rather than failed -- eight
// red lines that all mean "the dataset is not here" read as a regression.
let HAVE_RECORDS = true;
try {
  const r = await fetch("http://localhost:3000/api/fire/records");
  const d = await r.json();
  HAVE_RECORDS = !!d?.ok;
  if (!HAVE_RECORDS) console.log(`skip (no records): ${String(d?.reason).slice(0, 90)}`);
} catch { HAVE_RECORDS = false; }

const skip = (name) => console.log(`skip ${name}  needs the NDWS records`);

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 } });
p.on("pageerror", (e) => { console.log("PAGE ERROR: " + e.message); bad += 1; });
const text = async () => (await p.locator("body").innerText()).replace(/\s+/g, " ");

await p.goto("http://localhost:3000/wildfire", { waitUntil: "domcontentloaded" });
let t = "";
// Wait for the LIST, not the heading. The heading is static and paints
// immediately; the records take about six seconds cold, so asserting on the
// heading tested nothing and reported an empty list as a failure.
for (let i = 0; i < 90; i++) {
  t = await text();
  if (/fire \d+/.test(t)) break;
  await p.waitForTimeout(1000);
}
ok("the wildfire page loads", /Wildfire planner/.test(t));
if (HAVE_RECORDS) ok("it offers real fire records to pick", /fire \d+/.test(t));
else skip("it offers real fire records to pick");
ok("it says which half is live", /live and works anywhere|Observe.*live/i.test(t));

// The point of the rewrite: no reader should meet a metric name they cannot
// act on. These are the ones the panel used to print.
const jargon = ["AUC-PR", "Average precision", "Critical success index",
                "counterfactual", "population-weighted", "held-out"];
const found = jargon.filter((w) => new RegExp(w, "i").test(t));
ok("it uses no metric jargon a reader cannot act on",
   found.length === 0, found.length ? `found: ${found.join(", ")}` : "");

// --- forecast -------------------------------------------------------------
if (!HAVE_RECORDS) {
  for (const n of ["the forecast is checked against what happened",
                   "it reports what it missed and what it got wrong",
                   "it says plainly that this forecaster is not the best available",
                   "the plan is compared against sending the same crews out evenly",
                   "it says nobody can know what would really have happened",
                   "it reports what the plan spent against its budget"]) skip(n);
  console.log(bad ? `\n${bad} failed` : "\nall checks passed (record tabs skipped)");
  await b.close();
  process.exit(bad ? 1 : 0);
}
await p.getByRole("button", { name: /Forecast this fire/ }).click().catch(() => {});
for (let i = 0; i < 120; i++) {
  t = await text();
  if (/Average precision/.test(t)) break;
  await p.waitForTimeout(1000);
}
// The claim, not the metric name. The panel used to say "Average precision
// 0.368", which is a number a reader cannot act on; it now says how much of
// tomorrow's fire it caught.
ok("the forecast is checked against what happened",
   /It got right/.test(t) && /caught \d+% of tomorrow/.test(t));
ok("it reports what it missed and what it got wrong",
   /It warned wrongly/.test(t) && /places that burned/.test(t));
// The claim that would be easiest to leave out, and most flattering to omit.
ok("it says plainly that this forecaster is not the best available",
   /not as good/i.test(t) && /best\s+one anyone has built/i.test(t));

// --- plan -----------------------------------------------------------------
await p.getByRole("button", { name: "plan", exact: true }).click().catch(() => {});
await p.waitForTimeout(600);
await p.getByRole("button", { name: /Plan the breaks/ }).click().catch(() => {});
for (let i = 0; i < 200; i++) {
  t = await text();
  if (/Against spreading evenly/.test(t)) break;
  await p.waitForTimeout(1000);
}
ok("the plan is compared against sending the same crews out evenly",
   /Better than even cover/.test(t) && /same crews, chosen places/.test(t));
ok("it says nobody can know what would really have happened",
   /nobody can know what would really have happened/i.test(t));

const m = t.match(/used ([\d.]+)% of the area · (inside the budget|OVER BUDGET)/);
ok("it reports what the plan spent against its budget", !!m, m ? m[0] : "");

console.log(bad ? `\n${bad} failed` : "\nall checks passed");
await b.close();
process.exit(bad ? 1 : 0);
