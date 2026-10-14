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
ok("it offers real fire records to pick", /fire \d+/.test(t));
ok("it says which half is live", /live and works anywhere|Observe.*live/i.test(t));

// --- forecast -------------------------------------------------------------
await p.getByRole("button", { name: /Forecast this fire/ }).click().catch(() => {});
for (let i = 0; i < 120; i++) {
  t = await text();
  if (/Average precision/.test(t)) break;
  await p.waitForTimeout(1000);
}
ok("the forecast is scored, not just drawn", /Average precision/.test(t));
ok("it reports misses as well as hits", /Missed/.test(t) && /False alarms/.test(t));
// The claim that would be easiest to leave out, and most flattering to omit.
ok("it states that the forecaster is behind the field",
   /behind the field/i.test(t));

// --- plan -----------------------------------------------------------------
await p.getByRole("button", { name: "plan", exact: true }).click().catch(() => {});
await p.waitForTimeout(600);
await p.getByRole("button", { name: /Plan the breaks/ }).click().catch(() => {});
for (let i = 0; i < 200; i++) {
  t = await text();
  if (/Against spreading evenly/.test(t)) break;
  await p.waitForTimeout(1000);
}
ok("the plan is compared against spreading evenly, not only against nothing",
   /Against spreading evenly/.test(t));
ok("it says the counterfactual is unobservable",
   /counterfactual cannot be observed/i.test(t));

const m = t.match(/spent ([\d.]+)% of the patch · (within budget|OVER BUDGET)/);
ok("it reports what the plan spent against its budget", !!m, m ? m[0] : "");

console.log(bad ? `\n${bad} failed` : "\nall checks passed");
await b.close();
process.exit(bad ? 1 : 0);
