// The trip figure, from a run whose box actually contains a detected channel.
//
// The obvious pair does not work and that is worth recording. Geocoding
// "Lismore NSW" -> "Coraki NSW" gives a box in which `river_inflow` finds no
// through-channel, so the driver falls back to rainfall and nothing floods --
// which is what the previously committed 95_trip.png shows, and why it was the
// wrong case for the caption. Probed offline before spending a solve:
//
//   Lismore(geo) -> Coraki(geo)   no channel      -> rainfall
//   Lismore(geo) -> Woodburn      west edge, 1 cell (marginal)
//   Hospital     -> Coraki(geo)   north edge, 20 cells, outflow west
//
// so this drives the third. And the detector firing turned out to be necessary
// and not sufficient, exactly as suspected: on that box bankfull comes out at
// 234 m3/s, the storm preset takes it to 585, and 17.8 km of a 1,309 km network
// goes under -- 1.4%, no deadline, a dry route. A correct answer about a flood
// that is not the one anybody is asking about.
//
// So the discharge is now typed in rather than inherited from the terrain, which
// is a control the trip view did not have and should: without it the view can
// only model the flood the DEM implies. 3,000 m3/s is the order of the 2022
// Wilsons River event and is STATED here rather than estimated.
//
// The script reports the driver note, the fallback, the brief, the deadline and
// the cut length, and dumps the panel when extraction fails -- which the first
// version did not, leaving a NOT USABLE verdict with no evidence attached.
import { chromium } from "playwright";

const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 180)));
const text = async () => await p.locator("aside").innerText();

await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2000);
await p.getByRole("button", { name: "Plan a trip" }).click();
await p.waitForTimeout(1500);

async function pick(idx, query, match) {
  const box = p.locator("aside input").nth(idx);
  await box.fill(query);
  for (let i = 0; i < 10; i++) {
    await p.waitForTimeout(1600);
    const hit = p.locator("aside button").filter({ hasText: match });
    if (await hit.count()) { await hit.first().click(); return true; }
  }
  return false;
}

if (!await pick(0, "Lismore Base Hospital", /Lismore Base Hospital/)) {
  console.log("FAIL: origin not geocoded"); await b.close(); process.exit(1);
}
await p.waitForTimeout(1200);
if (!await pick(1, "Coraki NSW", /Coraki/)) {
  console.log("FAIL: destination not geocoded"); await b.close(); process.exit(1);
}
await p.waitForTimeout(1000);

// State the design discharge. Without this the view inherits a 234 m3/s
// bankfull estimate and models a flood two orders below the event.
const q = p.locator("aside input#trip-q");
if (await q.count()) {
  await q.fill("3000");
  console.log("discharge: 3000 m3/s (stated, not estimated)");
} else {
  console.log("WARNING: no discharge field; the run will inherit the estimate");
}
await p.waitForTimeout(600);
await p.locator("aside button").filter({ hasText: /Find a way through/ }).first().click();

let txt = "";
for (let i = 0; i < 70; i++) {
  await p.waitForTimeout(5000);
  txt = await text();
  if (/What to do|No written summary/.test(txt)) { console.log(`answered ~${5*(i+1)}s`); break; }
  if (/could not plan|not reachable/i.test(txt)) break;
}
await p.waitForTimeout(3500);
txt = await text();

await p.screenshot({ path: `${OUT}/96_trip_river.png` });
console.log("wrote 96_trip_river");

const grab = (re) => { const m = txt.match(re); return m ? m[0].trim() : null; };
const driver   = grab(/[\d,]+ m3\/s on the [a-z]+ edge[^\n]*/);
const fallback = /applied as rainfall instead/.test(txt);
const deadline = grab(/How long this stays possible[\s\S]{0,40}/);
const cutline  = grab(/Up to [\d.]+ km[^\n]*/);
const brief    = grab(/(Leave now|Leaving now|No route)[^\n]*/);

console.log("DRIVER   :", driver ?? "(none reported)");
console.log("FALLBACK :", fallback ? "YES — rainfall, wrong case again" : "no, river driver held");
console.log("BRIEF    :", brief ?? "(none)");
console.log("DEADLINE :", deadline ? deadline.replace(/\s+/g, " ") : "(none shown)");
console.log("CUT      :", cutline ?? "(none)");
console.log(errs.length ? "ERRORS: " + [...new Set(errs)].slice(0, 3).join(" ;; ") : "no page errors");

// A figure is usable only if the flood it shows is a flood: a correct answer
// about 1.4% of a network is not what the caption claims.
const cutPeak = Number((txt.match(/Up to ([\d.]+) km/) ?? [])[1] ?? 0);
const net = Number((txt.match(/of the ([\d,]+) km network/) ?? [])[1]?.replace(/,/g, "") ?? 0);
const frac = net ? cutPeak / net : 0;
console.log(`CUT FRACTION: ${(100 * frac).toFixed(1)}% of the network at peak`);
const usable = !fallback && !!brief && frac > 0.05;
if (!usable) {
  console.log("--- panel, for evidence ---");
  console.log(txt.split("\n").filter(Boolean).slice(0, 30).join(" | ").slice(0, 900));
}
console.log(usable ? "USABLE as the trip figure" : "NOT USABLE — see above");
await b.close();
