// Wireframe 1a: the live column, the roads-out list and the hydrograph.
//
// Live against the real service. What is under test is that the things on
// screen are observations with their provenance attached, and that the two
// claims the wireframe makes which we could not originally support -- a named
// flood category and a hydrograph against flood levels -- are now sourced.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
let fails = 0;
const ok = (n, c, extra = "") => { if (!c) fails++; console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`); };

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1700, height: 1250 }, deviceScaleFactor: 2 });
p.on("pageerror", (e) => { console.log("PAGE ERROR:", e.message); fails++; });
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.getByRole("button", { name: "Austin, Texas", exact: true })
  .waitFor({ state: "visible", timeout: 120000 });

const aside = p.locator("aside");
let t = "";
for (let i = 0; i < 60; i++) {
  await p.waitForTimeout(2000);
  t = await aside.innerText();
  if (/Layers/.test(t) && /reads/.test(t)) break;
}

// --- the header -----------------------------------------------------------
const head = await p.locator("header").innerText();
ok("the header says what is live and how often", /LIVE/.test(head) && /every 15 min/.test(head),
   head.replace(/\n/g, " ").slice(0, 70));

// --- the gauge headline ---------------------------------------------------
ok("it reports an instrument, not a model", /now · observed/.test(t));
ok("it names the river's direction", /River (rising|falling|steady)/.test(t),
   (t.match(/River (rising|falling|steady)/) || [""])[0]);
ok("it reads a stage in metres", /reads \d+\.\d{2} m/.test(t),
   (t.match(/reads [\d.]+ m[^\n]*/) || [""])[0].slice(0, 60));
// The wireframe asks for "above the minor flood level". That is an NWS
// category and it is now fetched, so the panel must either use one or say
// why it cannot for this gauge.
ok("it places the reading against the weather service's levels, or says it cannot",
   /(above|Below) the (action|minor|moderate|MAJOR)/.test(t)
   || /publishes no flood levels for this gauge/.test(t),
   (t.match(/(above|Below) the [^.]*\.|publishes no flood levels[^.]*\./) || [""])[0].slice(0, 80));

// --- layers ---------------------------------------------------------------
ok("every layer names its source and cadence",
   /every 15 min · USGS/.test(t) && /every 6 min/.test(t));
ok("a layer with no feed is disabled with the reason, not hidden",
   /no verified public feed wired up yet/.test(t));

// --- roads out ------------------------------------------------------------
ok("the roads-out list is present", /Roads out/.test(t),
   (t.match(/Roads out[^\n]*/) || [""])[0]);
ok("it states the vehicle and the closing depth", /closed over \d+ cm/.test(t),
   (t.match(/closed over \d+ cm/) || [""])[0]);
ok("it is sorted by the worst point, and says so",
   /deepest first/.test(t) || /No named road/.test(t));

// --- the hydrograph -------------------------------------------------------
const hyd = p.locator("svg[data-chart=hydrograph]");
ok("a hydrograph is drawn under the map", (await hyd.count()) === 1);
if (await hyd.count()) {
  const bb = await hyd.boundingBox();
  const ab = await aside.boundingBox();
  ok("and it runs the full width", bb.width > ab.width * 1.5,
     `${Math.round(bb.width)} px vs ${Math.round(ab.width)} px`);
  const label = await hyd.getAttribute("aria-label");
  ok("it is described for a screen reader", /Three days of stage at/.test(label),
     (label || "").slice(0, 80));
  const svgText = await hyd.evaluate((e) => e.textContent ?? "");
  ok("it plots against flood levels where the gauge has them, or says it has none",
     /minor flood level/.test(label) || /minor flood/.test(svgText)
     || /no flood levels published for this gauge/.test(svgText),
     (label || "").slice(-50));
}

// --- the map legend and the street chip ----------------------------------
ok("the depth ramp has a legend", (await p.getByText(/^0 m$/).count()) > 0);
ok("Street sits with the other view chips",
   (await p.getByRole("button", { name: "Zoom to street level" }).count()) === 1);
await p.screenshot({ path: `${OUT}/93_live.png` });

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
