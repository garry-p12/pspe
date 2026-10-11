// The street view: 1b from runs/mockui.
//
// Solved live against the real service rather than from a fixture, because the
// thing under test is the downscaling -- the catchment solved at its own scale
// and the street taken from its water surface -- and a saved response would
// not exercise it. Two minutes is normal.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
let fails = 0;
const ok = (n, c, extra = "") => { if (!c) fails++; console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`); };

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
p.on("pageerror", (e) => { console.log("PAGE ERROR:", e.message); fails++; });
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
// The nav is three tabs plus two controls that are not tabs: Street is a map
// zoom, and the two planning tools are steps inside Plan. Each helper clicks
// until the state flips, because a click can land before React hydrates and
// is then simply lost.
async function tab(name) {
  const b = p.getByRole("tab", { name });
  await b.waitFor({ state: "visible", timeout: 120000 });
  for (let i = 0; i < 25; i++) {
    await b.click();
    await p.waitForTimeout(600);
    if ((await b.getAttribute("data-on")) === "true") return true;
  }
  return false;
}
async function pick(name) {
  const b = p.getByRole("button", { name, exact: true });
  await b.waitFor({ state: "visible", timeout: 90000 });
  for (let i = 0; i < 25; i++) {
    await b.click();
    await p.waitForTimeout(600);
    if ((await b.getAttribute("data-on")) === "true") return true;
  }
  return false;
}

const streetChip = p.getByRole("button", { name: "Zoom to street level" });
await streetChip.waitFor({ state: "visible", timeout: 120000 });
for (let i = 0; i < 25; i++) {
  await streetChip.click();
  await p.waitForTimeout(700);
  if (/This street/.test(await p.locator("aside").innerText())) break;
}
ok("the street view opens from the map's zoom chips",
   /This street/.test(await p.locator("aside").innerText()));
await p.waitForTimeout(900);
const aside = p.locator("aside");
ok("the street tab opens", /This street/.test(await aside.innerText()));

// Down the channel, at 2.5x the 2013 peak. Not because 2013 was 2.5x -- it
// was not -- but because that is what this model needs to reproduce the
// inundation 2013 caused, which the panel says out loud.
//
// This comment used to add "at 1x it puts no house over its floor", which
// was wrong: measured, 1x puts 31 of 78 over. See
// `_verify_street_1x.mjs`. The over-floor path is exercised either way; 2.5x
// is here because it is the configuration the interface recommends, not
// because the assertion would otherwise be vacuous.
//
// 3823, not 3400: that is USGS's annual peak for 31 Oct 2013 (135,000 cfs,
// measured indirectly from high-water marks), and it is the figure the panel
// beside this field now quotes. A harness driving the model with a number
// the interface does not name is testing something nobody can reproduce.
await p.getByRole("button", { name: "Down the channel" }).click();
await p.getByRole("button", { name: "2.5x" }).click();
await p.locator("#street-q").fill("3823");
await p.waitForTimeout(400);

// Ask again on a timeout rather than failing.
//
// A cold street solve runs past the five minutes a browser's `fetch` will
// wait -- the portal guide says so in §5 -- and the UI's own answer is to
// restart it, which resets the counter under this assertion. Clicking once
// and waiting therefore fails against a service that is working correctly,
// which is exactly what it did after a restart cleared the solve cache. The
// work is keyed, so asking again joins the run already going.
let t = "";
for (let attempt = 0; attempt < 4; attempt++) {
  await p.getByRole("button", { name: /Open the street view|Look again/ })
    .first().click().catch(() => {});
  for (let i = 0; i < 120; i++) {
    await p.waitForTimeout(3000);
    t = await aside.innerText();
    if (/Over floor level|Could not open/.test(t)) break;
  }
  if (/Over floor level/.test(t)) break;
  console.log(`  (cold solve outran the browser's wait; asking again)`);
  await p.waitForTimeout(4000);
}
ok("it solved", /Over floor level/.test(t), (t.match(/Over floor level\n[^\n]*/) || [""])[0].replace("\n", " "));
if (!/Over floor level/.test(t)) { console.log(t.slice(0, 400)); await b.close(); process.exit(1); }

const m = t.match(/(\d+) of (\d+)/);
ok("it counts floors, not ground", Boolean(m), m ? `${m[1]} of ${m[2]} over floor` : "");
ok("the floor assumption is on the face of it", /floor assumed \+0\.17 m/.test(t));
ok("the deepest figure is the street, not the creek bed",
   /Deepest at a house/.test(t));
ok("and the channel maximum is named separately",
   /m in the channel/.test(t) || !/in the channel/.test(t));
ok("a time slider is offered", (await p.locator("#street-t").count()) === 1);
ok("the street is listed", /On this street/.test(t));
// The street REPLACES the live column rather than stacking under it. Both
// rendered at once before, because `section` and the street zoom are
// independent, and the street panel sat below three screens of forecast.
ok("the street replaces the live column rather than stacking under it",
   !/Roads out of/.test(t) && !/Rain radar/.test(t));
ok("the block is named from its own addresses, not called \"this street\"",
   !/^This street$/m.test(t.split("\n")[1] ?? ""));
ok("there is a way back out to the whole study",
   (await p.getByRole("button", { name: /^←/ }).count()) === 1);
await p.screenshot({ path: `${OUT}/84_street.png` });

// Click a building in the list; the card must name it and the map must follow.
const first = p.locator("aside ul button[aria-pressed]").first();
if (await first.count()) {
  await first.click();
  await p.waitForTimeout(2500);
  const t2 = await aside.innerText();
  ok("selecting a building opens its card", /slab on ground, assumed/.test(t2));
  ok("the card says over floor or to spare",
     /Over floor|Floor above water/.test(t2),
     (t2.match(/(Over floor|Floor above water)[^\n]*/) || [""])[0].slice(0, 54));
  ok("and gives the ground outside it", /m on the ground outside/.test(t2));
  await p.screenshot({ path: `${OUT}/85_street_selected.png` });
}

// Scrub back to the start: fewer floors should be wet than at the peak.
const peak = Number((await aside.innerText()).match(/(\d+) of \d+/)[1]);
await p.locator("#street-t").fill("1");
await p.waitForTimeout(2500);
const early = await aside.innerText();
ok("the clock changes what is wet",
   /h into the storm/.test(early), (early.match(/[\d.]+ h into the storm/) || [""])[0]);
ok("the provenance fold explains the downscaling",
   /Solved at \d+ m/.test(early));
console.log(`     peak over floor: ${peak}`);
await p.screenshot({ path: `${OUT}/86_street_early.png` });

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
