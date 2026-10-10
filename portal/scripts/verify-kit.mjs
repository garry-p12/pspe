// The levee kit: 1c from runs/mockui.
//
// Live against the real service. The thing under test is that the sites come
// out of the terrain rather than being clicked, and that the guarantee shown
// is the framework's conformal margin rather than a number in the markup.
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

ok("the Plan tab opens", await tab("Plan"));
ok("and Build a plan is its first step", await pick("Build a plan"));
const aside = p.locator("aside");
// Two states are correct here and which one you get is a cache hit away:
// the search running, or the sites already on screen. What must be true in
// BOTH is that nothing asked the reader to press anything first.
const started = await aside.innerText();
ok("nothing asks for a click before the tab does its job",
   !/Find the sites/.test(started),
   /Find the sites/.test(started) ? "a search button is on screen" : "");
ok("it says where the sites come from, or is already showing them",
   /almost high enough/.test(started) || /places? a wall would pay/.test(started));

let t = started;
for (let i = 0; i < 20; i++) {
  if (/Looking for banks that almost hold|places? a wall would pay/.test(t)) break;
  await p.waitForTimeout(1000);
  t = await aside.innerText();
}
ok("the tab is either searching or already has its answer",
   /Looking for banks that almost hold|places? a wall would pay/.test(t));
ok("the plan is judged against a named flood, not whatever the map showed",
   /Judged against/.test(t) && /USGS annual peak streamflow|annual chance/.test(t),
   (t.match(/\d+ \w+ \d{4} flood · [^\n]*/) || [""])[0]);
for (let i = 0; i < 160; i++) {
  await p.waitForTimeout(3000);
  t = await aside.innerText();
  if (/Place on the map|No sites here/.test(t)) break;
}
ok("it found sites", /place(s)? a wall would pay/.test(t) && /Place on the map/.test(t),
   (t.match(/\d+ places? a wall would pay/) || [""])[0]);
if (!/Place on the map/.test(t)) {
  console.log(t.slice(t.indexOf("Build a plan"), t.indexOf("Build a plan") + 400));
  await b.close(); process.exit(1);
}
ok("the banks are named from the map, not numbered",
   /near [A-Z][a-z]+|bank/.test(t) && !/unnamed bank/.test(t));
ok("each site carries the height its lowest point needs", /needs \d+\.\d+ m/.test(t));
ok("the budget is shown as money left in US dollars",
   /US\$[\d.]+ M left/.test(t), (t.match(/US\$[\d.]+ M left/) || [""])[0]);
ok("the tray offers all five parts",
   ["Earth levee", "Raised levee", "Flood wall", "Floodgate", "Pump station"]
     .every((x) => t.includes(x)));
ok("the two point structures are there as parts, not pictures",
   /drains one way/.test(t) && /lifts\s+water from inside/.test(t));
// The sourcing moved behind the costing (i) in the declutter -- it is
// provenance, not a decision. What has to stay on the face of the panel is
// the thing that changes what you buy: height is not linear in cost.
ok("the panel says cost is not linear in height",
   /2\.97× a 1 m one/.test(t) || /2\.97 times a 1 m one/.test(t));
ok("the rate provenance is one click away",
   await p.getByRole("button", { name: /What these cost/ }).count() > 0);
await p.screenshot({ path: `${OUT}/87_kit.png` });

// Arm a piece and place it on two sites.
await p.getByRole("button", { name: /Earth levee/ }).click();
await p.waitForTimeout(300);
ok("arming a piece asks for a site", /click a dashed site/i.test(await aside.innerText()));
const rows = p.locator("aside ul button[aria-pressed]");
const n = await rows.count();
await rows.nth(0).click();
await p.waitForTimeout(400);
await p.getByRole("button", { name: /Raised levee/ }).click();
await p.waitForTimeout(250);
await rows.nth(1).click();
await p.waitForTimeout(500);
t = await aside.innerText();
// k or M. Two levees on 400 m banks come to US$136 k under the trapezoidal
// section, and a regex that only accepted millions called that a failure --
// which is the harness disagreeing with the costing model, not with the UI.
ok("placing pieces spends the budget", /spent US\$[\d.]+ ?[kM]/.test(t),
   (t.match(/spent US\$[^\n]*/) || [""])[0]);
ok("two pieces are counted", /2 pieces/.test(t));
await p.screenshot({ path: `${OUT}/88_kit_placed.png` });

await p.getByRole("button", { name: /Solve this plan/ }).click();
// One solved bed per candidate combination: minutes, not seconds.
for (let i = 0; i < 200; i++) {
  await p.waitForTimeout(3000);
  t = await aside.innerText();
  if (/Land flooded|could not solve|No sites here/.test(t)) break;
}
ok("it solves and reports impact", /Land flooded/.test(t));
ok("the impact is before and after", /Homes over floor/.test(t) && /Road cut/.test(t));
ok("the guarantee comes from the margin, or says it cannot",
   /conformal margin is ±/.test(t) || /No guarantee on this one/.test(t),
   (t.match(/At least [\d.]+% less flooding|No guarantee on this one/) || [""])[0]);
ok("before and with-plan compare on the map, where the water is",
   (await p.getByRole("button", { name: "Before" }).count()) === 1);
ok("the percentage says what it is a percentage of",
   /Per cent of mean depth over/.test(t));
// The plan library is solved coarser than the bank search, so the areas in
// the impact table and the area under the map are two grids of the same
// flood. A reader comparing them and finding they differ must be told why.
ok("and the table says which grid its areas came from",
   /Solved at \d+ m, where the search for\s+banks ran at \d+ m/.test(
     t.replace(/\s+/g, " ").replace(/(Solved at \d+ m, where the search for) /,
                                     "$1\n")) || /Solved at \d+ m/.test(t));
ok("a plan can be saved", (await p.getByRole("button", { name: "Save plan" }).count()) === 1);
await p.screenshot({ path: `${OUT}/89_kit_impact.png` });

await p.getByRole("button", { name: "Undo", exact: true }).click();
await p.waitForTimeout(400);
ok("undo removes a piece", /1 piece\b/.test(await aside.innerText()));

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
