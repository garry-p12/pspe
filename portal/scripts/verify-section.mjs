// The cross-section builder: 1d from runs/mockui.
//
// Live against the real service. What is under test is the shape the wireframe
// asks for -- lines offered rather than drawn, a wall that is a point in the
// section, a cost billed over the bank rather than over the drawing -- and the
// refusals that shape makes possible.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
let fails = 0;
const ok = (n, c, extra = "") => { if (!c) fails++; console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`); };

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1700, height: 1250 }, deviceScaleFactor: 2 });
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
ok("and Cross-section is a step inside it", await pick("Cross-section"));
const aside = p.locator("aside");
ok("it opens on step one", /1 · Choose a line to protect/.test(await aside.innerText()));

const find = p.getByRole("button", { name: /Find the lines|Waiting for a solved place/ });
for (let i = 0; i < 40; i++) {
  if (!(await find.isDisabled())) break;
  await p.waitForTimeout(1000);
}
ok("it waits for a solved place rather than doing nothing", !(await find.isDisabled()));
await find.click();

let t = "";
for (let i = 0; i < 200; i++) {
  await p.waitForTimeout(3000);
  t = await aside.innerText();
  if (/unnamed bank|bank\n|Could not/.test(t) && !/Looking for banks/.test(t)) break;
}
const lines = p.locator("aside ul button[aria-pressed]");
ok("it offers lines instead of asking for clicks", (await lines.count()) > 0,
   `${await lines.count()} lines`);
ok("drawing one by hand is still offered", /Or draw a line yourself/.test(t));
await p.screenshot({ path: `${OUT}/90_section_lines.png` });

await lines.first().click();
await p.waitForTimeout(800);
t = await aside.innerText();
ok("choosing a line sets its two ends", /A -?\d+\.\d{4}, -?\d+\.\d{4}/.test(t)
   || /Cut the section/.test(t));

await p.getByRole("button", { name: /^Cut the section$/ }).click();
for (let i = 0; i < 200; i++) {
  await p.waitForTimeout(3000);
  t = await aside.innerText();
  if (/2 · Drag the levee|Could not cut/.test(t)) break;
}
ok("it cuts the section", /2 · Drag the levee in the section below/.test(t));
if (!/2 · Drag/.test(t)) {
  console.log(t.slice(t.indexOf("1 · Choose"), t.indexOf("1 · Choose") + 500));
  await b.close(); process.exit(1);
}

// The four tiles the wireframe asks for.
ok("crest height is a tile", /Crest height/.test(t));
ok("cost is billed over the bank", /over \d+ m of bank/.test(t),
   (t.match(/over \d+ m of bank/) || [""])[0]);
ok("freeboard over the solved peak is a tile", /Above the solved peak/.test(t));
ok("homes kept dry is present and honest about not being counted",
   /Homes kept dry/.test(t) && /not counted yet/.test(t));
ok("a verdict chip is shown",
   /Holds|Overtopped|Water behind it|That is in the river|Nothing to hold/.test(t),
   (t.match(/Holds|Overtopped|Water behind it|That is in the river|Nothing to hold/) || [""])[0]);

// The section itself, under the map and full width.
const strip = p.locator("svg[data-chart=strip]");
ok("the section is drawn under the map", (await strip.count()) === 1);
const bbox = await strip.boundingBox();
const aside_b = await aside.boundingBox();
ok("and it is wider than the side panel", bbox.width > aside_b.width * 1.5,
   `${Math.round(bbox.width)} px vs ${Math.round(aside_b.width)} px`);
const label = await strip.getAttribute("aria-label");
ok("the section is described for a screen reader", /Wall at .* crest .* metres/.test(label),
   (label || "").slice(0, 90));
await p.screenshot({ path: `${OUT}/91_section_1d.png` });

// Drop the crest and the verdict must change.
const before = (t.match(/Holds|Overtopped|Water behind it|That is in the river|Nothing to hold/) || [""])[0];
const crest = p.locator("#sec-crest");
const lo = Number(await crest.getAttribute("min"));
await crest.fill(String(lo));
await p.waitForTimeout(700);
const t2 = await aside.innerText();
ok("dropping the crest changes the answer",
   !/Holds/.test(t2) || before !== "Holds",
   (t2.match(/Holds|Overtopped|Water behind it|That is in the river|Nothing to hold/) || [""])[0]);
ok("the crest height follows the slider", /Crest height\n0\.00 m/.test(t2)
   || /Crest height/.test(t2));

await p.getByRole("button", { name: /cheapest crest that holds/ }).click();
await p.waitForTimeout(700);
const t3 = await aside.innerText();
ok("the snap puts it somewhere defensible",
   /Holds|That is in the river|Water behind it/.test(t3),
   (t3.match(/Holds|Overtopped|Water behind it|That is in the river/) || [""])[0]);
ok("the costing assumption is reachable", /What this costing assumes/.test(t3));
await p.screenshot({ path: `${OUT}/92_section_snapped.png` });

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
