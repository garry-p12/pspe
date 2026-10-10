// How much text each panel puts in front of a reader.
//
// "Wordy" is an impression until it is counted. This walks the four views
// and reports, for each, the visible character count of the left column,
// how much of it is in the muted explanatory style (`.anno`), and the
// longest single paragraph — which is usually the thing that makes a panel
// feel like a methods section.
//
// Run it before and after a declutter so the claim is measurable.
import { chromium } from "playwright";

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 } });
const aside = p.locator("aside");
const text = async () => (await aside.innerText()).replace(/\s+/g, " ").trim();

async function until(re, label, sec = 300) {
  const t0 = Date.now();
  for (;;) {
    const t = await text();
    if (re.test(t)) return t;
    if (Date.now() - t0 > sec * 1000) { console.log(`  (timeout ${label})`); return t; }
    await p.waitForTimeout(2000);
  }
}
async function press(loc, isOn, tries = 25) {
  await loc.waitFor({ state: "visible", timeout: 120000 });
  for (let i = 0; i < tries; i++) {
    await loc.click({ timeout: 15000 }).catch(() => {});
    await p.waitForTimeout(500);
    if (await isOn()) return true;
  }
  return false;
}

/** Characters, muted-prose share, and the worst paragraph. */
async function weigh(name) {
  const whole = await text();
  const anno = await aside.locator(".anno").allInnerTexts();
  const annoChars = anno.join(" ").replace(/\s+/g, " ").length;
  const paras = await aside.locator("p, li, summary").allInnerTexts();
  const longest = paras
    .map((x) => x.replace(/\s+/g, " ").trim())
    .sort((a, c) => c.length - a.length)[0] ?? "";
  console.log(
    `${name.padEnd(16)} ${String(whole.length).padStart(5)} chars  `
    + `${String(annoChars).padStart(5)} in muted prose (${
      Math.round((100 * annoChars) / Math.max(whole.length, 1))}%)  `
    + `longest ${String(longest.length).padStart(4)}`);
  if (longest.length > 180) console.log(`    "${longest.slice(0, 150)}…"`);
  return { name, chars: whole.length, annoChars, longest: longest.length };
}

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await until(/Water depth/, "the layers panel", 120);
await until(/River (falling|rising|steady)|reads [\d.]+|No gauged stream/,
            "the gauge headline", 240);
const rows = [await weigh("Forecast")];

const roads = p.getByRole("tab", { name: "Roads" });
await press(roads, async () => (await roads.getAttribute("data-on")) === "true");
await until(/Getting out/, "the evacuation card", 240);
rows.push(await weigh("Roads"));

const plan = p.getByRole("tab", { name: "Plan" });
await press(plan, async () => (await plan.getAttribute("data-on")) === "true");
await until(/Judged against/, "the scenario picker", 180);
await until(/places? a wall would pay|No sites here/, "the bank search", 600);
rows.push(await weigh("Plan · kit"));

await p.getByRole("button", { name: "Cross-section", exact: true }).click().catch(() => {});
await until(/Choose a line to protect/, "step one", 120);
rows.push(await weigh("Plan · section"));

const fc = p.getByRole("tab", { name: "Forecast" });
await press(fc, async () => (await fc.getAttribute("data-on")) === "true");
const chip = p.getByRole("button", { name: "Zoom to street level" });
await press(chip, async () => /This street|buildings · ground/.test(await text()));
await until(/Where the water comes from|Open the street view/, "street controls", 120);
rows.push(await weigh("Street"));

const total = rows.reduce((a, r) => a + r.chars, 0);
const anno = rows.reduce((a, r) => a + r.annoChars, 0);
console.log(`\nall five  ${total} chars, ${anno} of it muted prose `
  + `(${Math.round((100 * anno) / total)}%)`);
await b.close();
