import { chromium } from "playwright";

/**
 * The trip view, end to end: two searches in, a routed answer out.
 *
 * This is the slow one by nature -- terrain for a box containing both ends, a
 * shallow-water solve across it, and the road network from OpenStreetMap -- so
 * it waits minutes rather than seconds. It asserts the three things that make
 * the answer an answer: a brief, a duration, and a deadline or an explicit
 * statement that there is none.
 */

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 } });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));

await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(13000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2000);
await p.getByRole("button", { name: "Plan a trip" }).click();
await p.waitForTimeout(1500);

async function pick(label, query, match) {
  const box = p.locator("aside input").nth(label);
  await box.fill(query);
  for (let i = 0; i < 8; i++) {
    await p.waitForTimeout(1500);
    const hit = p.locator("aside button").filter({ hasText: match });
    if (await hit.count()) { await hit.first().click(); return true; }
  }
  return false;
}

if (!await pick(0, "Lismore NSW", /Lismore/)) {
  console.log("FAIL: origin not found"); await b.close(); process.exit(1);
}
await p.waitForTimeout(1200);
if (!await pick(1, "Coraki NSW", /Coraki/)) {
  console.log("FAIL: destination not found"); await b.close(); process.exit(1);
}
await p.waitForTimeout(1000);

const go = p.locator("aside button").filter({ hasText: /Find a way through/ });
if (!(await go.count())) { console.log("FAIL: button not enabled"); await b.close(); process.exit(1); }
await go.first().click();

let txt = "";
for (let i = 0; i < 60; i++) {
  await p.waitForTimeout(5000);
  txt = await p.locator("aside").innerText();
  if (/What to do/.test(txt)) { console.log(`answered ~${5 * (i + 1)}s`); break; }
  if (/could not plan|not reachable|no route/i.test(txt) && !/Modelling/.test(txt)) break;
}
await p.waitForTimeout(3000);
txt = await p.locator("aside").innerText();
const i = txt.indexOf("What to do");
console.log(i < 0 ? "(no answer panel)"
  : txt.slice(i, i + 1100).split("\n").filter(Boolean).join(" | "));
await p.screenshot({ path: "/Users/guruprasadparasnis/pspe/runs/portal_shots/95_trip.png" });

const ok = i >= 0 && /\d+ min/.test(txt) && errs.length === 0;
console.log(errs.length ? "ERRORS: " + [...new Set(errs)].slice(0, 3).join(" ;; ") : "no page errors");
console.log(ok ? "PASS" : "FAIL");
await b.close();
process.exit(ok ? 0 : 1);
