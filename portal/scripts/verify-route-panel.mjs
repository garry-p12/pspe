import { chromium } from "playwright";

/**
 * The route card in the Anywhere view, end to end.
 *
 * Solves a town, opens the build section, and asserts the card is there with
 * one of the answers it is allowed to give. An earlier version of this script
 * polled for the road network and then reported on a stale snapshot, which
 * failed while the application was fine; it now reads the panel once, after
 * waiting, and prints what it saw either way -- a check that can fail without
 * telling you what it saw is worse than no check.
 */

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 } });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));

await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(13000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2000);
await p.getByRole("button", { name: "Anywhere" }).click();
await p.waitForTimeout(2000);
await p.getByPlaceholder(/Search a town/i).first().fill("Lismore NSW");
await p.waitForTimeout(5000);
const sug = p.locator("aside button").filter({ hasText: /Lismore/ });
if (!(await sug.count())) { console.log("FAIL: no geocode suggestion"); await b.close(); process.exit(1); }
await sug.first().click();

let solved = false;
for (let i = 0; i < 30; i++) {
  await p.waitForTimeout(4000);
  if (/km²/.test(await p.locator("aside").innerText())) {
    solved = true; console.log(`solved ~${4 * (i + 1)}s`); break;
  }
}
if (!solved) { console.log("FAIL: solve did not finish"); await b.close(); process.exit(1); }

// The section buttons are "Forecast / Roads / Plan".
await p.locator("aside button").filter({ hasText: /^Plan$/ }).first().click();

// Roads load for this section too, but Overpass takes 20-60 s for a town box.
let txt = "";
for (let i = 0; i < 25; i++) {
  await p.waitForTimeout(4000);
  txt = await p.locator("aside").innerText();
  if (/road segments/.test(txt)) { console.log(`roads ~${4 * (i + 1)}s`); break; }
}
await p.waitForTimeout(2000);
txt = await p.locator("aside").innerText();

const i = txt.indexOf("Getting out");
const card = i < 0 ? "" : txt.slice(i, i + 700).split("\n").filter(Boolean).join(" | ");
console.log(card || "(no card)");
await p.screenshot({ path: "/Users/guruprasadparasnis/pspe/runs/portal_shots/90_route.png" });

// It must say something, and the something has to be an answer: a time, or
// that there is no way out, or that the network has not arrived.
const answered = /\d+\s*min|No way out by road|Open the roads view first|No trip to assess/.test(card);
console.log(errs.length ? "ERRORS: " + [...new Set(errs)].slice(0, 3).join(" ;; ") : "no page errors");
console.log(i >= 0 && answered && errs.length === 0 ? "PASS" : "FAIL");
await b.close();
process.exit(i >= 0 && answered && errs.length === 0 ? 0 : 1);
