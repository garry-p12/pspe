import { chromium } from "playwright";

/**
 * The route card on the district, where the flood is the 2022 event.
 *
 * No solve and no Overpass call: the network, the depth grid and the settlement
 * all ship with the page, so this is the fast check that the router, the card
 * and the drawn path work. Under the reference field 213 km of the 287 km
 * network is over 25 cm, so the expected answer at car depth is that there is
 * no way out -- and that is an answer, not a failure.
 */

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 } });
const errs = []; p.on("pageerror", (e) => errs.push(String(e).slice(0, 180)));
await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2500);
for (const n of [/what do we build/i, /build/i, /plan/i]) {
  const btn = p.locator("aside button").filter({ hasText: n });
  if (await btn.count()) { await btn.first().click(); break; }
}
let txt = "";
for (let i = 0; i < 12; i++) {
  await p.waitForTimeout(3000);
  txt = await p.locator("aside").innerText();
  if (/road segments/.test(txt)) break;
}
const i = txt.indexOf("Getting out");
console.log(i < 0 ? "FAIL: no card" : txt.slice(i, i + 600).split("\n").filter(Boolean).join(" | "));

// And at a depth a truck can cross, where a route should exist.
const truck = p.locator("aside button").filter({ hasText: /4WD|truck/i });
if (await truck.count()) {
  await truck.first().click();
  await p.waitForTimeout(2500);
  const t2 = await p.locator("aside").innerText();
  const j = t2.indexOf("Getting out");
  console.log("--- at 4WD depth ---");
  console.log(t2.slice(j, j + 600).split("\n").filter(Boolean).join(" | "));
}
await p.screenshot({ path: "/Users/guruprasadparasnis/pspe/runs/portal_shots/91_route_district.png" });
console.log(errs.length ? "ERRORS: " + [...new Set(errs)].slice(0, 3).join(" ;; ") : "no page errors");
await b.close();
