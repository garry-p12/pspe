// Re-capture the figures the two documents cite, against the interface as it
// now is.
//
// Why this exists: 40_forecast, 44_roads and 45_plan_result were captured on
// 2026-09-28, before the per-option water, the 3D water body, the confidence
// control and the route card went in -- and 45_plan_result is the workshop
// paper's "single most important figure". A paper that cites a screenshot of a
// superseded interface is making a claim about a tool that no longer exists.
//
// Two gotchas inherited from poster-shots.mjs, both still true:
//   1. Section tabs are <button role="tab">; an explicit role overrides the
//      implicit one, so getByRole("button", {name:"Plan"}) matches nothing.
//   2. The Sentinel-1 control is disabled until the scenario has bounds.
import { chromium } from "playwright";

const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 160)));
const shot = async (n) => { await p.screenshot({ path: `${OUT}/${n}.png` }); console.log("wrote", n); };
const text = async () => (await p.locator("aside").innerText().catch(() => "")) ;
const step = async (name, fn) => {
  try { await fn(); } catch (e) { console.log(`SKIP ${name}:`, String(e).split("\n")[0].slice(0, 110)); }
};

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(15000);
await shot("40_forecast");

await step("plan section", async () => {
  await p.getByRole("tab", { name: "Plan" }).click();
  await p.waitForTimeout(4000);
  // The district ships its options, so this needs no solve.
  const row = p.locator("aside ul li button[aria-pressed]");
  for (let i = 0; i < 10 && (await row.count()) === 0; i++) await p.waitForTimeout(1500);
  if (await row.count()) { await row.first().click(); await p.waitForTimeout(3500); }
  await shot("45_plan_result");
  console.log("PLAN:", (await text()).split("\n").filter(Boolean).slice(0, 10).join(" | "));
});

await step("3D water and levee walls", async () => {
  // The body of water is on by default; the flat paint is the fallback. Shoot
  // both so the figure can show the difference the volume makes.
  const solid = p.locator("button").filter({ hasText: /^(Solid|Water|3D)/i });
  await shot("60_water_solid");
  if (await solid.count()) {
    await solid.first().click(); await p.waitForTimeout(2500);
    await shot("62_water_flat");
    await solid.first().click(); await p.waitForTimeout(2000);
  }
});

await step("confidence control", async () => {
  const sl = p.locator("aside input[type=range]");
  const n = await sl.count();
  console.log("range inputs in panel:", n);
  if (n) {
    // Push confidence past what 30 held-out runs can support: the tool must
    // refuse rather than extrapolate. That refusal is the paper's argument.
    const last = sl.nth(n - 1);
    await last.evaluate((el) => {
      el.value = el.max;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await p.waitForTimeout(2500);
    await shot("51_certainty");
    const t = await text();
    const m = t.match(/[^\n]*(refus|cannot|not attainable|more runs|96\.8)[^\n]*/i);
    console.log("REFUSAL LINE:", m ? m[0].trim().slice(0, 150) : "(none found)");
  }
});

await step("budget trace", async () => {
  const t = await text();
  if (/trace|step|allocat/i.test(t)) { await shot("50_budget_trace"); }
  else console.log("budget trace: no matching copy in the open section");
});

await step("roads", async () => {
  await p.getByRole("tab", { name: "Roads" }).click();
  await p.waitForTimeout(6000);
  for (let i = 0; i < 12; i++) {
    if (/km cut|cut\b|road/i.test(await text())) break;
    await p.waitForTimeout(2500);
  }
  await shot("44_roads");
  console.log("ROADS:", (await text()).split("\n").filter(Boolean).slice(0, 8).join(" | "));
});

await step("getting out, district", async () => {
  await p.getByRole("tab", { name: "Plan" }).click();
  await p.waitForTimeout(3000);
  const t = await text();
  if (t.includes("Getting out")) {
    await shot("91_route_district");
    const i = t.indexOf("Getting out");
    console.log("ROUTE CARD:", t.slice(i, i + 320).split("\n").filter(Boolean).join(" | "));
  } else console.log("getting out: card not present (roads may not have loaded)");
});

console.log(errs.length ? "ERRORS: " + [...new Set(errs)].slice(0, 3).join(" ;; ") : "no page errors");
await b.close();
