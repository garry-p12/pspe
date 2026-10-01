import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 160)));
await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(16000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(6000);
console.log("water-depth toggle:", await p.getByRole("button", { name: "Water depth" }).count());
await p.screenshot({ path: `${OUT}/60_water_solid.png` });
console.log("wrote 60_water_solid.png");

// Pick the best plan so levee walls appear.
await p.getByRole("button", { name: /Find the best plan/i }).first().click();
for (let i = 0; i < 25; i++) {
  await p.waitForTimeout(4000);
  if (/stand behind/i.test(await p.locator("aside").innerText())) break;
}
await p.waitForTimeout(3000);
await p.screenshot({ path: `${OUT}/61_levee_walls.png` });
console.log("wrote 61_levee_walls.png");

await p.getByRole("button", { name: "Water depth" }).click();
await p.waitForTimeout(3000);
await p.screenshot({ path: `${OUT}/62_water_flat.png` });
console.log("wrote 62_water_flat.png");
if (errs.length) console.log("ERRORS:", [...new Set(errs)].slice(0, 3));
else console.log("no page errors");
await b.close();
