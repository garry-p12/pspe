// The Texas regions on the Observe tab: a catastrophic fire, a local one, and
// a quiet week. The third is the point as much as the first two.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const OUT = "/Users/guruprasadparasnis/pspe/docs/images/portal-guide";
mkdirSync(OUT, { recursive: true });
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 }, deviceScaleFactor: 2 });
const text = async () => (await p.locator("body").innerText()).replace(/\s+/g, " ");
await p.goto("http://localhost:3000/wildfire", { waitUntil: "domcontentloaded" });
for (let i = 0; i < 60; i++) { if (/burning now|Smokehouse/.test(await text())) break; await p.waitForTimeout(1000); }
await p.getByRole("button", { name: "observe", exact: true }).click().catch(() => {});
await p.waitForTimeout(600);

const shots = [
  [/Smokehouse Creek/, "33-fire-texas-smokehouse.jpg"],
  [/Bastrop, September/, "34-fire-texas-bastrop.jpg"],
  [/Austin, right now/, "35-fire-texas-austin.jpg"],
];
for (const [name, file] of shots) {
  await p.getByRole("button", { name }).click().catch(() => {});
  for (let i = 0; i < 180; i++) {
    const t = await text();
    if (/cells|nothing alight|No live feed/.test(t)) break;
    await p.waitForTimeout(1000);
  }
  await p.waitForTimeout(1000);
  await p.screenshot({ path: `${OUT}/${file}`, type: "jpeg", quality: 84 });
  console.log("shot " + file);
}
await b.close();
