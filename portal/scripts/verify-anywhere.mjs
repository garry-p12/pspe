import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
const errs = []; p.on("pageerror", (e) => errs.push(String(e).slice(0, 140)));
await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(15000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(3000);
await p.getByRole("button", { name: "Anywhere" }).click();
await p.waitForTimeout(2500);
const box = p.getByPlaceholder(/Search a town|town|place/i);
await box.first().fill("Lismore NSW");
await p.waitForTimeout(5000);
const sug = p.locator("aside button").filter({ hasText: /Lismore/ });
console.log("suggestions:", await sug.count());
if (await sug.count()) {
  await sug.first().click();
  for (let i = 0; i < 40; i++) {
    await p.waitForTimeout(5000);
    const t = await p.locator("aside").innerText();
    if (/km²|Flooded|flooded|Candidate sites/i.test(t)) { console.log("solved at", 5*(i+1), "s"); break; }
  }
  await p.waitForTimeout(8000);
  await p.screenshot({ path: `${OUT}/70_anywhere_town.png` });
  console.log("wrote 70_anywhere_town.png");
  const t = await p.locator("aside").innerText();
  console.log("panel:", t.split("\n").filter(Boolean).slice(0, 12).join(" | "));
}
if (errs.length) console.log("ERRORS:", [...new Set(errs)].slice(0, 3)); else console.log("no page errors");
await b.close();
