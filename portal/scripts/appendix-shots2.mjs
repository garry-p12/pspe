import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
const errs = []; p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(11000);

await p.getByRole("button", { name: "Appraised options" }).click();
await p.waitForTimeout(1500);
await p.getByRole("button").filter({ hasText: "BEST VALUE" }).first().click();
await p.waitForTimeout(4000);
// Whatever the tool says about the measure it just evaluated.
await p.locator("aside").evaluate((el) => { el.scrollTop = el.scrollHeight; });
await p.waitForTimeout(1200);
await p.screenshot({ path: `${OUT}/05_detail.png` });
const aside = (await p.locator("aside").innerText()).split("\n").filter(Boolean);
console.log("DETAIL:", aside.slice(-26).join(" | "));

// The global path: analyse a place the tool was never configured for.
await p.getByRole("button", { name: "Anywhere" }).click();
await p.waitForTimeout(1200);
await p.getByPlaceholder(/Search a town/).fill("Cedar Rapids Iowa");
await p.waitForTimeout(4000);
const sug = p.locator("aside button").filter({ hasText: /Cedar Rapids/ });
console.log("suggestions:", await sug.count());
if (await sug.count()) {
  await sug.first().click();
  for (let i = 0; i < 40; i++) {
    await p.waitForTimeout(5000);
    const t = await p.locator("aside").innerText();
    if (/km²|Flooded|flooded/.test(t)) { console.log("solved at", 5 * (i + 1), "s"); break; }
  }
  await p.waitForTimeout(12000);
  await p.screenshot({ path: `${OUT}/06_anywhere.png` });
  console.log("ANYWHERE:", (await p.locator("aside").innerText()).split("\n").filter(Boolean).slice(0, 24).join(" | "));
}
if (errs.length) console.log("ERRORS:", [...new Set(errs)].slice(0, 3));
await b.close();
