import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 950 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));
await p.goto("http://localhost:3100/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(6000);
await p.getByRole("button", { name: "Anywhere" }).click();
await p.waitForTimeout(800);
await p.getByPlaceholder(/Search a town/).fill("Nashville Tennessee");
await p.waitForTimeout(3500);
const opts = await p.locator("aside button").filter({ hasText: /Nashville/ }).count();
console.log("geocode suggestions:", opts);
if (opts > 0) {
  await p.locator("aside button").filter({ hasText: /Nashville/ }).first().click();
  // Solve takes about a minute.
  await p.waitForTimeout(95000);
}
await p.screenshot({ path: "/tmp/portal-shots/anywhere.png" });
const txt = await p.locator("aside").innerText();
console.log(txt.split("\n").filter(Boolean).slice(0, 22).join("\n"));
if (errs.length) console.log("ERRORS:", [...new Set(errs)].slice(0, 3));
await b.close();
