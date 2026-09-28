/** Capture the portal states that stand behind the appendix figure.
 *  Each shot is one stage of the loop as an operator actually meets it, so the
 *  figure shows the tool rather than a rendering of the tool. */
import { chromium } from "playwright";

const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(11000);          // terrain tiles + forecast fetch

await p.screenshot({ path: `${OUT}/01_forecast.png` });
console.log("01 forecast");

await p.getByRole("button", { name: "Appraised options" }).click();
await p.waitForTimeout(2500);
await p.screenshot({ path: `${OUT}/02_options.png` });
console.log("02 options");

const best = p.getByRole("button").filter({ hasText: "BEST VALUE" });
if (await best.count()) {
  await best.first().click();
  await p.waitForTimeout(4000);
  await p.screenshot({ path: `${OUT}/03_chosen.png` });
  console.log("03 chosen:", (await p.locator("aside").innerText()).split("\n").filter(Boolean).slice(0, 18).join(" | "));
} else { console.log("!! no BEST VALUE button"); }

await p.getByRole("button", { name: "Build your own" }).click();
await p.waitForTimeout(2500);
await p.screenshot({ path: `${OUT}/04_build.png` });
console.log("04 build:", (await p.locator("aside").innerText()).split("\n").filter(Boolean).slice(0, 14).join(" | "));

if (errs.length) console.log("ERRORS:", [...new Set(errs)].slice(0, 3));
await b.close();
