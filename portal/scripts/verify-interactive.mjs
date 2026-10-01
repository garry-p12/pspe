// Verify the five interactive additions render and respond.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 140)));
p.on("response", (r) => { if (r.status() >= 500) errs.push(`${r.status()} ${r.url().slice(0,70)}`); });
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);

const text = async () => (await p.locator("aside").innerText().catch(() => ""));
console.log("FORECAST panel mentions river confidence:",
  /know the river/i.test(await text()));
console.log("quiet-day pointer:", /Nothing much coming/i.test(await text()));
console.log("observe button label:", /satellite saw|Loading the map/i.test(await text()));

await p.getByRole("tab", { name: "Plan" }).click();
await p.waitForTimeout(5000);
const t2 = await text();
console.log("budget trace present:", /What more money buys/i.test(t2));
console.log("knee callout:", /goes\s+flat|stops buying/i.test(t2));
await p.screenshot({ path: `${OUT}/50_budget_trace.png` });

await p.getByRole("button", { name: /Find the best plan/i }).first().click();
for (let i = 0; i < 30; i++) {
  await p.waitForTimeout(4000);
  if (/stand behind/i.test(await text())) break;
}
const t3 = await text();
console.log("\ncertainty control:", /How sure do you need to be/i.test(t3));
console.log("plain-language level:", /times in/i.test(t3));
console.log("harmful block:", /make it worse/i.test(t3));
await p.locator("aside").evaluate((el) => { el.scrollTop = el.scrollHeight; }).catch(() => {});
await p.waitForTimeout(1500);
await p.screenshot({ path: `${OUT}/51_certainty.png` });
console.log("\nPANEL TAIL:\n  " + t3.split("\n").filter(Boolean).slice(-16).join("\n  "));
if (errs.length) console.log("\nERRORS:", [...new Set(errs)].slice(0, 4));
await b.close();
