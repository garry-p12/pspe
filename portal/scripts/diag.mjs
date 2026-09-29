import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
const errs = [], fails = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 160)));
p.on("requestfailed", (r) => fails.push(`${r.url().slice(0, 90)} ${r.failure()?.errorText}`));
p.on("response", (r) => { if (r.status() >= 400) fails.push(`${r.status()} ${r.url().slice(0, 90)}`); });
await p.goto("http://localhost:3000/", { waitUntil: "networkidle", timeout: 60000 }).catch(e => console.log("goto:", String(e).slice(0,80)));
await p.waitForTimeout(20000);
const btns = async () => [...new Set((await p.locator("button").allInnerTexts()).map(t => t.trim()).filter(Boolean))];
console.log("BUTTONS@load:", (await btns()).join(" | "));
const plan = p.getByRole("button", { name: "Plan", exact: true });
console.log("Plan tab:", await plan.count());
if (await plan.count()) {
  await plan.first().click();
  await p.waitForTimeout(8000);
  console.log("BUTTONS@plan:", (await btns()).slice(0, 18).join(" | "));
  await p.screenshot({ path: `${OUT}/36_diag_plan.png` });
  console.log("wrote 36_diag_plan.png");
}
if (errs.length) console.log("PAGE ERRORS:", [...new Set(errs)].slice(0, 4));
if (fails.length) console.log("NET FAILURES:", [...new Set(fails)].slice(0, 6));
await b.close();
