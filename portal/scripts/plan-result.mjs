// Capture the plan RESULT with its conformal margin -- the figure both documents
// are built around and the one no existing screenshot shows.
//
// The previous attempt polled a loose regex that matched on the first tick while
// the solver was still running, and screenshotted the options list instead.
// Here: wait specifically for the "at least" guarantee wording, allow 3 minutes,
// and scroll the panel to the bottom so the margin and attribution are visible.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click();
await p.waitForTimeout(5000);

const before = (await p.locator("aside").innerText()).length;
await p.getByRole("button", { name: /Find the best plan/i }).first().click();
console.log("clicked; waiting for the guarantee wording…");

let done = false;
for (let i = 0; i < 36; i++) {                 // up to 3 minutes
  await p.waitForTimeout(5000);
  const t = await p.locator("aside").innerText().catch(() => "");
  if (/at least/i.test(t)) { console.log("guarantee appeared at", 5 * (i + 1), "s"); done = true; break; }
  if (i % 4 === 3) console.log(`  …${5 * (i + 1)}s, panel ${t.length} chars (was ${before})`);
}
if (!done) console.log("no guarantee wording after 180s — capturing whatever is on screen");

await p.locator("aside").evaluate((el) => { el.scrollTop = el.scrollHeight; }).catch(() => {});
await p.waitForTimeout(2500);
await p.screenshot({ path: `${OUT}/45_plan_result.png` });
console.log("wrote 45_plan_result.png");
console.log("PANEL:\n  " + (await p.locator("aside").innerText()).split("\n").filter(Boolean).join("\n  "));
await b.close();
