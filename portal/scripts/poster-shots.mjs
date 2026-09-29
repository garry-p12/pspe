// Fresh captures for docs/POSTER.md and docs/WORKSHOP_PAPER.md.
//
// Two gotchas cost several attempts, both recorded here so the next person
// does not repeat them:
//   1. The section tabs are <button role="tab">. An explicit role overrides the
//      implicit one, so getByRole("button", {name:"Plan"}) matches NOTHING while
//      the element's innerText is still "Plan". Use getByRole("tab").
//   2. The Sentinel-1 control is disabled={busy || !bounds}; it needs the
//      scenario loaded before it will click.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
const errs = [], fails = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 160)));
p.on("response", (r) => { if (r.status() >= 400) fails.push(`${r.status()} ${r.url().slice(0, 80)}`); });
const shot = async (n) => { await p.screenshot({ path: `${OUT}/${n}.png` }); console.log("wrote", n); };
const panel = async () => (await p.locator("aside").innerText().catch(() => "")).split("\n").filter(Boolean);
const step = async (name, fn) => { try { await fn(); } catch (e) { console.log(`SKIP ${name}:`, String(e).split("\n")[0].slice(0, 110)); } };

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await shot("40_forecast");

await step("observe", async () => {
  const btn = p.getByRole("button", { name: /Check the latest Sentinel-1 pass/i });
  if (await btn.first().isEnabled()) {
    await btn.first().click();
    for (let i = 0; i < 20; i++) {
      await p.waitForTimeout(5000);
      if (/km²|observed|no pass|Observed/i.test((await panel()).join(" "))) break;
    }
    await p.waitForTimeout(2000);
    await shot("41_observe");
    console.log("OBSERVE:", (await panel()).slice(0, 12).join(" | "));
  } else console.log("observe: button still disabled (needs bounds)");
});

await step("plan", async () => {
  await p.getByRole("tab", { name: "Plan" }).click();
  await p.waitForTimeout(5000);
  await shot("42_plan_options");
  console.log("OPTIONS:", (await panel()).slice(0, 22).join(" | "));

  const find = p.getByRole("button", { name: /Find the best plan/i });
  console.log("find-best:", await find.count());
  if (await find.count()) {
    await find.first().click();
    for (let i = 0; i < 30; i++) {
      await p.waitForTimeout(4000);
      if (/at least|nine times|guarantee|Recommended/i.test((await panel()).join(" "))) {
        console.log("result at", 4 * (i + 1), "s"); break;
      }
    }
    await p.waitForTimeout(3000);
    await shot("43_plan_result");
    console.log("RESULT:\n  " + (await panel()).slice(0, 28).join("\n  "));
  }
});

await step("roads", async () => {
  await p.getByRole("tab", { name: "Roads" }).click();
  await p.waitForTimeout(9000);
  await shot("44_roads");
});

if (errs.length) console.log("PAGE ERRORS:", [...new Set(errs)].slice(0, 3));
if (fails.length) console.log("NET:", [...new Set(fails)].slice(0, 4));
await b.close();
