// The conditional band's three states, rendered.
//
// `bandForPlan` returns one of: exact (a single site at full height, where the
// prediction IS the solver run that measured it), a figure (the stratum carries
// enough runs), or a refusal (it does not). Only the first two have ever been
// seen on screen -- the refusal path was written against the measured fact that
// three-site plans have 8 calibration runs where 9 are needed, and never
// exercised. A UI state that has never rendered is a liability.
//
// The stratum a plan lands in is driven by the BUDGET: a small budget buys one
// site, A$20M buys two, and a larger budget pushes the greedy search to three
// or more, which is where the evidence runs out.
import { chromium } from "playwright";

const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1400 }, deviceScaleFactor: 2 });

// Drive the BUDGET SLIDER rather than the URL. `?budget=` is read once on
// mount, and clicking "find the best plan" immediately after navigation races
// that effect -- the first version of this script did exactly that and got an
// identical 2-site plan at every budget from 8 to 200, which looked like the
// planner saturating rather than the harness failing.
const findPlan = async (budget) => {
  await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
  await p.waitForTimeout(14000);
  await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
  await p.waitForTimeout(3000);
  const slider = p.locator("#budget");
  if (await slider.count()) {
    await slider.fill(String(budget));
    await p.waitForTimeout(1500);
  }
  await p.getByRole("button", { name: /Find the best plan/i }).first().click();
  for (let i = 0; i < 30; i++) {
    await p.waitForTimeout(4000);
    if (/stand behind/i.test(await p.locator("aside").innerText())) return true;
  }
  return false;
};

// Scope every match to the conditional block. Testing the whole aside for the
// word "exact" matched the verify notice -- "a full hydrodynamic run of this
// exact height vector" -- and reported every plan as exact.
const read = async () => {
  const t = await p.locator("aside").innerText();
  const blk = (/Checked against[\s\S]{0,420}/.exec(t) || [""])[0];
  return {
    sites:    (/Checked against (\d+)-site floods only/.exec(blk) || [])[1] ?? null,
    state:    /can't say/.test(blk) ? "REFUSES"
            : /\bexact\b/.test(blk.split("\n").slice(0, 2).join("\n")) ? "exact"
            : /(\d+)%/.test(blk.split("\n").slice(0, 2).join("\n")) ? "figure"
            : "absent",
    needs:    (/need (\d+)\b/.exec(blk) || [])[1] ?? null,
    have:     (/We have (\d+) past flood/.exec(blk) || [])[1] ?? null,
    promises: /A plan we can promise more for/.test(t),
  };
};

let sawRefusal = false, sawExact = false, sawFigure = false, sawAlt = false;
// 8 -> one site at a partial height (refusal), 15 -> one site at full
// height (exact), 20+ -> two sites (figure). Above 60 the UI clamps.
for (const budget of [8, 12, 15, 20, 45]) {
  if (!(await findPlan(budget))) { console.log(`A$${budget}M: no plan returned`); continue; }
  const r = await read();
  console.log(`A$${budget}M  sites=${r.sites ?? "-"}  band=${r.state}` +
              (r.needs ? `  (have ${r.have}, need ${r.needs})` : "") +
              (r.promises ? "  [alt plan offered]" : ""));
  if (r.state === "REFUSES" && !sawRefusal) {
    sawRefusal = true;
    await p.screenshot({ path: `${OUT}/56_conditional_refusal.png` });
  }
  if (r.state === "exact" && !sawExact) {
    sawExact = true;
    await p.screenshot({ path: `${OUT}/57_conditional_exact.png` });
  }
  if (r.state === "figure" && !sawFigure) {
    sawFigure = true;
    await p.screenshot({ path: `${OUT}/58_conditional_figure.png` });
  }
  if (r.promises && !sawAlt) {
    sawAlt = true;
    await p.screenshot({ path: `${OUT}/59_promise_more.png` });
  }
}
console.log("\nstates rendered:",
  { exact: sawExact, figure: sawFigure, refusal: sawRefusal, alternativePlan: sawAlt });
if (!sawRefusal) console.log("WARNING: the refusal path still never rendered.");
await b.close();
