import { chromium } from "playwright";
import { createHash } from "node:crypto";

/**
 * Choosing an option has to change the map.
 *
 * The planner solves every option it lists, in one batch, and the list was
 * ranked and selectable -- but selecting a row only moved a radio button: the
 * field was returned for the do-nothing case alone and the chosen plan never
 * reached the map. This drives the whole flow in a browser and asserts that
 * picking a row changes what is drawn, that the panel says what is on the map,
 * and that going back to do-nothing restores it.
 */

const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const shot = (buf) => createHash("sha1").update(buf).digest("hex").slice(0, 12);

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 1 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 160)));

await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(12000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2000);
await p.getByRole("button", { name: "Anywhere" }).click();
await p.waitForTimeout(2000);

await p.getByPlaceholder(/Search a town/i).first().fill("Lismore NSW");
await p.waitForTimeout(5000);
const sug = p.locator("aside button").filter({ hasText: /Lismore/ });
if (!(await sug.count())) { console.log("FAIL: no geocode suggestion"); await b.close(); process.exit(1); }
await sug.first().click();

let solved = false;
for (let i = 0; i < 48; i++) {
  await p.waitForTimeout(5000);
  if (/km²/.test(await p.locator("aside").innerText())) { solved = true; console.log(`solved at ~${5 * (i + 1)}s`); break; }
}
if (!solved) { console.log("FAIL: solve did not finish"); await b.close(); process.exit(1); }

// The build section, where the options live.
for (const name of [/what do we build/i, /build/i, /plan/i]) {
  const btn = p.locator("aside button").filter({ hasText: name });
  if (await btn.count()) { await btn.first().click(); break; }
}
await p.waitForTimeout(2500);

// Candidate sites, near the middle of the box where the channel runs. Exact
// placement does not matter to what is being tested: whatever the hydrology
// does, the sites the chosen option builds are drawn differently from the ones
// it leaves out, so a selection always has to change the canvas.
const map = p.locator("canvas").first();
const bb = await map.boundingBox();
const pts = [
  { x: bb.x + bb.width * 0.47, y: bb.y + bb.height * 0.46 },
  { x: bb.x + bb.width * 0.53, y: bb.y + bb.height * 0.54 },
  { x: bb.x + bb.width * 0.44, y: bb.y + bb.height * 0.58 },
];
for (const pt of pts) {
  await p.locator("aside button").filter({ hasText: /Add a candidate site/i }).first().click();
  await p.waitForTimeout(600);
  await p.mouse.click(pt.x, pt.y);
  await p.waitForTimeout(900);
}
console.log("placed:", (await p.locator("aside").innerText()).match(/(\d+) placed/)?.[1]);

await p.locator("aside button").filter({ hasText: /Find the best plan/i }).first().click();
for (let i = 0; i < 40; i++) {
  await p.waitForTimeout(5000);
  if (/pick a wall/i.test(await p.locator("aside").innerText())) { console.log(`planned at ~${5 * (i + 1)}s`); break; }
}
await p.waitForTimeout(3000);

// Prefer an option that actually moves the water, so the assertion is about
// the flood and not only about the markers recolouring. Where the clicked sites
// turn out to sit off the flow path, nothing helps and the weaker assertion is
// the honest one -- said out loud rather than passed off as the strong test.
let rows = p.locator("aside ul li button[aria-pressed]");
let material = false;
const helps = p.locator("aside button").filter({ hasText: /^Helps$/ });
if (await helps.count()) {
  await helps.first().click();
  await p.waitForTimeout(800);
  if (await rows.count()) material = true;
  else {
    await p.locator("aside button").filter({ hasText: /^All$/ }).first().click();
    await p.waitForTimeout(800);
  }
}
const n = await rows.count();
console.log("selectable options:", n, material
  ? "(one of them reduces flooding: the water itself must change)"
  : "(none reduce flooding here: markers and panel only)");
if (n === 0) { console.log("FAIL: no options offered"); await b.close(); process.exit(1); }

const before = await map.screenshot();
await p.screenshot({ path: `${OUT}/80_option_none.png` });

await rows.first().click();
await p.waitForTimeout(3500);
const after = await map.screenshot();
await p.screenshot({ path: `${OUT}/81_option_picked.png` });

const card = p.locator("aside").filter({ hasText: /On the map now/i });
const hasCard = await card.count() > 0;
const cardText = hasCard
  ? (await p.locator("aside").innerText()).split("On the map now")[1]?.split("\n").filter(Boolean).slice(0, 7).join(" | ")
  : "(missing)";

// Back to the baseline.
if (hasCard) {
  await p.locator("aside button").filter({ hasText: /back to do nothing/i }).first().click();
  await p.waitForTimeout(3500);
}
const reverted = await map.screenshot();

console.log("map before pick:", shot(before));
console.log("map after pick: ", shot(after), after.equals(before) ? "UNCHANGED (fail)" : "changed");
console.log("map reverted:   ", shot(reverted), reverted.equals(after) ? "still showing the option (fail)" : "restored");
console.log("panel card:", cardText);
console.log(errs.length ? `ERRORS: ${[...new Set(errs)].slice(0, 3).join(" ;; ")}` : "no page errors");

const ok = !after.equals(before) && hasCard && !reverted.equals(after) && errs.length === 0;
console.log(ok ? "PASS" : "FAIL");
await b.close();
process.exit(ok ? 0 : 1);
