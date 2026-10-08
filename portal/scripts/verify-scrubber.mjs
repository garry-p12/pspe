// The timeline scrubber: does clicking a bar change the water on the map?
//
// The failure that matters is cosmetic success -- the bar highlights, the
// caption updates, and the overlay texture never changes because the swapped
// depth_png never reaches the map. So this compares the rendered canvas, not
// the panel text.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2500);
await p.getByRole("button", { name: /plan a trip/i }).click().catch(() => {});
await p.waitForTimeout(2500);

// Drive the trip the evacuation results use.
const type = async (ph, value) => {
  const f = p.getByPlaceholder(ph).first();
  if (!(await f.count())) { console.log("no field:", ph); return false; }
  await f.click();
  await f.fill(value);
  await p.waitForTimeout(2600);                 // geocoder round-trip
  const opt = p.locator("li,[role=option]").filter({ hasText: new RegExp(value, "i") }).first();
  if (await opt.count()) await opt.click();
  else { await p.keyboard.press("ArrowDown"); await p.keyboard.press("Enter"); }
  await p.waitForTimeout(900);
  return true;
};
await type("Town, suburb or address…", "Lismore");
await type("Hospital, evacuation centre, town…", "Coraki");
// An explicit discharge. Left blank, the service estimates from terrain and
// the river driver falls back to rainfall on this box -- which floods nothing,
// so cut_km_peak is 0 and the timeline chart never renders. The evacuation
// experiments all pass 3000 m3/s here for the same reason.
const q = p.getByPlaceholder("estimate from the terrain").first();
if (await q.count()) { await q.fill("3000"); await p.waitForTimeout(600); }
else console.log("no discharge field found");

const go = p.getByRole("button", { name: /find a way through/i }).first();
console.log("submit enabled:", await go.count() ? !(await go.isDisabled()) : "no button");
if (await go.count()) await go.click();
for (let i = 0; i < 70; i++) {
  await p.waitForTimeout(5000);
  const t = await p.locator("aside").innerText();
  if (/Roads lost as the flood arrives/i.test(t)) break;
  if (i % 6 === 5) console.log(`  t+${(i+1)*5}s: ` + t.split("\n").filter(Boolean).slice(-2).join(" | ").slice(0,130));
}
console.log("--- panel tail ---");
console.log((await p.locator("aside").innerText()).split("\n").filter(Boolean).slice(-14).join("\n"));
const bars = p.locator("aside button[aria-pressed]");
const n = await bars.count();
console.log("scrubbable bars:", n);
if (!n) { console.log("no trip result — cannot verify"); await b.close(); process.exit(0); }

const canvasHash = async () => p.evaluate(() => {
  const c = document.querySelector("canvas");
  if (!c) return "nocanvas";
  const d = c.toDataURL("image/png");
  let h = 0; for (let i = 0; i < d.length; i += 97) h = (h * 31 + d.charCodeAt(i)) | 0;
  return String(h);
});
const peak = await canvasHash();
console.log("peak overlay hash:", peak);
const seen = new Set([peak]);
for (const i of [0, Math.floor(n / 2), n - 1]) {
  await bars.nth(i).click();
  await p.waitForTimeout(3500);
  const h = await canvasHash();
  const cap = (await p.locator("aside").innerText()).match(/Showing [^\n]*/);
  console.log(`  bar ${i}: hash ${h}${h === peak ? "  <-- SAME AS PEAK" : ""}  | ${cap ? cap[0] : ""}`);
  seen.add(h);
  if (i === n - 1) await p.screenshot({ path: `${OUT}/66_scrubber_late.png` });
  if (i === 0) await p.screenshot({ path: `${OUT}/67_scrubber_early.png` });
}
console.log("\ndistinct map states:", seen.size, seen.size > 2 ? "— the overlay follows the scrubber" : "— OVERLAY NOT CHANGING");
await b.close();
