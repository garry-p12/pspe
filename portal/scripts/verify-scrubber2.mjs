// Verify the scrubber's UI logic against a REAL solved response.
//
// The trip tab's default 130 m cells fall back to rainfall on this route and
// report 0 km cut, so the timeline chart never renders through the normal path
// -- a separate, pre-existing issue. This serves the genuine 200 m solve
// (driver river, 7,500 m3/s, 113.2 km cut, 9 frames) so the frontend is
// exercised on real solver output rather than on invented numbers.
import { chromium } from "playwright";
import { readFileSync } from "node:fs";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const body = readFileSync("/tmp/route_frames.json", "utf8");
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
await p.route("**/route", r => r.fulfill({ status: 200, contentType: "application/json", body }));
await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(2000);
await p.getByRole("button", { name: /plan a trip/i }).click().catch(() => {});
await p.waitForTimeout(2000);
const type = async (ph, v) => {
  const f = p.getByPlaceholder(ph).first();
  if (!(await f.count())) return;
  await f.click(); await f.fill(v); await p.waitForTimeout(2600);
  const o = p.locator("li,[role=option]").filter({ hasText: new RegExp(v, "i") }).first();
  if (await o.count()) await o.click(); else { await p.keyboard.press("ArrowDown"); await p.keyboard.press("Enter"); }
  await p.waitForTimeout(800);
};
await type("Town, suburb or address…", "Lismore");
await type("Hospital, evacuation centre, town…", "Coraki");
// Wait for both ends to resolve: the geocoder is a network round-trip and the
// submit button stays disabled until it lands. A fixed sleep here flakes.
const go = p.getByRole("button", { name: /find a way through|choose both ends/i }).first();
for (let i = 0; i < 20; i++) {
  if (await go.count() && !(await go.isDisabled())) break;
  await p.waitForTimeout(1500);
}
if (await go.isDisabled()) { console.log("both ends never resolved — geocoder down?"); await b.close(); process.exit(1); }
await go.click();
for (let i = 0; i < 24; i++) {
  await p.waitForTimeout(2500);
  if (/Roads lost as the flood arrives/i.test(await p.locator("aside").innerText())) break;
}
const bars = p.locator("aside button[aria-pressed]");
const n = await bars.count();
console.log("scrubbable bars:", n);
if (!n) { console.log("chart did not render"); await b.close(); process.exit(1); }

const hash = async () => p.evaluate(() => {
  const c = document.querySelector("canvas");
  if (!c) return "nocanvas";
  const d = c.toDataURL("image/png");
  let h = 0; for (let i = 0; i < d.length; i += 97) h = (h * 31 + d.charCodeAt(i)) | 0;
  return String(h);
});
// Settle before sampling the baseline too. Sampling it immediately caught the
// scene still streaming terrain, so "peak" differed on every run while the
// post-toggle state was identical across runs -- the baseline was the unstable
// half of the comparison, not the thing under test.
const settled = async () => {
  let h = "", prev = "";
  for (let i = 0; i < 12; i++) {
    await p.waitForTimeout(2000);
    h = await hash();
    if (h === prev) return h;
    prev = h;
  }
  return h;
};
const peak = await settled();
console.log("peak overlay (settled):", peak);
const seen = new Set([peak]);
for (const i of [0, Math.floor(n / 2), n - 1]) {
  await bars.nth(i).click();
  await p.waitForTimeout(3000);
  const h = await hash();
  const cap = ((await p.locator("aside").innerText()).match(/Showing [^\n]*/) || [""])[0];
  console.log(`  bar ${i}: ${h}${h === peak ? "  <-- same as peak" : ""}  | ${cap}`);
  seen.add(h);
  if (i === 0) await p.screenshot({ path: `${OUT}/67_scrubber_early.png` });
  if (i === n - 1) await p.screenshot({ path: `${OUT}/66_scrubber_late.png` });
}
// Toggle off deliberately: select, settle, deselect, settle. The first
// version waited 500 ms between the two clicks and 2.5 s after, and reported a
// failure that was its own impatience -- the terrain layer reloads its texture
// asynchronously.
await bars.nth(0).click();
await p.waitForTimeout(3000);
const picked = await hash();
await bars.nth(0).click();
// Hash repeatedly until the scene stops moving. A single sample after a fixed
// wait catches deck.gl mid-transition and reports a mismatch that settles a
// moment later -- which is what the first version of this check did.
const back = await settled();
console.log(`toggle off: picked ${picked} -> settled ${back}, peak ${peak}`);
console.log("back to peak:", back === peak);
const cap2 = ((await p.locator("aside").innerText()).match(/Showing [^\n]*/) || [""])[0];
console.log("caption after toggle off:", cap2 || "(none — expected, peak has no caption)");
console.log("\ndistinct map states:", seen.size,
  seen.size > 2 ? "-- the overlay follows the scrubber" : "-- OVERLAY NOT CHANGING");
await b.close();
