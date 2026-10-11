// The street multiplier's justification, asserted against the model.
//
// The panel's (i) used to say "at this discharge with no multiplier this
// model puts no house in the block over its floor". That is the reason a
// reader is given for the only free knob in the street view, and it was
// false: at 3,823 m3/s -- USGS's indirect annual peak for 31 Oct 2013 --
// the model puts 31 of 78 over. The knob is still defensible as calibration
// against what 2013 did, but not for that reason.
//
// It went unnoticed because the figure in the sentence and the claim about
// the figure were edited at different times. This pins both.
import { chromium } from "playwright";

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 } });
const aside = p.locator("aside");
const text = async () => (await aside.innerText()).replace(/\s+/g, " ");

async function until(re, label, sec = 400) {
  const t0 = Date.now();
  for (;;) {
    const t = await text();
    if (re.test(t)) return t;
    if (Date.now() - t0 > sec * 1000) { console.log(`TIMEOUT ${label}`); return t; }
    await p.waitForTimeout(2000);
  }
}
async function press(loc, isOn, tries = 30) {
  await loc.waitFor({ state: "visible", timeout: 120000 });
  for (let i = 0; i < tries; i++) {
    await loc.click({ timeout: 15000 }).catch(() => {});
    await p.waitForTimeout(500);
    if (await isOn()) return true;
  }
  return false;
}

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await until(/Water depth/, "layers");
const chip = p.getByRole("button", { name: "Zoom to street level" });
await press(chip, async () => /This street|buildings · ground/.test(await text()));
await until(/Where the water comes from|Open the street view/, "street controls", 120);

await p.getByRole("button", { name: "Down the channel" }).click().catch(() => {});
await p.getByRole("button", { name: "As gauged" }).click().catch(() => {});   // 1x
await p.locator("#street-q").fill("3823").catch(() => {});
await p.waitForTimeout(600);
for (let i = 0; i < 5; i++) {
  await p.getByRole("button", { name: /Open the street view|Look again/ })
    .first().click().catch(() => {});
  const t = await until(/Over floor level|Could not open/, "solved street", 330);
  if (/Over floor level/.test(t)) break;
  console.log(`  attempt ${i + 1} timed out; asking again`);
  await p.waitForTimeout(4000);
}
const t = await text();
const m = t.match(/Over floor level (\d+) of (\d+)/);
const deep = t.match(/Deepest at a house ([\d.]+) m/);
console.log("panel:", t.slice(0, 260));
if (m) {
  const [, over, total] = m;
  console.log(`\nat 1x, 3823 m3/s: ${over} of ${total} over floor` +
              (deep ? `, deepest at a house ${deep[1]} m` : ""));
  let bad = 0;
  const ok = (name, cond, note = "") => {
    console.log(`${cond ? "ok  " : "FAIL"} ${name}${note ? "  " + note : ""}`);
    if (!cond) bad += 1;
  };
  ok("1x floods this block rather than leaving it dry",
     Number(over) > 0,
     `${over} of ${total} over floor`);
  ok("and the panel no longer claims otherwise",
     !/no house/i.test(await p.locator("body").innerText()));
  ok("1x is materially less than the 2.5x the panel recommends",
     Number(over) < 60, `${over} vs 60 at 2.5x`);
  console.log(bad ? `\n${bad} failed` : "\nall checks passed");
  await b.close();
  process.exit(bad ? 1 : 0);
} else {
  console.log("FAIL — could not read the tile");
  await b.close();
  process.exit(1);
}
await b.close();
