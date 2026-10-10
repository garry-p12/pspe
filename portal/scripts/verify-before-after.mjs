// "Before" and "With plan" have to show two different floods.
//
// They were a pair of pills that wrote `showPlan`, and nothing in the file
// read it except the two buttons' own styling: the Plan tab painted the bank
// search's baseline field in both states. The map never changed, so the one
// control on the page whose entire job is to show what a plan buys showed
// nothing, and a reader clicking between them would conclude the plan had no
// effect.
//
// The check is pixels, not state. Switch, read the WebGL buffer, and require
// the two frames to differ -- unless the solve genuinely found no option
// better than doing nothing, which happens on this reach and which the page
// now says out loud rather than silently repainting the same water.
import { chromium } from "playwright";

let fails = 0;
const ok = (n, c, extra = "") => {
  if (!c) fails++;
  console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`);
};

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 } });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e.message).slice(0, 160)));

const aside = p.locator("aside");
const text = async () => (await aside.innerText()).replace(/\s+/g, " ");
async function until(re, label, sec = 300) {
  const t0 = Date.now();
  for (;;) {
    const t = await text();
    if (re.test(t)) return t;
    if (Date.now() - t0 > sec * 1000) {
      console.log(`TIMEOUT ${label} after ${sec}s — ${t.slice(0, 200)}`);
      return t;
    }
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
/** A cheap perceptual hash of the water on the map. */
const waterHash = () => p.evaluate(() => {
  const c = document.querySelector("canvas");
  const g = c.getContext("webgl2") || c.getContext("webgl");
  const px = new Uint8Array(c.width * c.height * 4);
  g.readPixels(0, 0, c.width, c.height, g.RGBA, g.UNSIGNED_BYTE, px);
  let wet = 0, sum = 0;
  for (let i = 0; i < px.length; i += 4) {
    const s = px[i + 2] - Math.max(px[i], px[i + 1]);
    if (s > 25) { wet++; sum += s; }
  }
  return { wet, sum };
});

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await until(/Water depth/, "the layers panel", 120);

const plan = p.getByRole("tab", { name: "Plan" });
await press(plan, async () => (await plan.getAttribute("data-on")) === "true");
await until(/Judged against/, "the scenario picker", 180);
await until(/places? a wall would pay/, "the bank search", 600);

// Arm a piece and place two, which is the minimum the solver accepts.
await p.getByRole("button", { name: /Earth levee/ }).first().click().catch(() => {});
await p.waitForTimeout(500);
const rows = p.locator("aside ul button[aria-pressed]");
await rows.nth(0).click().catch(() => {});
await p.waitForTimeout(600);
await p.getByRole("button", { name: /Raised levee/ }).first().click().catch(() => {});
await p.waitForTimeout(500);
await rows.nth(1).click().catch(() => {});
const placed = await until(/2 pieces/, "two placed pieces", 60);
ok("two pieces are on the map", /2 pieces/.test(placed));

await p.getByRole("button", { name: /^Solve this plan$/ }).click().catch(() => {});
const solved = await until(/Land flooded|could not solve|No sites here/,
                           "the solved plan", 900);
ok("the plan solved", /Land flooded/.test(solved),
   (solved.match(/Land flooded [^A-Z]{0,40}/) || [""])[0]);

const pair = p.locator("button", { hasText: /^With plan$/ });
ok("the Before / With plan pair is on the map", await pair.count() > 0);

await p.getByRole("button", { name: "Before", exact: true }).click();
await p.waitForTimeout(2500);
const before = await waterHash();
await p.getByRole("button", { name: "With plan", exact: true }).click();
await p.waitForTimeout(2500);
const after = await waterHash();

const nullResult = await p.getByText("no option beat doing nothing").count() > 0;
console.log(`before ${JSON.stringify(before)}  after ${JSON.stringify(after)}  nullResult=${nullResult}`);

if (nullResult) {
  ok("a null result is stated, not painted over",
     before.wet === after.wet && before.sum === after.sum,
     "identical water, and the page says so");
} else {
  ok("the two states paint different water",
     before.wet !== after.wet || before.sum !== after.sum,
     `wet ${before.wet} -> ${after.wet}`);
}

// Changing the plan must retire the answer to the old one.
await p.getByRole("button", { name: "Clear", exact: true }).click().catch(() => {});
await p.waitForTimeout(1200);
const cleared = await text();
ok("clearing the pieces retires the solved readout",
   !/Land flooded/.test(cleared) && /Place at least two pieces|Solve this plan/.test(cleared),
   (cleared.match(/Place at least two pieces|Solve this plan|Land flooded/) || [""])[0]);

ok("no page errors", errs.length === 0, errs.join(" | "));
console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
