// The levee kit: 1c from runs/mockui.
//
// Live against the real service. The thing under test is that the sites come
// out of the terrain rather than being clicked, and that the guarantee shown
// is the framework's conformal margin rather than a number in the markup.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
let fails = 0;
const ok = (n, c, extra = "") => { if (!c) fails++; console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`); };

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
p.on("pageerror", (e) => { console.log("PAGE ERROR:", e.message); fails++; });
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });

// Click until the state flips: a click can land before React hydrates.
async function pick(name) {
  const btn = p.getByRole("button", { name, exact: true });
  await btn.waitFor({ state: "visible", timeout: 120000 });
  for (let i = 0; i < 25; i++) {
    await btn.click();
    await p.waitForTimeout(700);
    if ((await btn.getAttribute("data-on")) === "true") return true;
  }
  return false;
}
ok("the kit tab can be selected", await pick("Levee kit"));
const aside = p.locator("aside");
ok("it explains where sites come from",
   /almost high enough/.test(await aside.innerText()));

const find = p.getByRole("button", { name: /Find the sites|Waiting for a solved place/ });
// The study loads asynchronously and the kit needs its box.
for (let i = 0; i < 40; i++) {
  if (!(await find.isDisabled())) break;
  await p.waitForTimeout(1000);
}
ok("the button waits for a place rather than doing nothing", !(await find.isDisabled()));
await find.click();
let t = "";
for (let i = 0; i < 160; i++) {
  await p.waitForTimeout(3000);
  t = await aside.innerText();
  if (/Pick a piece|No sites here/.test(t)) break;
}
ok("it found sites", /place(s)? a wall would pay/.test(t) && /Pick a piece/.test(t),
   (t.match(/\d+ places? a wall would pay/) || [""])[0]);
if (!/Pick a piece/.test(t)) {
  console.log(t.slice(t.indexOf("Build a plan"), t.indexOf("Build a plan") + 400));
  await b.close(); process.exit(1);
}
ok("each site carries the height its lowest point needs", /needs \d+\.\d+ m/.test(t));
ok("the budget is shown as money left", /M left/.test(t), (t.match(/A\$[\d.]+ M left/) || [""])[0]);
ok("the tray offers parts with unit rates", /Earth levee/.test(t) && /\/m\/m/.test(t));
await p.screenshot({ path: `${OUT}/87_kit.png` });

// Arm a piece and place it on two sites.
await p.getByRole("button", { name: /Earth levee/ }).click();
await p.waitForTimeout(300);
ok("arming a piece asks for a site", /click a dashed site/i.test(await aside.innerText()));
const rows = p.locator("aside ul button[aria-pressed]");
const n = await rows.count();
await rows.nth(0).click();
await p.waitForTimeout(400);
await p.getByRole("button", { name: /Raised levee/ }).click();
await p.waitForTimeout(250);
await rows.nth(1).click();
await p.waitForTimeout(500);
t = await aside.innerText();
ok("placing pieces spends the budget", /spent A\$[\d.]+ M/.test(t), (t.match(/spent A\$[^\n]*/) || [""])[0]);
ok("two pieces are counted", /2 pieces/.test(t));
await p.screenshot({ path: `${OUT}/88_kit_placed.png` });

await p.getByRole("button", { name: /Solve this plan/ }).click();
// One solved bed per candidate combination: minutes, not seconds.
for (let i = 0; i < 200; i++) {
  await p.waitForTimeout(3000);
  t = await aside.innerText();
  if (/Land flooded|could not solve|No sites here/.test(t)) break;
}
ok("it solves and reports impact", /Land flooded/.test(t));
ok("the impact is before and after", /Homes over floor/.test(t) && /Road cut/.test(t));
ok("the guarantee comes from the margin, or says it cannot",
   /conformal margin is ±/.test(t) || /No guarantee on this one/.test(t),
   (t.match(/At least [\d.]+% less flooding|No guarantee on this one/) || [""])[0]);
ok("before and with-plan can be compared", (await p.getByRole("button", { name: "Before" }).count()) === 1);
await p.screenshot({ path: `${OUT}/89_kit_impact.png` });

await p.getByRole("button", { name: "Undo", exact: true }).click();
await p.waitForTimeout(400);
ok("undo removes a piece", /1 piece\b/.test(await aside.innerText()));

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
