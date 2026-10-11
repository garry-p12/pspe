// The Operate pill: does it exist, and does it refuse honestly when empty?
//
// The panel's whole claim is that it runs the structures ON THE MAP. With
// nothing placed there is no schedule to choose, and saying so is the
// behaviour worth asserting -- a demonstration polder would be fiction, and
// the easiest fiction to ship by accident.
import { chromium } from "playwright";

let bad = 0;
const ok = (name, cond, note = "") => {
  console.log(`${cond ? "ok  " : "FAIL"} ${name}${note ? "  " + note : ""}`);
  if (!cond) bad += 1;
};

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 } });
p.on("pageerror", (e) => { console.log("PAGE ERROR: " + e.message); bad += 1; });
const aside = p.locator("aside");
const text = async () => (await aside.innerText()).replace(/\s+/g, " ");

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
for (let i = 0; i < 90 && !/Water depth/.test(await text()); i++) await p.waitForTimeout(1000);

const plan = p.getByRole("tab", { name: "Plan" });
for (let i = 0; i < 30; i++) {
  await plan.click().catch(() => {});
  await p.waitForTimeout(500);
  if ((await plan.getAttribute("data-on")) === "true") break;
}
ok("the Plan tab opens", (await plan.getAttribute("data-on")) === "true");

const pill = p.getByRole("button", { name: "Operate", exact: true });
ok("an Operate pill is offered beside Build a plan and Cross-section",
   await pill.count() > 0);

if (await pill.count()) {
  for (let i = 0; i < 20; i++) {
    await pill.click().catch(() => {});
    await p.waitForTimeout(400);
    if ((await pill.getAttribute("data-on")) === "true") break;
  }
  ok("it opens", (await pill.getAttribute("data-on")) === "true");
  const t = await text();
  ok("it explains what a gate and a pump each cost",
     /gate drains for nothing|seals when it is high/i.test(t));
  // Nothing placed yet, so it must say what is missing rather than show a demo.
  ok("with nothing placed it says so instead of inventing a basin",
     /Nothing to operate yet/i.test(t));
  ok("and names what would make it answerable",
     /Floodgate/.test(t) && /Pump station/.test(t));
  ok("it does not offer a Run button with nothing to run",
     (await p.getByRole("button", { name: /Run the event/ }).count()) === 0);
}

console.log(bad ? `\n${bad} failed` : "\nall checks passed");
await b.close();
process.exit(bad ? 1 : 0);
