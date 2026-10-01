import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3000/?view=plan&budget=20", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(4000);
await p.getByRole("button", { name: /Find the best plan/i }).first().click();
for (let i = 0; i < 30; i++) {
  await p.waitForTimeout(4000);
  if (/stand behind/i.test(await p.locator("aside").innerText())) break;
}
const grab = async () => {
  const t = await p.locator("aside").innerText();
  const m = t.match(/What we'll stand behind[^\n]*\n([^\n]+)/);
  const lvl = t.match(/How sure do you need to be\s*\n?([^\n]*)/);
  return { floor: m ? m[1].trim() : "?", level: (lvl ? lvl[1] : "").trim() };
};
const slider = p.locator("#certainty");
console.log("certainty slider found:", await slider.count());
for (const v of [70, 90, 95, 99]) {
  await slider.fill(String(v));
  await p.waitForTimeout(3500);
  const g = await grab();
  const t = await p.locator("aside").innerText();
  const refused = /would need about (\d+) test floods/.exec(t);
  console.log(`  asked ${v}%  ->  stand behind ${g.floor}` +
              (refused ? `   REFUSED: needs ${refused[1]} test floods` : ""));
}
await p.screenshot({ path: "/Users/guruprasadparasnis/pspe/runs/portal_shots/52_refusal.png" });
await b.close();
