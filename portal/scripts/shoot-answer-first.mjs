import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1400 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3000/?view=plan", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);
await p.getByRole("tab", { name: "Plan" }).click().catch(() => {});
await p.waitForTimeout(3000);
await p.locator("#budget").fill("20");
await p.waitForTimeout(1200);
await p.screenshot({ path: `${OUT}/64_answer_first_before_plan.png` });
await p.getByRole("button", { name: /find the best plan/i }).first().click();
for (let i = 0; i < 30; i++) {
  await p.waitForTimeout(4000);
  if (/stand behind/i.test(await p.locator("aside").innerText())) break;
}
await p.waitForTimeout(1500);
await p.screenshot({ path: `${OUT}/65_answer_first.png` });
// How far down the panel is the answer now?
const pos = await p.evaluate(() => {
  const el = [...document.querySelectorAll("aside *")]
    .find(e => /stand behind/i.test(e.textContent || "") && e.children.length < 4);
  const aside = document.querySelector("aside");
  if (!el || !aside) return null;
  return { answerTop: Math.round(el.getBoundingClientRect().top),
           asideTop: Math.round(aside.getBoundingClientRect().top),
           viewport: window.innerHeight };
});
console.log("answer position:", JSON.stringify(pos));
console.log("above the fold:", pos && pos.answerTop < pos.viewport);
const folds = await p.locator("details.fold").count();
console.log("folded sections:", folds);
for (let i = 0; i < folds; i++) {
  console.log("  -", (await p.locator("details.fold summary").nth(i).innerText()).replace(/\n/g, " | "));
}
await b.close();
