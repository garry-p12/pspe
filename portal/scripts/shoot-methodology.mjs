// The methodology pages, which the two documents never cited and should.
//
// Correcting an error: a previous pass assumed the panel's single range input
// was a confidence control and captured it as "51_certainty". It is the budget
// slider. The margin's own interactive page is /methodology/margin, whose
// slider is `f` -- the fraction of the achievable gain you are willing to spend
// on a guarantee -- which is a different parameter and a better figure.
import { chromium } from "playwright";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 160)));

for (const [route, name] of [
  ["/methodology/margin", "53_margin_explorer"],
  ["/methodology/limits", "54_limits"],
  ["/methodology/flood", "55_flood_method"],
]) {
  try {
    await p.goto(`http://localhost:3000${route}`, { waitUntil: "domcontentloaded" });
    await p.waitForTimeout(7000);
    await p.screenshot({ path: `${OUT}/${name}.png`, fullPage: false });
    const body = (await p.locator("body").innerText()).split("\n").filter(Boolean);
    console.log(`${name}: ${body.slice(0, 6).join(" | ").slice(0, 190)}`);
  } catch (e) {
    console.log(`SKIP ${name}:`, String(e).split("\n")[0].slice(0, 100));
  }
}
// Move the f slider to both ends, so the figure can show that the crossing moves.
try {
  await p.goto("http://localhost:3000/methodology/margin", { waitUntil: "domcontentloaded" });
  await p.waitForTimeout(6000);
  const sl = p.locator("input[type=range]").first();
  for (const [v, tag] of [["0.1", "low"], ["0.6", "high"]]) {
    await sl.evaluate((el, val) => {
      el.value = val;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }, v);
    await p.waitForTimeout(1800);
    await p.screenshot({ path: `${OUT}/53_margin_f_${tag}.png` });
    console.log(`wrote 53_margin_f_${tag} (f=${v})`);
  }
} catch (e) { console.log("SKIP f sweep:", String(e).split("\n")[0].slice(0, 100)); }

console.log(errs.length ? "ERRORS: " + [...new Set(errs)].slice(0, 3).join(" ;; ") : "no page errors");
await b.close();
