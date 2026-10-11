// The warning panel: does it say how much notice there is, and where there
// is none?
//
// The Roads tab already answers "which roads close". This asserts the second
// question is on screen and answered honestly -- a median alone would
// describe the warning by the floods nobody needed warning for.
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

const roads = p.getByRole("tab", { name: "Roads" });
for (let i = 0; i < 30; i++) {
  await roads.click().catch(() => {});
  await p.waitForTimeout(500);
  if ((await roads.getAttribute("data-on")) === "true") break;
}
ok("the Roads tab opens", (await roads.getAttribute("data-on")) === "true");

// Reading the gauge record is several USGS round-trips on a cold cache.
let t = "";
for (let i = 0; i < 180; i++) {
  t = await text();
  if (/Warning time/.test(t) && !/reading the gauge record/.test(t)) break;
  await p.waitForTimeout(1000);
}
ok("a warning time is reported beside the closures", /Warning time/.test(t));
ok("it names the gauge it came from", /Twin Creeks|from .+ Ck/.test(t));
// Scoped to the headline, not to the aside. The first version of this
// check matched `/\d+ h|\d+ min/` anywhere in the panel, which the event
// list satisfies on its own -- it passed while the headline read "0 gauge
// on this reach". A check that cannot fail is worth less than no check,
// because it reports health it never looked at.
const lag = (await aside.locator("[data-warning-lag]").innerText().catch(() => "")).trim();
ok("it gives a figure, not a verdict",
   /^(\d+ h( \d+ min)?|\d+ min)$/.test(lag), `headline: ${lag || "(absent)"}`);
// Not just "how many" but what OF. The sample is the N largest floods on
// record, so "of 6 since 2004" read as "there were 6 floods since 2004" --
// false by nearly four times, since the outlet has 22 annual peaks since the
// upstream gauge was built. The word "largest" is the whole correction.
ok("it says how many floods that figure rests on, and of what",
   /median on the \d+ it recorded, of the \d+ largest since \d{4}/.test(t));
// The honest half: the failures have to be visible, not averaged away.
ok("it reports the floods with no warning at all",
   /No warning on the/.test(t) && /no usable trace/.test(t));
ok("it names 2013 and 2015 specifically",
   /2013-10-31/.test(t) && /2015-10-30/.test(t));
ok("it does not count floods predating the gauge as failures",
   !/1921|1929|1941/.test(t));

console.log(bad ? `\n${bad} failed` : "\nall checks passed");
await b.close();
process.exit(bad ? 1 : 0);
