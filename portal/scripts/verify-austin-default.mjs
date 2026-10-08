// The portal opens on Austin, with the shipped study already solved.
//
// Checked against the bundle on disk rather than against typed-in numbers, so
// a rebuilt study cannot silently drift away from what the page claims.
import { chromium } from "playwright";
import { readFileSync } from "node:fs";
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";
const bundle = JSON.parse(readFileSync("public/data/austin/study.json", "utf8"));
const sk = bundle.study.skill;

let fails = 0;
const ok = (n, c, extra = "") => { if (!c) fails++; console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`); };

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
p.on("pageerror", (e) => { console.log("PAGE ERROR:", e.message); fails++; });
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(14000);

const aside = p.locator("aside");
let t = await aside.innerText();
ok("the study row names Austin first", /Study[\s\S]{0,40}Austin, Texas/.test(t));
ok("Austin is the selected study",
   await p.getByRole("button", { name: "Austin, Texas", exact: true })
     .getAttribute("data-on") === "true");
ok("Richmond is still reachable", await p.getByRole("button", { name: /Richmond Valley/ }).count() === 1);
ok("the header names the place",
   /Onion Creek, Austin, Texas/.test(await p.locator("header").innerText()),
   (await p.locator("header").innerText()).replace(/\n/g, " ").slice(0, 70));
ok("it opens on a solved place, not a search box", !/Search for a town/i.test(t.slice(0, 400)));

// The answer lives under Plan.
await p.getByRole("tab", { name: "Plan" }).click();
await p.waitForTimeout(2500);
t = await aside.innerText();
ok("the evidence label is PROJECTION, not a bare map", /PROJECTION/.test(t));
ok("the skill score is on the face of it",
   new RegExp(`CSI ${sk.csi.toFixed(2)}`).test(t), `CSI ${sk.csi.toFixed(2)}`);
ok("it says how much of FEMA's floodplain it finds",
   new RegExp(`${Math.round(sk.pod * 100)}%`).test(t));
ok("and how much of ours falls outside",
   new RegExp(`${Math.round(sk.far * 100)}%`).test(t));
ok("the terrain is named and it is bare earth", /3DEP bare earth/.test(t));
ok("the flooded area matches the bundle",
   t.includes(String(Math.round(bundle.flooded_km2))), `${bundle.flooded_km2} km2`);
ok("the caveats are reachable", /What this answer is not/.test(t));

// The map drew something.
const hash = await p.evaluate(() => {
  const c = document.querySelector("canvas");
  if (!c) return "nocanvas";
  const d = c.toDataURL("image/png");
  let h = 0; for (let i = 0; i < d.length; i += 97) h = (h * 31 + d.charCodeAt(i)) | 0;
  return String(h);
});
ok("the map rendered", hash !== "nocanvas" && hash !== "0", hash);
await p.screenshot({ path: `${OUT}/80_austin_default.png` });

// Switching to Richmond must still work.
await p.getByRole("button", { name: /Richmond Valley/ }).click();
await p.waitForTimeout(4000);
ok("Richmond still opens", await p.getByRole("button", { name: "This district" }).count() === 1);
await p.screenshot({ path: `${OUT}/81_richmond_study.png` });

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
