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
// There is no study switch any more: the portal ships one place. What has to
// be true is that nothing of the Richmond archive is left in the running app.
ok("Richmond is gone from the portal",
   !/Richmond/i.test(await aside.innerText())
   && !/Richmond/i.test(await p.locator("header").innerText()));
ok("the nav is the three tabs the design has, with no Study or Area rows",
   (await p.getByRole("tab").count()) === 3
   && !/\bArea\b/.test(await aside.innerText()));
ok("the header names the place",
   /Onion Creek, Austin, Texas/.test(await p.locator("header").innerText()),
   (await p.locator("header").innerText()).replace(/\n/g, " ").slice(0, 70));
ok("it opens on a solved place, not a search box", !/Search for a town/i.test(t.slice(0, 400)));

// The evidence card sits with the layer it describes, in Forecast.
t = await aside.innerText();
ok("the evidence label is PROJECTION, not a bare map", /PROJECTION/.test(t));
ok("the skill score is on the face of it",
   new RegExp(`CSI ${sk.csi.toFixed(2)}`).test(t), `CSI ${sk.csi.toFixed(2)}`);
ok("it says how much of FEMA's floodplain it finds",
   new RegExp(`${Math.round(sk.pod * 100)}%`).test(t));
ok("and how much of ours falls outside",
   new RegExp(`${Math.round(sk.far * 100)}%`).test(t));
ok("the terrain is named and it is bare earth", /3DEP bare earth/.test(t));
ok("the flooded area is stated beside the layer that draws it",
   t.includes(String(bundle.flooded_km2)), `${bundle.flooded_km2} km2`);
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

// Clicking the map must answer where the reader can see the answer. Both the
// "Here" readout and the run scrubber asked for the bottom-left corner, and
// the scrubber is painted second, so the reading sat underneath it and the
// map looked like it had ignored the click.
{
  const cbox = await p.locator("canvas").first().boundingBox();
  const wet = await p.evaluate(() => {
    const c = document.querySelector("canvas");
    const g = c.getContext("webgl2") || c.getContext("webgl");
    const px = new Uint8Array(c.width * c.height * 4);
    g.readPixels(0, 0, c.width, c.height, g.RGBA, g.UNSIGNED_BYTE, px);
    let best = null, bestScore = 0;
    for (let y = 0; y < c.height; y += 4) {
      for (let x = 0; x < c.width; x += 4) {
        const i = (y * c.width + x) * 4;
        const sc = px[i + 2] - Math.max(px[i], px[i + 1]);
        if (sc > bestScore) { bestScore = sc; best = [x, y]; }
      }
    }
    if (!best) return null;
    const s = c.width / c.clientWidth;
    return { x: best[0] / s, y: (c.height - best[1]) / s };
  });
  ok("the map has water on it to click", Boolean(wet));
  if (wet) {
    await p.mouse.click(cbox.x + wet.x, cbox.y + wet.y);
    const card = p.locator("[data-readout]");
    await card.waitFor({ state: "visible", timeout: 15000 }).catch(() => {});
    const shown = await card.count();
    ok("clicking the map answers with a depth", shown > 0);
    if (shown) {
      ok("and the reading is a depth, not an empty card",
         /of water on the ground|no water in this cell/.test(await card.innerText()));
      const r = await card.boundingBox();
      const scr = await p.locator("[data-scrubber]").boundingBox().catch(() => null);
      const clear = !scr || r.y + r.height <= scr.y + 1 || r.x + r.width <= scr.x + 1;
      ok("and nothing is painted over it", clear,
         scr ? `readout ${Math.round(r.y + r.height)} vs scrubber top ${Math.round(scr.y)}` : "no scrubber");
    }
  }
}

// There is nothing to switch to. The portal ships one place, and the check
// that matters is that the Plan tab reaches the planning tools rather than an
// archive's precomputed option library.
await p.getByRole("tab", { name: "Plan" }).click();
await p.waitForTimeout(2000);
ok("Plan opens on the planning tools, not an archive library",
   /Build a plan|Cross-section/.test(await aside.innerText()));
await p.screenshot({ path: `${OUT}/81_plan_tab.png` });

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
