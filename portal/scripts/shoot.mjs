import { chromium } from "playwright";
import { mkdirSync } from "fs";

const OUT = "/tmp/portal-shots";
mkdirSync(OUT, { recursive: true });
const routes = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 950 },
  deviceScaleFactor: 2,
});
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));

for (const r of routes) {
  const name = r === "/" ? "home" : r.replace(/\//g, "_").replace(/^_/, "");
  // networkidle never settles: MapLibre streams basemap tiles continuously.
  await page.goto(`http://localhost:3100${r}`, { waitUntil: "domcontentloaded", timeout: 45000 });
  await page.waitForTimeout(9000);
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: false });
  console.log(`shot ${r} -> ${OUT}/${name}.png`);
}
if (errors.length) {
  console.log("\nCONSOLE ERRORS:");
  for (const e of [...new Set(errors)].slice(0, 12)) console.log("  " + e);
} else {
  console.log("\nno console errors");
}
await browser.close();
