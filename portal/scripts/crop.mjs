import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 950 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3100/margin", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(6000);
const el = await p.locator("section").nth(1);
await el.screenshot({ path: "/tmp/portal-shots/margin-chart.png" });
console.log("cropped chart panel");
await b.close();
