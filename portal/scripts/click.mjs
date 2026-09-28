import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3100/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(9000);
// Click in the middle of the flooded floodplain.
await p.mouse.click(900, 480);
await p.waitForTimeout(2500);
await p.screenshot({ path: "/tmp/portal-shots/click.png" });
const txt = await p.locator("aside").innerText();
console.log(txt.split("\n").slice(0, 18).join("\n"));
await b.close();
