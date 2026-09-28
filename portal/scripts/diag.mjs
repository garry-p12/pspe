import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 950 } });
const failed = [];
p.on("response", (r) => { if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`); });
await p.goto("http://localhost:3100/margin", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(6000);
console.log("FAILED REQUESTS:", failed.length ? failed : "none");
const info = await p.evaluate(async () => {
  const r = await fetch("/data/metrics.json");
  const m = await r.json();
  return {
    ok: r.ok,
    keys: Object.keys(m),
    precondLen: (m.precond_synthetic || []).length,
    firstRow: (m.precond_synthetic || [])[0],
    elasticityLen: (m.elasticity?.rows || []).length,
  };
});
console.log("FETCH FROM PAGE:", JSON.stringify(info, null, 2));
const paths = await p.locator("svg .recharts-line-curve").count();
console.log("recharts line curves in DOM:", paths);
await b.close();
