import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 950 } });
await p.goto("http://localhost:3100/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(7000);
for (const name of ["Any water", "Car", "4WD / truck", "Impassable"]) {
  await p.getByRole("button", { name, exact: true }).click();
  await p.waitForTimeout(1800);
  const txt = await p.locator("aside").innerText();
  const m = txt.match(/([\d.]+)\s*km\s*cut of\s*([\d.]+)\s*km/);
  const pct = txt.match(/(\d+)% of the mapped network/);
  const worst = txt.split("WORST AFFECTED")[1]?.trim().split("\n")[0] ?? "—";
  console.log(`  ${name.padEnd(12)} ${m ? `${m[1]} km cut of ${m[2]}` : "no reading"}  ${pct ? pct[1] + "%" : ""}   worst: ${worst}`);
}
await b.close();
