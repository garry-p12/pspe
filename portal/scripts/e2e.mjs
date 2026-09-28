import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 950 } });
const errs = [];
p.on("pageerror", (e) => errs.push("PAGE: " + String(e).slice(0, 160)));
p.on("console", (m) => { if (m.type() === "error" && !m.text().includes("503")) errs.push("CON: " + m.text().slice(0, 160)); });

const step = async (label, fn) => {
  try { await fn(); console.log(`  ok   ${label}`); }
  catch (e) { console.log(`  FAIL ${label}: ${String(e).slice(0, 110)}`); }
};

await p.goto("http://localhost:3100/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(7000);

await step("district tab renders", async () => {
  await p.getByText("Road access in this event").waitFor({ timeout: 5000 });
});
await step("forecast panel loaded", async () => {
  await p.getByText(/flooding forecast|Worth watching|Significant/).first().waitFor({ timeout: 8000 });
});
await step("road threshold recomputes", async () => {
  // The reading spans two elements ("72 km" + "cut of 287 km"), so match on the
  // panel's text rather than a single node, and assert the number actually MOVES.
  const read = async () => {
    const txt = await p.locator("aside").innerText();
    const m = txt.match(/([\d.]+)\s*km\s*cut of/);
    return m ? Number(m[1]) : NaN;
  };
  await p.getByRole("button", { name: "Any water", exact: true }).click();
  await p.waitForTimeout(1600);
  const wet = await read();
  await p.getByRole("button", { name: "Impassable", exact: true }).click();
  await p.waitForTimeout(1600);
  const deep = await read();
  if (!(deep < wet)) throw new Error(`expected fewer km cut at depth: ${wet} -> ${deep}`);
});
await step("map click inspects a point", async () => {
  await p.mouse.click(1000, 500);
  await p.waitForTimeout(2000);
  await p.getByText("Selected location").waitFor({ timeout: 4000 });
});
await step("anywhere tab opens", async () => {
  await p.getByRole("button", { name: "Anywhere" }).click();
  await p.getByPlaceholder(/Search a town/).waitFor({ timeout: 4000 });
});
await step("geocode returns results", async () => {
  await p.getByPlaceholder(/Search a town/).fill("Cologne Germany");
  await p.waitForTimeout(3500);
  const n = await p.locator("aside button").filter({ hasText: /Cologne|Köln/ }).count();
  if (n === 0) throw new Error("no suggestions");
});
await step("methodology reachable", async () => {
  await p.getByRole("link", { name: /How this works/ }).click();
  await p.waitForTimeout(2500);
  await p.getByText(/The loop|climate/i).first().waitFor({ timeout: 5000 });
});
console.log("\nerrors:", errs.length ? [...new Set(errs)].slice(0, 5) : "none");
await b.close();
