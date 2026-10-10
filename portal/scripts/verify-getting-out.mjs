// "Getting out": can anyone leave this place by road, and is the answer true.
//
// Replaces verify-route-district.mjs, which drove `?view=plan` and looked for
// this card there. Both halves of that had rotted. The card is in the Roads
// tab now -- it was mounted inside `AnywherePanel` under `section === "plan"`
// while that component renders only under `section === "forecast"`, two
// conditions that cannot both hold, so it had been dead since the navigation
// became three tabs. And the expectations were the Richmond archive's ("213 km
// of the 287 km network"), which is not the place the portal ships.
//
// The check that matters is the one that caught a real falsehood: the panel
// used to pick the Euclidean-nearest dry junction and, if the router could not
// reach THAT node, report "No way out by road". On Onion Creek, with 30 km of
// a 900 km network cut, that is what it said. A place is cut off when no dry
// ground is reachable, not when one arbitrary node is.
import { chromium } from "playwright";

let fails = 0;
const ok = (n, c, extra = "") => {
  if (!c) fails++;
  console.log(`${c ? "ok  " : "FAIL"} ${n}${extra ? `  ${extra}` : ""}`);
};

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1480, height: 1040 } });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e.message).slice(0, 160)));
await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });

const aside = p.locator("aside");
const text = async () => (await aside.innerText()).replace(/\s+/g, " ");

const tab = p.getByRole("tab", { name: "Roads" });
await tab.waitFor({ state: "visible", timeout: 120000 });
for (let i = 0; i < 25; i++) {
  await tab.click().catch(() => {});
  await p.waitForTimeout(600);
  if ((await tab.getAttribute("data-on")) === "true") break;
}

let t = "";
for (let i = 0; i < 60; i++) {
  t = await text();
  if (/Getting out/.test(t)) break;
  await p.waitForTimeout(2000);
}
ok("the evacuation card is reachable at all", /Getting out/.test(t));
ok("it names the place people are leaving", /From .+ to the nearest dry ground/.test(t),
   (t.match(/From [^.]{0,60}/) || [""])[0]);
ok("it says how much network it read", /\d[\d,]* road segments/.test(t),
   (t.match(/[\d,]+ road segments/) || [""])[0]);

const verdict = (t.match(/Road out is open|No way out by road/) || [""])[0];
ok("it reaches a verdict", Boolean(verdict), verdict);

// The falsehood this harness exists for. Onion Creek has ~30 km of 898 km
// cut; "no way out" there would be wrong, and it was what the panel said.
const cut = t.match(/(\d[\d.]*) km of the local network/);
if (verdict === "No way out by road" && cut) {
  ok("and \"no way out\" is not claimed over a mostly-open network",
     Number(cut[1]) > 400,
     `${cut[1]} km cut — too little of the network to justify that verdict`);
} else {
  ok("and \"no way out\" is not claimed over a mostly-open network", true, verdict);
}
if (verdict === "Road out is open") {
  ok("an open road is named and timed",
     /\d+ (min|h)\b/.test(t) && /via /.test(t),
     (t.match(/via [^.]{0,70}/) || [""])[0]);
}

// Two questions, two thresholds, and the simpler one first.
ok("the trip planner is on the same tab, below it",
   /Plan a trip/.test(t) && t.indexOf("Getting out") < t.indexOf("Plan a trip"));

// Moving this threshold must not retune the Forecast tab's road list, which
// answers a different question about a different network.
const before = (await p.getByRole("tab", { name: "Forecast" }).isVisible()) ? true : false;
ok("the Forecast tab is still reachable from here", before);

ok("no page errors", errs.length === 0, errs.join(" | "));
console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
