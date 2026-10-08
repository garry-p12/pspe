// The cross-section view, driven through the real UI on a real solved section.
//
// The solve itself is served from a saved /section response rather than run
// live, for the same reason verify-scrubber2.mjs does it: the point here is
// whether the panel reads the profile correctly, and a 28 s solve inside a
// browser test buys nothing except flake. The response is genuine -- 7.1 km
// across the Wilsons River floodplain at Lismore, 80 m cells, river driver.
//
//   node scripts/verify-section.mjs <section_response.json>
import { chromium } from "playwright";
import { readFileSync } from "node:fs";

const json = process.argv[2];
if (!json) { console.error("usage: node verify-section.mjs <section_response.json>"); process.exit(2); }
const body = readFileSync(json, "utf8");
const OUT = "/Users/guruprasadparasnis/pspe/runs/portal_shots";

let fails = 0;
const ok = (name, cond, extra = "") => {
  if (!cond) fails++;
  console.log(`${cond ? "ok  " : "FAIL"} ${name}${extra ? `  ${extra}` : ""}`);
};

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
p.on("pageerror", (e) => { console.log("PAGE ERROR:", e.message); fails++; });
await p.route("**/api/section", (r) =>
  r.fulfill({ status: 200, contentType: "application/json", body }));

await p.goto("http://localhost:3000/", { waitUntil: "domcontentloaded" });
await p.waitForTimeout(12000);

await p.getByRole("button", { name: "Cross-section" }).click();
await p.waitForTimeout(800);
const aside = p.locator("aside");
ok("panel invites a line", /Click the map where the section should start/i
   .test(await aside.innerText()));

// Two clicks on the map, in order. The canvas is deck.gl's; a click on it
// resolves to a coordinate, which is what the picker consumes -- but only
// below the horizon. The scene is pitched 48 degrees, so a click in the upper
// third of it returns no coordinate at all and the picker correctly ignores
// it. The first version of this check clicked at 40% height and reported a
// broken picker that was its own aim.
const canvas = p.locator("canvas").first();
const box = await canvas.boundingBox();
await p.mouse.click(box.x + box.width * 0.45, box.y + box.height * 0.55);
await p.waitForTimeout(500);
ok("first click sets A and asks for B",
   /Now click the far side of the floodplain/i.test(await aside.innerText()));
await p.mouse.click(box.x + box.width * 0.60, box.y + box.height * 0.72);
await p.waitForTimeout(500);
const ends = await aside.innerText();
ok("both ends are shown", /A -?\d+\.\d{4}, \d+\.\d{4}/.test(ends) && /B -?\d+\.\d{4}/.test(ends));

// A third click must NOT throw the line away: picking disarms itself.
await p.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.62);
await p.waitForTimeout(400);
const aCoord = (ends.match(/A (-?\d+\.\d{4}, \d+\.\d{4})/) || [])[1];
ok("a later map click does not restart the line",
   Boolean(aCoord) && (await aside.innerText()).includes(aCoord), aCoord ?? "(A never set)");

await p.getByRole("button", { name: /^Cut the section$/ }).click();
for (let i = 0; i < 20; i++) {
  await p.waitForTimeout(500);
  if (/This crest holds|Overtopped|open end|stayed dry/i.test(await aside.innerText())) break;
}
const t = await aside.innerText();

// --- the answer, against the figures verify-section-math.mjs pins ----------
ok("verdict is 'holds'", /This embankment holds/.test(t));
ok("crest elevation stated", /11\.57 m/.test(t), (t.match(/crest at \d+\.\d+ m/i) || [""])[0]);
ok("cost stated", /A\$11\.2 M/.test(t));
ok("crest length stated", /1\.62 km/.test(t));
ok("max height stated", /3\.91 m/.test(t));
ok("fill area stated", /4305 m²/.test(t));
ok("the landward end ties in, the other meets the river",
   /ties into ground at 12\.3 m/.test(t) && /meets the channel bank/.test(t));
ok("nine runs filtered to six reaches", /crosses 6 separate bodies of water/.test(t));
ok("the channel splits the reach into two panels", /Which side of the channel/.test(t));
ok("the expensive mistake is priced", /A\$50\.7 M/.test(t) && /A\$11\.2 M/.test(t));
ok("what is being priced is stated", /tie-back embankment/.test(t));
// The flood on the chart is the one solved without the wall. Saying so is not
// optional: the picture otherwise shows an embankment standing in the water it
// is meant to stop, and a reader is entitled to conclude the model ran with it.
ok("it says the flood was not re-solved with the wall",
   /solved WITHOUT this wall/.test(t) && /does not re-solve/.test(t));
ok("a zero-freeboard default says it is not a design standard",
   /not a design standard/.test(t) && /0\.3 to 0\.6 m of freeboard/.test(t));
// Not inside the provenance fold: a river driver that found no channel falls
// back to rainfall, and that changes what a dry stretch of this profile MEANS.
// It has to be readable without opening anything.
ok("the driver that actually ran is on the face of the answer",
   /Forced by river: 18,750 m3\/s on the east edge/.test(t),
   (t.match(/Forced by [^\n]*/) || [""])[0].slice(0, 60));

// --- the chart -------------------------------------------------------------
const svg = p.locator("aside svg[role=img]");
ok("chart rendered", await svg.count() === 1);
const paths = await p.locator("aside svg[role=img] path").count();
ok("ground, water runs and the embankment are all drawn", paths >= 8, `${paths} paths`);
const label = await svg.getAttribute("aria-label");
ok("chart is described for a screen reader",
   /holds back peaks at 11\.57 metres\. Crest at 11\.6 metres over 1\.62 km: holds/.test(label), label);
await p.screenshot({ path: `${OUT}/70_section_holds.png` });

// --- the controls ----------------------------------------------------------
// Drop the crest a metre and the verdict has to change, with the overtopping
// depth and its station named.
await p.locator("#sec-crest").fill("10.6");
await p.waitForTimeout(500);
const t2 = await aside.innerText();
ok("dropping the crest overtops it", /Overtopped by 0\.97 m/.test(t2),
   (t2.match(/Overtopped by [\d.]+ m/) || [""])[0]);
ok("it says where", /along the line/.test(t2));
ok("it says how far below the peak", /0\.97 m below the solved peak/.test(t2));
ok("the way back is offered", /Back to the cheapest crest that holds/.test(t2));
await p.screenshot({ path: `${OUT}/71_section_overtops.png` });

await p.getByRole("button", { name: /Back to the cheapest crest that holds/ }).click();
await p.waitForTimeout(400);
ok("and it works", /This embankment holds/.test(await aside.innerText()));

// Pull the landward end in off the bank: the wall still clears the water, and
// the panel has to say it has an open end rather than that it holds.
await p.locator("#sec-w").fill("6");
await p.waitForTimeout(500);
const t3 = await aside.innerText();
ok("a wall cut short of the bank reports an open end",
   /It holds, but it has an open end/.test(t3));
ok("the shortfall is in metres", /3\.44 m below\s+the crest/.test(t3),
   (t3.match(/[\d.]+ m below\s+the crest/) || [""])[0]);
ok("and it is not called a failure", !/fails/i.test(t3.split("Where these numbers")[0]));
await p.screenshot({ path: `${OUT}/72_section_open_end.png` });

// Drag the far end over the river: a continuous embankment there is a dam, and
// the panel has to refuse it rather than quote a price.
await p.getByRole("button", { name: /Back to the cheapest crest that holds/ }).click();
await p.waitForTimeout(400);
await p.locator("#sec-e").fill("45");
await p.waitForTimeout(500);
const t5 = await aside.innerText();
ok("an alignment over the channel is refused", /That is a dam, not a levee/.test(t5));
ok("the refusal names the channel", /1\.68 km/.test(t5) && /2\.39 km/.test(t5),
   (t5.match(/runs over the channel between[^.]*/) || [""])[0].slice(0, 70));
await p.screenshot({ path: `${OUT}/76_section_dam.png` });
await p.getByRole("button", { name: /Back to the cheapest crest that holds/ }).click();
await p.waitForTimeout(400);

// Another reach is another design.
await p.getByRole("button", { name: /4\.52 km – 4\.72 km/ }).click().catch(() => {});
await p.waitForTimeout(600);
const t4 = await aside.innerText();
ok("switching reach re-answers", /8\.80 m/.test(t4) && !/A\$11\.2 M/.test(t4),
   (t4.match(/A\$[\d.]+ M/) || [""])[0]);

// --- the line that came back dry ------------------------------------------
// The case a live run hit on the first try: a river driver whose derived box
// contains no through-channel falls back to rainfall, which spread over the
// whole box floods almost nothing. The profile then looks identical to "this
// river does not reach here", and the panel has to say which it is. The
// fixture is the same real response with its water removed and the service's
// own fallback note in place.
const dryBody = JSON.stringify({
  ...JSON.parse(body),
  profile: JSON.parse(body).profile.map((q) => ({ ...q, water_m: null, depth_m: 0 })),
  peak_water_m: null, wet_fraction: 0, driver: "rain",
  driver_note: "No through-channel crosses this box, so the storm is applied as rainfall instead.",
});
await p.unroute("**/api/section");
await p.route("**/api/section", (r) =>
  r.fulfill({ status: 200, contentType: "application/json", body: dryBody }));
await p.locator("aside summary").filter({ hasText: "The line and the storm" }).click();
await p.waitForTimeout(400);
await p.getByRole("button", { name: /^Cut it again$/ }).click();
for (let i = 0; i < 20; i++) {
  await p.waitForTimeout(500);
  if (/stayed dry/.test(await aside.innerText())) break;
}
const t6 = await aside.innerText();
ok("a dry line says so", /This line stayed dry/.test(t6));
ok("and does not offer a design", !/Cost, indicative/.test(t6));
ok("and names the fallback rather than letting it pass as a river answer",
   /You asked for a river, and none was found/.test(t6)
   && /not evidence the river does not reach it/.test(t6));
await p.screenshot({ path: `${OUT}/77_section_dry.png` });

console.log(fails ? `\n${fails} FAILED` : "\nall checks passed");
await b.close();
process.exit(fails ? 1 : 0);
