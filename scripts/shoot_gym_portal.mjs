/**
 * Capture the /gym page for the paper's figures.
 *
 *   cd portal && npx next build && npx next start -p 3021   # NOT `next dev`
 *   node ../scripts/shoot_gym_portal.mjs <out-dir>
 *
 * It must be the production server. Under Playwright the dev server's HMR
 * WebSocket fails its handshake, React never hydrates, and the page sits on
 * "Loading..." forever -- which is what the first attempt captured.
 *
 * Run from `portal/`, so `playwright` resolves from its node_modules.
 */
import { chromium } from 'playwright';
const OUT = process.argv[2];
const b = await chromium.launch({ channel: 'chrome' });
const page = await b.newPage({ viewport: { width: 1460, height: 1000 }, deviceScaleFactor: 2 });
page.on('console', m => { if (m.type() === 'error') console.log('  console:', m.text().slice(0,90)); });

async function shot(name) {
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log('  ->', name);
}
async function pick(label) {
  const btn = page.locator('aside button', { hasText: label }).first();
  await btn.click({ timeout: 20000 });
  // wait for the scene to arrive
  await page.waitForFunction(() => !!document.querySelector('svg[role="img"] polygon'),
                             null, { timeout: 90000 });
}

await page.goto('http://127.0.0.1:3021/gym', { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => !!document.querySelector('svg[role="img"] polygon'),
                           null, { timeout: 90000 });
await shot('gym-pumps');

for (const [label, name] of [['Spend the defence budget', 'gym-defend'],
                             ['When to close the road', 'gym-crossing'],
                             ['Cut firebreaks', 'gym-firebreak'],
                             ['The one with a known answer', 'gym-toy']]) {
  try { await pick(label); await shot(name); }
  catch (e) { console.log(`  ${name} FAILED: ${e.message.split('\n')[0].slice(0,70)}`); }
}
try {
  await pick('Run the pumps');
  await page.locator('button', { hasText: 'Skip to end' }).first().click();
  await page.waitForTimeout(12000);
  await shot('gym-played');
} catch (e) { console.log('  played FAILED:', e.message.split('\n')[0].slice(0,70)); }
await b.close();
