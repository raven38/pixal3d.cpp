// Usage:
// node web/partial_e2e/run_playwright.js <flow.gguf> <shape_dec.gguf> <tex_dec.gguf> <fixture_dir> [concat.npy]
// Serve repo root first, e.g. python3 -m http.server 8199, then set PIXAL3D_BASE_URL if needed.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const [flow, shape, tex, fixtureDir, concat] = process.argv.slice(2);
  if (!flow || !shape || !tex || !fixtureDir) throw new Error('need flow shape_dec tex_dec fixture_dir [concat]');
  const fixture = fs.readdirSync(fixtureDir).filter(x => x.endsWith('.npy')).map(x => path.join(fixtureDir, x));
  const browser = await chromium.launch({ headless: false, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('[browser]', m.text()));
  const base = process.env.PIXAL3D_BASE_URL || 'http://127.0.0.1:8199/web/partial_e2e/';
  await page.goto(base);
  await page.setInputFiles('#flow', flow);
  await page.setInputFiles('#shape', shape);
  await page.setInputFiles('#tex', tex);
  await page.setInputFiles('#fixture', fixture);
  if (concat) await page.setInputFiles('#concat', concat);
  await page.click('#run');
  await page.waitForFunction(() => document.querySelector('#log').textContent.includes('PARTIAL_E2E_RESULT:'), null, { timeout: 3 * 60 * 60 * 1000 });
  const log = await page.textContent('#log');
  console.log(log);
  const ok = log.includes('PARTIAL_E2E_RESULT: OK');
  if (ok) {
    const href = await page.getAttribute('#download', 'href');
    const bytes = await page.evaluate(async (u) => Array.from(new Uint8Array(await (await fetch(u)).arrayBuffer())), href);
    fs.writeFileSync('partial_e2e.glb', Buffer.from(bytes));
    console.log('wrote partial_e2e.glb', bytes.length, 'bytes');
  }
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
