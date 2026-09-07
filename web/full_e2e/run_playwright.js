// fixture-input full-model E2E を Chrome/WebGPU で回す harness。
// 推論は一切 JS 側に無い（WASM の C ABI を叩くだけ）。
// Usage:
//   node web/full_e2e/run_playwright.js <models_dir> <fixture_dir> [seed] [--tex-cond-fixture]
// 先にリポジトリルートを配信すること（例: python3 -m http.server 8199）。
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const argv = process.argv.slice(2).filter(a => a !== '--tex-cond-fixture');
  const useFixtureCond = process.argv.includes('--tex-cond-fixture');
  const [modelsDir, fixtureDir, seed] = argv;
  if (!modelsDir || !fixtureDir) throw new Error('need <models_dir> <fixture_dir> [seed] [--tex-cond-fixture]');
  const models = fs.readdirSync(modelsDir).filter(x => x.endsWith('.gguf')).map(x => path.join(modelsDir, x));
  const fixture = fs.readdirSync(fixtureDir).filter(x => x.endsWith('.npy')).map(x => path.join(fixtureDir, x));
  const browser = await chromium.launch({ headless: false, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('[browser]', m.text()));
  await page.goto(process.env.PIXAL3D_BASE_URL || 'http://127.0.0.1:8199/web/full_e2e/');
  await page.setInputFiles('#models', models);
  await page.setInputFiles('#fixture', fixture);
  if (seed) await page.fill('#seed', String(seed));
  if (useFixtureCond) await page.check('#texfixture');
  await page.click('#run');
  // ページ内 #log は worker からの postMessage で伸びるので、進捗を定期的に吐く
  let shown = 0;
  const tick = setInterval(async () => {
    try {
      const t = await page.textContent('#log');
      if (t && t.length > shown) { process.stdout.write(t.slice(shown)); shown = t.length; }
    } catch (_) { /* ページ遷移中などは黙って次の tick へ */ }
  }, 15000);
  await page.waitForFunction(() => document.querySelector('#log').textContent.includes('FULL_FIXTURE_RESULT:'),
                             null, { timeout: 6 * 60 * 60 * 1000 });
  clearInterval(tick);
  const log = await page.textContent('#log');
  console.log(log);
  const ok = log.includes('FULL_FIXTURE_RESULT: OK');
  if (ok) {
    const href = await page.getAttribute('#download', 'href');
    const bytes = await page.evaluate(async u => Array.from(new Uint8Array(await (await fetch(u)).arrayBuffer())), href);
    fs.writeFileSync('full_e2e.glb', Buffer.from(bytes));
    console.log('wrote full_e2e.glb', bytes.length, 'bytes');
  }
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
