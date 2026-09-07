// 実入力（transforms.json + pre-matted RGBA）の E2E を Chrome/WebGPU で回す harness。
// models_dir に texture 2本（pixal3d_tex_flow_1024_mv.gguf / tex_dec.gguf）が入っていれば
// live Texture conditioning 込みの true full E2E、無ければ geometry E2E になる。
// Usage:
//   node web/real_e2e/run_playwright.js <models_dir> <views_dir> [seed]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const [modelsDir, viewsDir, seed] = process.argv.slice(2);
  if (!modelsDir || !viewsDir) throw new Error('need <models_dir> <views_dir> [seed]');
  const models = fs.readdirSync(modelsDir).filter(x => x.endsWith('.gguf')).map(x => path.join(modelsDir, x));
  // #views は webkitdirectory なのでディレクトリパスを渡す
  const views = path.resolve(viewsDir);
  const browser = await chromium.launch({ headless: false, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('[browser]', m.text()));
  await page.goto(process.env.PIXAL3D_BASE_URL || 'http://127.0.0.1:8199/web/real_e2e/');
  await page.setInputFiles('#models', models);
  await page.setInputFiles('#views', views);
  if (seed) await page.fill('#seed', String(seed));
  await page.click('#run');
  // ページ内 #log は worker からの postMessage で伸びるので、進捗を定期的に吐く
  let shown = 0;
  const tick = setInterval(async () => {
    try {
      const t = await page.textContent('#log');
      if (t && t.length > shown) { process.stdout.write(t.slice(shown)); shown = t.length; }
    } catch (_) { /* ページ遷移中などは黙って次の tick へ */ }
  }, 15000);
  await page.waitForFunction(() => /REAL_(GEOMETRY|FULL)_RESULT:/.test(document.querySelector('#log').textContent),
                             null, { timeout: 6 * 60 * 60 * 1000 });
  clearInterval(tick);
  const log = await page.textContent('#log');
  console.log(log);
  const ok = /REAL_(GEOMETRY|FULL)_RESULT: OK/.test(log);
  if (ok) {
    const href = await page.getAttribute('#download', 'href');
    const name = await page.getAttribute('#download', 'download');
    const bytes = await page.evaluate(async u => Array.from(new Uint8Array(await (await fetch(u)).arrayBuffer())), href);
    fs.writeFileSync(name || 'real_e2e.glb', Buffer.from(bytes));
    console.log('wrote', name, bytes.length, 'bytes');
  }
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
