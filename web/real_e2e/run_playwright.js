// 実入力（transforms.json + pre-matted RGBA）の E2E を Chrome/WebGPU で回す harness。
// models_dir に texture 2本（pixal3d_tex_flow_1024_mv.gguf / tex_dec.gguf）が入っていれば
// live Texture conditioning 込みの true full E2E、無ければ geometry E2E になる。
// Usage:
//   node web/real_e2e/run_playwright.js <models_dir> <views_dir> [seed]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// 失敗経路でも必ず Chromium を落とす。落とし損ねると node が待ち続け、
// Chromium が GPU を掴んだまま何時間も残ってマシンを重くする（実測 5 時間）。
let browser = null;
const shutdown = async () => { try { if (browser) await browser.close(); } catch (_) {} browser = null; };
for (const sig of ['SIGINT', 'SIGTERM'])
  process.on(sig, async () => { await shutdown(); process.exit(130); });

(async () => {
  const [modelsDir, viewsDir, seed] = process.argv.slice(2);
  if (!modelsDir || !viewsDir) throw new Error('need <models_dir> <views_dir> [seed]');
  const models = fs.readdirSync(modelsDir).filter(x => x.endsWith('.gguf')).map(x => path.join(modelsDir, x));
  // #views は webkitdirectory なのでディレクトリパスを渡す
  const views = path.resolve(viewsDir);
  browser = await chromium.launch({ headless: false, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  const readBlob = async (u) => {
    // 大きい GLB（geometry only は 100 MB 超）は Array.from で RangeError になるので、
    // 8 MB ずつ base64 にして取り出す。
    const CH = 8 * 1024 * 1024;
    const size = await page.evaluate(async (url) => (await (await fetch(url)).blob()).size, u);
    const parts = [];
    for (let off = 0; off < size; off += CH) {
      const b64 = await page.evaluate(async ([url, o, n]) => {
        const buf = await (await fetch(url)).arrayBuffer();
        const view = new Uint8Array(buf, o, Math.min(n, buf.byteLength - o));
        let s = '';
        for (let i = 0; i < view.length; i += 0x8000) s += String.fromCharCode.apply(null, view.subarray(i, i + 0x8000));
        return btoa(s);
      }, [u, off, CH]);
      parts.push(Buffer.from(b64, 'base64'));
    }
    return Buffer.concat(parts);
  };
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
  await page.waitForFunction(() => { const t = document.querySelector('#log').textContent;
    return /REAL_(GEOMETRY|FULL)_RESULT:/.test(t) || /ERROR:|EXCEPTION|Aborted\(\)|RuntimeError/.test(t); },
                             null, { timeout: 90 * 60 * 1000 });
  clearInterval(tick);
  const log = await page.textContent('#log');
  console.log(log);
  const ok = /REAL_(GEOMETRY|FULL)_RESULT: OK/.test(log);
  if (ok) {
    const href = await page.getAttribute('#download', 'href');
    const name = await page.getAttribute('#download', 'download');
    const buf = await readBlob(href);
    fs.writeFileSync(name || 'real_e2e.glb', buf);
    console.log('wrote', name, buf.length, 'bytes');
  }
  await shutdown();
  process.exit(ok ? 0 : 1);
})().catch(async e => { console.error(e); await shutdown(); process.exit(1); });
