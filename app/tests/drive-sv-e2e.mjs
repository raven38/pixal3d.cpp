// A10 用ドライバ: Studio のフロントエンド（vite preview の同一ビルド）を headless Chromium で
// 操作し、Tauri の Studio が spawn した実 trellis-server に対して Pixal3D SV 生成を 1 回回す。
// 結果 GLB は `pixal3d-sv-result` イベントから受け取ってファイルへ書く。
//
//   STUDIO_URL=http://127.0.0.1:4173 SERVER_PORT=8080 IMAGE=... OUT_DIR=... node tests/drive-sv-e2e.mjs
//
// 判定はしない（IoU ゲートと 6 視点レンダは別ツール）。ここは「UI 経由で完走し GLB を得る」だけ。
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const url = process.env.STUDIO_URL || 'http://127.0.0.1:4173';
const port = Number(process.env.SERVER_PORT || 8080);
const image = process.env.IMAGE;
const outDir = process.env.OUT_DIR;
const fovDeg = process.env.FOV_DEG || '20';
const seed = process.env.SEED || '42';
if (!image || !outDir) { console.error('IMAGE and OUT_DIR are required'); process.exit(2); }
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('[console.error]', m.text()); });
await page.addInitScript((p) => {
  localStorage.setItem('trellis.config', JSON.stringify({ host: '127.0.0.1', port: p }));
  localStorage.removeItem('trellis.mode');
}, port);

let resolveResult; const result = new Promise((r) => { resolveResult = r; });
await page.exposeFunction('__svResult', (b64, meta) => resolveResult({ b64, meta }));
await page.exposeFunction('__svError', (msg) => resolveResult({ error: msg }));
await page.addInitScript(() => {
  window.addEventListener('pixal3d-sv-result', async (e) => {
    const d = e.detail;
    const buf = new Uint8Array(await d.glb.arrayBuffer());
    let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    window.__svResult(btoa(s), { name: d.name, fovRad: d.fovRad, seed: d.seed, bytes: buf.length });
  });
  window.addEventListener('pixal3d-sv-error', (e) => window.__svError(String(e.detail)));
});

await page.goto(url, { waitUntil: 'load' });
await page.locator('#mode-pixal3d-sv').check();
await page.locator('#svp-file').setInputFiles([image]);
await page.locator('#svp-fov').fill(fovDeg);
await page.locator('#svp-seed').fill(seed);
await page.waitForFunction(() => !document.querySelector('#svp-generate')?.disabled, null, { timeout: 60000 });
console.log('availability:', await page.locator('#svp-avail').innerText());
console.log('preflight:', (await page.locator('#svp-preflight').innerText()).replace(/\n/g, ' | '));
await page.screenshot({ path: path.join(outDir, 'ui_before_generate.png'), fullPage: true });

const t0 = Date.now();
console.log('click Generate at', new Date().toISOString());
await page.locator('#svp-generate').click();
await page.waitForFunction(() => document.querySelector('#svp-generate')?.disabled, null, { timeout: 10000 });
const stageLog = fs.createWriteStream(path.join(outDir, 'ui_stage.log'));
const timer = setInterval(async () => {
  try { stageLog.write(`${((Date.now() - t0) / 1000).toFixed(0)}s ${await page.locator('#svp-stage').innerText()}\n`); } catch {}
}, 10000);

const r = await Promise.race([result, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout 60 min')), 60 * 60 * 1000))]).catch((e) => ({ error: String(e) }));
clearInterval(timer); stageLog.end();
const wall = (Date.now() - t0) / 1000;
if (r.error) { console.log('SV_E2E_FAILED', r.error, `wall=${wall}s`); await browser.close(); process.exit(1); }
const glbPath = path.join(outDir, 'out.glb');
fs.writeFileSync(glbPath, Buffer.from(r.b64, 'base64'));
console.log('result:', JSON.stringify(r.meta), `wall=${wall.toFixed(1)}s`);
console.log('stage:', await page.locator('#svp-stage').innerText());
await page.waitForTimeout(3000);
await page.screenshot({ path: path.join(outDir, 'ui_after_generate.png'), fullPage: true });
console.log('SV_E2E_DONE', glbPath);
await browser.close();
