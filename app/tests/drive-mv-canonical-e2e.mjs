// A12 用ドライバ（drive-sv-e2e.mjs の MV canonical 版）: Studio のフロントエンド（vite preview の同一ビルド）を headless Chromium で
// 操作し、Tauri の Studio が spawn した実 trellis-server に対して Pixal3D MV canonical 4-view（transforms.json 無し）生成を 1 回回す。
// 結果 GLB は `pixal3d-mv-result` イベントから受け取ってファイルへ書く。
//
//   STUDIO_URL=http://127.0.0.1:4173 SERVER_PORT=8080 IMAGES=front.png,right.png,back.png,left.png MESH_SCALE=1.0 OUT_DIR=... node tests/drive-mv-canonical-e2e.mjs
//
// 判定はしない（IoU ゲートと 6 視点レンダは別ツール）。ここは「UI 経由で完走し GLB を得る」だけ。
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const url = process.env.STUDIO_URL || 'http://127.0.0.1:4173';
const port = Number(process.env.SERVER_PORT || 8080);
const images = (process.env.IMAGES || '').split(',').filter(Boolean);
const outDir = process.env.OUT_DIR;
const meshScale = process.env.MESH_SCALE || '';
const seed = process.env.SEED || '42';
if (images.length !== 4 || !outDir || !meshScale) { console.error('IMAGES (4, ordered front,right,back,left), MESH_SCALE and OUT_DIR are required'); process.exit(2); }
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('[console.error]', m.text()); });
await page.addInitScript((p) => {
  localStorage.setItem('trellis.config', JSON.stringify({ host: '127.0.0.1', port: p }));
  localStorage.removeItem('trellis.mode');
}, port);

let resolveResult; const result = new Promise((r) => { resolveResult = r; });
await page.exposeFunction('__mvResult', (b64, meta) => resolveResult({ b64, meta }));
await page.exposeFunction('__mvError', (msg) => resolveResult({ error: msg }));
await page.addInitScript(() => {
  window.addEventListener('pixal3d-mv-result', async (e) => {
    const d = e.detail;
    const buf = new Uint8Array(await d.glb.arrayBuffer());
    let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    window.__mvResult(btoa(s), { name: d.name, meshScale: d.meshScale, resolution: d.resolution, seed: d.seed, canonical: d.canonical, bytes: buf.length });
  });
  window.addEventListener('pixal3d-mv-error', (e) => window.__mvError(String(e.detail)));
});

await page.goto(url, { waitUntil: 'load' });
await page.locator('#mode-pixal3d-mv').check();
await page.locator('#mvcal-files').setInputFiles(images);
await page.locator('#mvcal-body').waitFor({ state: 'visible' });
await page.locator('#mvcal-num').fill(meshScale);
await page.locator('#mvcal-num').dispatchEvent('input');
await page.locator('#mvcal-res').selectOption('1024');
await page.locator('#mvcal-seed').fill(seed);
await page.locator('#mvcal-order-ok').check();
await page.waitForFunction(() => !document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 60000 });
console.log('cards:', (await page.locator('.mvcal-view .mvcal-name').allTextContents()).join(', '));
console.log('preflight:', (await page.locator('#mvcal-preflight').innerText()).replace(/\n/g, ' | '));
await page.screenshot({ path: path.join(outDir, 'ui_before_generate.png'), fullPage: true });

const t0 = Date.now();
console.log('click Generate at', new Date().toISOString());
await page.locator('#mvcal-generate').click();
await page.waitForFunction(() => document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 10000 });
const stageLog = fs.createWriteStream(path.join(outDir, 'ui_stage.log'));
const timer = setInterval(async () => {
  try { stageLog.write(`${((Date.now() - t0) / 1000).toFixed(0)}s ${await page.locator('#mvcal-stage').innerText()}\n`); } catch {}
}, 10000);

const r = await Promise.race([result, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout 60 min')), 60 * 60 * 1000))]).catch((e) => ({ error: String(e) }));
clearInterval(timer); stageLog.end();
const wall = (Date.now() - t0) / 1000;
if (r.error) { console.log('MV_E2E_FAILED', r.error, `wall=${wall}s`); await browser.close(); process.exit(1); }
const glbPath = path.join(outDir, 'out.glb');
fs.writeFileSync(glbPath, Buffer.from(r.b64, 'base64'));
console.log('result:', JSON.stringify(r.meta), `wall=${wall.toFixed(1)}s`);
console.log('stage:', await page.locator('#mvcal-stage').innerText());
await page.waitForTimeout(3000);
await page.screenshot({ path: path.join(outDir, 'ui_after_generate.png'), fullPage: true });
console.log('MV_E2E_DONE', glbPath);
await browser.close();
