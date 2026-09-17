// 本番 URL のデプロイ後スモーク（docs/runbooks/web-deploy.md）。
// test_headless.mjs は「SV manifest が未設定であること」を assert するので本番には使えない。
// ここでは SV が明示注入された配信物に対して次を確認する:
//   1. index.html に MV base URL と SV の 2 URL が注入されている
//   2. real_e2e/build-info.json が同梱され、配信されている wasm の SHA256 と一致する
//   3. #input-mode が見え、SV が選べ、SV manifest が SV family として解決し #download-models が有効
//   4. MV に戻すと MV manifest が解決し #download-models が有効
// 実 GPU は使わない（headless Chromium、WebGPU の可否はログに出すだけ）。
//   node web/app/smoke_production.mjs https://pixal3d-web.raven38.workers.dev/
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

const base = (process.argv[2] || process.env.WEB_APP_URL || '').replace(/\/?$/, '/');
if (!base) throw new Error('usage: smoke_production.mjs <app url>');
const checks = [];
const check = (ok, label, detail = '') => { checks.push({ ok, label, detail }); console.log(`${ok ? 'ok ' : 'NG '}: ${label}${detail ? ` (${detail})` : ''}`); if (!ok) process.exitCode = 1; };

const html = await (await fetch(base, { cache: 'no-store' })).text();
const urlOf = (key) => (html.match(new RegExp(`window\\.${key} = "([^"]+)"`)) || [])[1] || '';
const mvBase = urlOf('PIXAL3D_MODEL_BASE_URL'), svManifest = urlOf('PIXAL3D_SV_MODEL_MANIFEST_URL'), svBase = urlOf('PIXAL3D_SV_MODEL_BASE_URL');
check(!!mvBase, 'MV model base URL injected', mvBase);
check(!!svManifest && !!svBase, 'SV manifest + base URL injected', `${svManifest} | ${svBase}`);
check(html.includes('id="input-mode"'), 'index.html carries the input-mode selector');

const info = await (await fetch(`${base}real_e2e/build-info.json`, { cache: 'no-store' })).json().catch(() => null);
check(!!info?.files?.['pixal3d_real_geometry.wasm']?.sha256, 'real_e2e/build-info.json served', info ? `commit ${String(info.source_commit).slice(0, 12)} · ${info.emcc}` : 'missing');
const wasm = Buffer.from(await (await fetch(`${base}real_e2e/pixal3d_real_geometry.wasm`, { cache: 'no-store' })).arrayBuffer());
const wasmSha = createHash('sha256').update(wasm).digest('hex');
check(wasmSha === info?.files?.['pixal3d_real_geometry.wasm']?.sha256, 'served wasm SHA256 matches build-info', `${wasmSha} · ${wasm.length} B`);

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
page.on('console', (m) => { if (/release manifest|WebGPU/.test(m.text())) console.log(`   [page] ${m.text()}`); });
await page.goto(base, { waitUntil: 'networkidle' });
await page.waitForFunction(() => /WebGPU/.test(document.querySelector('#webgpu')?.textContent || ''));
console.log(`   preflight: ${await page.locator('#webgpu').textContent()}`);
check(await page.locator('#input-mode').isVisible(), 'input-mode selector visible');
const initialMode = await page.locator('#input-mode').inputValue();
check(initialMode === (svManifest ? 'sv' : 'mv'), 'default input mode follows SV configuration', `default=${initialMode}`);

const logText = async () => (await page.locator('#log').textContent().catch(() => '')) || '';
for (const mode of ['sv', 'mv']) {
  await page.locator('#input-mode').selectOption(mode);
  const expected = mode === 'sv' ? 'pixal3d-sv-q8_0' : 'pixal3d-q8_0';
  await page.waitForFunction((needle) => (document.querySelector('#log')?.textContent || '').includes(`release manifest ${needle}`), expected, { timeout: 30_000 }).catch(() => {});
  const resolved = (await logText()).includes(`release manifest ${expected}`);
  check(resolved, `${mode.toUpperCase()} release manifest resolves to ${expected}`);
  check(!(await page.locator('#download-models').isDisabled()), `${mode.toUpperCase()}: Download release models enabled`);
  check(!!(await page.locator(mode === 'sv' ? '#sv-panel' : '#mv-panel').isVisible()), `${mode.toUpperCase()} panel shown`);
}
await browser.close();
const failed = checks.filter((c) => !c.ok).length;
console.log(failed ? `WEB_PRODUCTION_SMOKE_FAILED ${failed}/${checks.length}` : `WEB_PRODUCTION_SMOKE_OK ${checks.length} checks`);
