// Studio の headless UI スモーク（.github/workflows/studio-headless.yml）。
// vite preview（browser mode）に対して Playwright/Chromium で次を見る:
//   1. 3 モード（TRELLIS.2 / Pixal3D SV / Pixal3D MV）の切替でパネルが排他表示される
//   2. MV + transforms.json: mesh_scale 欠落は Generate 無効・スライダ無効・既定値を捏造しない、
//      明示入力で通る、並べ替えが保存 JSON に反映される（0.9.0 からの回帰テスト）
//   3. MV canonical rig（transforms.json 無し）: ちょうど 4 枚のときだけ受け付け、
//      並び順の明示確認 + 明示 mesh_scale が揃うまで Generate 無効。5 枚は拒否
//   4. SV パネル: 不透明画像は拒否、FOV 範囲外は拒否、/capabilities が無いと Generate 無効
//   5. busy ゲート（stub サーバ）: busy=true で全モードの Generate 無効、
//      Stop waiting 後は busy=false かつ completed 増まで無効のまま
import http from 'node:http';
import { chromium } from 'playwright';

// 2x2 の PNG。rgbaPng は alpha 付き、opaquePng は alpha 無し（RGB）。
const rgbaPng = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAF0lEQVR4nAXBAQEAAAjDIG7/zhMW0CF4IBYDAUWmUmEAAAAASUVORK5CYII=', 'base64');
const opaquePng = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGM4UREARAwQCgAtrgZBSJ9wGQAAAABJRU5ErkJggg==', 'base64');
const transforms = {
  camera_angle_x: 0.7,
  // Intentionally omit mesh_scale first: preflight must block Generate until the user types it.
  frames: [
    { file_path: 'front.png', transform_matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]] },
    { file_path: 'side.png',  transform_matrix: [[0,0,1,2],[0,1,0,0],[-1,0,0,0],[0,0,0,1]] },
  ],
};

const fail = (msg) => { throw new Error(msg); };
const png = (name, buf = rgbaPng) => ({ name, mimeType: 'image/png', buffer: buf });

// ---- stub trellis-server: /health と /capabilities だけ。生成は永遠に応答しない ----------
const stub = { busy: false, completed: 0, sv: { configured: true, available: true, model_set: 'pixal3d-sv-q8_0', version: 'v1', model_family: 'sv' } };
const pending = new Set();
const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  if (req.url === '/health') { res.writeHead(200, { 'Content-Type': 'text/plain' }); res.end('ok'); return; }
  if (req.url === '/capabilities') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ busy: stub.busy, completed: stub.completed, mv: { configured: true, available: true, model_set: 'pixal3d-f16', version: 'v1', model_family: 'mv' }, sv: stub.sv }));
    return;
  }
  if (req.url === '/generate-sv' && req.method === 'POST') {
    // 生成中の振る舞い: busy にして応答を保留する（Stop waiting 側のテストが使う）。
    stub.busy = true;
    pending.add(res);
    req.resume();
    return;
  }
  res.writeHead(404); res.end();
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
const url = process.env.STUDIO_URL || 'http://127.0.0.1:4173';
// browser mode の設定は localStorage。stub サーバを指すようにしてから読み込む。
await page.addInitScript((p) => {
  localStorage.setItem('trellis.config', JSON.stringify({ host: '127.0.0.1', port: p }));
  localStorage.removeItem('trellis.mode');
}, port);
await page.goto(url, { waitUntil: 'networkidle' });

const visible = async (id) => page.locator(`#${id}`).isVisible();
const pickMode = async (mode) => {
  await page.locator(`#mode-${mode}`).check();
};

// ---- 1. モード切替 ------------------------------------------------------------------
if (!(await visible('trellis2-panel'))) fail('TRELLIS.2 panel must be the default');
if (await visible('sv-mount') || await visible('mv-calibration-mount')) fail('SV/MV panels must start hidden');
await pickMode('pixal3d-sv');
if (!(await visible('sv-mount')) || await visible('trellis2-panel')) fail('SV mode must show only the SV panel');
await pickMode('pixal3d-mv');
if (!(await visible('mv-calibration-mount')) || await visible('sv-mount')) fail('MV mode must show only the MV panel');
console.log('ok   mode switching');

// ---- 2. MV + transforms.json（0.9.0 回帰）-------------------------------------------
const input = page.locator('#mvcal-files');
await input.setInputFiles([
  { name: 'transforms.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(transforms)) },
  png('front.png'), png('side.png'),
]);
await page.locator('#mvcal-body').waitFor({ state: 'visible' });
if (await page.locator('.mvcal-view').count() !== 2) fail('expected two matched MV views');
await page.waitForFunction(() => document.querySelector('#mvcal-preflight')?.textContent?.includes('mesh_scale'));
if (!(await page.locator('#mvcal-generate').isDisabled())) fail('Generate must be disabled while mesh_scale is missing');
if (!(await page.locator('#mvcal-range').isDisabled())) fail('mesh_scale slider must stay disabled while the explicit value is missing');
if ((await page.locator('#mvcal-num').inputValue()) !== '') fail('missing mesh_scale must not be replaced by a numeric default');
await page.locator('#mvcal-range').dispatchEvent('input');
if (!(await page.locator('#mvcal-generate').isDisabled())) fail('disabled slider event must not clear mesh_scale preflight error');
if ((await page.locator('#mvcal-num').inputValue()) !== '') fail('disabled slider must not synthesize mesh_scale');
await page.locator('#mvcal-num').fill('0.250');
await page.locator('#mvcal-num').dispatchEvent('input');
await page.waitForFunction(() => !document.querySelector('#mvcal-preflight')?.textContent?.includes('error(s)'));
// サーバ online（stub）+ idle なので押せる。最初のポーリング（4 s 周期）まで待つ。
await page.waitForFunction(() => !document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 15000 });
if (await page.locator('#mvcal-range').isDisabled()) fail('slider should be enabled only after explicit manual scale exists');
if (await page.locator('#mvcal-order-confirm').isVisible()) fail('order confirmation belongs to the canonical rig only');
await page.locator('#mvcal-res').selectOption('1536');
if (await page.locator('#mvcal-res').inputValue() !== '1536') fail('resolution selector did not accept 1536');
const cards = page.locator('.mvcal-view');
await cards.nth(0).dragTo(cards.nth(1));
const downloadPromise = page.waitForEvent('download');
await page.locator('#mvcal-download').click();
const dl = await downloadPromise;
if (dl.suggestedFilename() !== 'transforms.json') fail('unexpected transforms download name');
const { readFile } = await import('node:fs/promises');
const saved = JSON.parse(await readFile(await dl.path(), 'utf8'));
if (Math.abs(saved.mesh_scale - 0.25) > 1e-6) fail(`mesh_scale not patched: ${saved.mesh_scale}`);
if (saved.frames?.[0]?.file_path !== 'side.png') fail('drag reorder was not reflected in frames order');
console.log('ok   MV with transforms.json (0.9.0 regression)');

// ---- 3. MV canonical rig（transforms.json 無し）--------------------------------------
await input.setInputFiles([png('a.png'), png('b.png'), png('c.png'), png('d.png'), png('e.png')]);
await page.waitForFunction(() => document.querySelector('#mvcal-files-summary')?.textContent?.includes('exactly 4'));
if (await page.locator('#mvcal-body').isVisible()) fail('five images without transforms.json must be rejected');

await input.setInputFiles([png('a.png'), png('b.png'), png('c.png'), png('d.png')]);
await page.locator('#mvcal-body').waitFor({ state: 'visible' });
if (await page.locator('.mvcal-view').count() !== 4) fail('canonical rig must show four cards');
if (!(await page.locator('#mvcal-order-confirm').isVisible())) fail('canonical rig must ask to confirm the view order');
if (await page.locator('#mvcal-download').isVisible()) fail('no transforms.json can be saved for the canonical rig (poses live in C++)');
if ((await page.locator('#mvcal-num').inputValue()) !== '') fail('canonical rig must not assume a mesh_scale');
await page.waitForFunction(() => document.querySelector('#mvcal-preflight')?.textContent?.includes('mesh_scale'));
if (!(await page.locator('#mvcal-generate').isDisabled())) fail('canonical rig: Generate must stay disabled without mesh_scale + order confirmation');
await page.locator('#mvcal-num').fill('0.206');
await page.locator('#mvcal-num').dispatchEvent('input');
await page.waitForFunction(() => document.querySelector('#mvcal-preflight')?.textContent?.includes('confirm that the cards are ordered'));
if (!(await page.locator('#mvcal-generate').isDisabled())) fail('canonical rig: Generate must stay disabled until the order is confirmed');
await page.locator('#mvcal-order-ok').check();
await page.waitForFunction(() => !document.querySelector('#mvcal-preflight')?.textContent?.includes('error(s)'));
await page.waitForFunction(() => !document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 15000 });
const firstCard = await page.locator('.mvcal-view .mvcal-name').first().textContent();
if (!firstCard?.startsWith('front')) fail(`canonical cards must be labelled by pose, got ${firstCard}`);
console.log('ok   MV canonical rig preflight');

// ---- 4. SV パネル ----------------------------------------------------------------------
await pickMode('pixal3d-sv');
const svFile = page.locator('#svp-file');
await svFile.setInputFiles([png('opaque.png', opaquePng)]);
await page.waitForFunction(() => document.querySelector('#svp-preflight')?.textContent?.includes('fully opaque'));
if (!(await page.locator('#svp-generate').isDisabled())) fail('SV: an opaque image must not be generatable');
await svFile.setInputFiles([png('matte.png')]);
await page.waitForFunction(() => document.querySelector('#svp-preflight')?.textContent?.includes('real alpha matte'));
await page.locator('#svp-fov').fill('200');
await page.locator('#svp-fov').dispatchEvent('input');
await page.waitForFunction(() => document.querySelector('#svp-preflight')?.textContent?.includes('between 0 and 180'));
if (!(await page.locator('#svp-generate').isDisabled())) fail('SV: FOV 200 deg must be rejected');
await page.locator('#svp-fov').fill('20');
await page.locator('#svp-fov').dispatchEvent('input');
await page.waitForFunction(() => document.querySelector('#svp-avail')?.textContent?.includes('ready'), null, { timeout: 15000 });
await page.waitForFunction(() => !document.querySelector('#svp-generate')?.disabled, null, { timeout: 15000 });
console.log('ok   SV panel preflight + availability');

// sv.available=false → 理由を表示して Generate 無効
stub.sv = { configured: true, available: false, reason: 'missing model file: pixal3d_tex_flow_1024_sv.gguf' };
await page.waitForFunction(() => document.querySelector('#svp-avail')?.textContent?.includes('missing model file'), null, { timeout: 15000 });
if (!(await page.locator('#svp-generate').isDisabled())) fail('SV: unavailable model set must disable Generate');
stub.sv = { configured: true, available: true, model_set: 'pixal3d-sv-q8_0', version: 'v1', model_family: 'sv' };
await page.waitForFunction(() => !document.querySelector('#svp-generate')?.disabled, null, { timeout: 15000 });
console.log('ok   SV availability follows /capabilities');

// ---- 5. busy ゲートと Stop waiting -------------------------------------------------------
stub.busy = true;
await page.waitForFunction(() => document.querySelector('#svp-generate')?.disabled, null, { timeout: 15000 });
await pickMode('pixal3d-mv');
if (!(await page.locator('#mvcal-generate').isDisabled())) fail('busy=true must disable Generate in MV mode too');
await page.waitForFunction(() => document.querySelector('#mvcal-blocked')?.textContent?.includes('busy'));
stub.busy = false;
await pickMode('pixal3d-sv');
await page.waitForFunction(() => !document.querySelector('#svp-generate')?.disabled, null, { timeout: 15000 });

// Generate → stub は応答を保留（busy=true 相当）→ Stop waiting → completed が増えるまで無効
await page.locator('#svp-generate').click();
await page.waitForFunction(() => !document.querySelector('#svp-stop')?.disabled);
await page.locator('#svp-stop').click();
await page.waitForFunction(() => document.querySelector('#svp-stage')?.textContent?.includes('stopped waiting'));
if (!(await page.locator('#svp-generate').isDisabled())) fail('after Stop waiting Generate must stay disabled while the server is busy');
// サーバが終わる: busy=false だが completed は据え置き → まだ無効（完了を確認できていない）
stub.busy = false;
await page.waitForTimeout(5000);
if (!(await page.locator('#svp-generate').isDisabled())) fail('busy=false alone must not re-enable Generate; completed must advance');
stub.completed += 1;
for (const res of pending) { res.writeHead(500, { 'Content-Type': 'application/json' }); res.end('{"error":"stub"}'); }
pending.clear();
await page.waitForFunction(() => !document.querySelector('#svp-generate')?.disabled, null, { timeout: 15000 });
console.log('ok   busy gate + Stop waiting re-enable');

console.log('HEADLESS_MV_PREFLIGHT_OK');
await browser.close();
server.close();
