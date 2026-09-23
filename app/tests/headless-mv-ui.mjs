// Studio の headless UI スモーク（.github/workflows/studio-headless.yml）。
// vite preview（browser mode）に対して Playwright/Chromium で次を見る:
//   1. 4 modes (TRELLIS.2 single/MV, Pixal3D SV/MV) switch panels exclusively
//   2. MV + transforms.json: mesh_scale 欠落は Generate 無効・スライダ無効・既定値を捏造しない、
//      明示入力で通る、並べ替えが保存 JSON に反映される（0.9.0 からの回帰テスト）
//   3. MV canonical rig（transforms.json 無し）: ちょうど 4 枚のときだけ受け付け、
//      並び順の明示確認 + 明示 mesh_scale が揃うまで Generate 無効。5 枚は拒否
//   4. SV パネル: 不透明画像は拒否、FOV 範囲外は拒否、/capabilities が無いと Generate 無効
//   5. busy ゲート（stub サーバ）: busy=true で全モードの Generate 無効、
//      Stop waiting 後は busy=false かつ completed 増まで無効のまま
//   6. MV 完了パス（#25）: /generate-mv の結果が viewer / gallery / Save GLB… に届き、
//      黙ってダウンロードされず、gallery から開き直しても TRELLIS.2 の入力を汚さない
//   7. サーバ消失（#36、browser mode）: 生成中に stub が接続を切る（/health も落ちる、/generate-mv は
//      応答しない）→ /health 3 回連続失敗で err toast、進行状態が畳まれ、サーバ復帰後に Generate が
//      再有効化される。先に、短い断（/health 1 回失敗）では生成が捨てられず完了することを見る
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

// 三角形 1 枚の untextured GLB。/generate-mv の stub 応答（model-viewer がロードできる最小構成）。
function minimalGlb() {
  const bin = Buffer.alloc(36);
  [[0, 0, 0], [1, 0, 0], [0, 1, 0]].flat().forEach((v, i) => bin.writeFloatLE(v, i * 4));
  const json = Buffer.from(JSON.stringify({
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] }],
    bufferViews: [{ buffer: 0, byteLength: bin.length }],
    buffers: [{ byteLength: bin.length }],
  }));
  const pad = Buffer.alloc((4 - (json.length % 4)) % 4, 0x20);
  const chunk = (type, body) => {
    const h = Buffer.alloc(8);
    h.writeUInt32LE(body.length, 0);
    h.write(type, 4, 'ascii');
    return Buffer.concat([h, body]);
  };
  const chunks = Buffer.concat([chunk('JSON', Buffer.concat([json, pad])), chunk('BIN\0', bin)]);
  const header = Buffer.alloc(12);
  header.write('glTF', 0, 'ascii');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + chunks.length, 8);
  return Buffer.concat([header, chunks]);
}

// ---- stub trellis-server: /health, /capabilities, /generate-mv（即答）。/generate-sv は永遠に応答しない ----
const stub = { busy: false, completed: 0, mvRequests: 0, t2mvRequests: 0, dead: false, holdMv: false, sv: { configured: true, available: true, model_set: 'pixal3d-sv-q8_0', version: 'v1', model_family: 'sv' } };
const pending = new Set();
const pendingMv = new Set();
const server = http.createServer((req, res) => {
  // dead: サーバが落ちた状態。新しい接続は即切断（fetch は失敗）。保留中の応答はそのまま
  // 放置する（#36 で観測した「fetch が永遠に返らない」状態を再現する）。
  if (stub.dead) { req.socket.destroy(); return; }
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  if (req.url === '/health') { res.writeHead(200, { 'Content-Type': 'text/plain' }); res.end('ok'); return; }
  if (req.url === '/capabilities') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ busy: stub.busy, completed: stub.completed, mv: { configured: true, available: true, model_set: 'pixal3d-f16', version: 'v1', model_family: 'mv' }, sv: stub.sv, trellis2_mv: { available: true, max_images: 8, modes: ['stochastic','multidiffusion'] } }));
    return;
  }
  if (req.url === '/generate-trellis2-mv' && req.method === 'POST') {
    stub.t2mvRequests++;
    stub.completed++;
    req.resume();
    req.on('end', () => { res.writeHead(200, { 'Content-Type': 'model/gltf-binary' }); res.end(minimalGlb()); });
    return;
  }
  if (req.url === '/generate-mv' && req.method === 'POST') {
    stub.mvRequests++;
    if (stub.holdMv) { pendingMv.add(res); req.resume(); return; }
    stub.completed++;
    req.resume();
    req.on('end', () => { res.writeHead(200, { 'Content-Type': 'model/gltf-binary' }); res.end(minimalGlb()); });
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
if (await visible('trellis2-mv-mount') || await visible('sv-mount') || await visible('mv-calibration-mount')) fail('non-default panels must start hidden');
await pickMode('trellis2-mv');
if (!(await visible('trellis2-mv-mount')) || await visible('trellis2-panel')) fail('TRELLIS.2 MV mode must show only its panel');
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

// ---- 6. MV 完了パス（#25）-----------------------------------------------------------
// 結果は main.ts 経由で viewer / gallery / Save GLB… に届き、黙ってダウンロードされない。
// stub が /generate-mv に三角形 1 枚の GLB を返す（model-viewer がロードできる最小構成）。
await pickMode('pixal3d-mv');
await input.setInputFiles([
  { name: 'transforms.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ ...transforms, mesh_scale: 0.25 })) },
  png('front.png'), png('side.png'),
]);
await page.locator('#mvcal-body').waitFor({ state: 'visible' });
await page.waitForFunction(() => !document.querySelector('#mvcal-preflight')?.textContent?.includes('error(s)'));
await page.locator('#mvcal-res').selectOption('1536');
await page.waitForFunction(() => !document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 15000 });
const silentDownload = page.waitForEvent('download', { timeout: 2500 }).then(() => true, () => false);
await page.locator('#mvcal-generate').click();
await page.waitForFunction(() => document.querySelector('#mvcal-stage')?.textContent === 'complete', null, { timeout: 15000 });
if (stub.mvRequests !== 1) fail(`expected one /generate-mv request, saw ${stub.mvRequests}`);
await page.waitForFunction(() => document.querySelector('#viewer-caption')?.textContent?.includes('mesh_scale 0.250'), null, { timeout: 10000 });
const caption = await page.locator('#viewer-caption').textContent();
if (!caption.startsWith('Pixal3D MV · 1536 · seed 42')) fail(`unexpected MV caption: ${caption}`);
if (await page.locator('#save-glb').isDisabled()) fail('Save GLB… must be enabled after an MV result');
if (await page.locator('#reset-view').isDisabled()) fail('Reset view must be enabled after an MV result');
if (await page.locator('.gitem').count() !== 1) fail('MV result must be registered in the gallery');
const gmeta = await page.locator('.gitem .gmeta').textContent();
if (gmeta !== 'MV 1536') fail(`unexpected gallery meta: ${gmeta}`);
if (await silentDownload) fail('MV result must not trigger a silent browser download');
// Save GLB…（browser mode）が明示のダウンロード。名前は MV 実行に由来する。
const savePromise = page.waitForEvent('download');
await page.locator('#save-glb').click();
const savedGlb = await savePromise;
if (savedGlb.suggestedFilename() !== 'multiview_1536_seed42_scale0.250.glb') fail(`unexpected Save GLB name: ${savedGlb.suggestedFilename()}`);
// gallery から MV レコードを開き直しても TRELLIS.2 のドロップゾーンを上書きしない。
// loadRecord は model-viewer がロードできないと UI に触らず戻るので、この段だけ WebGL が要る
// （headless Chromium は SwiftShader で持つ）。無い環境では明示してスキップする。
const hasWebgl = await page.evaluate(() => !!document.createElement('canvas').getContext('webgl2'));
if (hasWebgl) {
  await page.locator('.gitem').first().click();
  await page.waitForFunction(() => document.querySelector('#viewer-caption')?.textContent?.startsWith('Pixal3D MV · multiview ·'), null, { timeout: 10000 });
  if (!(await page.locator('#input-preview').evaluate((el) => el.classList.contains('hidden')))) fail('loading an MV record must not populate the single-image dropzone');
} else {
  console.log('HEADLESS_MV_REOPEN_SKIPPED (no WebGL in this browser)');
}
console.log('ok   MV completion path (viewer / gallery / Save GLB…, no silent download)');

console.log('HEADLESS_MV_COMPLETION_OK');

// ---- 7. サーバ消失（#36）---------------------------------------------------------
// MV パネル（section 6 の入力のまま）で Generate → stub は /generate-mv を保留 → stub が「死ぬ」
// （新規接続は切断、保留中の応答は返さない）。browser mode では pollHealth が生成中の offline を
// 2 回続けて見たら handleServerLost: err toast、fetch abort、進行状態を畳む。
const startHeldMv = async (expectRequests) => {
  await page.waitForFunction(() => !document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 15000 });
  stub.holdMv = true;
  await page.locator('#mvcal-generate').click();
  await page.waitForFunction(() => !document.querySelector('#mvcal-cancel')?.disabled);
  for (let i = 0; i < 40 && pendingMv.size === 0; i++) await new Promise((r) => setTimeout(r, 250));
  if (pendingMv.size !== 1 || stub.mvRequests !== expectRequests) fail(`expected /generate-mv request #${expectRequests} to be held, saw ${stub.mvRequests} (held ${pendingMv.size})`);
};
// 7a. 短い断: /health が 1 回落ちる（4 s 周期のポーリング 1 回分）だけでは生成を捨てない。
await startHeldMv(2);
stub.dead = true;
await page.waitForTimeout(4500);
stub.dead = false;
await page.waitForTimeout(4500);
if (await page.locator('#mvcal-cancel').isDisabled()) fail('a single missed health poll must not abandon the generation');
if ([...await page.locator('.toast.err').allTextContents()].some((t) => t.includes('stopped responding'))) fail('no server-lost toast for a short outage');
stub.holdMv = false;
stub.completed++;
for (const res of pendingMv) { res.writeHead(200, { 'Content-Type': 'model/gltf-binary' }); res.end(minimalGlb()); }
pendingMv.clear();
await page.waitForFunction(() => document.querySelector('#mvcal-stage')?.textContent === 'complete', null, { timeout: 15000 });
console.log('ok   short health outage keeps the generation (no false server-lost)');

// 7b. 本当に消えた: /health が 3 回続けて落ちる → toast、進行状態が畳まれる。
await startHeldMv(3);
stub.dead = true;
await page.waitForFunction(() => [...document.querySelectorAll('.toast.err')].some((t) => t.textContent.includes('server stopped responding during generation')), null, { timeout: 25000 });
await page.waitForFunction(() => document.querySelector('#mvcal-cancel')?.disabled, null, { timeout: 5000 });
const lostStage = await page.locator('#mvcal-stage').textContent();
if (!lostStage.includes('server stopped responding')) fail(`MV stage should name the lost server, got: ${lostStage}`);
if (!(await page.locator('#mvcal-generate').isDisabled())) fail('Generate must stay disabled while the server is gone');
if (await page.locator('#server-label').textContent() !== 'offline') fail('status must show offline after the server was lost');
// サーバ復帰（Settings → Restart server 相当）: 保留していた応答は返らないまま、Generate が戻る
stub.dead = false;
stub.holdMv = false;
for (const res of pendingMv) res.socket.destroy();
pendingMv.clear();
await page.waitForFunction(() => !document.querySelector('#mvcal-generate')?.disabled, null, { timeout: 15000 });
if (await page.locator('#server-label').textContent() !== 'ready') fail('status must be ready again once the server answers');
console.log('ok   server lost mid-generation → toast, state cleared, Generate re-enabled after restart');

console.log('HEADLESS_SERVER_LOST_OK');

// ---- 8. TRELLIS.2 multiview completion ----------------------------------------------
await pickMode('trellis2-mv');
const t2file = page.locator('#t2mvp-file');
await t2file.setInputFiles([png('view0.png'), png('view1.png')]);
await page.locator('#t2mvp-fusion').selectOption('multidiffusion');
await page.locator('#t2mvp-res').selectOption('1024');
await page.waitForFunction(() => !document.querySelector('#t2mvp-generate')?.disabled, null, { timeout: 15000 });
await page.locator('#t2mvp-generate').click();
await page.waitForFunction(() => document.querySelector('#t2mvp-stage')?.textContent === 'complete', null, { timeout: 15000 });
if (stub.t2mvRequests !== 1) fail(`expected one /generate-trellis2-mv request, saw ${stub.t2mvRequests}`);
await page.waitForFunction(() => document.querySelector('#viewer-caption')?.textContent?.includes('TRELLIS.2 MV'), null, { timeout: 10000 });
const t2cap = await page.locator('#viewer-caption').textContent();
if (!t2cap.includes('2 views') || !t2cap.includes('multidiffusion')) fail(`TRELLIS.2 MV caption missing metadata: ${t2cap}`);
if (await page.locator('#save-glb').isDisabled()) fail('Save GLB must be enabled after TRELLIS.2 MV result');
console.log('ok   TRELLIS.2 MV completion path');
console.log('HEADLESS_TRELLIS2_MV_OK');

await browser.close();
server.close();
