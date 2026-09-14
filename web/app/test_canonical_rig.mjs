// 合成リグ（transforms.json 無しの 4 視点）の回帰テスト。
//   1. 公式 cyclops の 4 枚 + mesh_scale=1.0 が、同梱 transforms.json を全フィールド再現する
//   2. 合成モードでは mesh_scale を明示確定するまで生成できない
//   3. 4 枚以外は明示エラーで拒否される
//   4. カードの並べ替えで姿勢ではなく画像が動く
//
//   node web/app/test_canonical_rig.mjs [views_dir]
// views_dir 既定は Pixal3D の assets/mv_images/example（4 枚 + transforms.json）。
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { chromium } from 'playwright';

const viewsDir = process.argv[2] || process.env.PIXAL3D_VIEWS_DIR;
if (!viewsDir || !fs.existsSync(path.join(viewsDir, 'transforms.json'))) {
  console.log('SKIP: need a views dir with transforms.json (arg 1 or PIXAL3D_VIEWS_DIR)');
  process.exit(0);
}
const reference = JSON.parse(fs.readFileSync(path.join(viewsDir, 'transforms.json'), 'utf8'));
const images = reference.frames.map((f) => path.join(viewsDir, f.file_path));

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../..');
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
  const file = path.join(root, rel);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end('no');
  const ext = path.extname(file);
  const type = ext === '.html' ? 'text/html' : ext === '.js' ? 'text/javascript' : ext === '.json' ? 'application/json'
    : ext === '.wasm' ? 'application/wasm' : 'application/octet-stream';
  res.writeHead(200, { 'content-type': type });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const APP = `http://127.0.0.1:${server.address().port}/web/app/index.html`;

let failed = false;
const check = (ok, label, detail = '') => {
  console.log(`${ok ? 'ok' : 'FAIL'}: ${label}${detail ? ` (${detail})` : ''}`);
  if (!ok) failed = true;
};

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('pageerror', (e) => check(false, 'pageerror', e.message));
await page.goto(APP, { waitUntil: 'domcontentloaded' });
const viewInput = page.locator('input[accept="image/*,.json,application/json"]');

// ---- 1/2. 4 枚・JSON 無し ------------------------------------------------
await viewInput.setInputFiles(images);
await page.waitForFunction(() => document.querySelectorAll('.cal-view').length === 4);

const calState = () => page.evaluate(() => ({ ready: window.__calibration.isReady(),
  synthetic: window.__calibration.isSynthetic(), confirmed: window.__calibration.isScaleConfirmed() }));
let st = await calState();
check(st.synthetic === true, 'synthetic mode is active without transforms.json');
check(st.ready === false, 'synthetic mode is not ready before scale confirmation', JSON.stringify(st));
check(/confirm mesh_scale/i.test((await page.locator('#cal-badge').textContent()) || ''),
  'the badge asks for an explicit mesh_scale');

await page.fill('#cal-num', String(reference.mesh_scale));
await page.locator('#cal-num').dispatchEvent('input');
st = await calState();
check(st.ready === false && st.confirmed === false, 'typing a scale alone does not arm generation', JSON.stringify(st));
await page.locator('#cal-confirm').click();
st = await calState();
check(st.ready === true && st.confirmed === true, 'confirming the scale arms generation', JSON.stringify(st));

const [download] = await Promise.all([page.waitForEvent('download'), page.click('#cal-save')]);
const generated = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
check(generated.camera_angle_x === reference.camera_angle_x, 'camera_angle_x matches',
  `${generated.camera_angle_x} vs ${reference.camera_angle_x}`);
check(generated.mesh_scale === reference.mesh_scale, 'mesh_scale matches',
  `${generated.mesh_scale} vs ${reference.mesh_scale}`);
check(generated.frames.length === reference.frames.length, 'frame count matches');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
generated.frames.forEach((f, i) => {
  check(f.file_path === reference.frames[i].file_path, `frame ${i} file_path matches`);
  check(same(f.transform_matrix, reference.frames[i].transform_matrix), `frame ${i} transform_matrix matches`);
});

// ---- 4. 並べ替えは画像を動かし、姿勢は位置に固定 --------------------------
const titlesBefore = await page.locator('.cal-view').evaluateAll((cs) => cs.map((c) => c.title));
await page.evaluate(() => {
  const cs = [...document.querySelectorAll('.cal-view')];
  cs[0].dispatchEvent(new DragEvent('dragstart', { bubbles: true }));
  cs[2].dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true }));
});
await page.waitForTimeout(150);
const titlesAfter = await page.locator('.cal-view').evaluateAll((cs) => cs.map((c) => c.title));
const labelsAfter = await page.locator('.cal-view span').allInnerTexts();
check(titlesAfter[0] === titlesBefore[1] && titlesAfter[2] === titlesBefore[0], 'reordering moves the images');
check(/front/i.test(labelsAfter[0]) && /back/i.test(labelsAfter[2]), 'poses stay bound to card position');

// ---- 3. 4 枚以外は拒否 ----------------------------------------------------
await viewInput.setInputFiles(images.slice(0, 3));
await page.waitForTimeout(300);
check((await calState()).ready === false, 'three views are rejected');
check(/transforms\.json missing/i.test((await page.locator('#cal-badge').textContent()) || ''),
  'three views report the missing transforms.json explicitly');

await browser.close();
server.close();
console.log(failed ? 'WEB_CANONICAL_RIG_FAILED' : 'WEB_CANONICAL_RIG_OK');
process.exit(failed ? 1 : 0);
