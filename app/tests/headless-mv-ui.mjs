import { chromium } from 'playwright';

const rgbaPng = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAF0lEQVR4nAXBAQEAAAjDIG7/zhMW0CF4IBYDAUWmUmEAAAAASUVORK5CYII=', 'base64');
const transforms = {
  camera_angle_x: 0.7,
  // Intentionally omit mesh_scale first: preflight must block Generate until the user types it.
  frames: [
    { file_path: 'front.png', transform_matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]] },
    { file_path: 'side.png',  transform_matrix: [[0,0,1,2],[0,1,0,0],[-1,0,0,0],[0,0,0,1]] },
  ],
};

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:4173', { waitUntil: 'networkidle' });
const input = page.locator('#mvcal-files');
await input.setInputFiles([
  { name: 'transforms.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(transforms)) },
  { name: 'front.png', mimeType: 'image/png', buffer: rgbaPng },
  { name: 'side.png', mimeType: 'image/png', buffer: rgbaPng },
]);
await page.locator('#mvcal-body').waitFor({ state: 'visible' });
if (await page.locator('.mvcal-view').count() !== 2) throw new Error('expected two matched MV views');
await page.waitForFunction(() => document.querySelector('#mvcal-preflight')?.textContent?.includes('mesh_scale'));
if (!(await page.locator('#mvcal-generate').isDisabled())) throw new Error('Generate must be disabled while mesh_scale is missing');
if (!(await page.locator('#mvcal-range').isDisabled())) throw new Error('mesh_scale slider must stay disabled while the explicit value is missing');
if ((await page.locator('#mvcal-num').inputValue()) !== '') throw new Error('missing mesh_scale must not be replaced by a numeric default');

// A synthetic slider event must not turn the missing value into a guessed 0.2.
await page.locator('#mvcal-range').dispatchEvent('input');
if (!(await page.locator('#mvcal-generate').isDisabled())) throw new Error('disabled slider event must not clear mesh_scale preflight error');
if ((await page.locator('#mvcal-num').inputValue()) !== '') throw new Error('disabled slider must not synthesize mesh_scale');

// User calibration explicitly supplies the missing scale and clears the hard error.
await page.locator('#mvcal-num').fill('0.250');
await page.locator('#mvcal-num').dispatchEvent('input');
await page.waitForFunction(() => !document.querySelector('#mvcal-preflight')?.textContent?.includes('error(s)'));
if (await page.locator('#mvcal-generate').isDisabled()) throw new Error('valid calibrated input should pass preflight');
if (await page.locator('#mvcal-range').isDisabled()) throw new Error('slider should be enabled only after explicit manual scale exists');

await page.locator('#mvcal-res').selectOption('1536');
if (await page.locator('#mvcal-res').inputValue() !== '1536') throw new Error('resolution selector did not accept 1536');
const cards = page.locator('.mvcal-view');
await cards.nth(0).dragTo(cards.nth(1));
const downloadPromise = page.waitForEvent('download');
await page.locator('#mvcal-download').click();
const dl = await downloadPromise;
if (dl.suggestedFilename() !== 'transforms.json') throw new Error('unexpected transforms download name');
const p = await dl.path();
const { readFile } = await import('node:fs/promises');
const saved = JSON.parse(await readFile(p, 'utf8'));
if (Math.abs(saved.mesh_scale - 0.25) > 1e-6) throw new Error(`mesh_scale not patched: ${saved.mesh_scale}`);
if (saved.frames?.[0]?.file_path !== 'side.png') throw new Error('drag reorder was not reflected in frames order');
console.log('HEADLESS_MV_PREFLIGHT_OK');

// ---- MV completion path (#25): the result must reach the viewer, gallery and
// "Save GLB…" through main.ts, and must NOT be silently downloaded.
// vite preview has no trellis-server, so mock /generate-mv with a minimal GLB.
function minimalGlb() {
  // One untextured triangle: enough for model-viewer to load and frame it.
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
const glb = minimalGlb();
let mvRequests = 0;
await page.route('**/generate-mv', async (route) => {
  mvRequests++;
  await route.fulfill({ status: 200, contentType: 'model/gltf-binary', body: glb });
});
const silentDownload = page.waitForEvent('download', { timeout: 2500 }).then(() => true, () => false);
await page.locator('#mvcal-generate').click();
await page.waitForFunction(() => document.querySelector('#mvcal-stage')?.textContent === 'complete', null, { timeout: 15000 });
if (mvRequests !== 1) throw new Error(`expected one /generate-mv request, saw ${mvRequests}`);
await page.waitForFunction(() => document.querySelector('#viewer-caption')?.textContent?.includes('mesh_scale 0.250'), null, { timeout: 10000 });
const caption = await page.locator('#viewer-caption').textContent();
if (!caption.includes('1536') || !caption.includes('seed 42')) throw new Error(`unexpected MV caption: ${caption}`);
if (await page.locator('#save-glb').isDisabled()) throw new Error('Save GLB… must be enabled after an MV result');
if (await page.locator('#reset-view').isDisabled()) throw new Error('Reset view must be enabled after an MV result');
if (await page.locator('.gitem').count() !== 1) throw new Error('MV result must be registered in the gallery');
const gmeta = await page.locator('.gitem .gmeta').textContent();
if (gmeta !== '1536 · MV') throw new Error(`unexpected gallery meta: ${gmeta}`);
if (await silentDownload) throw new Error('MV result must not trigger a silent browser download');
// Save GLB… (browser mode) is the explicit download, named after the MV run.
const savePromise = page.waitForEvent('download');
await page.locator('#save-glb').click();
const savedGlb = await savePromise;
if (savedGlb.suggestedFilename() !== 'multiview_1536_seed42_scale0.250.glb') throw new Error(`unexpected Save GLB name: ${savedGlb.suggestedFilename()}`);
// Re-opening the MV record from the gallery must not overwrite the single-image dropzone.
// loadRecord bails out before touching any UI when model-viewer can't render, so this
// step needs WebGL (SwiftShader in headless Chromium); skip it explicitly otherwise.
const hasWebgl = await page.evaluate(() => !!document.createElement('canvas').getContext('webgl2'));
if (hasWebgl) {
  await page.locator('.gitem').first().click();
  await page.waitForFunction(() => document.querySelector('#viewer-caption')?.textContent?.startsWith('multiview ·'), null, { timeout: 10000 });
  if (!(await page.locator('#input-preview').evaluate((el) => el.classList.contains('hidden')))) throw new Error('loading an MV record must not populate the single-image dropzone');
} else {
  console.log('HEADLESS_MV_REOPEN_SKIPPED (no WebGL in this browser)');
}
console.log('HEADLESS_MV_COMPLETION_OK');
await browser.close();
