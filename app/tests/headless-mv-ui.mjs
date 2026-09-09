import { chromium } from 'playwright';

const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mP8z/C/HwMDAwMjIAMAJxQEAf3p9NQAAAAASUVORK5CYII=', 'base64');
const transforms = {
  camera_angle_x: 0.7,
  mesh_scale: 0.206,
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
  { name: 'front.png', mimeType: 'image/png', buffer: png },
  { name: 'side.png', mimeType: 'image/png', buffer: png },
]);
await page.locator('#mvcal-body').waitFor({ state: 'visible' });
if (await page.locator('.mvcal-view').count() !== 2) throw new Error('expected two matched MV views');
await page.locator('#mvcal-num').fill('0.250');
await page.locator('#mvcal-num').dispatchEvent('input');
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
console.log('HEADLESS_MV_UI_OK');
await browser.close();
