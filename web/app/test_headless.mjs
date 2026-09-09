import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

// Scope: this is intentionally a fast UI + OPFS + manifest-integrity gate.
// Worker/WebGPU/model-viewer are mocked; the heavy C ABI/WebGPU runtime is covered
// by the separate real_e2e harness, not by this test.
const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAF0lEQVR4nAXBAQEAAAjDIG7/zhMW0CF4IBYDAUWmUmEAAAAASUVORK5CYII=',
  'base64',
);

const mockModels = [
  { name: 'mock-image.gguf', role: 'image_encoder', buffer: Buffer.from('mock-image-model') },
  { name: 'mock-shape.gguf', role: 'shape_decoder', buffer: Buffer.from('mock-shape-model') },
];
const manifest = {
  schema_version: 1,
  model_set: 'pixal3d-test',
  version: 'test-1',
  files: mockModels.map((m) => ({
    name: m.name,
    role: m.role,
    required: true,
    size_bytes: m.buffer.length,
    sha256: createHash('sha256').update(m.buffer).digest('hex'),
  })),
};

const transforms = {
  camera_angle_x: 0.7,
  mesh_scale: 0.206,
  frames: [
    { file_path: 'front.png', transform_matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]] },
    { file_path: 'side.png', transform_matrix: [[0,0,1,2],[0,1,0,0],[-1,0,0,0],[0,0,0,1]] },
  ],
};

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.addInitScript(() => {
  Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true });
  if (!customElements.get('model-viewer')) {
    customElements.define('model-viewer', class extends HTMLElement {
      set src(v) { this._src = v; }
      jumpCameraToGoal() {}
    });
  }
  window.__mockMode = 'success';
  window.__lastPayload = null;
  class MockWorker {
    constructor(url) {
      this.url = String(url);
      this.onmessage = null;
      this.onerror = null;
    }
    postMessage(p) {
      window.__lastPayload = {
        workerUrl: this.url,
        views: p.views.map((f) => f.name),
        models: p.models.map((f) => f.name),
        seed: p.seed,
        resolution: p.resolution,
        modelSet: p.modelManifest?.model_set,
        modelVersion: p.modelManifest?.version,
        roles: p.modelManifest?.files?.map((f) => f.role),
      };
      setTimeout(() => this.onmessage?.({ data: { type: 'log', text: 'mock texture flow' } }), 5);
      setTimeout(() => {
        if (window.__mockMode === 'error') {
          this.onmessage?.({ data: { type: 'error', text: 'mock failure' } });
        } else {
          this.onmessage?.({
            data: {
              type: 'done',
              report: 'REAL_FULL_E2E_RESULT: OK',
              glb: new Uint8Array([103, 108, 84, 70]).buffer,
            },
          });
        }
      }, 15);
    }
    terminate() {}
  }
  window.Worker = MockWorker;
});

await page.goto(process.env.WEB_APP_URL || 'http://127.0.0.1:8199/web/app/', { waitUntil: 'networkidle' });

// Installation must exercise the real OPFS writer + incremental SHA-256 path.
const modelFiles = [
  {
    name: 'pixal3d-models.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(manifest)),
  },
  ...mockModels.map((m) => ({
    name: m.name,
    mimeType: 'application/octet-stream',
    buffer: m.buffer,
  })),
];
await page.locator('#model-files').setInputFiles(modelFiles);
await page.waitForFunction(() => document.querySelector('#model-status')?.textContent?.includes('Ready'));
const statusText = await page.locator('#model-status').textContent();
if (!statusText?.includes('pixal3d-test') || !statusText?.includes('test-1')) {
  throw new Error(`manifest identity not surfaced in ready state: ${statusText}`);
}

const viewInput = page.locator('input[accept="image/*,.json,application/json"]');
await viewInput.setInputFiles([
  { name: 'transforms.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(transforms)) },
  { name: 'front.png', mimeType: 'image/png', buffer: png },
  { name: 'side.png', mimeType: 'image/png', buffer: png },
]);
await page.waitForFunction(() => document.querySelectorAll('.cal-view').length === 2);
const cards = page.locator('.cal-view');
await cards.nth(0).dragTo(cards.nth(1));
await page.waitForFunction(() => !document.querySelector('#run')?.disabled);
await page.locator('#run').click();
await page.waitForFunction(() => document.querySelector('#progress-text')?.textContent === 'complete');

const payload = await page.evaluate(() => window.__lastPayload);
if (payload.views[1] !== 'side.png' || payload.views[2] !== 'front.png') {
  throw new Error('drag reorder did not reach Worker payload');
}
if (payload.models.length !== mockModels.length) {
  throw new Error(`verified cached model set not passed to Worker: ${payload.models}`);
}
if (payload.resolution !== 1024) throw new Error(`resolution contract mismatch: ${payload.resolution}`);
if (payload.modelSet !== manifest.model_set || payload.modelVersion !== manifest.version) {
  throw new Error('model manifest was not forwarded to Worker');
}
if (!payload.roles.includes('image_encoder') || !payload.roles.includes('shape_decoder')) {
  throw new Error(`manifest roles not forwarded: ${payload.roles}`);
}
if (!payload.workerUrl.includes('/web/real_e2e/worker.js')) {
  throw new Error(`unexpected Worker URL: ${payload.workerUrl}`);
}
if (await page.locator('#download').isDisabled()) throw new Error('GLB download not enabled after success');

const downloadPromise = page.waitForEvent('download');
await page.locator('#download').click();
const dl = await downloadPromise;
if (dl.suggestedFilename() !== 'pixal3d.glb') throw new Error('unexpected GLB filename');

// Failure path must leave a visible user-facing state and re-enable Generate.
await page.evaluate(() => { window.__mockMode = 'error'; });
await page.locator('#run').click();
await page.waitForFunction(() => document.querySelector('#progress-text')?.textContent === 'failed');
if (await page.locator('#run').isDisabled()) throw new Error('Generate stayed disabled after Worker failure');

console.log('WEB_PRODUCTION_UI_OPFS_MOCK_INFERENCE_OK');
await browser.close();
