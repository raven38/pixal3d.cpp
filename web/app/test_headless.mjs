import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAF0lEQVR4nAXBAQEAAAjDIG7/zhMW0CF4IBYDAUWmUmEAAAAASUVORK5CYII=', 'base64');
const localModels = [
  { name: 'mock-image.gguf', role: 'image_encoder', buffer: Buffer.from('mock-image-model') },
  { name: 'mock-shape.gguf', role: 'shape_decoder', buffer: Buffer.from('mock-shape-model') },
];
const localManifest = {
  schema_version: 1, model_set: 'pixal3d-test-mv', model_family: 'mv', version: 'test-1',
  files: localModels.map((m) => ({ name: m.name, role: m.role, required: true, size_bytes: m.buffer.length, sha256: createHash('sha256').update(m.buffer).digest('hex') })),
};
const svLocalModels = [
  { name: 'mock-sv-image.gguf', role: 'image_encoder', buffer: Buffer.from('mock-sv-image-model') },
  { name: 'mock-sv-shape.gguf', role: 'shape_decoder', buffer: Buffer.from('mock-sv-shape-model') },
];
const svLocalManifest = {
  schema_version: 1, model_set: 'pixal3d-test-sv', model_family: 'sv', version: 'test-1',
  files: svLocalModels.map((m) => ({ name: m.name, role: m.role, required: true, size_bytes: m.buffer.length, sha256: createHash('sha256').update(m.buffer).digest('hex') })),
};
const roles = ['image_encoder','naf','ss_flow','ss_decoder','shape_flow_512','shape_decoder','shape_flow_1024','texture_flow_1024','texture_decoder'];
const remoteModels = roles.map((role, i) => ({ name: role.includes('flow') ? `remote-${i}-${role}_mv.gguf` : `remote-${i}.gguf`, role, buffer: Buffer.from(`remote-model-${i}-${role}`) }));
const remoteManifest = {
  schema_version: 1, model_set: 'pixal3d-remote-test', model_family: 'mv', version: 'v1',
  files: remoteModels.map((m) => ({ name: m.name, role: m.role, required: true, size_bytes: m.buffer.length, sha256: createHash('sha256').update(m.buffer).digest('hex') })),
};
const transforms = {
  camera_angle_x: 0.7, mesh_scale: 0.206,
  frames: [
    { file_path: 'front.png', transform_matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]] },
    { file_path: 'side.png', transform_matrix: [[0,0,1,2],[0,1,0,0],[-1,0,0,0],[0,0,0,1]] },
  ],
};

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.route('https://models.test/**', async (route) => {
  const url = new URL(route.request().url());
  const headers = { 'access-control-allow-origin': '*' };
  if (url.pathname === '/manifest.json') {
    await route.fulfill({ status: 200, headers: { ...headers, 'content-type': 'application/json' }, body: JSON.stringify(remoteManifest) });
    return;
  }
  const name = decodeURIComponent(url.pathname.split('/').pop());
  const model = remoteModels.find((m) => m.name === name);
  if (model) await route.fulfill({ status: 200, headers, body: model.buffer });
  else await route.fulfill({ status: 404, headers, body: 'missing' });
});

await page.addInitScript(() => {
  Object.defineProperty(navigator, 'gpu', {
    configurable: true,
    value: {
      requestAdapter: async () => ({
        limits: { maxBufferSize: 4 * 1024 * 1024 * 1024, maxStorageBufferBindingSize: 1024 * 1024 * 1024, maxComputeWorkgroupsPerDimension: 65535 },
        info: { vendor: 'mock', architecture: 'mock', device: 'mock', description: 'Playwright mock adapter' },
        features: new Set(['shader-f16']),
        requestDevice: async (desc) => {
          if (!(desc?.requiredFeatures || []).includes('shader-f16')) throw new Error('CONTRACT: shader-f16 not requested');
          if (!desc?.requiredLimits?.maxBufferSize) throw new Error('CONTRACT: requiredLimits missing');
          return { destroy() {} };
        },
      }),
    },
  });
  if (!customElements.get('model-viewer')) customElements.define('model-viewer', class extends HTMLElement { set src(v) { this._src = v; } jumpCameraToGoal() {} });
  window.__mockMode = 'success'; window.__lastPayload = null;
  class MockWorker {
    constructor(url) { this.url=String(url); this.onmessage=null; this.onerror=null; }
    async postMessage(p) {
      const tfFile = p.views.find((f) => f.name === 'transforms.json');
      let transforms = null;
      if (tfFile) { try { transforms = JSON.parse(await tfFile.text()); } catch {} }
      window.__lastPayload={
        workerUrl:this.url,
        views:p.views.map(f=>f.name),
        models:p.models.map(f=>f.name),
        seed:p.seed,
        resolution:p.resolution,
        inputMode:p.inputMode,
        modelSet:p.modelManifest?.model_set,
        modelFamily:p.modelManifest?.model_family,
        modelVersion:p.modelManifest?.version,
        roles:p.modelManifest?.files?.map(f=>f.role),
        transforms,
      };
      setTimeout(()=>this.onmessage?.({data:{type:'log',text:'mock texture flow'}}),5);
      setTimeout(()=>window.__mockMode==='error'?this.onmessage?.({data:{type:'error',text:'mock failure'}}):this.onmessage?.({data:{type:'done',report:'REAL_FULL_E2E_RESULT: OK',glb:new Uint8Array([103,108,84,70]).buffer}}),15);
    }
    terminate() {}
  }
  window.Worker=MockWorker;
});

await page.goto(process.env.WEB_APP_URL || 'http://127.0.0.1:8199/web/app/', { waitUntil: 'networkidle' });
await page.waitForFunction(() => document.querySelector('#preflight-status')?.textContent?.includes('WebGPU'));
if ((await page.locator('#preflight-status').textContent())?.includes('WebGPU unavailable')) throw new Error('mock WebGPU preflight failed');
if (await page.locator('#input-mode').inputValue() !== 'mv') throw new Error('MV must remain the default when no public SV manifest is configured');

// Import modules relative to the page so this same smoke covers source web/app/ and built web/dist/.
const remoteStatus = await page.evaluate(async () => {
  const mod = await import(new URL('release_store.js', location.href).href);
  const s = await mod.installReleaseModels({ manifestUrl: 'https://models.test/manifest.json', modelBaseUrl: 'https://models.test/files' });
  return { ready:s.ready, modelSet:s.modelSet, version:s.version, files:s.files };
});
if (!remoteStatus.ready || remoteStatus.modelSet !== remoteManifest.model_set || remoteStatus.files !== 9) throw new Error(`remote release install failed: ${JSON.stringify(remoteStatus)}`);

await page.evaluate(async () => { const mod = await import(new URL('release_store.js', location.href).href); await mod.deleteCachedModels(); });
const deleted = await page.evaluate(async () => { const mod = await import(new URL('model_store.js', location.href).href); return mod.cacheStatus(); });
if (deleted.ready) throw new Error('Pixal3D OPFS cache remained Ready after delete');

const modelFiles = [{ name:'pixal3d-models.json', mimeType:'application/json', buffer:Buffer.from(JSON.stringify(localManifest)) }, ...localModels.map(m=>({name:m.name,mimeType:'application/octet-stream',buffer:m.buffer}))];
await page.locator('#model-files').setInputFiles(modelFiles);
await page.waitForFunction(() => document.querySelector('#model-status')?.textContent?.includes('Ready'));
const statusText=await page.locator('#model-status').textContent();
if(!statusText?.includes('pixal3d-test-mv')||!statusText?.includes('MV'))throw new Error(`MV manifest identity not surfaced: ${statusText}`);

const viewInput=page.locator('input[accept="image/*,.json,application/json"]');
await viewInput.setInputFiles([{name:'transforms.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(transforms))},{name:'front.png',mimeType:'image/png',buffer:png},{name:'side.png',mimeType:'image/png',buffer:png}]);
await page.waitForFunction(()=>document.querySelectorAll('.cal-view').length===2);
const cards=page.locator('.cal-view'); await cards.nth(0).dragTo(cards.nth(1));
await page.waitForFunction(()=>!document.querySelector('#run')?.disabled); await page.locator('#run').click();
await page.waitForFunction(()=>document.querySelector('#progress-text')?.textContent==='complete');
let payload=await page.evaluate(()=>window.__lastPayload);
if(payload.views[1]!=='side.png'||payload.views[2]!=='front.png')throw new Error('drag reorder did not reach Worker payload');
if(payload.models.length!==localModels.length)throw new Error(`cached MV model set not passed: ${payload.models}`);
if(payload.resolution!==1024||payload.inputMode!=='mv'||payload.modelSet!==localManifest.model_set||payload.modelFamily!=='mv')throw new Error('MV worker manifest/resolution contract mismatch');
if(!payload.workerUrl.endsWith('/real_e2e/worker.js'))throw new Error(`unexpected Worker URL: ${payload.workerUrl}`);
if(await page.locator('#download').isDisabled())throw new Error('GLB download not enabled');
const downloadPromise=page.waitForEvent('download'); await page.locator('#download').click(); if((await downloadPromise).suggestedFilename()!=='pixal3d.glb')throw new Error('unexpected GLB filename');

await page.evaluate(()=>{window.__mockMode='error';}); await page.locator('#run').click();
await page.waitForFunction(()=>document.querySelector('#progress-text')?.textContent==='failed');
if(await page.locator('#run').isDisabled())throw new Error('Generate stayed disabled after Worker failure');

await page.locator('#input-mode').selectOption('sv');
await page.waitForFunction(()=>!document.querySelector('#sv-panel')?.hidden);
await page.waitForFunction(()=>document.querySelector('#model-status')?.textContent?.includes('SV input requires'));
if(!(await page.locator('#download-models').isDisabled()))throw new Error('SV release download should be disabled without a configured manifest');
if(!((await page.locator('#model-family-note').textContent())||'').includes('MV weights are never substituted'))throw new Error('missing fail-closed SV model guidance');

const svFiles = [{ name:'pixal3d-models.json', mimeType:'application/json', buffer:Buffer.from(JSON.stringify(svLocalManifest)) }, ...svLocalModels.map(m=>({name:m.name,mimeType:'application/octet-stream',buffer:m.buffer}))];
await page.locator('#model-files').setInputFiles(svFiles);
await page.waitForFunction(() => document.querySelector('#model-status')?.textContent?.includes('Ready') && document.querySelector('#model-status')?.textContent?.includes('SV'));
await page.locator('#sv-image').setInputFiles({name:'portrait.png',mimeType:'image/png',buffer:png});
await page.waitForFunction(()=>document.querySelector('#sv-file')?.textContent?.includes('ready'));
await page.waitForFunction(()=>!document.querySelector('#run')?.disabled);
await page.evaluate(()=>{window.__mockMode='success';});
await page.locator('#run').click();
await page.waitForFunction(()=>document.querySelector('#progress-text')?.textContent==='complete');
payload=await page.evaluate(()=>window.__lastPayload);
if(payload.inputMode!=='sv'||payload.modelSet!==svLocalManifest.model_set||payload.modelFamily!=='sv')throw new Error(`SV model family did not reach worker: ${JSON.stringify(payload)}`);
if(JSON.stringify(payload.views)!==JSON.stringify(['transforms.json','input.png']))throw new Error(`unexpected SV staged files: ${JSON.stringify(payload.views)}`);
if(!payload.transforms||payload.transforms.frames?.length!==1||payload.transforms.frames[0].file_path!=='input.png')throw new Error('SV one-frame transforms contract missing');
if(payload.transforms.mesh_scale!==1)throw new Error(`SV mesh_scale is not canonical 1.0: ${payload.transforms.mesh_scale}`);
if(Math.abs(payload.transforms.camera_angle_x-0.3490658503988659)>1e-12)throw new Error(`SV default FOV mismatch: ${payload.transforms.camera_angle_x}`);
const distance=-payload.transforms.frames[0].transform_matrix[1][3];
const expectedDistance=1/(2*Math.tan(payload.transforms.camera_angle_x/2));
if(Math.abs(distance-expectedDistance)>1e-12)throw new Error(`SV camera distance mismatch: ${distance} vs ${expectedDistance}`);

console.log('WEB_RELEASE_OPFS_PREFLIGHT_MV_AND_SV_MOCK_INFERENCE_OK');
await browser.close();
