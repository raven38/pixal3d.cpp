import { initMeshScaleCalibration } from '../real_e2e/calibration.js';
import { cacheStatus, installFiles, loadModelSet } from './model_store.js';
import {
  DEFAULT_MANIFEST_URL,
  deleteCachedModels,
  fetchReleaseManifest,
  installReleaseModels,
  resolveModelSource,
  saveModelBaseUrl,
} from './release_store.js';
import { requestPersistentStorage, storagePreflight, webgpuPreflight } from './preflight.js';

const $ = (s) => document.querySelector(s);
const log = (t) => { $('#log').textContent += String(t) + '\n'; };

const viewInput = document.createElement('input');
viewInput.type = 'file';
viewInput.multiple = true;
viewInput.accept = 'image/*,.json,application/json';
viewInput.hidden = true;
document.body.appendChild(viewInput);
const calibrator = initMeshScaleCalibration({ input: viewInput, mount: $('#calibration') });

let worker = null;
let timer = null;
let currentGlb = null;
let currentUrl = null;
let modelsReady = false;
let gpuReady = false;
let preflightReady = false;
let releaseManifest = null;
let installAbort = null;

const fmtBytes = (n) => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = Number(n || 0), i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i < 2 ? 0 : 2)} ${units[i]}`;
};
const fmtTime = (ms) => {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

function setRunEnabled() {
  $('#run').disabled = !(modelsReady && gpuReady && preflightReady && calibrator.isReady() && !worker && !installAbort);
}

async function refreshPreflight(cache = null) {
  const gpu = await webgpuPreflight();
  gpuReady = gpu.ok;
  $('#webgpu').textContent = gpu.ok ? 'WebGPU ready' : 'WebGPU unavailable';
  $('#webgpu').classList.toggle('warn', !gpu.ok);

  let storage = null;
  if (releaseManifest) storage = await storagePreflight(releaseManifest, cache || await cacheStatus());
  preflightReady = gpu.ok && (!storage || storage.ok || modelsReady);

  const lines = [];
  lines.push(gpu.ok
    ? `✓ WebGPU · max buffer ${fmtBytes(gpu.limits.maxBufferSize)} · storage binding ${fmtBytes(gpu.limits.maxStorageBufferBindingSize)}`
    : `✕ ${gpu.message}`);
  if (storage) {
    lines.push(storage.ok
      ? `✓ Storage · ${fmtBytes(storage.storage.available)} available · ${fmtBytes(storage.additional)} additional model bytes`
      : `✕ Storage · ${fmtBytes(storage.storage.available)} available · ${fmtBytes(storage.requiredWithMargin)} required incl. margin`);
  }
  $('#preflight-status').textContent = lines.join('\n');
  log(`preflight gpu=${gpu.code}${storage ? ` storage=${storage.code} quota=${storage.storage.quota} usage=${storage.storage.usage} additional=${storage.additional}` : ''}`);
  setRunEnabled();
}

async function refreshModels() {
  const s = await cacheStatus();
  modelsReady = s.ready;
  $('#model-status').textContent = s.ready
    ? `Ready · ${s.modelSet} ${s.version} · ${fmtBytes(s.bytes)} · ${s.files} verified model(s)`
    : `Not ready · ${s.missing.length} required item(s) missing/invalid · ${fmtBytes(s.bytes)} verified`;
  $('#model-bar').style.width = s.ready ? '100%' : '0%';
  $('#delete-models').disabled = !s.manifest || !!installAbort || !!worker;
  await refreshPreflight(s);
}

async function ensureViewer() {
  if (!customElements.get('model-viewer')) {
    try { await import('../../app/public/vendor/model-viewer.min.js'); }
    catch (e) { log('model-viewer load failed: ' + e); return null; }
  }
  let mv = $('#viewer model-viewer');
  if (!mv) {
    mv = document.createElement('model-viewer');
    mv.setAttribute('camera-controls', '');
    mv.setAttribute('shadow-intensity', '1');
    mv.setAttribute('interaction-prompt', 'none');
    $('#viewer').replaceChildren(mv);
  }
  return mv;
}

async function showGlb(glb) {
  currentGlb = glb;
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = URL.createObjectURL(glb);
  const mv = await ensureViewer();
  if (mv) mv.src = currentUrl;
  $('#download').disabled = false;
  $('#reset').disabled = !mv;
}

$('#install-models').onclick = () => $('#model-files').click();
$('#model-files').onchange = async () => {
  const files = [...$('#model-files').files];
  if (!files.length) return;
  $('#install-models').disabled = true;
  try {
    await installFiles(files, (p) => {
      const filePart = p.fileTotal ? ` · ${fmtBytes(p.fileDone)} / ${fmtBytes(p.fileTotal)}` : '';
      $('#model-status').textContent = `Verifying + caching ${p.name}${filePart}`;
      $('#model-bar').style.width = `${Math.min(100, p.total ? p.done / p.total * 100 : 0)}%`;
    });
    await refreshModels();
  } catch (e) {
    $('#model-status').textContent = String(e?.message || e);
  } finally {
    $('#install-models').disabled = false;
    $('#model-files').value = '';
  }
};

$('#model-base-url').value = resolveModelSource().modelBaseUrl;
$('#model-base-url').addEventListener('change', () => saveModelBaseUrl($('#model-base-url').value));

$('#download-models').onclick = async () => {
  if (worker || installAbort) return;
  const source = resolveModelSource();
  const entered = $('#model-base-url').value.trim();
  if (entered) { saveModelBaseUrl(entered); source.modelBaseUrl = entered; }
  installAbort = new AbortController();
  $('#download-models').disabled = true;
  $('#install-models').disabled = true;
  setRunEnabled();
  try {
    await requestPersistentStorage();
    await installReleaseModels({
      manifestUrl: source.manifestUrl,
      modelBaseUrl: source.modelBaseUrl,
      signal: installAbort.signal,
      onProgress: (p) => {
        $('#model-status').textContent = `Downloading + verifying ${p.name} · ${fmtBytes(p.fileDone)} / ${fmtBytes(p.fileTotal)}`;
        $('#model-bar').style.width = `${Math.min(100, p.total ? p.done / p.total * 100 : 0)}%`;
      },
    });
    await refreshModels();
  } catch (e) {
    $('#model-status').textContent = e?.name === 'AbortError' ? 'Model download cancelled' : String(e?.message || e);
  } finally {
    installAbort = null;
    $('#download-models').disabled = false;
    $('#install-models').disabled = false;
    setRunEnabled();
  }
};

$('#delete-models').onclick = async () => {
  if (worker || installAbort) return;
  $('#delete-models').disabled = true;
  try {
    await deleteCachedModels();
    $('#model-status').textContent = 'Pixal3D browser model cache deleted';
    await refreshModels();
  } catch (e) {
    $('#model-status').textContent = `Cache delete failed: ${e?.message || e}`;
  }
};

$('#run').onclick = async () => {
  if (worker || +$('#resolution').value !== 1024 || !modelsReady || !gpuReady || !preflightReady || !calibrator.isReady()) return;
  const modelSet = await loadModelSet().catch((e) => { log(e); return null; });
  if (!modelSet) return;
  const views = calibrator.getPatchedFiles();
  const seed = Math.max(0, +$('#seed').value || 1);
  worker = new Worker(new URL('../real_e2e/worker.js', import.meta.url));
  setRunEnabled();
  $('#cancel').disabled = false;
  $('#delete-models').disabled = true;
  const started = Date.now();
  $('#progress-text').textContent = 'starting… · 0:00';
  timer = setInterval(() => {
    $('#progress-text').textContent = $('#progress-text').textContent.split(' · ')[0] + ' · ' + fmtTime(Date.now() - started);
  }, 1000);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') {
      log(m.text);
      const text = String(m.text || '').trim();
      if (text) $('#progress-text').textContent = text + ' · ' + fmtTime(Date.now() - started);
    } else if (m.type === 'error') { log(m.text); finish('failed'); }
    else { log(m.report); showGlb(new Blob([m.glb], { type: 'model/gltf-binary' })); finish('complete'); }
  };
  worker.onerror = (e) => { log(e.message); finish('failed'); };
  worker.postMessage({ models: modelSet.models, modelManifest: modelSet.manifest, views, seed, resolution: 1024 });
};

function finish(label) {
  if (timer) clearInterval(timer);
  timer = null;
  if (worker) { worker.terminate(); worker = null; }
  $('#cancel').disabled = true;
  $('#progress-text').textContent = label;
  void refreshModels();
}

$('#cancel').onclick = () => finish('cancelled');
$('#download').onclick = () => {
  if (!currentGlb) return;
  const a = document.createElement('a'); a.href = currentUrl; a.download = 'pixal3d.glb'; a.click();
};
$('#reset').onclick = () => {
  const mv = $('#viewer model-viewer'); if (!mv) return;
  mv.cameraOrbit = 'auto auto auto'; mv.fieldOfView = 'auto'; mv.jumpCameraToGoal?.();
};

(async () => {
  try {
    releaseManifest = await fetchReleaseManifest(resolveModelSource().manifestUrl || DEFAULT_MANIFEST_URL);
    log(`release manifest ${releaseManifest.model_set} ${releaseManifest.version}`);
  } catch (e) {
    log(`release manifest unavailable: ${e?.message || e}`);
  }
  await refreshModels();
  const mo = new MutationObserver(setRunEnabled);
  mo.observe($('#calibration'), { subtree: true, childList: true, attributes: true });
  setRunEnabled();
})();
