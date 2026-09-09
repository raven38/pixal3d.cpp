import { initMeshScaleCalibration } from '../real_e2e/calibration.js';
import { cacheStatus, installFiles, loadModelSet } from './model_store.js';

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

const fmtBytes = (n) => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i < 2 ? 0 : 2)} ${units[i]}`;
};
const fmtTime = (ms) => {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

function setRunEnabled() {
  $('#run').disabled = !(modelsReady && calibrator.isReady() && navigator.gpu && !worker);
}

async function refreshModels() {
  const s = await cacheStatus();
  modelsReady = s.ready;
  $('#model-status').textContent = s.ready
    ? `Ready · ${s.modelSet} ${s.version} · ${fmtBytes(s.bytes)} · ${s.files} verified model(s)`
    : `Not ready · ${s.missing.length} required item(s) missing/invalid · ${fmtBytes(s.bytes)} verified`;
  $('#model-bar').style.width = s.ready ? '100%' : '0%';
  setRunEnabled();
}

async function ensureViewer() {
  if (!customElements.get('model-viewer')) {
    try {
      await import('../../app/public/vendor/model-viewer.min.js');
    } catch (e) {
      log('model-viewer load failed: ' + e);
      return null;
    }
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

$('#run').onclick = async () => {
  if (worker || +$('#resolution').value !== 1024 || !modelsReady || !calibrator.isReady()) return;
  const modelSet = await loadModelSet().catch((e) => { log(e); return null; });
  if (!modelSet) return;

  const views = calibrator.getPatchedFiles();
  const seed = Math.max(0, +$('#seed').value || 1);
  worker = new Worker(new URL('../real_e2e/worker.js', import.meta.url));
  setRunEnabled();
  $('#cancel').disabled = false;
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
    } else if (m.type === 'error') {
      log(m.text);
      finish('failed');
    } else {
      log(m.report);
      showGlb(new Blob([m.glb], { type: 'model/gltf-binary' }));
      finish('complete');
    }
  };
  worker.onerror = (e) => { log(e.message); finish('failed'); };
  worker.postMessage({
    models: modelSet.models,
    modelManifest: modelSet.manifest,
    views,
    seed,
    resolution: 1024,
  });
};

function finish(label) {
  if (timer) clearInterval(timer);
  timer = null;
  if (worker) { worker.terminate(); worker = null; }
  $('#cancel').disabled = true;
  $('#progress-text').textContent = label;
  setRunEnabled();
}

$('#cancel').onclick = () => finish('cancelled');
$('#download').onclick = () => {
  if (!currentGlb) return;
  const a = document.createElement('a');
  a.href = currentUrl;
  a.download = 'pixal3d.glb';
  a.click();
};
$('#reset').onclick = () => {
  const mv = $('#viewer model-viewer');
  if (!mv) return;
  mv.cameraOrbit = 'auto auto auto';
  mv.fieldOfView = 'auto';
  mv.jumpCameraToGoal?.();
};

(async () => {
  const gpu = !!navigator.gpu;
  $('#webgpu').textContent = gpu ? 'WebGPU ready' : 'WebGPU unavailable';
  $('#webgpu').classList.toggle('warn', !gpu);
  await refreshModels();
  const mo = new MutationObserver(setRunEnabled);
  mo.observe($('#calibration'), { subtree: true, childList: true, attributes: true });
  setRunEnabled();
})();
