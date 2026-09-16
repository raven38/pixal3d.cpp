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
import {
  DEFAULT_SV_FOV,
  makeSingleViewTransforms,
  modelFamilyForManifest,
  prepareSingleViewFiles,
  resolveSvModelSource,
  saveSvModelBaseUrl,
  validateSingleViewFov,
} from './single_view.js';

const $ = (s) => document.querySelector(s);
const log = (t) => { $('#log').textContent += String(t) + '\n'; };

const viewInput = document.createElement('input');
viewInput.type = 'file';
viewInput.multiple = true;
viewInput.accept = 'image/*,.json,application/json';
viewInput.hidden = true;
document.body.appendChild(viewInput);
const calibrator = initMeshScaleCalibration({ input: viewInput, mount: $('#calibration') });
window.__calibration = calibrator;

let worker = null;
let timer = null;
let currentGlb = null;
let currentUrl = null;
let modelsReady = false;
let gpuReady = false;
let preflightReady = false;
let releaseManifest = null;
let installAbort = null;
let svPrepared = null;
let svPreparing = false;

const urlMode = new URLSearchParams(location.search).get('mode');
const svConfigured = !!resolveSvModelSource().manifestUrl;
let inputMode = (urlMode === 'sv' || urlMode === 'mv') ? urlMode : (svConfigured ? 'sv' : 'mv');
$('#input-mode').value = inputMode;

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
const familyName = (mode) => mode === 'sv' ? 'SV' : 'MV';
const inputReady = () => inputMode === 'sv' ? !!svPrepared && !svPreparing : calibrator.isReady();
const inputFiles = () => inputMode === 'sv' ? (svPrepared?.files || []) : calibrator.getPatchedFiles();

function activeModelSource() {
  return inputMode === 'sv' ? resolveSvModelSource() : resolveModelSource();
}

function saveActiveModelBaseUrl(url) {
  if (inputMode === 'sv') saveSvModelBaseUrl(url);
  else saveModelBaseUrl(url);
}

function setRunEnabled() {
  $('#run').disabled = !(modelsReady && gpuReady && preflightReady && inputReady() && !worker && !installAbort);
}

function currentFovRad() {
  const deg = Number($('#sv-fov-deg').value);
  return validateSingleViewFov(deg * Math.PI / 180);
}

function updateSvSummary() {
  try {
    const fov = currentFovRad();
    const distance = 1 / (2 * Math.tan(fov / 2));
    $('#sv-summary').textContent = `Canonical front camera · FOV ${(fov * 180 / Math.PI).toFixed(1)}° · mesh_scale 1.0 · distance ${distance.toFixed(4)}`;
  } catch (e) {
    $('#sv-summary').textContent = String(e?.message || e);
  }
}

function rebuildSvCamera() {
  updateSvSummary();
  if (!svPrepared?.image) { setRunEnabled(); return; }
  try {
    const transforms = makeSingleViewTransforms(svPrepared.image.name, currentFovRad());
    const json = new File([JSON.stringify(transforms, null, 2) + '\n'], 'transforms.json', { type: 'application/json' });
    svPrepared = { ...svPrepared, transforms, files: [json, svPrepared.image] };
    $('#sv-file').textContent = `${svPrepared.image.name} · ready · ${(transforms.camera_angle_x * 180 / Math.PI).toFixed(1)}°`;
  } catch (e) {
    svPrepared = null;
    $('#sv-file').textContent = String(e?.message || e);
  }
  setRunEnabled();
}

async function ingestSingleView(file) {
  svPrepared = null;
  if (!file) { $('#sv-file').textContent = 'No image selected.'; setRunEnabled(); return; }
  svPreparing = true;
  $('#sv-file').textContent = `Preprocessing ${file.name}…`;
  setRunEnabled();
  try {
    svPrepared = await prepareSingleViewFiles(file, currentFovRad());
    $('#sv-file').textContent = `${file.name} → ${svPrepared.image.name} · ${svPrepared.image.size.toLocaleString()} bytes · ready`;
    updateSvSummary();
  } catch (e) {
    svPrepared = null;
    $('#sv-file').textContent = String(e?.message || e);
  } finally {
    svPreparing = false;
    setRunEnabled();
  }
}

const svInput = $('#sv-image');
const svDrop = $('#sv-drop');
svDrop.onclick = () => svInput.click();
svDrop.onkeydown = (e) => { if (e.key === 'Enter') svInput.click(); };
svInput.onchange = () => void ingestSingleView(svInput.files?.[0]);
['dragenter', 'dragover'].forEach((ev) => svDrop.addEventListener(ev, (e) => { e.preventDefault(); svDrop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => svDrop.addEventListener(ev, (e) => { e.preventDefault(); svDrop.classList.remove('drag'); }));
svDrop.addEventListener('drop', (e) => void ingestSingleView(e.dataTransfer?.files?.[0]));
$('#sv-fov-deg').addEventListener('change', rebuildSvCamera);
$('#sv-reset-fov').onclick = () => { $('#sv-fov-deg').value = (DEFAULT_SV_FOV * 180 / Math.PI).toFixed(1); rebuildSvCamera(); };

function updateModeUi() {
  $('#sv-panel').hidden = inputMode !== 'sv';
  $('#mv-panel').hidden = inputMode !== 'mv';
  $('#input-family').textContent = familyName(inputMode);
  $('#model-family-required').textContent = `${familyName(inputMode)} required`;
  const source = activeModelSource();
  $('#model-base-url').value = source.modelBaseUrl || '';
  $('#model-source-label').firstChild.textContent = `${familyName(inputMode)} model download base URL\n      `;
  if (inputMode === 'sv' && !source.manifestUrl) {
    $('#model-family-note').textContent = 'Public SV model download is not configured yet. Install a local verified SV manifest + GGUF set, or provide ?sv_manifest_url=...&sv_model_base_url=... . MV weights are never substituted.';
  } else {
    $('#model-family-note').textContent = 'Release models are streamed into origin-owned OPFS and become Ready only after exact size + SHA-256 verification.';
  }
  $('#download-models').disabled = !source.manifestUrl || !!installAbort || !!worker;
}

async function refreshActiveReleaseManifest() {
  releaseManifest = null;
  const source = activeModelSource();
  if (!source.manifestUrl) {
    log(`${familyName(inputMode)} release manifest is not configured`);
    return;
  }
  try {
    const manifest = await fetchReleaseManifest(source.manifestUrl || (inputMode === 'mv' ? DEFAULT_MANIFEST_URL : undefined));
    const family = modelFamilyForManifest(manifest);
    if (family !== inputMode) throw new Error(`configured ${familyName(inputMode)} manifest resolves to ${family ? family.toUpperCase() : 'unknown'} weights`);
    releaseManifest = manifest;
    log(`release manifest ${manifest.model_set} ${manifest.version} (${familyName(inputMode)})`);
  } catch (e) {
    log(`release manifest unavailable: ${e?.message || e}`);
  }
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
    ? `✓ WebGPU · max buffer ${fmtBytes(gpu.limits.maxBufferSize)} · storage binding ${fmtBytes(gpu.limits.maxStorageBufferBindingSize)} · device ok (${(gpu.requiredFeatures || []).join(', ')})`
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
  const family = modelFamilyForManifest(s.manifest);
  modelsReady = s.ready && family === inputMode;
  if (s.ready && family === inputMode) {
    $('#model-status').textContent = `Ready · ${s.modelSet} ${s.version} · ${familyName(family)} · ${fmtBytes(s.bytes)} · ${s.files} verified model(s)`;
  } else if (s.ready) {
    $('#model-status').textContent = `Installed ${s.modelSet} ${s.version} (${family ? familyName(family) : 'unknown family'}), but ${familyName(inputMode)} input requires a ${familyName(inputMode)} model set.`;
  } else {
    $('#model-status').textContent = `Not ready · ${s.missing.length} required item(s) missing/invalid · ${fmtBytes(s.bytes)} verified`;
  }
  $('#model-bar').style.width = modelsReady ? '100%' : '0%';
  $('#delete-models').disabled = !s.manifest || !!installAbort || !!worker;
  updateModeUi();
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

$('#input-mode').addEventListener('change', async () => {
  inputMode = $('#input-mode').value === 'sv' ? 'sv' : 'mv';
  updateModeUi();
  await refreshActiveReleaseManifest();
  await refreshModels();
});

$('#install-models').onclick = () => $('#model-files').click();
$('#model-files').onchange = async () => {
  const files = [...$('#model-files').files];
  if (!files.length) return;
  $('#install-models').disabled = true;
  try {
    const mf = files.find((f) => f.name === 'pixal3d-models.json');
    if (!mf) throw new Error('Select pixal3d-models.json together with the GGUF files');
    const manifest = JSON.parse(await mf.text());
    const family = modelFamilyForManifest(manifest);
    if (family !== inputMode) throw new Error(`Selected model set is ${family ? familyName(family) : 'unknown family'}; ${familyName(inputMode)} mode requires ${familyName(inputMode)} weights`);
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

$('#model-base-url').addEventListener('change', () => saveActiveModelBaseUrl($('#model-base-url').value));

$('#download-models').onclick = async () => {
  if (worker || installAbort) return;
  const source = activeModelSource();
  if (!source.manifestUrl) {
    $('#model-status').textContent = inputMode === 'sv'
      ? 'Public SV model manifest is not configured. Use Install local verified set… or provide sv_manifest_url + sv_model_base_url.'
      : 'Release model manifest is not configured.';
    return;
  }
  const entered = $('#model-base-url').value.trim();
  if (entered) { saveActiveModelBaseUrl(entered); source.modelBaseUrl = entered; }
  if (!releaseManifest || modelFamilyForManifest(releaseManifest) !== inputMode) {
    await refreshActiveReleaseManifest();
    if (!releaseManifest) { $('#model-status').textContent = `No valid ${familyName(inputMode)} release manifest is available.`; return; }
  }
  installAbort = new AbortController();
  $('#download-models').disabled = true;
  $('#install-models').disabled = true;
  setRunEnabled();
  try {
    await requestPersistentStorage();
    const installed = await installReleaseModels({
      manifestUrl: source.manifestUrl,
      modelBaseUrl: source.modelBaseUrl,
      signal: installAbort.signal,
      onProgress: (p) => {
        $('#model-status').textContent = `Downloading + verifying ${p.name} · ${fmtBytes(p.fileDone)} / ${fmtBytes(p.fileTotal)}`;
        $('#model-bar').style.width = `${Math.min(100, p.total ? p.done / p.total * 100 : 0)}%`;
      },
    });
    const family = modelFamilyForManifest(installed.manifest);
    if (family !== inputMode) {
      await deleteCachedModels();
      throw new Error(`Downloaded manifest changed family to ${family || 'unknown'}; cache removed instead of running the wrong weights`);
    }
    await refreshModels();
  } catch (e) {
    $('#model-status').textContent = e?.name === 'AbortError' ? 'Model download cancelled' : String(e?.message || e);
  } finally {
    installAbort = null;
    $('#install-models').disabled = false;
    updateModeUi();
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
  if (worker || +$('#resolution').value !== 1024 || !modelsReady || !gpuReady || !preflightReady || !inputReady()) return;
  const modelSet = await loadModelSet().catch((e) => { log(e); return null; });
  if (!modelSet) return;
  const family = modelFamilyForManifest(modelSet.manifest);
  if (family !== inputMode) {
    log(`refusing inference: ${family || 'unknown'} model family with ${inputMode} input mode`);
    await refreshModels();
    return;
  }
  const views = inputFiles();
  const seed = Math.max(0, +$('#seed').value || 1);
  worker = new Worker(new URL('../real_e2e/worker.js', import.meta.url));
  setRunEnabled();
  $('#cancel').disabled = false;
  $('#delete-models').disabled = true;
  const started = Date.now();
  const genProgress = makeGenProgress();
  let progressLabel = 'starting…';
  const render = () => { $('#progress-text').textContent = progressLabel + ' · ' + fmtTime(Date.now() - started); };
  render();
  timer = setInterval(render, 1000);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') {
      log(m.text);
      const text = String(m.text || '').trim();
      const staged = genProgress(text);
      if (staged) { progressLabel = staged; render(); }
      else if (text) { progressLabel = text; render(); }
    } else if (m.type === 'error') { log(m.text); finish('failed'); }
    else { log(m.report); showGlb(new Blob([m.glb], { type: 'model/gltf-binary' })); finish('complete'); }
  };
  worker.onerror = (e) => { log(e.message); finish('failed'); };
  worker.postMessage({ models: modelSet.models, modelManifest: modelSet.manifest, views, seed, resolution: 1024, inputMode });
};

const FLOW_STAGES = [
  { name: 'Sparse-structure flow', weight: 194.9 },
  { name: 'Shape-512 flow', weight: 165.7 },
  { name: 'Shape-1024 flow', weight: 1476.3 },
  { name: 'Texture flow', weight: 1042.4 },
];
const TAIL_WEIGHT = 800;
const TOTAL_WEIGHT = FLOW_STAGES.reduce((a, s) => a + s.weight, 0) + TAIL_WEIGHT;

function makeGenProgress() {
  let stage = -1;
  let lastDone = -1;
  const setBar = (frac) => {
    const el = document.getElementById('gen-bar');
    if (el) el.style.width = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + '%';
  };
  setBar(0);
  return (text) => {
    const m = /\[flow\]\s+\[[#.]+\]\s+(\d+)\/(\d+)\s+([\d.]+)s\s+(.*)$/.exec(text || '');
    if (!m) return null;
    const done = +m[1], steps = +m[2], eta = m[4].trim();
    if (stage < 0 || done < lastDone) stage++;
    lastDone = done;
    if (stage >= FLOW_STAGES.length) return null;
    const before = FLOW_STAGES.slice(0, stage).reduce((a, s) => a + s.weight, 0);
    const frac = (before + FLOW_STAGES[stage].weight * (steps ? done / steps : 0)) / TOTAL_WEIGHT;
    setBar(frac);
    return `${FLOW_STAGES[stage].name} ${done}/${steps} · ${eta} · overall ${(frac * 100).toFixed(0)}%`;
  };
}

function finish(label) {
  if (timer) clearInterval(timer);
  timer = null;
  if (worker) { worker.terminate(); worker = null; }
  $('#cancel').disabled = true;
  $('#progress-text').textContent = label;
  const gb = document.getElementById('gen-bar');
  if (gb) gb.style.width = label === 'complete' ? '100%' : '0%';
  setRunEnabled();
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
  updateModeUi();
  updateSvSummary();
  await refreshActiveReleaseManifest();
  await refreshModels();
  document.addEventListener('pixal3d-calibration-change', setRunEnabled);
  const mo = new MutationObserver(setRunEnabled);
  mo.observe($('#calibration'), { subtree: true, childList: true, attributes: true });
  setRunEnabled();
})();
