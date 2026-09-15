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
// テスト用フック（test_canonical_rig.mjs）。UI だけでは観測できない isReady() の遷移を直接検証する。
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
  const genProgress = makeGenProgress();
  // 進捗文字列は状態として持ち、タイマーは経過時間だけを付け直す。
  // （以前は表示文字列を ' · ' で切っていたので、段名以外の ETA / overall が 1 秒で消えていた）
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
  worker.postMessage({ models: modelSet.models, modelManifest: modelSet.manifest, views, seed, resolution: 1024 });
};

// 生成の進捗バー。C++ (src/flow_runner.cpp) が flow の各ステップで stdout へ
//   [flow] [####................]  5/12   42.3s  ~58s left
// を出すので、それを読んで全体進捗に直す。flow は SS / Shape-512 / Shape-1024 /
// Texture の 4 段あり各 12 step で、段ごとの所要が大きく違う（公式 cyclops の実測で
// 194.9 / 165.7 / 1476.3 / 1042.4 秒）。段を均等に扱うとバーが跳ねるので実測秒で重み付けし、
// flow 後段の decode + postprocess ぶんも残しておく。
const FLOW_STAGES = [
  { name: 'Sparse-structure flow', weight: 194.9 },
  { name: 'Shape-512 flow', weight: 165.7 },
  { name: 'Shape-1024 flow', weight: 1476.3 },
  { name: 'Texture flow', weight: 1042.4 },
];
const TAIL_WEIGHT = 800; // decode + postprocess（生成全体 3688s と flow 合計 2879s の差）
const TOTAL_WEIGHT = FLOW_STAGES.reduce((a, s) => a + s.weight, 0) + TAIL_WEIGHT;

function makeGenProgress() {
  let stage = -1;      // 現在の flow 段
  let lastDone = -1;   // 段の切り替わり検出用
  const setBar = (frac) => {
    const el = document.getElementById('gen-bar');
    if (el) el.style.width = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + '%';
  };
  setBar(0);
  return (text) => {
    const m = /\[flow\]\s+\[[#.]+\]\s+(\d+)\/(\d+)\s+([\d.]+)s\s+(.*)$/.exec(text || '');
    if (!m) return null;
    const done = +m[1], steps = +m[2], eta = m[4].trim();
    // 段の切り替えは done が「減った」ことで検出する（12→0 も 12→1 も拾う）。
    // 同じ値の再受信（0/12 が 2 回来る等）では進めない。
    if (stage < 0 || done < lastDone) stage++;
    lastDone = done;
    if (stage >= FLOW_STAGES.length) {
      // flow 4 段を越えた行は想定外。バーは flow 完了位置のまま、行はそのまま見せる。
      return null;
    }
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
  // ボタンの状態は同期的に戻す。refreshModels() は await を挟むので、それ任せにすると
  // 「失敗表示は出ているが Generate はまだ disabled」という窓ができる（CI で顕在化した）。
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
  try {
    releaseManifest = await fetchReleaseManifest(resolveModelSource().manifestUrl || DEFAULT_MANIFEST_URL);
    log(`release manifest ${releaseManifest.model_set} ${releaseManifest.version}`);
  } catch (e) {
    log(`release manifest unavailable: ${e?.message || e}`);
  }
  await refreshModels();
  // 合成リグの mesh_scale 確定は DOM 変化に頼らず明示イベントで拾う
  document.addEventListener('pixal3d-calibration-change', setRunEnabled);
  const mo = new MutationObserver(setRunEnabled);
  mo.observe($('#calibration'), { subtree: true, childList: true, attributes: true });
  setRunEnabled();
})();
