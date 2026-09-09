import { Sha256 } from './sha256.js';
import { cacheStatus, MODEL_MANIFEST_NAME } from './model_store.js';
import { storagePreflight } from './preflight.js';

const ROOT = 'pixal3d-models-v1';
export const DEFAULT_MANIFEST_URL = '/models/pixal3d-q8_0-v1/pixal3d-models.json';

function safeName(name) {
  return typeof name === 'string' && name.length > 0 && name !== '.' && name !== '..' && !name.includes('/') && !name.includes('\\');
}

export function validateReleaseManifest(m) {
  if (!m || m.schema_version !== 1 || typeof m.model_set !== 'string' || typeof m.version !== 'string' || !Array.isArray(m.files)) {
    throw new Error('Invalid Pixal3D release manifest');
  }
  const names = new Set();
  const roles = new Set();
  for (const f of m.files) {
    if (!safeName(f?.name)) throw new Error(`Unsafe model filename: ${String(f?.name)}`);
    if (names.has(f.name)) throw new Error(`Duplicate model filename: ${f.name}`);
    if (typeof f.role !== 'string' || !f.role || roles.has(f.role)) throw new Error(`Missing/duplicate model role: ${String(f.role)}`);
    if (f.required !== true) throw new Error(`Release model must be required: ${f.name}`);
    if (!Number.isSafeInteger(f.size_bytes) || f.size_bytes <= 0) throw new Error(`Invalid size_bytes: ${f.name}`);
    if (typeof f.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(f.sha256)) throw new Error(`Invalid sha256: ${f.name}`);
    names.add(f.name); roles.add(f.role);
  }
  if (m.files.length !== 9) throw new Error(`Release manifest must contain exactly 9 required models; got ${m.files.length}`);
  return m;
}

async function originRoot() {
  if (!navigator.storage?.getDirectory) throw new Error('OPFS is unavailable in this browser');
  return navigator.storage.getDirectory();
}

async function modelDir(create = true) {
  const root = await originRoot();
  return root.getDirectoryHandle(ROOT, { create });
}

async function removeRootIfPresent() {
  const root = await originRoot();
  try { await root.removeEntry(ROOT, { recursive: true }); }
  catch (e) { if (e?.name !== 'NotFoundError') throw e; }
}

export async function deleteCachedModels() {
  await removeRootIfPresent();
  return cacheStatus();
}

export async function fetchReleaseManifest(url = DEFAULT_MANIFEST_URL, signal) {
  const res = await fetch(url, { cache: 'no-store', signal });
  if (!res.ok) throw new Error(`Model manifest fetch failed (HTTP ${res.status})`);
  return validateReleaseManifest(await res.json());
}

export function resolveModelSource() {
  const q = new URLSearchParams(location.search);
  const manifestUrl = q.get('manifest_url') || window.PIXAL3D_MODEL_MANIFEST_URL || DEFAULT_MANIFEST_URL;
  const modelBaseUrl = q.get('model_base_url') || window.PIXAL3D_MODEL_BASE_URL || localStorage.getItem('pixal3d.modelBaseUrl') || '';
  return { manifestUrl, modelBaseUrl };
}

export function saveModelBaseUrl(url) {
  const v = String(url || '').trim();
  if (v) localStorage.setItem('pixal3d.modelBaseUrl', v);
  else localStorage.removeItem('pixal3d.modelBaseUrl');
}

async function writeManifestLast(d, manifest) {
  const h = await d.getFileHandle(MODEL_MANIFEST_NAME, { create: true });
  const w = await h.createWritable({ keepExistingData: false });
  await w.write(JSON.stringify(manifest, null, 2) + '\n');
  await w.close();
}

async function downloadOne(d, ent, url, onProgress, aggregate, signal) {
  const res = await fetch(url, { cache: 'no-store', signal });
  if (!res.ok || !res.body) throw new Error(`Download failed for ${ent.name} (HTTP ${res.status})`);
  const h = await d.getFileHandle(ent.name, { create: true });
  const w = await h.createWritable({ keepExistingData: false });
  const reader = res.body.getReader();
  const hash = new Sha256();
  let bytes = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      hash.update(value);
      await w.write(value);
      bytes += value.byteLength;
      onProgress({ name: ent.name, fileDone: bytes, fileTotal: ent.size_bytes, done: aggregate.done + bytes, total: aggregate.total, phase: 'download' });
    }
    await w.close();
  } catch (e) {
    try { await reader.cancel(); } catch {}
    try { await w.abort(); } catch {}
    try { await d.removeEntry(ent.name); } catch {}
    throw e;
  }
  const digest = hash.hex();
  if (bytes !== ent.size_bytes || digest !== ent.sha256) {
    try { await d.removeEntry(ent.name); } catch {}
    throw new Error(`Downloaded model failed verification: ${ent.name} (size ${bytes}/${ent.size_bytes}, sha256 ${digest}/${ent.sha256})`);
  }
  aggregate.done += bytes;
}

export async function installReleaseModels({ manifestUrl, modelBaseUrl, onProgress = () => {}, signal } = {}) {
  const manifest = await fetchReleaseManifest(manifestUrl || DEFAULT_MANIFEST_URL, signal);
  const current = await cacheStatus();
  if (current.ready && current.modelSet === manifest.model_set && current.version === manifest.version) return current;

  if (!modelBaseUrl) {
    throw new Error('Model download base URL is not configured. Set PIXAL3D_MODEL_BASE_URL, ?model_base_url=..., or use local verified install.');
  }

  const storage = await storagePreflight(manifest, current);
  if (!storage.ok) throw new Error(storage.message);

  // A different or incomplete set is never mixed with the release set. The namespace
  // is origin-owned and contains only Pixal3D-managed files, so recursive removal is safe.
  await removeRootIfPresent();
  const d = await modelDir(true);
  const aggregate = {
    done: 0,
    total: manifest.files.reduce((s, f) => s + f.size_bytes, 0),
  };
  for (const ent of manifest.files) {
    const url = `${String(modelBaseUrl).replace(/\/$/, '')}/${encodeURIComponent(ent.name)}`;
    await downloadOne(d, ent, url, onProgress, aggregate, signal);
  }
  await writeManifestLast(d, manifest);
  return cacheStatus();
}
