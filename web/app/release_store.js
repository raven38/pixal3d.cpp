import { Sha256 } from './sha256.js';
import { cacheStatus, MODEL_MANIFEST_NAME } from './model_store.js';
import { storagePreflight } from './preflight.js';

const ROOT = 'pixal3d-models-v1';
const PENDING_MANIFEST_NAME = '.pixal3d-pending.json';
export const DEFAULT_MANIFEST_URL = '/models/pixal3d-q8_0-v1/pixal3d-models.json';

function safeName(name) {
  return typeof name === 'string' && name.length > 0 && name !== '.' && name !== '..' && !name.includes('/') && !name.includes('\\');
}

export function validateReleaseManifest(m) {
  if (!m || m.schema_version !== 1 || typeof m.model_set !== 'string' || typeof m.version !== 'string' || !Array.isArray(m.files)) throw new Error('Invalid Pixal3D release manifest');
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

function manifestIdentity(m) {
  return m ? `${m.model_set}\n${m.version}\n${m.files.map((f) => `${f.name}:${f.role}:${f.required}:${f.sha256}:${f.size_bytes}`).join('\n')}` : '';
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
  try { await root.removeEntry(ROOT, { recursive: true }); } catch (e) { if (e?.name !== 'NotFoundError') throw e; }
}
async function removeIfPresent(d, name) {
  try { await d.removeEntry(name); } catch (e) { if (e?.name !== 'NotFoundError') throw e; }
}
async function readJsonIfPresent(d, name) {
  try { return JSON.parse(await (await (await d.getFileHandle(name)).getFile()).text()); } catch { return null; }
}
async function writeJson(d, name, value) {
  const h = await d.getFileHandle(name, { create: true });
  const w = await h.createWritable({ keepExistingData: false });
  await w.write(JSON.stringify(value, null, 2) + '\n');
  await w.close();
}

export async function deleteCachedModels() {
  await removeRootIfPresent();
  return cacheStatus();
}

export async function fetchReleaseManifest(url = DEFAULT_MANIFEST_URL, signal) {
  const res = await fetch(url, { cache: 'no-store', signal, referrerPolicy: 'no-referrer' });
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
  if (v) localStorage.setItem('pixal3d.modelBaseUrl', v); else localStorage.removeItem('pixal3d.modelBaseUrl');
}

async function occupiedTargetBytes(d, manifest) {
  let bytes = 0;
  for (const ent of manifest.files) {
    try {
      const f = await (await d.getFileHandle(ent.name)).getFile();
      bytes += Math.min(f.size, ent.size_bytes);
    } catch {}
  }
  return bytes;
}

async function verifyExisting(d, ent, signal) {
  let f;
  try { f = await (await d.getFileHandle(ent.name)).getFile(); } catch { return false; }
  if (f.size !== ent.size_bytes) return false;
  const reader = f.stream().getReader();
  const hash = new Sha256();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      hash.update(value);
    }
  } finally { try { reader.releaseLock(); } catch {} }
  return hash.hex() === ent.sha256;
}

async function downloadOne(d, ent, url, onProgress, aggregate, signal) {
  if (await verifyExisting(d, ent, signal)) {
    aggregate.done += ent.size_bytes;
    onProgress({ name: ent.name, fileDone: ent.size_bytes, fileTotal: ent.size_bytes, done: aggregate.done, total: aggregate.total, phase: 'cached-verified' });
    return;
  }
  await removeIfPresent(d, ent.name);
  const res = await fetch(url, { cache: 'no-store', signal, referrerPolicy: 'no-referrer' });
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
      hash.update(value); await w.write(value); bytes += value.byteLength;
      onProgress({ name: ent.name, fileDone: bytes, fileTotal: ent.size_bytes, done: aggregate.done + bytes, total: aggregate.total, phase: 'download' });
    }
    await w.close();
  } catch (e) {
    try { await reader.cancel(); } catch {}
    try { await w.abort(); } catch {}
    await removeIfPresent(d, ent.name);
    throw e;
  }
  const digest = hash.hex();
  if (bytes !== ent.size_bytes || digest !== ent.sha256) {
    await removeIfPresent(d, ent.name);
    throw new Error(`Downloaded model failed verification: ${ent.name} (size ${bytes}/${ent.size_bytes}, sha256 ${digest}/${ent.sha256})`);
  }
  aggregate.done += bytes;
}

export async function installReleaseModels({ manifestUrl, modelBaseUrl, onProgress = () => {}, signal } = {}) {
  const manifest = await fetchReleaseManifest(manifestUrl || DEFAULT_MANIFEST_URL, signal);
  const current = await cacheStatus();
  if (current.ready && manifestIdentity(current.manifest) === manifestIdentity(manifest)) return current;
  if (!modelBaseUrl) throw new Error('Model download base URL is not configured. Set PIXAL3D_MODEL_BASE_URL, ?model_base_url=..., or use local verified install.');

  let d;
  try { d = await modelDir(false); } catch { d = await modelDir(true); }
  const pendingRaw = await readJsonIfPresent(d, PENDING_MANIFEST_NAME);
  const storedRaw = await readJsonIfPresent(d, MODEL_MANIFEST_NAME);
  let pending = null, storedManifest = null;
  try { if (pendingRaw) pending = validateReleaseManifest(pendingRaw); } catch {}
  try { if (storedRaw) storedManifest = validateReleaseManifest(storedRaw); } catch {}
  const targetId = manifestIdentity(manifest);
  const resumable = (pending && manifestIdentity(pending) === targetId) || (storedManifest && manifestIdentity(storedManifest) === targetId);
  if (!resumable && (pendingRaw || storedRaw)) {
    await removeRootIfPresent();
    d = await modelDir(true);
  }

  // Existing target files already consume quota. Credit only their occupied bytes;
  // a corrupt/partial file is removed before replacement, so this is conservative
  // for the net additional capacity needed to finish the set.
  const occupied = await occupiedTargetBytes(d, manifest);
  const storage = await storagePreflight(manifest, { manifest, bytes: occupied });
  if (!storage.ok) throw new Error(storage.message);

  await writeJson(d, PENDING_MANIFEST_NAME, manifest);
  await removeIfPresent(d, MODEL_MANIFEST_NAME);
  const aggregate = { done: 0, total: manifest.files.reduce((s, f) => s + f.size_bytes, 0) };
  for (const ent of manifest.files) {
    const url = `${String(modelBaseUrl).replace(/\/$/, '')}/${encodeURIComponent(ent.name)}`;
    await downloadOne(d, ent, url, onProgress, aggregate, signal);
  }
  await writeJson(d, MODEL_MANIFEST_NAME, manifest);
  await removeIfPresent(d, PENDING_MANIFEST_NAME);
  return cacheStatus();
}
