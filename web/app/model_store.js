const ROOT = 'pixal3d-models-v1';
export const MODEL_MANIFEST_NAME = 'pixal3d-models.json';
const MANIFEST_SCHEMA_VERSION = 1;
const CHUNK_SIZE = 8 * 1024 * 1024;

async function dir(create = true) {
  if (!navigator.storage?.getDirectory) throw new Error('OPFS is unavailable in this browser');
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(ROOT, { create });
}

function isSafeName(name) {
  return typeof name === 'string' && name.length > 0 && !name.includes('/') && !name.includes('\\') && name !== '.' && name !== '..';
}

function validateManifest(manifest) {
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) throw new Error('Invalid model manifest');
  if (manifest.schema_version !== MANIFEST_SCHEMA_VERSION) throw new Error(`Unsupported model manifest schema_version: ${manifest.schema_version}`);
  if (typeof manifest.model_set !== 'string' || !manifest.model_set) throw new Error('Model manifest model_set is missing');
  if (typeof manifest.version !== 'string' || !manifest.version) throw new Error('Model manifest version is missing');
  if (!Array.isArray(manifest.files) || !manifest.files.length) throw new Error('Model manifest files[] is missing');

  const seen = new Set();
  for (const [i, ent] of manifest.files.entries()) {
    if (!ent || typeof ent !== 'object') throw new Error(`Manifest files[${i}] must be an object`);
    if (!isSafeName(ent.name)) throw new Error(`Unsafe model filename in manifest: ${String(ent.name)}`);
    if (seen.has(ent.name)) throw new Error(`Duplicate model filename in manifest: ${ent.name}`);
    seen.add(ent.name);
    if (typeof ent.role !== 'string' || !ent.role) throw new Error(`Manifest role missing for ${ent.name}`);
    if (typeof ent.required !== 'boolean') throw new Error(`Manifest required flag missing for ${ent.name}`);
    if (!Number.isSafeInteger(ent.size_bytes) || ent.size_bytes <= 0) throw new Error(`Manifest size_bytes invalid for ${ent.name}`);
    if (typeof ent.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(ent.sha256)) throw new Error(`Manifest sha256 invalid for ${ent.name}`);
  }
  return manifest;
}

async function readStoredManifest(d) {
  const h = await d.getFileHandle(MODEL_MANIFEST_NAME);
  const f = await h.getFile();
  return validateManifest(JSON.parse(await f.text()));
}

async function removeIfPresent(d, name) {
  try {
    await d.removeEntry(name);
  } catch (e) {
    if (e?.name !== 'NotFoundError') throw e;
  }
}

// Minimal incremental SHA-256 so multi-GB GGUFs can be verified while streaming
// into OPFS. crypto.subtle.digest() is intentionally not used because it requires
// the complete file as one ArrayBuffer and would materialize gigabytes in memory.
class Sha256 {
  constructor() {
    this.h = new Uint32Array([
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]);
    this.buf = new Uint8Array(64);
    this.bufLen = 0;
    this.bytes = 0;
    this.w = new Uint32Array(64);
  }

  update(data) {
    let off = 0;
    this.bytes += data.byteLength;
    if (this.bufLen) {
      const n = Math.min(64 - this.bufLen, data.byteLength);
      this.buf.set(data.subarray(0, n), this.bufLen);
      this.bufLen += n;
      off += n;
      if (this.bufLen === 64) {
        this._block(this.buf, 0);
        this.bufLen = 0;
      }
    }
    while (off + 64 <= data.byteLength) {
      this._block(data, off);
      off += 64;
    }
    if (off < data.byteLength) {
      this.buf.set(data.subarray(off), 0);
      this.bufLen = data.byteLength - off;
    }
  }

  _block(data, off) {
    const w = this.w;
    for (let i = 0; i < 16; i++) {
      const j = off + i * 4;
      w[i] = ((data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | data[j + 3]) >>> 0;
    }
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15], y = w[i - 2];
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let [a, b, c, d, e, f, g, h] = this.h;
    for (let i = 0; i < 64; i++) {
      const s1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + s1 + ch + K[i] + w[i]) >>> 0;
      const s0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    this.h[0] = (this.h[0] + a) >>> 0;
    this.h[1] = (this.h[1] + b) >>> 0;
    this.h[2] = (this.h[2] + c) >>> 0;
    this.h[3] = (this.h[3] + d) >>> 0;
    this.h[4] = (this.h[4] + e) >>> 0;
    this.h[5] = (this.h[5] + f) >>> 0;
    this.h[6] = (this.h[6] + g) >>> 0;
    this.h[7] = (this.h[7] + h) >>> 0;
  }

  hex() {
    const tail = new Uint8Array(128);
    tail.set(this.buf.subarray(0, this.bufLen));
    tail[this.bufLen] = 0x80;
    const padded = this.bufLen + 1 <= 56 ? 64 : 128;
    const bitLo = (this.bytes * 8) >>> 0;
    const bitHi = Math.floor(this.bytes / 0x20000000) >>> 0;
    const p = padded - 8;
    tail[p] = bitHi >>> 24; tail[p + 1] = bitHi >>> 16; tail[p + 2] = bitHi >>> 8; tail[p + 3] = bitHi;
    tail[p + 4] = bitLo >>> 24; tail[p + 5] = bitLo >>> 16; tail[p + 6] = bitLo >>> 8; tail[p + 7] = bitLo;
    this._block(tail, 0);
    if (padded === 128) this._block(tail, 64);
    return [...this.h].map((x) => x.toString(16).padStart(8, '0')).join('');
  }
}

const K = new Uint32Array([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]);

async function parseSelectedManifest(files) {
  const manifestFile = [...files].find((f) => f.name === MODEL_MANIFEST_NAME);
  if (!manifestFile) throw new Error(`Select ${MODEL_MANIFEST_NAME} together with the GGUF files`);
  try {
    return validateManifest(JSON.parse(await manifestFile.text()));
  } catch (e) {
    throw new Error(`Invalid ${MODEL_MANIFEST_NAME}: ${e?.message || e}`);
  }
}

export async function cacheStatus() {
  try {
    const d = await dir(false);
    const manifest = await readStoredManifest(d);
    let bytes = 0, files = 0;
    const missing = [];
    for (const ent of manifest.files) {
      if (!ent.required) continue;
      try {
        const f = await (await d.getFileHandle(ent.name)).getFile();
        if (f.size !== ent.size_bytes) {
          missing.push(`${ent.name} (size mismatch)`);
          continue;
        }
        bytes += f.size;
        files++;
      } catch {
        missing.push(ent.name);
      }
    }
    return { ready: missing.length === 0, bytes, files, missing, version: manifest.version, modelSet: manifest.model_set, manifest };
  } catch {
    return { ready: false, bytes: 0, files: 0, missing: [MODEL_MANIFEST_NAME], version: null, modelSet: null, manifest: null };
  }
}

export async function installFiles(files, onProgress = () => {}) {
  const manifest = await parseSelectedManifest(files);
  const selected = new Map([...files].map((f) => [f.name, f]));
  const required = manifest.files.filter((ent) => ent.required);
  const missing = required.filter((ent) => !selected.has(ent.name)).map((ent) => ent.name);
  if (missing.length) throw new Error('Missing required models: ' + missing.join(', '));

  for (const ent of required) {
    const src = selected.get(ent.name);
    if (src.size !== ent.size_bytes) throw new Error(`Size mismatch for ${ent.name}: selected ${src.size}, manifest ${ent.size_bytes}`);
  }

  const d = await dir(true);
  let done = 0;
  const total = required.reduce((sum, ent) => sum + ent.size_bytes, 0);
  for (const ent of required) {
    const src = selected.get(ent.name);
    const handle = await d.getFileHandle(ent.name, { create: true });
    const writable = await handle.createWritable({ keepExistingData: false });
    const hash = new Sha256();
    let fileDone = 0;
    onProgress({ name: ent.name, done, total, fileDone, fileTotal: ent.size_bytes, phase: 'verifying' });
    try {
      for (let offset = 0; offset < src.size; offset += CHUNK_SIZE) {
        const chunk = new Uint8Array(await src.slice(offset, Math.min(src.size, offset + CHUNK_SIZE)).arrayBuffer());
        hash.update(chunk);
        await writable.write(chunk);
        fileDone += chunk.byteLength;
        onProgress({ name: ent.name, done: done + fileDone, total, fileDone, fileTotal: ent.size_bytes, phase: 'verifying' });
      }
      await writable.close();
    } catch (e) {
      try { await writable.abort(); } catch {}
      await removeIfPresent(d, ent.name);
      throw e;
    }

    if (hash.hex() !== ent.sha256) {
      await removeIfPresent(d, ent.name);
      throw new Error(`SHA-256 mismatch for ${ent.name}`);
    }
    done += ent.size_bytes;
  }

  // The manifest is the commit marker: write it only after every required GGUF
  // has passed size + SHA-256 verification. An interrupted install stays unready.
  const manifestHandle = await d.getFileHandle(MODEL_MANIFEST_NAME, { create: true });
  const manifestWritable = await manifestHandle.createWritable({ keepExistingData: false });
  await manifestWritable.write(JSON.stringify(manifest, null, 2) + '\n');
  await manifestWritable.close();
  return cacheStatus();
}

export async function loadModelSet() {
  const d = await dir(false);
  const manifest = await readStoredManifest(d);
  const models = [];
  for (const ent of manifest.files) {
    if (!ent.required) continue;
    const f = await (await d.getFileHandle(ent.name)).getFile();
    if (f.size !== ent.size_bytes) throw new Error(`Cached model size mismatch: ${ent.name}`);
    // Keep the OPFS File object directly. Re-wrapping multi-GB blobs in new File()
    // can force materialization in some browsers and is unnecessary for WORKERFS.
    models.push(f);
  }
  return { models, manifest };
}
