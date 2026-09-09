const STORAGE_MARGIN_BYTES = 512 * 1024 * 1024;

export async function storageEstimate() {
  if (!navigator.storage?.estimate) {
    return { supported: false, usage: 0, quota: 0, available: 0 };
  }
  const est = await navigator.storage.estimate();
  const usage = Number(est.usage || 0);
  const quota = Number(est.quota || 0);
  return {
    supported: true,
    usage,
    quota,
    available: Math.max(0, quota - usage),
  };
}

export function requiredModelBytes(manifest) {
  if (!manifest?.files) return 0;
  return manifest.files
    .filter((f) => f.required)
    .reduce((sum, f) => sum + Number(f.size_bytes || 0), 0);
}

export async function storagePreflight(manifest, cacheStatus = null) {
  const storage = await storageEstimate();
  const total = requiredModelBytes(manifest);
  const sameSet = !!(
    cacheStatus?.manifest &&
    cacheStatus.manifest.model_set === manifest?.model_set &&
    cacheStatus.manifest.version === manifest?.version
  );
  const verified = sameSet ? Number(cacheStatus.bytes || 0) : 0;
  const additional = Math.max(0, total - verified);
  const requiredWithMargin = additional + (additional > 0 ? STORAGE_MARGIN_BYTES : 0);

  if (!storage.supported) {
    return {
      ok: false,
      code: 'storage-estimate-unavailable',
      message: 'Browser storage quota cannot be estimated on this browser.',
      total,
      verified,
      additional,
      requiredWithMargin,
      storage,
    };
  }
  if (storage.available < requiredWithMargin) {
    return {
      ok: false,
      code: 'insufficient-storage',
      message: 'Not enough browser storage for the verified Pixal3D model set.',
      total,
      verified,
      additional,
      requiredWithMargin,
      storage,
    };
  }
  return {
    ok: true,
    code: additional > 0 ? 'download-required' : 'storage-ready',
    message: additional > 0 ? 'Storage capacity looks sufficient for the model download.' : 'Verified model set is already cached.',
    total,
    verified,
    additional,
    requiredWithMargin,
    storage,
  };
}

export async function requestPersistentStorage() {
  if (!navigator.storage?.persist) return false;
  try {
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}

export async function webgpuPreflight() {
  if (!navigator.gpu) {
    return {
      ok: false,
      code: 'webgpu-unavailable',
      message: 'WebGPU is unavailable. Use current Chrome/Chromium with WebGPU enabled, or use the Desktop app.',
      adapter: null,
      limits: null,
    };
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return {
      ok: false,
      code: 'adapter-error',
      message: `WebGPU adapter request failed: ${e?.message || e}`,
      adapter: null,
      limits: null,
    };
  }
  if (!adapter) {
    return {
      ok: false,
      code: 'adapter-unavailable',
      message: 'No WebGPU adapter is available on this device.',
      adapter: null,
      limits: null,
    };
  }

  const limits = {
    maxBufferSize: Number(adapter.limits?.maxBufferSize || 0),
    maxStorageBufferBindingSize: Number(adapter.limits?.maxStorageBufferBindingSize || 0),
    maxComputeWorkgroupsPerDimension: Number(adapter.limits?.maxComputeWorkgroupsPerDimension || 0),
  };

  // Do not invent an aggregate-VRAM threshold here: WebGPU does not expose one.
  // The C++ runtime's TRELLIS_DEVICE_BUDGET_MB / object-token preflight remains
  // the authoritative memory gate before the long neural stages execute.
  if (limits.maxBufferSize <= 0 || limits.maxStorageBufferBindingSize <= 0) {
    return {
      ok: false,
      code: 'invalid-webgpu-limits',
      message: 'The WebGPU adapter reports unusable buffer limits.',
      adapter,
      limits,
    };
  }

  let info = null;
  try {
    info = adapter.info ? {
      vendor: adapter.info.vendor || '',
      architecture: adapter.info.architecture || '',
      device: adapter.info.device || '',
      description: adapter.info.description || '',
    } : null;
  } catch {}

  return {
    ok: true,
    code: 'webgpu-ready',
    message: 'WebGPU adapter is available. Runtime memory budget checks still apply during generation.',
    adapter,
    info,
    limits,
  };
}
