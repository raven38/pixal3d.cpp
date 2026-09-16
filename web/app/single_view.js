export const DEFAULT_SV_FOV = 0.3490658503988659;
export const DEFAULT_SV_MESH_SCALE = 1.0;

export function validateSingleViewFov(fovRad) {
  const fov = Number(fovRad);
  if (!Number.isFinite(fov) || !(fov > 0 && fov < Math.PI)) {
    throw new Error(`single-view camera FOV must satisfy 0 < fov < pi radians; got ${String(fovRad)}`);
  }
  return fov;
}

export function makeSingleViewTransforms(imageName, fovRad = DEFAULT_SV_FOV) {
  const fov = validateSingleViewFov(fovRad);
  const meshScale = DEFAULT_SV_MESH_SCALE;
  const distance = 1 / (2 * meshScale * Math.tan(fov / 2));
  return {
    camera_angle_x: fov,
    mesh_scale: meshScale,
    frames: [{
      file_path: imageName,
      name: 'front (single-view canonical gauge)',
      transform_matrix: [
        [1, 0, 0, 0],
        [0, 0, -1, -distance],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
      ],
    }],
  };
}

export function modelFamilyForManifest(manifest) {
  const explicit = manifest?.model_family;
  if (explicit === 'sv' || explicit === 'mv') return explicit;
  if (!Array.isArray(manifest?.files)) return null;
  const flowRoles = new Set(['ss_flow', 'shape_flow_512', 'shape_flow_1024', 'texture_flow_1024']);
  const flows = manifest.files.filter((f) => flowRoles.has(f?.role));
  if (!flows.length) return null;
  const allSv = flows.every((f) => typeof f?.name === 'string' && f.name.includes('_sv.gguf'));
  const allMv = flows.every((f) => typeof f?.name === 'string' && f.name.includes('_mv.gguf'));
  if (allSv) return 'sv';
  if (allMv) return 'mv';
  return null;
}

export function resolveSvModelSource({ search, windowObj, storage } = {}) {
  const w = windowObj === undefined ? (typeof window !== 'undefined' ? window : null) : windowObj;
  const q = new URLSearchParams(search ?? w?.location?.search ?? '');
  const store = storage === undefined ? w?.localStorage : storage;
  const read = (key) => {
    try { return store?.getItem?.(key) || ''; } catch { return ''; }
  };
  return {
    manifestUrl: q.get('sv_manifest_url') || w?.PIXAL3D_SV_MODEL_MANIFEST_URL || read('pixal3d.svManifestUrl') || '',
    modelBaseUrl: q.get('sv_model_base_url') || w?.PIXAL3D_SV_MODEL_BASE_URL || read('pixal3d.svModelBaseUrl') || '',
  };
}

export function saveSvModelBaseUrl(url, storage = (typeof window !== 'undefined' ? window.localStorage : null)) {
  const v = String(url || '').trim();
  try {
    if (v) storage?.setItem?.('pixal3d.svModelBaseUrl', v);
    else storage?.removeItem?.('pixal3d.svModelBaseUrl');
  } catch {}
}

function canvas2d(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    return { canvas, ctx: canvas.getContext('2d', { willReadFrequently: true }) };
  }
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return { canvas, ctx: canvas.getContext('2d', { willReadFrequently: true }) };
  }
  throw new Error('Canvas API is unavailable');
}

async function canvasPngBlob(canvas) {
  if (typeof canvas.convertToBlob === 'function') return canvas.convertToBlob({ type: 'image/png' });
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('PNG encoding failed')), 'image/png'));
}

export async function preprocessSingleViewRgba(file) {
  if (!(file instanceof Blob)) throw new Error('single-view input must be an image File/Blob');
  if (typeof createImageBitmap !== 'function') throw new Error('createImageBitmap is unavailable');
  const bitmap = await createImageBitmap(file);
  try {
    const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
    const sw = Math.max(1, Math.floor(bitmap.width * scale));
    const sh = Math.max(1, Math.floor(bitmap.height * scale));
    const resized = canvas2d(sw, sh);
    if (!resized.ctx) throw new Error('2D canvas context is unavailable');
    resized.ctx.clearRect(0, 0, sw, sh);
    resized.ctx.drawImage(bitmap, 0, 0, sw, sh);
    const pixels = resized.ctx.getImageData(0, 0, sw, sh).data;
    let minX = sw, minY = sh, maxX = -1, maxY = -1;
    let hasTransparent = false;
    for (let y = 0; y < sh; ++y) {
      for (let x = 0; x < sw; ++x) {
        const a = pixels[(y * sw + x) * 4 + 3];
        if (a < 250) hasTransparent = true;
        if (a > 0.8 * 255) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (!hasTransparent) throw new Error('single-view input needs a real alpha matte (pre-matted RGBA)');
    if (maxX < minX || maxY < minY) throw new Error('alpha matte has no pixels above the Pixal3D foreground threshold (0.8)');

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const span = Math.max(maxX - minX, maxY - minY);
    let cropSize = Math.max(2, Math.floor(span * 1.1));
    cropSize = 2 * Math.max(1, Math.floor(cropSize / 2));
    const left = Math.floor(cx - cropSize / 2);
    const top = Math.floor(cy - cropSize / 2);

    const cropped = canvas2d(cropSize, cropSize);
    if (!cropped.ctx) throw new Error('2D canvas context is unavailable');
    cropped.ctx.clearRect(0, 0, cropSize, cropSize);
    cropped.ctx.drawImage(resized.canvas, left, top, cropSize, cropSize, 0, 0, cropSize, cropSize);
    const blob = await canvasPngBlob(cropped.canvas);
    return new File([blob], 'input.png', { type: 'image/png' });
  } finally {
    bitmap.close?.();
  }
}

export async function prepareSingleViewFiles(file, fovRad = DEFAULT_SV_FOV) {
  const image = await preprocessSingleViewRgba(file);
  const transforms = makeSingleViewTransforms(image.name, fovRad);
  const json = new File([JSON.stringify(transforms, null, 2) + '\n'], 'transforms.json', { type: 'application/json' });
  return { files: [json, image], transforms, image };
}
