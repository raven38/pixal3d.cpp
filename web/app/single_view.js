export const SINGLE_VIEW_MESH_SCALE = 1.0;
export const SINGLE_VIEW_MAX_SIDE = 1024;
export const SINGLE_VIEW_ALPHA_THRESHOLD = 0.8;
export const SINGLE_VIEW_CROP_MARGIN = 1.1;

const FLOW_ROLES = ['ss_flow', 'shape_flow_512', 'shape_flow_1024', 'texture_flow_1024'];
const IMAGE_RE = /\.(png|jpe?g|webp)$/i;

function baseName(path) {
  return String(path || '').replace(/\\/g, '/').split('/').pop();
}

export function modelVariant(manifest) {
  if (!manifest?.files || !Array.isArray(manifest.files)) return 'invalid';
  const byRole = new Map(manifest.files.map((entry) => [entry?.role, entry?.name]));
  const names = FLOW_ROLES.map((role) => byRole.get(role));
  if (names.some((name) => typeof name !== 'string' || !name)) return 'invalid';
  const sv = names.every((name) => /_sv\.gguf$/i.test(name));
  const mv = names.every((name) => /_mv\.gguf$/i.test(name));
  return sv ? 'sv' : (mv ? 'mv' : 'invalid');
}

export function classifyInput(files) {
  const list = [...(files || [])];
  const jsonFiles = list.filter((f) => baseName(f?.name).toLowerCase() === 'transforms.json' || String(f?.name || '').toLowerCase().endsWith('.json'));
  const images = list.filter((f) => String(f?.type || '').startsWith('image/') || IMAGE_RE.test(String(f?.name || '')));
  if (jsonFiles.length > 1) return { mode: null, images, jsonFile: null, error: 'Select at most one transforms.json.' };
  if (jsonFiles.length === 1) {
    if (!images.length) return { mode: null, images, jsonFile: jsonFiles[0], error: 'transforms.json needs at least one image.' };
    return { mode: 'mv-json', images, jsonFile: jsonFiles[0], error: null };
  }
  if (images.length === 1) return { mode: 'sv', images, jsonFile: null, error: null };
  if (images.length === 4) return { mode: 'mv-canonical', images, jsonFile: null, error: null };
  return {
    mode: null,
    images,
    jsonFile: null,
    error: `Without transforms.json select exactly one image (SV) or four turntable images (MV); found ${images.length}.`,
  };
}

export function validateFov(fov) {
  return Number.isFinite(fov) && fov > 0 && fov < Math.PI;
}

export function singleViewDistance(fov, meshScale = SINGLE_VIEW_MESH_SCALE) {
  if (!validateFov(fov)) throw new Error(`camera FOV must satisfy 0 < fov < pi; got ${fov}`);
  if (!Number.isFinite(meshScale) || meshScale <= 0) throw new Error(`mesh_scale must be finite and > 0; got ${meshScale}`);
  return 1 / (2 * meshScale * Math.tan(fov / 2));
}

export function singleViewTransforms(imageName, fov, meshScale = SINGLE_VIEW_MESH_SCALE) {
  const distance = singleViewDistance(fov, meshScale);
  return {
    camera_angle_x: fov,
    mesh_scale: meshScale,
    frames: [{
      file_path: imageName,
      name: 'front (estimated single view)',
      transform_matrix: [
        [1, 0, 0, 0],
        [0, 0, -1, -distance],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
      ],
    }],
  };
}

export function computeCropBounds({ x0, y0, x1, y1 }) {
  if (![x0, y0, x1, y1].every(Number.isFinite) || x1 < x0 || y1 < y0) throw new Error('invalid foreground bbox');
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const span = Math.max(x1 - x0, y1 - y0);
  let cropSize = Math.max(2, Math.trunc(span * SINGLE_VIEW_CROP_MARGIN));
  const half = Math.max(1, Math.trunc(cropSize / 2));
  cropSize = 2 * half;
  return { left: Math.floor(cx - half), top: Math.floor(cy - half), size: cropSize };
}

function canvasToPngFile(canvas, name = 'input.png') {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error('failed to encode processed PNG'));
      else resolve(new File([blob], name, { type: 'image/png' }));
    }, 'image/png');
  });
}

export async function preprocessSingleViewFile(file) {
  if (typeof document === 'undefined' || typeof createImageBitmap !== 'function') {
    throw new Error('single-view image preprocessing requires browser Canvas APIs');
  }
  const bitmap = await createImageBitmap(file);
  try {
    let width = bitmap.width, height = bitmap.height;
    const scale = Math.min(1, SINGLE_VIEW_MAX_SIDE / Math.max(width, height));
    const scaledW = Math.max(1, Math.trunc(width * scale));
    const scaledH = Math.max(1, Math.trunc(height * scale));
    const src = document.createElement('canvas');
    src.width = scaledW; src.height = scaledH;
    const sctx = src.getContext('2d', { willReadFrequently: true });
    sctx.clearRect(0, 0, scaledW, scaledH);
    sctx.drawImage(bitmap, 0, 0, scaledW, scaledH);
    const rgba = sctx.getImageData(0, 0, scaledW, scaledH);

    let alphaMin = 255, alphaMax = 0;
    let x0 = scaledW, y0 = scaledH, x1 = -1, y1 = -1;
    const threshold = SINGLE_VIEW_ALPHA_THRESHOLD * 255;
    for (let y = 0; y < scaledH; y++) for (let x = 0; x < scaledW; x++) {
      const a = rgba.data[(y * scaledW + x) * 4 + 3];
      alphaMin = Math.min(alphaMin, a); alphaMax = Math.max(alphaMax, a);
      if (a > threshold) {
        x0 = Math.min(x0, x); y0 = Math.min(y0, y);
        x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      }
    }
    if (alphaMin === 255) throw new Error('input image has no real alpha matte; provide pre-matted RGBA');
    if (alphaMax === 0) throw new Error('input image is fully transparent');
    if (x1 < x0 || y1 < y0) throw new Error('alpha matte has no pixels above the Pixal3D foreground threshold (0.8)');

    const { left, top, size } = computeCropBounds({ x0, y0, x1, y1, width: scaledW, height: scaledH });
    const cropped = document.createElement('canvas');
    cropped.width = size; cropped.height = size;
    const cctx = cropped.getContext('2d', { willReadFrequently: true });
    cctx.clearRect(0, 0, size, size);
    cctx.drawImage(src, -left, -top);
    const croppedRgba = cctx.getImageData(0, 0, size, size);

    // Explicit black composite for the camera estimator. Keep croppedRgba untouched
    // for Pixal3D so the existing worker/C++ alpha premultiplication sees the same RGB.
    const black = new ImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const a = croppedRgba.data[4*i+3] / 255;
      black.data[4*i] = Math.round(croppedRgba.data[4*i] * a);
      black.data[4*i+1] = Math.round(croppedRgba.data[4*i+1] * a);
      black.data[4*i+2] = Math.round(croppedRgba.data[4*i+2] * a);
      black.data[4*i+3] = 255;
    }
    const staged = await canvasToPngFile(cropped, 'input.png');
    return { file: staged, blackImageData: black, width: size, height: size, sourceWidth: width, sourceHeight: height };
  } finally {
    bitmap.close?.();
  }
}
