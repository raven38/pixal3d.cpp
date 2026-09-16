// Browser port of the camera-recovery convention used by Microsoft MoGe.
// MoGe predicts an affine-invariant point map. Camera focal/shift are recovered by
// minimizing |f * xy/(z+shift) - uv| in normalized view-plane coordinates.

function finite(v) { return Number.isFinite(v); }

export function normalizedViewPlaneUv(width, height, x, y) {
  if (!(width > 0 && height > 0)) throw new Error('invalid image dimensions');
  const aspect = width / height;
  const denom = Math.sqrt(1 + aspect * aspect);
  const spanX = aspect / denom;
  const spanY = 1 / denom;
  const sx = spanX * (width - 1) / width;
  const sy = spanY * (height - 1) / height;
  return [
    width === 1 ? 0 : -sx + 2 * sx * x / (width - 1),
    height === 1 ? 0 : -sy + 2 * sy * y / (height - 1),
  ];
}

function buildSupport(points, mask, width, height, maxSide = 64) {
  if (!points || points.length !== width * height * 3) throw new Error('point map shape mismatch');
  if (mask && mask.length !== width * height) throw new Error('mask shape mismatch');
  const stride = Math.max(1, Math.ceil(Math.max(width, height) / maxSide));
  const support = [];
  for (let y = 0; y < height; y += stride) for (let x = 0; x < width; x += stride) {
    const i = y * width + x;
    if (mask && !(mask[i] > 0.5)) continue;
    const px = points[3*i], py = points[3*i+1], pz = points[3*i+2];
    if (![px, py, pz].every(finite)) continue;
    const [u, v] = normalizedViewPlaneUv(width, height, x, y);
    support.push({ px, py, pz, u, v });
  }
  if (support.length < 4) throw new Error(`degenerate camera support: ${support.length} valid point(s)`);
  return support;
}

function focalAndError(support, shift) {
  let dot = 0, sq = 0;
  for (const p of support) {
    const z = p.pz + shift;
    if (!finite(z) || Math.abs(z) < 1e-7) return { focal: NaN, error: Infinity };
    const x = p.px / z, y = p.py / z;
    dot += x * p.u + y * p.v;
    sq += x * x + y * y;
  }
  if (!(sq > 1e-14)) return { focal: NaN, error: Infinity };
  const focal = dot / sq;
  if (!(focal > 0) || !finite(focal)) return { focal, error: Infinity };
  let error = 0;
  for (const p of support) {
    const z = p.pz + shift;
    const ex = focal * p.px / z - p.u;
    const ey = focal * p.py / z - p.v;
    error += ex * ex + ey * ey;
  }
  return { focal, error: error / (2 * support.length) };
}

function goldenMinimize(fn, lo, hi, iterations = 80) {
  const phi = (Math.sqrt(5) - 1) / 2;
  let a = lo, b = hi;
  let c = b - phi * (b - a), d = a + phi * (b - a);
  let fc = fn(c), fd = fn(d);
  for (let i = 0; i < iterations; i++) {
    if (fc <= fd) {
      b = d; d = c; fd = fc; c = b - phi * (b - a); fc = fn(c);
    } else {
      a = c; c = d; fc = fd; d = a + phi * (b - a); fd = fn(d);
    }
  }
  return fc <= fd ? c : d;
}

export function recoverFocalShift(points, mask, width, height, { maxSide = 64 } = {}) {
  const support = buildSupport(points, mask, width, height, maxSide);
  let minZ = Infinity, maxZ = -Infinity;
  for (const p of support) { minZ = Math.min(minZ, p.pz); maxZ = Math.max(maxZ, p.pz); }
  const zScale = Math.max(1e-3, maxZ - minZ, Math.abs(minZ), Math.abs(maxZ));
  const lower = -minZ + Math.max(1e-5, zScale * 1e-5);
  const upper = lower + Math.max(4, zScale * 12);

  // Coarse scan first. The joint focal/shift objective is one-dimensional but
  // can be very flat far from the optimum; a scan gives golden search a stable bracket.
  const bins = 128;
  let bestI = -1, bestShift = 0, best = { focal: NaN, error: Infinity };
  for (let i = 0; i <= bins; i++) {
    const s = lower + (upper - lower) * i / bins;
    const cur = focalAndError(support, s);
    if (cur.error < best.error) { best = cur; bestI = i; bestShift = s; }
  }
  if (bestI < 0 || !finite(best.error)) throw new Error('degenerate focal/shift objective');
  const step = (upper - lower) / bins;
  const lo = Math.max(lower, bestShift - step * 2);
  const hi = Math.min(upper, bestShift + step * 2);
  const shift = goldenMinimize((s) => focalAndError(support, s).error, lo, hi);
  const solved = focalAndError(support, shift);
  if (!(solved.focal > 0) || !finite(solved.error)) throw new Error('degenerate focal recovery result');
  return { focal: solved.focal, shift, error: solved.error, support: support.length };
}

// MoGe's recovered `focal` is relative to half the image diagonal. Convert to
// normalized fx where image width spans 1.0 and principal point is 0.5.
export function focalToNormalizedFx(focal, width, height) {
  if (!(focal > 0) || !finite(focal) || !(width > 0 && height > 0)) throw new Error('invalid focal or image size');
  const aspect = width / height;
  return focal / 2 * Math.sqrt(1 + aspect * aspect) / aspect;
}

export function focalToHorizontalFov(focal, width, height) {
  const fx = focalToNormalizedFx(focal, width, height);
  const fov = 2 * Math.atan(0.5 / fx);
  if (!(fov > 0 && fov < Math.PI) || !finite(fov)) throw new Error('invalid recovered horizontal FOV');
  return fov;
}
