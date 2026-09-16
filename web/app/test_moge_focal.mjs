import assert from 'node:assert/strict';
import { focalToHorizontalFov, focalToNormalizedFx, recoverFocalShift } from './moge_focal.js';

function normalizedUv(width, height, x, y) {
  const a = width / height;
  const denom = Math.sqrt(1 + a * a);
  const spanX = a / denom;
  const spanY = 1 / denom;
  return [
    -spanX * (width - 1) / width + (2 * spanX * (width - 1) / width) * x / (width - 1),
    -spanY * (height - 1) / height + (2 * spanY * (height - 1) / height) * y / (height - 1),
  ];
}

const width = 32, height = 24;
const focal = 0.83, shift = 0.37;
const points = new Float32Array(width * height * 3);
const mask = new Float32Array(width * height).fill(1);
for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
  const i = y * width + x;
  const [u, v] = normalizedUv(width, height, x, y);
  const z = 1.0 + 0.4 * Math.sin(x * 0.31) + 0.2 * Math.cos(y * 0.43);
  points[3*i] = u * (z + shift) / focal;
  points[3*i+1] = v * (z + shift) / focal;
  points[3*i+2] = z;
}
const recovered = recoverFocalShift(points, mask, width, height);
assert.ok(Math.abs(recovered.focal - focal) < 2e-3, recovered);
assert.ok(Math.abs(recovered.shift - shift) < 2e-3, recovered);

const fx = focalToNormalizedFx(focal, width, height);
const a = width / height;
const expectedFx = focal / 2 * Math.sqrt(1 + a*a) / a;
assert.ok(Math.abs(fx - expectedFx) < 1e-12);
const fov = focalToHorizontalFov(focal, width, height);
assert.ok(Math.abs(fov - 2 * Math.atan(0.5 / expectedFx)) < 1e-12);
assert.ok(fov > 0 && fov < Math.PI);

assert.throws(() => recoverFocalShift(new Float32Array(3), new Float32Array([0]), 1, 1), /support|degenerate/i);
console.log('MOGE_FOCAL_CONTRACT_OK');
