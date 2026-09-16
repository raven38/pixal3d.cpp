import assert from 'node:assert/strict';
import {
  classifyInput,
  modelVariant,
  singleViewDistance,
  singleViewTransforms,
  validateFov,
  computeCropBounds,
} from './single_view.js';

const manifest = (names) => ({ files: [
  { role: 'ss_flow', name: names[0] },
  { role: 'shape_flow_512', name: names[1] },
  { role: 'shape_flow_1024', name: names[2] },
  { role: 'texture_flow_1024', name: names[3] },
]});
const sv = manifest([
  'pixal3d_ss_flow_sv.gguf',
  'pixal3d_shape_flow_512_sv.gguf',
  'pixal3d_shape_flow_1024_sv.gguf',
  'pixal3d_tex_flow_1024_sv.gguf',
]);
const mv = manifest([
  'pixal3d_ss_flow_mv.gguf',
  'pixal3d_shape_flow_512_mv.gguf',
  'pixal3d_shape_flow_1024_mv.gguf',
  'pixal3d_tex_flow_1024_mv.gguf',
]);
assert.equal(modelVariant(sv), 'sv');
assert.equal(modelVariant(mv), 'mv');
assert.equal(modelVariant(manifest([
  'pixal3d_ss_flow_sv.gguf',
  'pixal3d_shape_flow_512_mv.gguf',
  'pixal3d_shape_flow_1024_sv.gguf',
  'pixal3d_tex_flow_1024_sv.gguf',
])), 'invalid');
assert.equal(modelVariant({ files: [] }), 'invalid');

const img = (name) => ({ name, type: 'image/png' });
const json = { name: 'transforms.json', type: 'application/json' };
assert.equal(classifyInput([img('front.png')]).mode, 'sv');
assert.equal(classifyInput([img('f.png'), img('r.png'), img('b.png'), img('l.png')]).mode, 'mv-canonical');
assert.equal(classifyInput([json, img('front.png')]).mode, 'mv-json');
assert.match(classifyInput([img('a.png'), img('b.png')]).error, /exactly one or four/i);

const fov = 0.3490658503988659;
assert.equal(validateFov(fov), true);
assert.equal(validateFov(0), false);
assert.equal(validateFov(Math.PI), false);
assert.equal(validateFov(Infinity), false);
const d = singleViewDistance(fov);
assert.ok(Math.abs(d - 2.8356409098088546) < 1e-9, d);
const tf = singleViewTransforms('input.png', fov);
assert.equal(tf.mesh_scale, 1.0);
assert.equal(tf.frames.length, 1);
assert.deepEqual(tf.frames[0].transform_matrix[0], [1,0,0,0]);
assert.deepEqual(tf.frames[0].transform_matrix[2], [0,1,0,0]);
assert.ok(Math.abs(tf.frames[0].transform_matrix[1][3] + d) < 1e-12);

// Foreground bbox [30,20]..[69,59] inclusive -> span 39, 1.1x -> 42 -> even 42.
const crop = computeCropBounds({ x0: 30, y0: 20, x1: 69, y1: 59, width: 100, height: 80 });
assert.deepEqual(crop, { left: 28, top: 18, size: 42 });

console.log('SINGLE_VIEW_CONTRACT_OK');
