import assert from 'node:assert/strict';
import { DEFAULT_SV_FOV, makeSingleViewTransforms, modelFamilyForManifest } from './single_view.js';

assert.equal(DEFAULT_SV_FOV, 0.3490658503988659);
const tf = makeSingleViewTransforms('input.png');
assert.equal(tf.camera_angle_x, DEFAULT_SV_FOV);
assert.equal(tf.mesh_scale, 1.0);
assert.equal(tf.frames.length, 1);
assert.equal(tf.frames[0].file_path, 'input.png');
const expectedDistance = 1 / (2 * Math.tan(DEFAULT_SV_FOV / 2));
assert.ok(Math.abs(tf.frames[0].transform_matrix[1][3] + expectedDistance) < 1e-12);
assert.deepEqual(tf.frames[0].transform_matrix[0], [1, 0, 0, 0]);
assert.deepEqual(tf.frames[0].transform_matrix[2], [0, 1, 0, 0]);

for (const bad of [0, -0.1, Math.PI, Math.PI + 0.01, NaN, Infinity]) {
  assert.throws(() => makeSingleViewTransforms('input.png', bad), /0 < fov < pi/);
}

const sv = { files: [
  { role:'ss_flow', name:'pixal3d_ss_flow_sv.gguf' },
  { role:'shape_flow_512', name:'pixal3d_shape_flow_512_sv.gguf' },
  { role:'shape_flow_1024', name:'pixal3d_shape_flow_1024_sv.gguf' },
  { role:'texture_flow_1024', name:'pixal3d_tex_flow_1024_sv.gguf' },
] };
const mv = { files: [
  { role:'ss_flow', name:'pixal3d_ss_flow_mv.gguf' },
  { role:'shape_flow_512', name:'pixal3d_shape_flow_512_mv.gguf' },
  { role:'shape_flow_1024', name:'pixal3d_shape_flow_1024_mv.gguf' },
  { role:'texture_flow_1024', name:'pixal3d_tex_flow_1024_mv.gguf' },
] };
assert.equal(modelFamilyForManifest(sv), 'sv');
assert.equal(modelFamilyForManifest(mv), 'mv');
assert.equal(modelFamilyForManifest({ model_family:'sv', files:[] }), 'sv');
assert.equal(modelFamilyForManifest({ files:[{role:'ss_flow',name:'weird.gguf'}] }), null);

console.log('WEB_SINGLE_VIEW_CONTRACT_OK');
