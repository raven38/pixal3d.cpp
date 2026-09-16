import assert from 'node:assert/strict';
import {
  DEFAULT_SV_FOV,
  makeSingleViewTransforms,
  modelFamilyForManifest,
  resolveSvModelSource,
} from './single_view.js';

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

const storage = new Map([
  ['pixal3d.svManifestUrl', 'https://stored.test/manifest.json'],
  ['pixal3d.svModelBaseUrl', 'https://stored.test/files'],
]);
const store = { getItem:(k)=>storage.get(k)||'' };
assert.deepEqual(
  resolveSvModelSource({ search:'', windowObj:null, storage:store }),
  { manifestUrl:'https://stored.test/manifest.json', modelBaseUrl:'https://stored.test/files' },
);
assert.deepEqual(
  resolveSvModelSource({
    search:'?sv_manifest_url=https%3A%2F%2Fquery.test%2Fm.json&sv_model_base_url=https%3A%2F%2Fquery.test%2Ffiles',
    windowObj:null,
    storage:store,
  }),
  { manifestUrl:'https://query.test/m.json', modelBaseUrl:'https://query.test/files' },
);
assert.deepEqual(
  resolveSvModelSource({ search:'', windowObj:{ PIXAL3D_SV_MODEL_MANIFEST_URL:'https://window.test/m.json', PIXAL3D_SV_MODEL_BASE_URL:'https://window.test/files' }, storage:null }),
  { manifestUrl:'https://window.test/m.json', modelBaseUrl:'https://window.test/files' },
);

console.log('WEB_SINGLE_VIEW_CONTRACT_OK');
