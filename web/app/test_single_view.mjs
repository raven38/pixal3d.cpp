import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_SV_FOV,
  makeSingleViewTransforms,
  modelFamilyForManifest,
  resolveSvModelSource,
} from './single_view.js';
import { manifestContractErrors, modelFilesForFamily } from './model_family.js';
import { validateReleaseManifest } from './release_store.js';
import { validateManifest } from './model_store.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');

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

// ---- name<->role / model_family 契約: Python tools/model_manifest.py と共有する conformance vector ----
// 期待値は family ('sv' | 'mv' | null)。Python 側は self-test で同じファイルを読んで照合する。
const vectors = JSON.parse(readFileSync(join(here, 'manifest_conformance.json'), 'utf8'));
assert.ok(Array.isArray(vectors.cases) && vectors.cases.length >= 18, 'conformance vector must not be empty');
for (const c of vectors.cases) {
  const got = modelFamilyForManifest(c.manifest);
  assert.equal(got, c.family, `conformance ${c.id}: expected ${c.family}, got ${got} (${c.note})`);
  const errors = manifestContractErrors(c.manifest);
  assert.equal(errors.length === 0, c.family !== null, `conformance ${c.id}: errors=${JSON.stringify(errors)}`);
}
const byId = Object.fromEntries(vectors.cases.map((c) => [c.id, c.manifest]));

// テーブルは Python model_files() と同じ 9 本・同じ role（family ごとに flow 名だけ違う）
for (const family of ['mv', 'sv']) {
  const table = modelFilesForFamily(family);
  assert.equal(Object.keys(table).length, 9);
  assert.equal(table[`pixal3d_ss_flow_${family}.gguf`].role, 'ss_flow');
  assert.equal(table[`pixal3d_tex_flow_1024_${family}.gguf`].role, 'texture_flow_1024');
  assert.equal(table['dinov3.gguf'].role, 'image_encoder');
}
assert.throws(() => modelFilesForFamily('xx'), /unknown model_family/);

// committed 3 manifest は全て契約を満たす（Python check-committed と同じ集合）
const committed = {
  'models/pixal3d-f16-v1/pixal3d-models.json': 'mv',
  'models/pixal3d-q8_0-v1/pixal3d-models.json': 'mv',
  'models/pixal3d-sv-q8_0-v1/pixal3d-models.json': 'sv',
};
for (const [rel, family] of Object.entries(committed)) {
  const m = JSON.parse(readFileSync(join(repoRoot, rel), 'utf8'));
  assert.equal(modelFamilyForManifest(m), family, rel);
  assert.deepEqual(manifestContractErrors(m), [], rel);
  assert.equal(validateReleaseManifest(m), m, `${rel} must pass the release validator`);
  assert.equal(validateManifest(m), m, `${rel} must pass the local validator`);
}
// 公開 SV manifest から model_family を外しても名前から sv と推定される（Python と同じ）
{
  const m = JSON.parse(readFileSync(join(repoRoot, 'models/pixal3d-sv-q8_0-v1/pixal3d-models.json'), 'utf8'));
  delete m.model_family;
  assert.equal(modelFamilyForManifest(m), 'sv');
}

// release validator: 契約違反は throw（レビュー指摘の foo.gguf 例）、余剰は「ちょうど 9 件」で拒否
assert.throws(() => validateReleaseManifest(byId.sv_explicit_unknown_name_on_flow_role), /ss_flow must be pixal3d_ss_flow_sv\.gguf/);
assert.throws(() => validateReleaseManifest(byId.sv_role_swap), /must have role/);
assert.throws(() => validateReleaseManifest(byId.sv_with_unknown_extra), /exactly 9/);
assert.throws(() => validateReleaseManifest(byId.sv_missing_tex_dec), /exactly 9/);
assert.equal(validateReleaseManifest(byId.sv_complete_inferred), byId.sv_complete_inferred);

// local validator (Install local verified set… / OPFS 読み戻し): 9 本完備も要求する。余剰は許容
assert.throws(() => validateManifest(byId.sv_explicit_unknown_name_on_flow_role), /ss_flow must be pixal3d_ss_flow_sv\.gguf/);
assert.throws(() => validateManifest(byId.sv_missing_tex_dec), /missing required model files: tex_dec\.gguf/);
assert.throws(() => validateManifest(byId.no_flows_explicit_sv), /missing required/);
assert.throws(() => validateManifest(byId.duplicate_role), /duplicate role/);
assert.equal(validateManifest(byId.sv_with_unknown_extra), byId.sv_with_unknown_extra);
assert.equal(validateManifest(byId.mv_complete_inferred), byId.mv_complete_inferred);

// 関数単体: files が無い / 空の manifest は family 不明（9 本完備でないので Python と同じく不合格）
assert.equal(modelFamilyForManifest({ model_family: 'sv', files: [] }), null);
assert.equal(modelFamilyForManifest({ model_family: 'sv' }), null);
assert.equal(modelFamilyForManifest(null), null);
assert.equal(modelFamilyForManifest({ files: [{ role: 'ss_flow', name: 'weird.gguf' }] }), null);

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

// ---- D9: C++ 実装（src/transforms_json.cpp の synthesize_single_view_gauge）との一致 ----
// 共有フィクスチャ tests/fixtures/sv_gauge/*.json は、この JS 実装から生成した gauge を
// そのまま保存したもの。C++ テスト（trellis-test-sv-input）も同じファイルを読んで
// 相対誤差 1e-6 で比較する。ここが落ちたら、JS を変えてフィクスチャを再生成していない
// （= 共有 C++ 側と無言で食い違った）ということ。
{
  const cases = [
    ['fov_20deg.json', 0.3490658503988659],
    ['fov_30deg.json', Math.PI / 6],
    ['fov_45deg.json', Math.PI / 4],
    ['fov_min.json', 1e-3],
    ['fov_max.json', Math.PI - 1e-3],
  ];
  for (const [file, fov] of cases) {
    const want = JSON.parse(readFileSync(join(repoRoot, 'tests', 'fixtures', 'sv_gauge', file), 'utf8'));
    const got = makeSingleViewTransforms(want.frames[0].file_path, fov);
    assert.deepEqual(got, want, `${file} no longer matches makeSingleViewTransforms; regenerate the shared fixture and re-run trellis-test-sv-input`);
  }
}

console.log('WEB_SINGLE_VIEW_CONTRACT_OK');
