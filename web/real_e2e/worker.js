importScripts('./pixal3d_real_geometry.js');

const GEOMETRY_ROLES = [
  'image_encoder',
  'naf',
  'ss_flow',
  'ss_decoder',
  'shape_flow_512',
  'shape_decoder',
  'shape_flow_1024',
];
const TEXTURE_ROLES = ['texture_flow_1024', 'texture_decoder'];

// Legacy developer harnesses predate the model manifest. Production web/app
// always supplies modelManifest and therefore does not duplicate filenames here.
const LEGACY_GEOMETRY = [
  'dinov3.gguf',
  'pixal3d_naf.gguf',
  'pixal3d_ss_flow_mv.gguf',
  'ss_dec.gguf',
  'pixal3d_shape_flow_512_mv.gguf',
  'shape_dec.gguf',
  'pixal3d_shape_flow_1024_mv.gguf',
];
const LEGACY_TEXTURE = ['pixal3d_tex_flow_1024_mv.gguf', 'tex_dec.gguf'];

function filesFromManifest(manifest, roles) {
  if (!manifest?.files) return null;
  const byRole = new Map(manifest.files.map((ent) => [ent.role, ent.name]));
  return roles.map((role) => {
    const name = byRole.get(role);
    if (!name) throw new Error(`model manifest missing required role ${role}`);
    return name;
  });
}

self.onmessage = async (ev) => {
  const { models, modelManifest, views, seed, resolution = 1024 } = ev.data;
  const log = (text) => self.postMessage({ type: 'log', text });
  try {
    if (resolution !== 1024) {
      throw new Error(`browser WebGPU runtime supports resolution=1024 only; got ${resolution}`);
    }

    const M = await createPixal3dRealGeometry({
      locateFile: (p) => new URL('./' + p, self.location.href).href,
      print: log,
      printErr: (t) => log('[stderr] ' + t),
    });
    const FS = M.FS, WORKERFS = M.WORKERFS;
    FS.mkdir('/models');
    FS.mount(WORKERFS, { files: models }, '/models');
    FS.mkdir('/views');
    FS.mount(WORKERFS, { files: views }, '/views');
    FS.mkdir('/out');

    const byName = Object.fromEntries(models.map((f) => [f.name, f]));
    const geometry = filesFromManifest(modelManifest, GEOMETRY_ROLES) || LEGACY_GEOMETRY;
    const texture = filesFromManifest(modelManifest, TEXTURE_ROLES) || LEGACY_TEXTURE;
    for (const name of geometry) {
      if (!byName[name]) throw new Error('missing model ' + name);
    }
    const full = texture.every((name) => !!byName[name]);
    const outPath = full ? '/out/real_full.glb' : '/out/real_geometry.glb';
    const files = full ? geometry.concat(texture) : geometry;
    const args = files.map((name) => '/models/' + name).concat(['/views', outPath, seed || 1]);
    const types = Array(files.length + 2).fill('string').concat(['number']);
    log(full ? 'live Texture conditioning included: full E2E' : 'geometry E2E only (texture models not provided)');
    const report = await M.ccall(
      full ? 'pixal3d_real_full_run' : 'pixal3d_real_geometry_run',
      'string',
      types,
      args,
      { async: true },
    );
    const glb = FS.readFile(outPath);
    self.postMessage({ type: 'done', report, glb: glb.buffer }, [glb.buffer]);
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e?.stack || e) });
  }
};
