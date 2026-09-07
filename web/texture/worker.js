// Worker side of the browser Texture-1024 harness. Texture Flow is temporarily emitted as a
// dedicated WASM module at the v0.6-v0.8 merge boundary so the already-validated decoder module
// remains untouched; inference is still the same shared C++/ggml WebGPU implementation.
importScripts('./pixal3d_texture.js');

self.onmessage = async (ev) => {
  const { flow, dinov3, naf, sample, cond, concat, nViews, ownCond } = ev.data;
  const post = (text) => self.postMessage({ type: 'log', text });
  try {
    post('worker: navigator.gpu present: ' + (!!self.navigator.gpu));
    const Module = await createPixal3dTexture({
      locateFile: (p) => new URL('./' + p, self.location.href).href,
      print: (t) => post(t),
      printErr: (t) => post('[stderr] ' + t),
    });
    const FS = Module.FS, WORKERFS = Module.WORKERFS;
    const modelFiles = [flow].concat(dinov3 ? [dinov3] : [], naf ? [naf] : []);
    FS.mkdir('/models'); FS.mount(WORKERFS, { files: modelFiles }, '/models');
    FS.mkdir('/sample'); FS.mount(WORKERFS, { files: sample }, '/sample');
    FS.mkdir('/cond');   FS.mount(WORKERFS, { files: cond }, '/cond');
    let concatPath = '';
    if (concat) { FS.mkdir('/concat'); FS.mount(WORKERFS, { files: [concat] }, '/concat'); concatPath = '/concat/' + concat.name; }
    post('worker: mounted /models/{' + modelFiles.map(f => f.name + ' (' + f.size + ' B)').join(', ') + '}, /sample (' + sample.length +
         '), /cond (' + cond.length + ')' + (concat ? ', /concat/' + concat.name : ''));
    const t0 = performance.now();
    const report = await Module.ccall('pixal3d_texture_run', 'string',
      ['string', 'string', 'string', 'string', 'string', 'number', 'number', 'string'],
      [dinov3 ? '/models/' + dinov3.name : '', naf ? '/models/' + naf.name : '', '/models/' + flow.name,
       '/cond', '/sample', nViews, ownCond, concatPath], { async: true });
    const wall = (performance.now() - t0) / 1000;
    const n = Module.ccall('pixal3d_texture_latent_size', 'number', [], []);
    const ptr = Module.ccall('pixal3d_texture_latent', 'number', [], []);
    const latent = new Float32Array(Module.HEAPF32.buffer, ptr, n).slice();
    const nSteps = Module.ccall('pixal3d_texture_n_steps', 'number', [], []);
    const sptr = Module.ccall('pixal3d_texture_steps', 'number', [], []);
    const steps = new Float32Array(Module.HEAPF32.buffer, sptr, n * nSteps).slice();
    self.postMessage({ type: 'done', report: report + 'browser wall ' + wall.toFixed(1) + ' s\n', latent, steps, nSteps });
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.stack ? e.stack : e) });
  }
};
