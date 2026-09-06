// Worker side of the browser Shape-512 harness: mount the File objects with WORKERFS (lazy reads
// via FileReaderSync, no copy into the wasm heap), call the single C entry point (JSPI async),
// and hand the report + latents back. Same module as web/ss (built by scripts/build_wasm_ss.sh);
// WebGPU is requested by the ggml backend inside it.
importScripts('../ss/pixal3d_ss.js');

self.onmessage = async (ev) => {
  const { dinov3, naf, flow, cond, sample, nViews, ownCond } = ev.data;
  const post = (text) => self.postMessage({ type: 'log', text });
  try {
    post('worker: navigator.gpu present: ' + (!!self.navigator.gpu));
    const Module = await createPixal3dSS({
      print: (t) => post(t),
      printErr: (t) => post('[stderr] ' + t),
    });
    const FS = Module.FS, WORKERFS = Module.WORKERFS;
    FS.mkdir('/models'); FS.mount(WORKERFS, { files: [dinov3, naf, flow] }, '/models');
    FS.mkdir('/cond');   FS.mount(WORKERFS, { files: cond }, '/cond');
    FS.mkdir('/sample'); FS.mount(WORKERFS, { files: sample }, '/sample');
    post('worker: mounted /models/{' + dinov3.name + ' (' + dinov3.size + ' B), ' + naf.name + ' (' + naf.size + ' B), ' +
         flow.name + ' (' + flow.size + ' B)}, /cond (' + cond.length + '), /sample (' + sample.length + ')');
    const t0 = performance.now();
    const report = await Module.ccall('pixal3d_shape512_run', 'string',
      ['string', 'string', 'string', 'string', 'string', 'number', 'number'],
      ['/models/' + dinov3.name, '/models/' + naf.name, '/models/' + flow.name, '/cond', '/sample', nViews, ownCond], { async: true });
    const wall = (performance.now() - t0) / 1000;
    const n = Module.ccall('pixal3d_shape512_latent_size', 'number', [], []);
    const ptr = Module.ccall('pixal3d_shape512_latent', 'number', [], []);
    const latent = new Float32Array(Module.HEAPF32.buffer, ptr, n).slice();
    const nSteps = Module.ccall('pixal3d_shape512_n_steps', 'number', [], []);
    const sptr = Module.ccall('pixal3d_shape512_steps', 'number', [], []);
    const steps = new Float32Array(Module.HEAPF32.buffer, sptr, n * nSteps).slice();
    self.postMessage({ type: 'done', report: report + 'browser wall ' + wall.toFixed(1) + ' s\n', latent, steps, nSteps });
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.stack ? e.stack : e) });
  }
};
