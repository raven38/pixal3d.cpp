// Worker side of the browser Shape-1024 harness: mount the File objects with WORKERFS (lazy reads
// via FileReaderSync, no copy into the wasm heap), call the single C entry point (JSPI async) --
// pixal3d_shape1024_run for the full path, pixal3d_shape_step_run for the checkpoint/resume probe
// (mode: 'step') -- and hand the report + latents back. Same module as web/ss (built by scripts/build_wasm_ss.sh);
// WebGPU is requested by the ggml backend inside it.
importScripts('../ss/pixal3d_ss.js');

self.onmessage = async (ev) => {
  const { dinov3, naf, flow, cond, sample, nViews, ownCond } = ev.data;
  const post = (text) => self.postMessage({ type: 'log', text });
  try {
    post('worker: navigator.gpu present: ' + (!!self.navigator.gpu));
    const Module = await createPixal3dSS({
      // the module lives in ../ss/ (one wasm for both stages); Emscripten resolves the .wasm
      // relative to this worker's URL unless told otherwise
      locateFile: (p) => new URL('../ss/' + p, self.location.href).href,
      print: (t) => post(t),
      printErr: (t) => post('[stderr] ' + t),
    });
    const FS = Module.FS, WORKERFS = Module.WORKERFS;
    if (ev.data.mode === 'step') {
      // checkpoint/resume probe: flow GGUF + hr_sample files (+ optional latents), one C call
      const { start, nSteps, repeat, inLatent, expLatent } = ev.data;
      FS.mkdir('/models'); FS.mount(WORKERFS, { files: [flow] }, '/models');
      FS.mkdir('/sample'); FS.mount(WORKERFS, { files: sample }, '/sample');
      FS.mkdir('/latent'); FS.mount(WORKERFS, { files: [inLatent, expLatent].filter(Boolean) }, '/latent');
      post('worker: mounted /models/' + flow.name + ' (' + flow.size + ' B), /sample (' + sample.length + ')' +
           (inLatent ? ', /latent/' + inLatent.name : '') + (expLatent ? ', /latent/' + expLatent.name : ''));
      const t0 = performance.now();
      const report = await Module.ccall('pixal3d_shape_step_run', 'string',
        ['string', 'string', 'number', 'number', 'number', 'string', 'string', 'number'],
        ['/models/' + flow.name, '/sample', 1, start, nSteps, inLatent ? '/latent/' + inLatent.name : '',
         expLatent ? '/latent/' + expLatent.name : '', repeat], { async: true });
      const wall = (performance.now() - t0) / 1000;
      const n = Module.ccall('pixal3d_shape512_latent_size', 'number', [], []);
      const ptr = Module.ccall('pixal3d_shape512_latent', 'number', [], []);
      const latent = new Float32Array(Module.HEAPF32.buffer, ptr, n).slice();
      self.postMessage({ type: 'done', report: report + 'browser wall ' + wall.toFixed(1) + ' s\n', latent, steps: new Float32Array(0), nSteps: 0 });
      return;
    }
    FS.mkdir('/models'); FS.mount(WORKERFS, { files: [dinov3, naf, flow] }, '/models');
    FS.mkdir('/cond');   FS.mount(WORKERFS, { files: cond }, '/cond');
    FS.mkdir('/sample'); FS.mount(WORKERFS, { files: sample }, '/sample');
    post('worker: mounted /models/{' + dinov3.name + ' (' + dinov3.size + ' B), ' + naf.name + ' (' + naf.size + ' B), ' +
         flow.name + ' (' + flow.size + ' B)}, /cond (' + cond.length + '), /sample (' + sample.length + ')');
    const t0 = performance.now();
    const report = await Module.ccall('pixal3d_shape1024_run', 'string',
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
