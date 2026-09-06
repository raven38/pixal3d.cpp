// Worker side of the browser SS harness: mount the File objects with WORKERFS (lazy reads via
// FileReaderSync, no copy into the wasm heap), call the single C entry point (JSPI async), and
// hand the report + latents back. WebGPU is requested by the ggml backend inside the module.
importScripts('pixal3d_ss.js');

self.onmessage = async (ev) => {
  const { dinov3, ssFlow, cond, sample, nViews } = ev.data;
  const post = (text) => self.postMessage({ type: 'log', text });
  try {
    post('worker: navigator.gpu present: ' + (!!self.navigator.gpu));
    const Module = await createPixal3dSS({
      print: (t) => post(t),
      printErr: (t) => post('[stderr] ' + t),
    });
    const FS = Module.FS, WORKERFS = Module.WORKERFS;
    FS.mkdir('/models'); FS.mount(WORKERFS, { files: [dinov3, ssFlow] }, '/models');
    FS.mkdir('/cond');   FS.mount(WORKERFS, { files: cond }, '/cond');
    FS.mkdir('/sample'); FS.mount(WORKERFS, { files: sample }, '/sample');
    post('worker: mounted /models/' + dinov3.name + ' (' + dinov3.size + ' B), /models/' + ssFlow.name +
         ' (' + ssFlow.size + ' B), /cond (' + cond.length + '), /sample (' + sample.length + ')');
    const t0 = performance.now();
    const report = await Module.ccall('pixal3d_ss_run', 'string',
      ['string', 'string', 'string', 'string', 'number'],
      ['/models/' + dinov3.name, '/models/' + ssFlow.name, '/cond', '/sample', nViews], { async: true });
    const wall = (performance.now() - t0) / 1000;
    const n = Module.ccall('pixal3d_ss_latent_size', 'number', [], []);
    const ptr = Module.ccall('pixal3d_ss_latent', 'number', [], []);
    const latent = new Float32Array(Module.HEAPF32.buffer, ptr, n).slice();
    const nSteps = Module.ccall('pixal3d_ss_n_steps', 'number', [], []);
    const sptr = Module.ccall('pixal3d_ss_steps', 'number', [], []);
    const steps = new Float32Array(Module.HEAPF32.buffer, sptr, n * nSteps).slice();
    self.postMessage({ type: 'done', report: report + 'browser wall ' + wall.toFixed(1) + ' s\n', latent, steps, nSteps });
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.stack ? e.stack : e) });
  }
};
