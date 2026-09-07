// Thin browser harness for the fixture-injected partial E2E C++ entry point.
importScripts('./pixal3d_partial_e2e.js');

self.onmessage = async (ev) => {
  const { texFlow, shapeDec, texDec, fixture, concat, resolution = 1024 } = ev.data;
  const log = (text) => self.postMessage({ type: 'log', text });
  try {
    log('worker: navigator.gpu=' + (!!self.navigator.gpu));
    const Module = await createPixal3dPartialE2E({
      locateFile: (p) => new URL('./' + p, self.location.href).href,
      print: (t) => log(t),
      printErr: (t) => log('[stderr] ' + t),
    });
    const FS = Module.FS, WORKERFS = Module.WORKERFS;
    FS.mkdir('/models');
    FS.mount(WORKERFS, { files: [texFlow, shapeDec, texDec] }, '/models');
    FS.mkdir('/fixture');
    FS.mount(WORKERFS, { files: fixture }, '/fixture');
    let concatPath = '';
    if (concat) {
      FS.mkdir('/concat'); FS.mount(WORKERFS, { files: [concat] }, '/concat');
      concatPath = '/concat/' + concat.name;
    }
    FS.mkdir('/out');
    const outPath = '/out/partial_e2e.glb';
    const t0 = performance.now();
    const report = await Module.ccall('pixal3d_partial_e2e_run', 'string',
      ['string','string','string','string','string','string','number'],
      ['/models/' + texFlow.name, '/models/' + shapeDec.name, '/models/' + texDec.name,
       '/fixture', concatPath, outPath, resolution], { async: true });
    const wall = (performance.now() - t0) / 1000;
    const glb = FS.readFile(outPath, { encoding: 'binary' });
    self.postMessage({ type: 'done', report: report + 'browser wall ' + wall.toFixed(1) + ' s\n', glb }, [glb.buffer]);
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.stack ? e.stack : e) });
  }
};
