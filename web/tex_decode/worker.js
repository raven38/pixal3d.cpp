// Worker side of the browser texture-decode harness: mount the File objects with WORKERFS (lazy
// reads via FileReaderSync, no copy into the wasm heap), call the single C entry point (JSPI
// async), read the attribute files it wrote to MEMFS and hand report + files back. Same module
// as web/ss, web/shape512 and web/shape_decode (built by scripts/build_wasm_ss.sh).
importScripts('../ss/pixal3d_ss.js');

self.onmessage = async (ev) => {
  const { shapeModel, texModel, fixture, res } = ev.data;
  const post = (text) => self.postMessage({ type: 'log', text });
  try {
    post('worker: navigator.gpu present: ' + (!!self.navigator.gpu));
    const Module = await createPixal3dSS({
      locateFile: (p) => new URL('../ss/' + p, self.location.href).href,
      print: (t) => post(t),
      printErr: (t) => post('[stderr] ' + t),
    });
    const FS = Module.FS, WORKERFS = Module.WORKERFS;
    FS.mkdir('/models'); FS.mount(WORKERFS, { files: [shapeModel, texModel] }, '/models');
    FS.mkdir('/fixture'); FS.mount(WORKERFS, { files: fixture }, '/fixture');
    FS.mkdir('/out');
    post('worker: mounted /models/' + shapeModel.name + ' (' + shapeModel.size + ' B), /models/' + texModel.name +
         ' (' + texModel.size + ' B), /fixture (' + fixture.length + ' files)');
    const t0 = performance.now();
    const report = await Module.ccall('pixal3d_tex_decode_run', 'string',
      ['string', 'string', 'string', 'number', 'string'],
      ['/models/' + shapeModel.name, '/models/' + texModel.name, '/fixture', res, '/out'], { async: true });
    const wall = (performance.now() - t0) / 1000;
    const files = {};
    const transfer = [];
    for (const name of ['attrs', 'coords']) {
      try {
        const bytes = FS.readFile('/out/' + name + '.npy');   // Uint8Array (MEMFS)
        files['browser_' + name + '.npy'] = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        transfer.push(files['browser_' + name + '.npy']);
      } catch (e) { post('worker: no /out/' + name + '.npy (' + e + ')'); }
    }
    self.postMessage({ type: 'done', report: report + 'browser wall ' + wall.toFixed(1) + ' s\n', files }, transfer);
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.stack ? e.stack : e) });
  }
};
