// Page side of the browser texture-decode harness: collect File objects, spawn the worker, print
// the report, offer the returned attribute files for download. No pipeline logic here.
(() => {
  const out = document.getElementById('out');
  const lines = [];
  function log(s) { lines.push(s); out.textContent = lines.join('\n'); }

  window.pixal3dResult = null;   // { report, files: {name: ArrayBuffer} }

  document.getElementById('run').addEventListener('click', () => {
    const shapeModel = document.getElementById('shape_model').files[0];
    const texModel = document.getElementById('tex_model').files[0];
    const fixture = Array.from(document.getElementById('fixture').files);
    const res = parseInt(document.getElementById('res').value, 10) || 0;
    if (!shapeModel || !texModel || fixture.length === 0) { log('need shape_dec.gguf, tex_dec.gguf and the fixture files'); document.title = 'RESULT: FAIL'; return; }
    log('navigator.gpu present (page): ' + (!!navigator.gpu));
    log('starting worker with ' + shapeModel.name + ', ' + texModel.name + ' and ' + fixture.length + ' fixture files, res=' + res);
    const worker = new Worker('worker.js');
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'log') { log(m.text); return; }
      if (m.type === 'done') {
        log(m.report);
        window.pixal3dResult = { report: m.report, files: m.files };
        for (const name of Object.keys(m.files)) {   // browser_attrs.npy etc.; the playwright driver saves these
          const a = document.createElement('a');
          a.href = URL.createObjectURL(new Blob([m.files[name]], { type: 'application/octet-stream' }));
          a.download = name; a.textContent = 'download ' + name; document.body.appendChild(a);
          a.click();
        }
        document.title = m.report.includes('RESULT: OK') ? 'RESULT: OK' : 'RESULT: FAIL';
        return;
      }
      if (m.type === 'error') { log('worker error: ' + m.text); document.title = 'RESULT: FAIL'; }
    };
    worker.onerror = (e) => { log('worker crashed: ' + e.message); document.title = 'RESULT: FAIL'; };
    worker.postMessage({ shapeModel, texModel, fixture, res });
  });
})();
