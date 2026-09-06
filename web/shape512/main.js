// Page side of the browser Shape-512 harness: collect File objects, spawn the worker, print the
// report. No pipeline logic here (docs/spec/31-webgpu-bringup.md §10).
(() => {
  const out = document.getElementById('out');
  const lines = [];
  function log(s) { lines.push(s); out.textContent = lines.join('\n'); }

  window.pixal3dResult = null;   // { report, latent: Float32Array [N*32], steps: Float32Array, nSteps }

  document.getElementById('run').addEventListener('click', () => {
    const models = Array.from(document.getElementById('models').files);
    const cond = Array.from(document.getElementById('cond').files);
    const sample = Array.from(document.getElementById('sample').files);
    const nViews = parseInt(document.getElementById('views').value, 10) || 0;
    const ownCond = document.getElementById('own').checked ? 1 : 0;
    const dinov3 = models.find(f => /dinov3/i.test(f.name));
    const naf = models.find(f => /naf/i.test(f.name));
    const flow = models.find(f => /shape_flow_512/i.test(f.name));
    if (!dinov3 || !naf || !flow) { log('need dinov3*.gguf, *naf*.gguf and *shape_flow_512*.gguf'); document.title = 'RESULT: FAIL'; return; }
    log('navigator.gpu present (page): ' + (!!navigator.gpu));
    log('starting worker with ' + models.length + ' model files, ' + cond.length + ' cond files, ' + sample.length + ' sample files, views=' + nViews + ' own_cond=' + ownCond);
    const worker = new Worker('worker.js');
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'log') { log(m.text); return; }
      if (m.type === 'done') {
        log(m.report);
        window.pixal3dResult = { report: m.report, latent: m.latent, steps: m.steps, nSteps: m.nSteps };
        document.title = m.report.includes('RESULT: OK') ? 'RESULT: OK' : 'RESULT: FAIL';
        return;
      }
      if (m.type === 'error') { log('worker error: ' + m.text); document.title = 'RESULT: FAIL'; }
    };
    worker.onerror = (e) => { log('worker crashed: ' + e.message); document.title = 'RESULT: FAIL'; };
    worker.postMessage({ dinov3, naf, flow, cond, sample, nViews, ownCond });
  });
})();
