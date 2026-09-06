// Page side of the browser Texture-1024 harness: collect File objects, spawn the worker, print the
// report. No pipeline logic here (docs/spec/32-texture-flow-webgpu-prep.md §5). Same shape as
// web/shape512/main.js.
(() => {
  const out = document.getElementById('out');
  const lines = [];
  function log(s) { lines.push(s); out.textContent = lines.join('\n'); }

  window.pixal3dResult = null;   // { report, latent: Float32Array [N*32], steps: Float32Array, nSteps }

  document.getElementById('run').addEventListener('click', () => {
    const models = Array.from(document.getElementById('models').files);
    const sample = Array.from(document.getElementById('sample').files);
    const cond = Array.from(document.getElementById('cond').files);
    const concatFiles = Array.from(document.getElementById('concat').files);
    const concat = concatFiles.length ? concatFiles[0] : null;
    const nViews = parseInt(document.getElementById('views').value, 10) || 0;
    const ownCond = document.getElementById('own').checked ? 1 : 0;
    const flow = models.find(f => /tex_flow/i.test(f.name));
    const dinov3 = models.find(f => /dinov3/i.test(f.name)) || null;
    const naf = models.find(f => /naf/i.test(f.name)) || null;
    if (!flow) { log('need *tex_flow*.gguf'); document.title = 'RESULT: FAIL'; return; }
    if (ownCond && (!dinov3 || !naf || !cond.length)) { log('own cond needs dinov3*.gguf, *naf*.gguf and the cond_slat files'); document.title = 'RESULT: FAIL'; return; }
    log('navigator.gpu present (page): ' + (!!navigator.gpu));
    log('starting worker with ' + models.length + ' model files, ' + sample.length + ' sample files, ' + cond.length + ' cond files, concat=' +
        (concat ? concat.name : '(fixture)') + ', views=' + nViews + ' own_cond=' + ownCond);
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
    worker.postMessage({ flow, dinov3, naf, sample, cond, concat, nViews, ownCond });
  });
})();
