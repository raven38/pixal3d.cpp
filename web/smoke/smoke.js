// Thin binding layer: load the WASM module, call the two exported C functions, print their
// text output. No pipeline logic here -- see src/test_webgpu_smoke.cpp for everything that
// actually runs on the ggml WebGPU backend (docs/spec/31-webgpu-bringup.md §8).
(async () => {
  const out = document.getElementById('out');
  const lines = [];
  function log(s) {
    lines.push(s);
    out.textContent = lines.join('\n');
  }

  log('navigator.gpu present: ' + (!!navigator.gpu));

  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const limitsObj = {};
        for (const k in adapter.limits) limitsObj[k] = adapter.limits[k];
        log('JS-side navigator.gpu.requestAdapter() limits (cross-check only; the WASM module');
        log('requests its own separate adapter/device -- see webgpu_smoke_limits() below):');
        log(JSON.stringify(limitsObj, null, 2));
        if (adapter.features) {
          log('JS-side adapter.features: ' + Array.from(adapter.features).join(', '));
        }
      } else {
        log('JS-side navigator.gpu.requestAdapter() returned null.');
      }
    } catch (e) {
      log('JS-side navigator.gpu.requestAdapter() threw: ' + e);
    }
  }

  log('');
  log('Loading WASM module (pixal3d_smoke.js/.wasm)...');
  let Module;
  try {
    Module = await createPixal3dSmoke();
  } catch (e) {
    log('Module load failed: ' + e);
    document.title = 'RESULT: FAIL';
    return;
  }
  log('Module loaded.');

  log('');
  log('Calling webgpu_smoke_limits() (async, JSPI)...');
  let limitsText;
  try {
    limitsText = await Module.ccall('webgpu_smoke_limits', 'string', [], [], { async: true });
  } catch (e) {
    limitsText = 'webgpu_smoke_limits() threw: ' + e;
  }
  log(limitsText);

  log('');
  log('Calling webgpu_smoke_run() (async, JSPI)...');
  let runText;
  try {
    runText = await Module.ccall('webgpu_smoke_run', 'string', [], [], { async: true });
  } catch (e) {
    runText = 'webgpu_smoke_run() threw: ' + e;
  }
  log(runText);

  const m = runText.match(/RESULT: (PASS|FAIL)/);
  document.title = m ? m[0] : 'RESULT: FAIL';
})();
