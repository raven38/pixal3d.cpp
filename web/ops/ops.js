// Thin binding layer for the browser build of src/test_webgpu_ops.cpp: load the WASM module,
// call webgpu_ops_run(args) (JSPI, suspends across the backend's async WebGPU waits) and mirror
// its stdout/stderr into the page. The test logic itself is the shared C++ file.
(async () => {
  const out = document.getElementById('out');
  const lines = [];
  function log(s) {
    lines.push(s);
    out.textContent = lines.join('\n');
  }
  const args = new URLSearchParams(location.search).get('args') || '--budget-mb 1400';
  log('args: ' + args);
  log('navigator.gpu present: ' + (!!navigator.gpu));

  let Module;
  try {
    Module = await createPixal3dOps({ print: log, printErr: log });
  } catch (e) {
    log('Module load failed: ' + e);
    document.title = 'RESULT: FAIL';
    return;
  }
  log('Module loaded; calling webgpu_ops_run() ...');
  let rc = 1;
  try {
    rc = await Module.ccall('webgpu_ops_run', 'number', ['string'], [args], { async: true });
  } catch (e) {
    log('webgpu_ops_run() threw: ' + e);
  }
  log('webgpu_ops_run() returned ' + rc);
  window.pixal3dOpsText = lines.join('\n');
  document.title = rc === 0 ? 'RESULT: PASS' : 'RESULT: FAIL';
})();
