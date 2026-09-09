const $ = (id) => document.getElementById(id);
const out = $('log');
const append = (s) => { out.textContent += s + (s.endsWith('\n') ? '' : '\n'); };
$('run').onclick = () => {
  out.textContent = '';
  $('download').hidden = true;
  const texFlow = $('flow').files[0], shapeDec = $('shape').files[0], texDec = $('tex').files[0];
  const fixture = [...$('fixture').files], concat = $('concat').files[0] || null;
  if (!texFlow || !shapeDec || !texDec || !fixture.length) { append('Select 3 GGUFs and fixture files.'); return; }
  const w = new Worker('worker.js');
  w.onmessage = (ev) => {
    if (ev.data.type === 'log') append(ev.data.text);
    else if (ev.data.type === 'error') { append('ERROR: ' + ev.data.text); w.terminate(); }
    else if (ev.data.type === 'done') {
      append(ev.data.report);
      const blob = new Blob([ev.data.glb], {type:'model/gltf-binary'});
      const a = $('download'); a.href = URL.createObjectURL(blob); a.download = 'partial_e2e.glb'; a.hidden = false; a.textContent = 'Download GLB (' + blob.size + ' bytes)';
      w.terminate();
    }
  };
  w.postMessage({ texFlow, shapeDec, texDec, fixture, concat, resolution: Number($('res').value) || 1024 });
};
