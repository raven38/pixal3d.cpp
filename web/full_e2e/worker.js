importScripts('./pixal3d_full_e2e.js');
self.onmessage = async (ev) => {
  const { models, fixture, seed } = ev.data;
  const log = (text) => self.postMessage({type:'log', text});
  try {
    const Module = await createPixal3dFullE2E({locateFile:p=>new URL('./'+p,self.location.href).href, print:log, printErr:t=>log('[stderr] '+t)});
    const FS=Module.FS, WORKERFS=Module.WORKERFS;
    FS.mkdir('/models'); FS.mount(WORKERFS,{files:models},'/models');
    FS.mkdir('/fixture'); FS.mount(WORKERFS,{files:fixture},'/fixture');
    FS.mkdir('/out');
    const by = Object.fromEntries(models.map(f=>[f.name,f]));
    const need = ['dinov3.gguf','pixal3d_naf.gguf','pixal3d_ss_flow_mv.gguf','ss_dec.gguf','pixal3d_shape_flow_512_mv.gguf','shape_dec.gguf','pixal3d_shape_flow_1024_mv.gguf','pixal3d_tex_flow_1024_mv.gguf','tex_dec.gguf'];
    for (const n of need) if(!by[n]) throw new Error('missing model file: '+n);
    const args = need.map(n=>'/models/'+n).concat(['/fixture','/out/full_fixture.glb',seed||1]);
    const report = await Module.ccall('pixal3d_full_fixture_run','string',Array(11).fill('string').concat(['number']).slice(0,12),args,{async:true});
    const glb = FS.readFile('/out/full_fixture.glb');
    self.postMessage({type:'done',report,glb:glb.buffer},[glb.buffer]);
  } catch(e) { self.postMessage({type:'error',text:String(e&&e.stack?e.stack:e)}); }
};
