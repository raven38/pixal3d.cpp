// transforms.json が無いときに仮定する標準リグ。公式 cyclops サンプルの transforms.json と
// 同一の行列（正面→右→背面→左、仰角 0、距離 3.119、camera_angle_x 20°）。自前で回転を組まず
// 検証済みの値をそのまま使う。Pixal3D はカメラ対応の投影で条件付けするので、実際の撮影姿勢が
// これから離れるほど劣化する（turntable / レンダ向け）。MV 重みは 4 視点前提で、視点数を
// 減らすと学習分布外で破綻する実測（4 視点が 85勝0敗 / 20勝0敗）があるため 4 枚に限定する。
const CANONICAL_RIG = {
  camera_angle_x: 0.3490658503988659,
  poses: [
    { name: 'front (azim 0)',   transform_matrix: [[1,0,0,0],[0,0,-1,-3.1192049980163574],[0,1,0,0],[0,0,0,1]] },
    { name: 'right (azim 90)',  transform_matrix: [[0,0,1,3.1192049980163574],[1,0,0,0],[0,1,0,0],[0,0,0,1]] },
    { name: 'back (azim 180)',  transform_matrix: [[-1,0,0,0],[0,0,1,3.1192049980163574],[0,1,0,0],[0,0,0,1]] },
    { name: 'left (azim 270)',  transform_matrix: [[0,0,-1,-3.1192049980163574],[-1,0,0,0],[0,1,0,0],[0,0,0,1]] },
  ],
};
export const CANONICAL_RIG_VIEWS = CANONICAL_RIG.poses.length;

export function initMeshScaleCalibration({ input, mount }) {
  let files = [], meta = null, initialScale = 0.2, scale = 0.2, objectUrls = [], views = [], dragging = -1;
  let synthetic = false; // transforms.json を持たず標準リグを合成しているか
  let scaleConfirmed = false; // 合成モードでユーザーが mesh_scale を明示確定したか
  const q = s => mount.querySelector(s);
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const base=p=>String(p||'').replace(/\\/g,'/').split('/').pop();
  mount.innerHTML = `<section class="cal"><div class="cal-head"><b>Multiview input & mesh scale</b><span id="cal-badge">select files</span></div><div id="cal-drop" class="cal-drop" tabindex="0">Drop RGBA views + transforms.json here<br><small>or click to choose files · 4 turntable views (front/right/back/left) work without transforms.json</small></div><div id="cal-summary" class="cal-msg"></div><div id="cal-grid" class="cal-grid"></div><div class="cal-msg">Drag cards to reorder. The patched transforms.json uses the same frame order.</div><div class="cal-controls"><input id="cal-range" type="range" min="0.02" max="2" step="0.001" value="0.2"><input id="cal-num" type="number" min="0.001" max="4" step="0.001" value="0.200"></div><div id="cal-msg" class="cal-msg">Blue guides are linked across views.</div><button id="cal-confirm" hidden>Confirm mesh_scale</button><button id="cal-save">Download patched transforms.json</button></section>`;
  const confirmBtn=q('#cal-confirm');
  const range=q('#cal-range'), num=q('#cal-num'), grid=q('#cal-grid'), badge=q('#cal-badge'), msg=q('#cal-msg'), drop=q('#cal-drop'), summary=q('#cal-summary');
  function cleanup(){objectUrls.forEach(URL.revokeObjectURL);objectUrls=[];views=[];}
  function orderedMeta(){
    if(synthetic){
      // i 番目のカードに i 番目の標準姿勢を割り当てる（並べ替えで姿勢は入れ替わらない）
      return {camera_angle_x:CANONICAL_RIG.camera_angle_x,mesh_scale:scale,
        frames:views.map((v,i)=>({file_path:v.file.name,name:CANONICAL_RIG.poses[i].name,transform_matrix:CANONICAL_RIG.poses[i].transform_matrix}))};
    }
    return {...meta,mesh_scale:scale,frames:views.map(v=>v.frame)};
  }
  function renderCards(){grid.innerHTML='';views.slice(0,8).forEach((v,i)=>{const c=document.createElement('div');c.className='cal-view';c.draggable=true;c.dataset.index=String(i);c.innerHTML=`<img><div class="guide"></div><span></span><b class="ord">${i+1}</b>`;c.querySelector('img').src=v.url;c.querySelector('span').textContent=synthetic?CANONICAL_RIG.poses[i].name:v.frame.file_path;c.title=v.file.name;c.ondragstart=()=>{dragging=i;c.classList.add('dragging');};c.ondragend=()=>{dragging=-1;c.classList.remove('dragging');};c.ondragover=e=>e.preventDefault();c.ondrop=e=>{e.preventDefault();const t=+c.dataset.index;if(dragging<0||t===dragging)return;const [m]=views.splice(dragging,1);views.splice(t,0,m);dragging=-1;renderCards();updateGuide();};grid.appendChild(c);});}
  function updateGuide(){if(synthetic)scaleConfirmed=false;scale=clamp(+num.value||initialScale,0.001,4);num.value=scale.toFixed(3);range.value=String(clamp(scale,0.02,2));const ratio=clamp(initialScale/scale,.22,2.2);grid.querySelectorAll('.guide').forEach(g=>{g.style.width=`${clamp(72*ratio,18,96)}%`;g.style.height=`${clamp(72*ratio,18,96)}%`;});if(synthetic&&!scaleConfirmed){badge.textContent='confirm mesh_scale';badge.className='warn';}else{badge.textContent=`mesh_scale ${scale.toFixed(3)}`;badge.className=scale<.02||scale>2?'warn':'ok';}if(confirmBtn)confirmBtn.hidden=!synthetic;if(confirmBtn)confirmBtn.disabled=scaleConfirmed;}
  range.oninput=()=>{num.value=(+range.value).toFixed(3);updateGuide();};num.oninput=updateGuide;
  confirmBtn.onclick=()=>{scaleConfirmed=true;confirmBtn.disabled=true;badge.textContent=`mesh_scale ${scale.toFixed(3)}`;badge.className=scale<.02||scale>2?'warn':'ok';msg.textContent=`mesh_scale ${scale.toFixed(3)} confirmed. Canonical rig, ${views.length} views.`;document.dispatchEvent(new CustomEvent('pixal3d-calibration-change'));};
  async function ingest(list){cleanup();files=[...list];const tf=files.find(f=>f.name==='transforms.json'||f.name.toLowerCase().endsWith('.json'));synthetic=false;const imgsOnly=files.filter(f=>f.type.startsWith('image/')||/\.(png|jpe?g|webp)$/i.test(f.name)).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));if(!tf){  if(imgsOnly.length!==CANONICAL_RIG_VIEWS){meta=null;grid.innerHTML='';badge.textContent='transforms.json missing';badge.className='warn';    summary.textContent=`${imgsOnly.length} image(s) without transforms.json`;    msg.textContent=`Without transforms.json exactly ${CANONICAL_RIG_VIEWS} views are required (front, right, back, left on a turntable). The multiview model needs all four; fewer views break down. Provide transforms.json for other camera setups.`;    return;}  synthetic=true;scaleConfirmed=false;initialScale=1;scale=initialScale;num.value=scale.toFixed(3);range.value=String(clamp(scale,.02,2));  imgsOnly.forEach((f,i)=>{const url=URL.createObjectURL(f);objectUrls.push(url);views.push({file:f,frame:{file_path:f.name,name:CANONICAL_RIG.poses[i].name,transform_matrix:CANONICAL_RIG.poses[i].transform_matrix},url});});  meta=orderedMeta();  summary.textContent=`${views.length} views · canonical rig (no transforms.json)`;  msg.textContent='No transforms.json: assuming a turntable rig — card 1 = front, 2 = right, 3 = back, 4 = left, elevation 0, FOV 20°. Reorder cards to match, then CONFIRM mesh_scale below (no default is assumed; the release path requires an explicit positive value).';badge.textContent='confirm mesh_scale';badge.className='warn';  renderCards();updateGuide();return;}try{meta=JSON.parse(await tf.text());}catch{meta=null;badge.textContent='invalid transforms.json';badge.className='warn';return;}if(!Array.isArray(meta.frames)||!meta.frames.length){meta=null;badge.textContent='no frames in transforms.json';badge.className='warn';return;}initialScale=Number.isFinite(meta.mesh_scale)&&meta.mesh_scale>0?meta.mesh_scale:.2;scale=initialScale;num.value=scale.toFixed(3);range.value=String(clamp(scale,.02,2));const by=new Map();files.filter(f=>f.type.startsWith('image/')||/\.(png|jpe?g|webp)$/i.test(f.name)).forEach(f=>{by.set(f.name,f);const rel=f.webkitRelativePath?.split('/').slice(1).join('/');if(rel)by.set(rel,f);});for(const fr of meta.frames){const key=String(fr.file_path||'').replace(/^\.\//,'');const f=by.get(fr.file_path)||by.get(key)||by.get(base(key));if(!f)continue;const url=URL.createObjectURL(f);objectUrls.push(url);views.push({file:f,frame:fr,url});}summary.textContent=`${views.length}/${meta.frames.length} views matched`;msg.textContent=views.length?`Loaded ${views.length} views. Adjust mesh_scale if needed, then run.`:'No images matched frames[].file_path';renderCards();updateGuide();}
  input.addEventListener('change',()=>ingest([...input.files]));
  drop.onclick=()=>input.click();drop.onkeydown=e=>{if(e.key==='Enter')input.click();};['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag');}));['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag');}));drop.addEventListener('drop',e=>ingest([...e.dataTransfer.files]));
  q('#cal-save').onclick=()=>{if(!meta)return;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(orderedMeta(),null,2)],{type:'application/json'}));a.download='transforms.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
  return { getPatchedFiles(){if(!meta)return files;const patched=new File([JSON.stringify(orderedMeta(),null,2)],'transforms.json',{type:'application/json'});const imgs=views.map(v=>v.file);return [patched,...imgs];}, getScale(){return scale;}, getViewCount(){return views.length;}, isReady(){return !!meta&&views.length>0&&(!synthetic||scaleConfirmed);}, isSynthetic(){return synthetic;}, isScaleConfirmed(){return scaleConfirmed;} };
}
