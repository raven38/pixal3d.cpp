import { generateMultiview, type MultiviewFile } from "./api";
import { listen } from "./tauri";

interface TransformFrame {
  file_path: string;
  camera_angle_x?: number;
  transform_matrix: number[][];
}
interface TransformMeta {
  camera_angle_x?: number;
  mesh_scale?: number;
  frames: TransformFrame[];
  [key: string]: unknown;
}

type ViewEntry = { file: File; frame: TransformFrame; url: string };

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const basename = (p: string) => p.replace(/\\/g, "/").split("/").pop() || p;

function injectStyle(): void {
  if (document.getElementById("mv-calibration-style")) return;
  const s = document.createElement("style");
  s.id = "mv-calibration-style";
  s.textContent = `
    .mvcal{margin-top:14px;border:1px solid #30343b;border-radius:12px;padding:12px;background:#15181d}
    .mvcal-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}
    .mvcal-title{font-weight:650}.mvcal-sub{font-size:12px;opacity:.7;margin:3px 0 10px}
    .mvcal-drop{border:1px dashed #4a5260;border-radius:10px;padding:14px;text-align:center;cursor:pointer;background:#111419}
    .mvcal-drop.drag{border-color:#50d2ff;background:#122029}.mvcal-files{font-size:11px;opacity:.75;margin-top:6px}
    .mvcal-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:10px 0}
    .mvcal-view{position:relative;aspect-ratio:1;background:#0e1014;border-radius:8px;overflow:hidden;border:1px solid #282c33;cursor:grab}
    .mvcal-view.dragging{opacity:.45}.mvcal-view img{width:100%;height:100%;object-fit:contain}.mvcal-guide{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:2px solid rgba(80,210,255,.9);box-shadow:0 0 0 9999px rgba(0,0,0,.08);pointer-events:none}
    .mvcal-name{position:absolute;left:6px;bottom:5px;background:#0009;padding:2px 5px;border-radius:4px;font-size:10px}.mvcal-order{position:absolute;right:6px;top:5px;background:#0009;padding:2px 5px;border-radius:4px;font-size:10px}
    .mvcal-row{display:grid;grid-template-columns:1fr 92px;gap:8px;align-items:center}.mvcal-row input[type=range]{width:100%}.mvcal-row input[type=number]{width:100%;box-sizing:border-box}
    .mvcal-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.mvcal-controls label{font-size:12px}.mvcal-controls select,.mvcal-controls input{width:100%;box-sizing:border-box;margin-top:4px}
    .mvcal-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}.mvcal-hint{font-size:11px;opacity:.7;margin-top:7px;line-height:1.35}
    .mvcal-status{font-size:12px;margin-top:8px}.mvcal-good{color:#75d49b}.mvcal-warn{color:#e6b85c}
    .mvcal-progress{margin-top:10px;padding:9px;border-radius:8px;background:#0f1217;border:1px solid #282c33}.mvcal-progress[hidden]{display:none}.mvcal-stage{font-size:12px}.mvcal-elapsed{font-size:11px;opacity:.7;margin-top:3px}
  `;
  document.head.appendChild(s);
}

async function alphaBounds(file: File): Promise<{ w: number; h: number } | null> {
  const bmp = await createImageBitmap(file).catch(() => null);
  if (!bmp) return null;
  const c = document.createElement("canvas"); c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true }); if (!ctx) return null;
  ctx.drawImage(bmp, 0, 0); const d = ctx.getImageData(0, 0, c.width, c.height).data;
  let xmin=c.width,ymin=c.height,xmax=-1,ymax=-1;
  for(let y=0;y<c.height;y++) for(let x=0;x<c.width;x++) { if(d[(y*c.width+x)*4+3] <= 16) continue; xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y); }
  bmp.close();
  if(xmax<xmin||ymax<ymin) return null;
  return { w:(xmax-xmin+1)/c.width, h:(ymax-ymin+1)/c.height };
}

export function mountMvCalibration(root: HTMLElement): void {
  injectStyle();
  root.innerHTML = `
    <div class="mvcal">
      <div class="mvcal-head"><div><div class="mvcal-title">Pixal3D multiview</div><div class="mvcal-sub">Drop views + transforms.json, reorder them, calibrate mesh scale, then generate.</div></div></div>
      <div id="mvcal-drop" class="mvcal-drop" tabindex="0">Drop RGBA views + transforms.json here<br><span class="muted">or click to choose files</span><div id="mvcal-files-summary" class="mvcal-files"></div></div>
      <input id="mvcal-files" type="file" multiple accept="image/*,.json,application/json" hidden>
      <div id="mvcal-body" hidden>
        <div id="mvcal-grid" class="mvcal-grid"></div>
        <div class="mvcal-hint">Drag view cards to reorder. The patched transforms.json will use this frame order.</div>
        <label>Mesh scale</label>
        <div class="mvcal-row"><input id="mvcal-range" type="range" min="0.02" max="2" step="0.001"><input id="mvcal-num" type="number" min="0.001" max="4" step="0.001"></div>
        <div id="mvcal-status" class="mvcal-status"></div>
        <div class="mvcal-controls">
          <label>Resolution<select id="mvcal-res"><option value="1024" selected>1024 · cascade</option><option value="1536">1536 · high</option></select></label>
          <label>Seed<input id="mvcal-seed" type="number" min="0" step="1" value="42"></label>
        </div>
        <div class="mvcal-hint">Blue guides change with scale across all views. Extreme values or inconsistent framing are flagged before generation.</div>
        <div class="mvcal-actions"><button id="mvcal-download" class="tool-btn">Save calibrated transforms.json</button><button id="mvcal-generate" class="primary">Generate MV 3D</button><button id="mvcal-cancel" class="link-btn" disabled>Cancel</button></div>
        <div id="mvcal-progress" class="mvcal-progress" hidden><div id="mvcal-stage" class="mvcal-stage">starting…</div><div id="mvcal-elapsed" class="mvcal-elapsed">0:00</div></div>
      </div>
    </div>`;

  const picker = root.querySelector<HTMLInputElement>("#mvcal-files")!;
  const drop = root.querySelector<HTMLElement>("#mvcal-drop")!;
  const summary = root.querySelector<HTMLElement>("#mvcal-files-summary")!;
  const body = root.querySelector<HTMLElement>("#mvcal-body")!;
  const grid = root.querySelector<HTMLElement>("#mvcal-grid")!;
  const range = root.querySelector<HTMLInputElement>("#mvcal-range")!;
  const num = root.querySelector<HTMLInputElement>("#mvcal-num")!;
  const status = root.querySelector<HTMLElement>("#mvcal-status")!;
  const gen = root.querySelector<HTMLButtonElement>("#mvcal-generate")!;
  const cancel = root.querySelector<HTMLButtonElement>("#mvcal-cancel")!;
  const res = root.querySelector<HTMLSelectElement>("#mvcal-res")!;
  const seed = root.querySelector<HTMLInputElement>("#mvcal-seed")!;
  const progress = root.querySelector<HTMLElement>("#mvcal-progress")!;
  const stage = root.querySelector<HTMLElement>("#mvcal-stage")!;
  const elapsed = root.querySelector<HTMLElement>("#mvcal-elapsed")!;

  let meta: TransformMeta | null = null;
  let views: ViewEntry[] = [];
  let imageFiles: File[] = [];
  let initialScale = 0.2;
  let dragging = -1;
  let activeAbort: AbortController | null = null;
  let timer: number | null = null;
  let generating = false;

  void listen<string>("server-log", (line) => {
    if (generating && String(line).trim()) stage.textContent = String(line).trim();
  });

  const cleanupViews = () => { views.forEach(v => URL.revokeObjectURL(v.url)); views = []; };
  const currentScale = () => clamp(Number(num.value) || initialScale, 0.001, 4);
  const orderedMeta = () => ({ ...meta!, mesh_scale: currentScale(), frames: views.map(v => v.frame) });
  const patchBlob = () => new Blob([JSON.stringify(orderedMeta(), null, 2)], { type:"application/json" });
  const fmt = (ms:number) => { const s=Math.floor(ms/1000); return `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`; };

  const renderCards = () => {
    grid.innerHTML = "";
    views.slice(0, 8).forEach((v, i) => {
      const card = document.createElement("div"); card.className="mvcal-view"; card.draggable=true; card.dataset.index=String(i);
      card.innerHTML=`<img src="${v.url}" alt="view ${i}"><div class="mvcal-guide"></div><div class="mvcal-name"></div><div class="mvcal-order">${i+1}</div>`;
      (card.querySelector(".mvcal-name") as HTMLElement).textContent=v.frame.file_path;
      card.addEventListener("dragstart",()=>{dragging=i;card.classList.add("dragging");});
      card.addEventListener("dragend",()=>{dragging=-1;card.classList.remove("dragging");});
      card.addEventListener("dragover",e=>e.preventDefault());
      card.addEventListener("drop",e=>{e.preventDefault(); const target=Number(card.dataset.index); if(dragging<0||dragging===target)return; const [m]=views.splice(dragging,1); views.splice(target,0,m); dragging=-1; renderCards(); void renderGuide();});
      grid.appendChild(card);
    });
  };

  const renderGuide = async () => {
    const s = currentScale(); num.value=s.toFixed(3); range.value=String(clamp(s,0.02,2));
    const ratio = clamp(initialScale / s, 0.22, 2.2);
    [...grid.querySelectorAll<HTMLElement>(".mvcal-guide")].forEach(g => { g.style.width=`${clamp(72*ratio,18,96)}%`; g.style.height=`${clamp(72*ratio,18,96)}%`; });
    const bounds = await Promise.all(views.slice(0,4).map(v => alphaBounds(v.file)));
    const cover = bounds.filter(Boolean).map(b => Math.max(b!.w,b!.h));
    const spread = cover.length ? Math.max(...cover)-Math.min(...cover) : 0;
    status.className = "mvcal-status " + (s<0.02||s>2||spread>0.35 ? "mvcal-warn":"mvcal-good");
    status.textContent = `scale ${s.toFixed(3)} · ${views.length} views${spread>0.35 ? " · view framing differs strongly; check cameras/FOV" : " · framing consistency looks usable"}`;
  };

  const reconcile = async () => {
    cleanupViews();
    if(!meta){ body.hidden=true; summary.textContent=`${imageFiles.length} image(s), transforms.json missing`; return; }
    const byName=new Map<string,File>(); imageFiles.forEach(f=>{byName.set(f.name,f); const rel=f.webkitRelativePath?.split("/").slice(1).join("/"); if(rel)byName.set(rel,f);});
    for(const fr of meta.frames){ const f=byName.get(fr.file_path)||byName.get(fr.file_path.replace(/^\.\//,""))||byName.get(basename(fr.file_path)); if(f)views.push({file:f,frame:fr,url:URL.createObjectURL(f)}); }
    summary.textContent=`${imageFiles.length} image(s) · ${views.length}/${meta.frames.length} matched`;
    if(!views.length){ body.hidden=true; status.textContent="No frame images matched transforms.json"; return; }
    initialScale=typeof meta.mesh_scale==="number"&&isFinite(meta.mesh_scale)&&meta.mesh_scale>0?meta.mesh_scale:0.2;
    num.value=initialScale.toFixed(3); range.value=String(clamp(initialScale,0.02,2));
    body.hidden=false; renderCards(); await renderGuide();
  };

  const ingest = async (list: File[]) => {
    const tf=list.find(f=>f.name==="transforms.json"||f.name.toLowerCase().endsWith(".json"));
    if(tf){ try{meta=JSON.parse(await tf.text()) as TransformMeta;}catch{status.textContent="Invalid transforms.json";return;} if(!Array.isArray(meta.frames)||!meta.frames.length){status.textContent="transforms.json has no frames";return;} }
    const imgs=list.filter(f=>f.type.startsWith("image/")||/\.(png|jpe?g|webp)$/i.test(f.name));
    if(imgs.length) imageFiles=imgs;
    await reconcile();
  };

  drop.addEventListener("click",()=>picker.click());
  drop.addEventListener("keydown",e=>{if((e as KeyboardEvent).key==="Enter")picker.click();});
  ["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("drag");}));
  ["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("drag");}));
  drop.addEventListener("drop",e=>void ingest([...(e as DragEvent).dataTransfer?.files??[]]));
  picker.addEventListener("change",()=>void ingest([...(picker.files??[])]));
  range.addEventListener("input",()=>{num.value=Number(range.value).toFixed(3); void renderGuide();});
  num.addEventListener("input",()=>void renderGuide());
  root.querySelector("#mvcal-download")!.addEventListener("click",()=>{ if(!meta)return; const a=document.createElement("a");a.href=URL.createObjectURL(patchBlob());a.download="transforms.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000); });
  cancel.addEventListener("click",()=>activeAbort?.abort());
  gen.addEventListener("click", async()=>{
    if(!meta||!views.length||generating)return;
    generating=true; activeAbort=new AbortController(); gen.disabled=true; cancel.disabled=false; progress.hidden=false; stage.textContent="starting…";
    const started=Date.now(); elapsed.textContent="0:00"; timer=window.setInterval(()=>elapsed.textContent=fmt(Date.now()-started),1000);
    try{
      const mv:MultiviewFile[]=views.map(v=>({name:v.frame.file_path,blob:v.file}));
      const resolution=Number(res.value)===1536?1536:1024;
      const seedValue=Math.max(0,Number(seed.value)||42);
      const {glb}=await generateMultiview(patchBlob(),mv,{seed:seedValue,resolution,uv:"xatlas",numViews:views.length},activeAbort.signal);
      stage.textContent="complete";
      window.dispatchEvent(new CustomEvent("pixal3d-mv-result",{detail:{glb,name:"multiview",meshScale:currentScale(),resolution,seed:seedValue}}));
    }catch(e){ if(activeAbort.signal.aborted) stage.textContent="cancelled"; else {stage.textContent="failed"; window.dispatchEvent(new CustomEvent("pixal3d-mv-error",{detail:String((e as Error).message||e)}));} }
    finally{generating=false;activeAbort=null;gen.disabled=false;cancel.disabled=true;if(timer)window.clearInterval(timer);timer=null;}
  });
}
