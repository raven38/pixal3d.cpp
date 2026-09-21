import { generateTrellis2Multiview, type Trellis2MvFusion } from "./api";
import { blockedReason, canGenerate, gateState, onServerLost, setLocalGenerating, stopWaiting, subscribeGate } from "./gate";
import { listen } from "./tauri";
import type { BgRemoval, Resolution, Uv } from "./types";

export interface Trellis2MvResultDetail {
  glb: Blob;
  input: Blob;
  name: string;
  files: File[];
  numViews: number;
  fusion: Trellis2MvFusion;
  resolution: Resolution;
  seed: number;
  bgRemoval: BgRemoval;
  uv: Uv;
}

function injectStyle(): void {
  if (document.getElementById("t2mv-panel-style")) return;
  const s = document.createElement("style");
  s.id = "t2mv-panel-style";
  s.textContent = `
    .t2mvp{margin-top:14px;border:1px solid #30343b;border-radius:12px;padding:12px;background:#15181d}
    .t2mvp-title{font-weight:650}.t2mvp-sub{font-size:12px;opacity:.72;margin:3px 0 10px;line-height:1.4}
    .t2mvp-drop{border:1px dashed #4a5260;border-radius:10px;padding:13px;text-align:center;cursor:pointer;background:#111419}
    .t2mvp-drop.drag{border-color:#50d2ff;background:#122029}
    .t2mvp-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:9px}
    .t2mvp-card{display:flex;align-items:center;gap:6px;border:1px solid #30343b;border-radius:8px;padding:7px;background:#101319;min-width:0}
    .t2mvp-card img{width:48px;height:48px;object-fit:cover;border-radius:5px;background:#080a0d}
    .t2mvp-name{font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
    .t2mvp-card button{padding:3px 6px;font-size:10px}
    .t2mvp-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.t2mvp-controls label{font-size:12px}
    .t2mvp-controls input,.t2mvp-controls select{width:100%;box-sizing:border-box;margin-top:4px}
    .t2mvp-status{font-size:11px;margin-top:9px;line-height:1.4}.t2mvp-status.good{color:#75d49b}.t2mvp-status.bad{color:#e6b85c}
    .t2mvp-actions{display:flex;gap:8px;margin-top:10px}.t2mvp-progress{margin-top:10px;padding:9px;border-radius:8px;background:#0f1217;border:1px solid #282c33}
    .t2mvp-progress[hidden]{display:none}.t2mvp-stage{font-size:12px}.t2mvp-elapsed{font-size:11px;opacity:.7;margin-top:3px}
  `;
  document.head.appendChild(s);
}

export function mountTrellis2MvPanel(root: HTMLElement): void {
  injectStyle();
  root.innerHTML = `
    <div class="t2mvp">
      <div class="t2mvp-title">TRELLIS.2 multiview</div>
      <div class="t2mvp-sub">
        2–8 images of the same object. This is pose-free multi-image conditioning:
        no transforms.json, camera pose, FOV or mesh_scale is used. Image order is explicit below.
      </div>
      <div id="t2mvp-drop" class="t2mvp-drop" tabindex="0">Drop 2–8 images here<br><span class="muted">or click to choose multiple files</span></div>
      <input id="t2mvp-file" type="file" accept="image/*" multiple hidden>
      <div id="t2mvp-list" class="t2mvp-list"></div>
      <div class="t2mvp-controls">
        <label>Fusion<select id="t2mvp-fusion"><option value="stochastic" selected>stochastic · cycle one view/step</option><option value="multidiffusion">multidiffusion · average all view predictions</option></select></label>
        <label>Resolution<select id="t2mvp-res"><option value="512">512</option><option value="1024" selected>1024 cascade</option><option value="1536">1536 cascade</option></select></label>
        <label>Background removal<select id="t2mvp-bg"><option value="auto" selected>auto</option><option value="birefnet">birefnet</option><option value="threshold">threshold</option></select></label>
        <label>UV<select id="t2mvp-uv"><option value="xatlas" selected>xatlas</option><option value="box">box</option></select></label>
        <label>Seed<input id="t2mvp-seed" type="number" min="0" step="1" value="42"></label>
      </div>
      <div id="t2mvp-status" class="t2mvp-status"></div>
      <div class="t2mvp-actions"><button id="t2mvp-generate" class="primary" disabled>Generate TRELLIS.2 MV</button><button id="t2mvp-stop" class="link-btn" disabled>Stop waiting</button></div>
      <div id="t2mvp-progress" class="t2mvp-progress" hidden><div id="t2mvp-stage" class="t2mvp-stage">starting…</div><div id="t2mvp-elapsed" class="t2mvp-elapsed">0:00</div></div>
    </div>`;

  const drop = root.querySelector<HTMLElement>("#t2mvp-drop")!;
  const picker = root.querySelector<HTMLInputElement>("#t2mvp-file")!;
  const list = root.querySelector<HTMLElement>("#t2mvp-list")!;
  const fusionEl = root.querySelector<HTMLSelectElement>("#t2mvp-fusion")!;
  const resEl = root.querySelector<HTMLSelectElement>("#t2mvp-res")!;
  const bgEl = root.querySelector<HTMLSelectElement>("#t2mvp-bg")!;
  const uvEl = root.querySelector<HTMLSelectElement>("#t2mvp-uv")!;
  const seedEl = root.querySelector<HTMLInputElement>("#t2mvp-seed")!;
  const status = root.querySelector<HTMLElement>("#t2mvp-status")!;
  const gen = root.querySelector<HTMLButtonElement>("#t2mvp-generate")!;
  const stop = root.querySelector<HTMLButtonElement>("#t2mvp-stop")!;
  const progress = root.querySelector<HTMLElement>("#t2mvp-progress")!;
  const stage = root.querySelector<HTMLElement>("#t2mvp-stage")!;
  const elapsed = root.querySelector<HTMLElement>("#t2mvp-elapsed")!;

  let files: File[] = [];
  let urls: string[] = [];
  let generating = false;
  let abort: AbortController | null = null;
  let lostMsg = "";
  let timer: number | null = null;

  const cleanupUrls = () => { urls.forEach((u) => URL.revokeObjectURL(u)); urls = []; };

  const capStatus = (): { ok: boolean; text: string } => {
    const cap = gateState().capabilities;
    if (!cap) {
      return gateState().serverOnline
        ? { ok: false, text: "This runtime does not advertise TRELLIS.2 multiview; update trellis-server." }
        : { ok: false, text: "Server is offline." };
    }
    if (!cap.trellis2_mv?.available) return { ok: false, text: "TRELLIS.2 multiview is not available in this runtime." };
    return { ok: true, text: `TRELLIS.2 multiview ready · up to ${cap.trellis2_mv.max_images || 8} images` };
  };

  const render = (): void => {
    cleanupUrls();
    list.replaceChildren();
    files.forEach((file, i) => {
      const card = document.createElement("div");
      card.className = "t2mvp-card";
      const img = document.createElement("img");
      const url = URL.createObjectURL(file); urls.push(url); img.src = url;
      const name = document.createElement("div");
      name.className = "t2mvp-name";
      name.textContent = `${i}: ${file.name}`;
      const up = document.createElement("button"); up.textContent = "↑"; up.disabled = i === 0;
      const down = document.createElement("button"); down.textContent = "↓"; down.disabled = i === files.length - 1;
      const del = document.createElement("button"); del.textContent = "✕";
      up.onclick = () => { [files[i-1], files[i]] = [files[i], files[i-1]]; render(); };
      down.onclick = () => { [files[i+1], files[i]] = [files[i], files[i+1]]; render(); };
      del.onclick = () => { files.splice(i, 1); render(); };
      card.append(img, name, up, down, del);
      list.appendChild(card);
    });
    const cap = capStatus();
    const countOk = files.length >= 2 && files.length <= 8;
    const blocked = blockedReason();
    status.className = `t2mvp-status ${cap.ok && countOk ? "good" : "bad"}`;
    const count = files.length ? `${files.length} image(s)` : "choose 2–8 images";
    status.textContent = cap.ok
      ? `${count}${blocked && !generating ? ` · Generate is paused: ${blocked}` : ""}`
      : cap.text;
    gen.disabled = generating || !cap.ok || !countOk || !canGenerate();
    stop.disabled = !generating;
  };

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming).filter((f) => f.type.startsWith("image/"));
    files = [...files, ...arr];
    render();
  };

  drop.addEventListener("click", () => picker.click());
  drop.addEventListener("keydown", (e) => { if ((e as KeyboardEvent).key === "Enter") picker.click(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => { const fs = (e as DragEvent).dataTransfer?.files; if (fs) addFiles(fs); });
  picker.addEventListener("change", () => { if (picker.files) addFiles(picker.files); picker.value = ""; });
  subscribeGate(render);

  void listen<string>("server-log", (line) => {
    if (generating && String(line).trim()) stage.textContent = String(line).trim();
  });

  const fmt = (ms: number) => {
    const sec = Math.floor(ms / 1000);
    return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
  };

  stop.addEventListener("click", () => { abort?.abort(); stopWaiting(); });
  onServerLost((msg) => {
    if (!generating || !abort) return;
    lostMsg = msg;
    abort.abort();
  });

  gen.addEventListener("click", async () => {
    if (generating || files.length < 2 || files.length > 8 || !canGenerate()) return;
    const seed = Math.max(0, Math.floor(Number(seedEl.value) || 42));
    const resolution = Number(resEl.value) as Resolution;
    const fusion = (fusionEl.value === "multidiffusion" ? "multidiffusion" : "stochastic") as Trellis2MvFusion;
    const bgRemoval = (["birefnet","threshold"].includes(bgEl.value) ? bgEl.value : "auto") as BgRemoval;
    const uv = (uvEl.value === "box" ? "box" : "xatlas") as Uv;

    generating = true;
    setLocalGenerating(true);
    lostMsg = "";
    abort = new AbortController();
    progress.hidden = false;
    stage.textContent = "starting…";
    const started = Date.now();
    timer = window.setInterval(() => { elapsed.textContent = fmt(Date.now() - started); }, 1000);
    render();

    try {
      const { glb } = await generateTrellis2Multiview(files, { fusion, resolution, seed, bgRemoval, uv }, abort.signal);
      stage.textContent = "complete";
      const detail: Trellis2MvResultDetail = {
        glb,
        input: files[0],
        name: "trellis2_multiview",
        files: [...files],
        numViews: files.length,
        fusion,
        resolution,
        seed,
        bgRemoval,
        uv,
      };
      window.dispatchEvent(new CustomEvent<Trellis2MvResultDetail>("trellis2-mv-result", { detail }));
    } catch (e) {
      if (abort.signal.aborted && lostMsg) stage.textContent = lostMsg;
      else if (abort.signal.aborted) stage.textContent = "stopped waiting — the server keeps computing this generation";
      else {
        stage.textContent = "failed";
        window.dispatchEvent(new CustomEvent<string>("trellis2-mv-error", { detail: String((e as Error).message || e) }));
      }
    } finally {
      generating = false;
      setLocalGenerating(false);
      abort = null;
      if (timer != null) window.clearInterval(timer);
      timer = null;
      render();
    }
  });

  render();
}
