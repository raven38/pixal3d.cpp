// Pixal3D single-view (SV) パネル。画像 1 枚 + FOV から POST /generate-sv を叩く。
//
// 前処理（クロップ）とカメラ合成はサーバ側の共有 C++ が行う。ここは
//   - pre-matted RGBA かどうかのプリフライト（mv_preflight.ts の alpha 検査を再利用）
//   - FOV の範囲検査（度で入力、ラジアンで送る。既定 20°）
//   - /capabilities の sv.available に従った Generate の有効/無効と理由の表示
//   - Stop waiting（応答待ちの中止。サーバは継続するので gate.ts が再有効化を判定）
// だけを持つ。解像度は本リリースでは 1024 固定なので選ばせない（設計書 D10）。

import { generateSingleView } from "./api";
import { blockedReason, canGenerate, gateState, setLocalGenerating, stopWaiting, subscribeGate } from "./gate";
import { imageAlphaStatus, type AlphaCheckResult } from "./mv_preflight";
import { listen } from "./tauri";

export const SV_DEFAULT_FOV_DEG = 20;

export interface SvResultDetail {
  glb: Blob;
  input: Blob;
  name: string;
  fovRad: number;
  seed: number;
}

function injectStyle(): void {
  if (document.getElementById("sv-panel-style")) return;
  const s = document.createElement("style");
  s.id = "sv-panel-style";
  s.textContent = `
    .svp{margin-top:14px;border:1px solid #30343b;border-radius:12px;padding:12px;background:#15181d}
    .svp-title{font-weight:650}.svp-sub{font-size:12px;opacity:.7;margin:3px 0 10px}
    .svp-drop{border:1px dashed #4a5260;border-radius:10px;padding:14px;text-align:center;cursor:pointer;background:#111419;position:relative;min-height:120px}
    .svp-drop.drag{border-color:#50d2ff;background:#122029}
    .svp-drop img{max-width:100%;max-height:220px;object-fit:contain;display:block;margin:0 auto}
    .svp-controls{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px}.svp-controls label{font-size:12px}.svp-controls input,.svp-controls select{width:100%;box-sizing:border-box;margin-top:4px}
    .svp-preflight{margin-top:10px;padding:8px;border:1px solid #30343b;border-radius:8px;font-size:11px}.svp-preflight ul{margin:5px 0 0;padding-left:18px}.svp-preflight .error{color:#ff8e8e}.svp-preflight .ok{color:#75d49b}.svp-preflight .warning{color:#e6b85c}
    .svp-avail{margin-top:8px;font-size:11px;line-height:1.4}.svp-avail.bad{color:#e6b85c}.svp-avail.good{color:#75d49b}
    .svp-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .svp-progress{margin-top:10px;padding:9px;border-radius:8px;background:#0f1217;border:1px solid #282c33}.svp-progress[hidden]{display:none}.svp-stage{font-size:12px}.svp-elapsed{font-size:11px;opacity:.7;margin-top:3px}
    .svp-hint{font-size:11px;opacity:.7;margin-top:7px;line-height:1.35}
  `;
  document.head.appendChild(s);
}

export function mountSvPanel(root: HTMLElement): void {
  injectStyle();
  root.innerHTML = `
    <div class="svp">
      <div class="svp-title">Pixal3D single view</div>
      <div class="svp-sub">One pre-matted RGBA image. The server crops it like the reference preprocess and synthesizes a front camera (mesh_scale 1.0).</div>
      <div id="svp-drop" class="svp-drop" tabindex="0"><span id="svp-drop-hint">Drop one RGBA image here<br><span class="muted">or click to choose a file</span></span><img id="svp-preview" alt="single-view input" hidden></div>
      <input id="svp-file" type="file" accept="image/png,image/webp" hidden>
      <div class="svp-controls">
        <label>Camera FOV (degrees)<input id="svp-fov" type="number" min="1" max="179" step="0.5" value="${SV_DEFAULT_FOV_DEG}"></label>
        <label>Seed<input id="svp-seed" type="number" min="0" step="1" value="42"></label>
        <label>UV<select id="svp-uv"><option value="xatlas" selected>xatlas</option><option value="box">box</option></select></label>
      </div>
      <div class="svp-hint">Resolution is fixed at 1024 for single view in this release. 20° matches the browser default; MoGe-based FOV estimation is a separate tool (tools/estimate_transforms_moge.py) and is not built in.</div>
      <div id="svp-preflight" class="svp-preflight" hidden></div>
      <div id="svp-avail" class="svp-avail"></div>
      <div class="svp-actions"><button id="svp-generate" class="primary" disabled>Generate SV 3D</button><button id="svp-stop" class="link-btn" disabled>Stop waiting</button></div>
      <div id="svp-progress" class="svp-progress" hidden><div id="svp-stage" class="svp-stage">starting…</div><div id="svp-elapsed" class="svp-elapsed">0:00</div></div>
    </div>`;

  const drop = root.querySelector<HTMLElement>("#svp-drop")!;
  const dropHint = root.querySelector<HTMLElement>("#svp-drop-hint")!;
  const preview = root.querySelector<HTMLImageElement>("#svp-preview")!;
  const picker = root.querySelector<HTMLInputElement>("#svp-file")!;
  const fovInput = root.querySelector<HTMLInputElement>("#svp-fov")!;
  const seedInput = root.querySelector<HTMLInputElement>("#svp-seed")!;
  const uvSelect = root.querySelector<HTMLSelectElement>("#svp-uv")!;
  const preflight = root.querySelector<HTMLElement>("#svp-preflight")!;
  const avail = root.querySelector<HTMLElement>("#svp-avail")!;
  const gen = root.querySelector<HTMLButtonElement>("#svp-generate")!;
  const stop = root.querySelector<HTMLButtonElement>("#svp-stop")!;
  const progress = root.querySelector<HTMLElement>("#svp-progress")!;
  const stage = root.querySelector<HTMLElement>("#svp-stage")!;
  const elapsed = root.querySelector<HTMLElement>("#svp-elapsed")!;

  let file: File | null = null;
  let alpha: AlphaCheckResult | null = null;
  let generating = false;
  let abort: AbortController | null = null;
  let timer: number | null = null;
  let previewUrl = "";

  void listen<string>("server-log", (line) => {
    if (generating && String(line).trim()) stage.textContent = String(line).trim();
  });

  const fovRad = (): number | null => {
    const deg = Number(fovInput.value);
    if (!Number.isFinite(deg) || !(deg > 0) || !(deg < 180)) return null;
    return (deg * Math.PI) / 180;
  };

  // 入力の問題を列挙する。空配列なら送れる。
  const problems = (): string[] => {
    const out: string[] = [];
    if (!file) out.push("choose one pre-matted RGBA image");
    else if (alpha === "opaque") out.push(`${file.name}: image is fully opaque; a real alpha matte is required`);
    else if (alpha === "decode_error") out.push(`${file.name}: image could not be decoded for alpha validation`);
    if (fovRad() == null) out.push("camera FOV must be between 0 and 180 degrees (exclusive)");
    return out;
  };

  const svAvailable = (): { ok: boolean; text: string } => {
    const cap = gateState().capabilities;
    if (!cap) {
      return gateState().serverOnline
        ? { ok: false, text: "This runtime does not report /capabilities (pre-0.10.0 trellis-server) — single view needs the 0.10.0 runtime." }
        : { ok: false, text: "Server is offline." };
    }
    if (!cap.sv.configured) return { ok: false, text: "Single-view model set is not installed. Add it with the installer (--model-manifest-sv) or set the SV models directory in Settings." };
    if (!cap.sv.available) return { ok: false, text: `Single-view model set is not usable: ${cap.sv.reason ?? "unknown reason"}` };
    return { ok: true, text: `Single-view model set ready: ${cap.sv.model_set ?? "?"} ${cap.sv.version ?? ""}`.trim() };
  };

  const render = (): void => {
    const issues = problems();
    if (file) {
      preflight.hidden = false;
      preflight.replaceChildren();
      const b = document.createElement("b");
      b.textContent = issues.length ? `${issues.length} error(s)` : "Preflight passed";
      preflight.appendChild(b);
      const ul = document.createElement("ul");
      if (issues.length) {
        for (const msg of issues) {
          const li = document.createElement("li");
          li.className = "error";
          li.textContent = `✕ ${msg}`;
          ul.appendChild(li);
        }
      } else {
        const li = document.createElement("li");
        li.className = "ok";
        li.textContent = `✓ ${file.name}: real alpha matte · FOV ${Number(fovInput.value)}°`;
        ul.appendChild(li);
      }
      preflight.appendChild(ul);
    } else {
      preflight.hidden = true;
    }
    const a = svAvailable();
    const blocked = blockedReason();
    avail.className = `svp-avail ${a.ok ? "good" : "bad"}`;
    avail.textContent = a.ok && blocked && !generating ? `${a.text} · Generate is paused: ${blocked}.` : a.text;
    gen.disabled = generating || !a.ok || issues.length > 0 || !canGenerate();
    stop.disabled = !generating;
  };

  const setFile = async (f: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    file = f;
    alpha = null;
    previewUrl = URL.createObjectURL(f);
    preview.src = previewUrl;
    preview.hidden = false;
    dropHint.hidden = true;
    render();
    alpha = await imageAlphaStatus(f);
    render();
  };

  drop.addEventListener("click", () => picker.click());
  drop.addEventListener("keydown", (e) => { if ((e as KeyboardEvent).key === "Enter") picker.click(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => {
    const f = (e as DragEvent).dataTransfer?.files?.[0];
    if (f) void setFile(f);
  });
  picker.addEventListener("change", () => { const f = picker.files?.[0]; if (f) void setFile(f); });
  fovInput.addEventListener("input", render);
  subscribeGate(render);

  const fmt = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  stop.addEventListener("click", () => {
    abort?.abort();
    stopWaiting();
  });

  gen.addEventListener("click", async () => {
    if (!file || generating || problems().length || !canGenerate()) return;
    const fov = fovRad();
    if (fov == null) return;
    const seed = Math.max(0, Math.floor(Number(seedInput.value) || 42));
    const uv = uvSelect.value === "box" ? "box" : "xatlas";
    generating = true;
    setLocalGenerating(true);
    abort = new AbortController();
    progress.hidden = false;
    stage.textContent = "starting…";
    const started = Date.now();
    elapsed.textContent = "0:00";
    timer = window.setInterval(() => { elapsed.textContent = fmt(Date.now() - started); }, 1000);
    render();
    try {
      const { glb } = await generateSingleView(file, { fov, seed, uv }, abort.signal);
      stage.textContent = "complete";
      const detail: SvResultDetail = { glb, input: file, name: file.name, fovRad: fov, seed };
      window.dispatchEvent(new CustomEvent<SvResultDetail>("pixal3d-sv-result", { detail }));
    } catch (e) {
      if (abort.signal.aborted) {
        stage.textContent = "stopped waiting — the server keeps computing this generation";
      } else {
        stage.textContent = "failed";
        window.dispatchEvent(new CustomEvent<string>("pixal3d-sv-error", { detail: String((e as Error).message || e) }));
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
