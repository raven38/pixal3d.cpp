import { CANONICAL_RIG_VIEWS, generateMultiview, type MultiviewFile } from "./api";
import { blockedReason, canGenerate, setLocalGenerating, stopWaiting, subscribeGate } from "./gate";
import { listen } from "./tauri";
import {
  hasErrors,
  imageAlphaStatus,
  validateCanonicalRig,
  validateMetadata,
  type AlphaCheckResult,
  type PreflightItem,
} from "./mv_preflight";

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

export interface MvResultDetail {
  glb: Blob;
  /** The first view, used as the gallery thumbnail source. */
  input: Blob;
  name: string;
  meshScale: number;
  resolution: 1024 | 1536;
  seed: number;
  /** true when no transforms.json was given (server-side canonical turntable rig). */
  canonical: boolean;
}

// transforms.json 無しのときにカードへ表示する姿勢名。姿勢そのもの（行列・FOV・距離）は
// 共有 C++（transforms_json.cpp）だけが持ち、ここには持たない。
const CANONICAL_NAMES = ["front", "right", "back", "left"] as const;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const basename = (p: string) => p.replace(/\\/g, "/").split("/").pop() || p;
const validScale = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v) && v > 0;

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
    .mvcal-preflight{margin-top:10px;padding:8px;border:1px solid #30343b;border-radius:8px;font-size:11px}.mvcal-preflight ul{margin:5px 0 0;padding-left:18px}.mvcal-preflight .error{color:#ff8e8e}.mvcal-preflight .warning{color:#e6b85c}.mvcal-preflight .ok{color:#75d49b}
    .mvcal-progress{margin-top:10px;padding:9px;border-radius:8px;background:#0f1217;border:1px solid #282c33}.mvcal-progress[hidden]{display:none}.mvcal-stage{font-size:12px}.mvcal-elapsed{font-size:11px;opacity:.7;margin-top:3px}
  `;
  document.head.appendChild(s);
}

async function alphaBounds(file: File): Promise<{ w: number; h: number } | null> {
  const bmp = await createImageBitmap(file).catch(() => null);
  if (!bmp) return null;
  try {
    const maxDim = 512;
    const scale = Math.min(1, maxDim / Math.max(bmp.width, bmp.height));
    const width = Math.max(1, Math.round(bmp.width * scale));
    const height = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(bmp, 0, 0, width, height);
    const d = ctx.getImageData(0, 0, width, height).data;
    let xmin = width, ymin = height, xmax = -1, ymax = -1;
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        if (d[(y * width + x) * 4 + 3] <= 16) continue;
        xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
        ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
      }
    }
    if (xmax < xmin || ymax < ymin) return null;
    return { w: (xmax - xmin + 1) / width, h: (ymax - ymin + 1) / height };
  } finally {
    bmp.close();
  }
}

export function mountMvCalibration(root: HTMLElement): void {
  injectStyle();
  root.innerHTML = `
    <div class="mvcal">
      <div class="mvcal-head"><div><div class="mvcal-title">Pixal3D multiview</div><div class="mvcal-sub">Drop views + transforms.json, reorder them, set mesh scale, then generate. Without transforms.json, exactly four turntable views (front, right, back, left) are accepted.</div></div></div>
      <div id="mvcal-drop" class="mvcal-drop" tabindex="0">Drop RGBA views (+ transforms.json) here<br><span class="muted">or click to choose files</span><div id="mvcal-files-summary" class="mvcal-files"></div></div>
      <input id="mvcal-files" type="file" multiple accept="image/*,.json,application/json" hidden>
      <div id="mvcal-body" hidden>
        <div id="mvcal-grid" class="mvcal-grid"></div>
        <div id="mvcal-order-hint" class="mvcal-hint">Drag view cards to reorder. The patched transforms.json will use this frame order.</div>
        <label id="mvcal-order-confirm" class="mvcal-hint" hidden><input id="mvcal-order-ok" type="checkbox"> I confirm the cards are ordered <b>front, right, back, left</b> (turntable, elevation 0, FOV 20°). Nothing is assumed otherwise.</label>
        <label>Mesh scale</label>
        <div class="mvcal-row"><input id="mvcal-range" type="range" min="0.02" max="2" step="0.001"><input id="mvcal-num" type="number" min="0.001" max="4" step="0.001" placeholder="required"></div>
        <div id="mvcal-status" class="mvcal-status"></div>
        <div class="mvcal-controls">
          <label>Resolution<select id="mvcal-res"><option value="1024" selected>1024 · cascade</option><option value="1536">1536 · high</option></select></label>
          <label>Seed<input id="mvcal-seed" type="number" min="0" step="1" value="42"></label>
        </div>
        <div id="mvcal-preflight" class="mvcal-preflight"></div>
        <div class="mvcal-hint">A missing mesh_scale is never guessed. Type a positive value explicitly; the slider is enabled only after that manual value exists.</div>
        <div role="note" aria-label="Multiview resource usage warning" class="mvcal-hint" style="margin-top:10px;padding:9px;border:1px solid #5a4930;border-radius:8px;background:#1a1711;color:#c9bda6"><strong style="color:#f0c978">⚠ Long-running generation</strong><div style="margin-top:4px">Multiview 1024 can take roughly <strong>10–30+ minutes</strong> on tested high-end hardware; 1536 may take substantially longer. High CPU/GPU and RAM/VRAM use are expected. Keep Trellis Studio open until the GLB is saved.</div></div>
        <div id="mvcal-blocked" class="mvcal-status mvcal-warn" hidden></div>
        <div class="mvcal-actions"><button id="mvcal-download" class="tool-btn">Save calibrated transforms.json</button><button id="mvcal-generate" class="primary" disabled>Generate MV 3D</button><button id="mvcal-cancel" class="link-btn" disabled>Stop waiting</button></div>
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
  const preflight = root.querySelector<HTMLElement>("#mvcal-preflight")!;
  const gen = root.querySelector<HTMLButtonElement>("#mvcal-generate")!;
  const cancel = root.querySelector<HTMLButtonElement>("#mvcal-cancel")!;
  const res = root.querySelector<HTMLSelectElement>("#mvcal-res")!;
  const seed = root.querySelector<HTMLInputElement>("#mvcal-seed")!;
  const progress = root.querySelector<HTMLElement>("#mvcal-progress")!;
  const stage = root.querySelector<HTMLElement>("#mvcal-stage")!;
  const elapsed = root.querySelector<HTMLElement>("#mvcal-elapsed")!;
  const orderHint = root.querySelector<HTMLElement>("#mvcal-order-hint")!;
  const orderConfirm = root.querySelector<HTMLElement>("#mvcal-order-confirm")!;
  const orderOk = root.querySelector<HTMLInputElement>("#mvcal-order-ok")!;
  const download = root.querySelector<HTMLButtonElement>("#mvcal-download")!;
  const blocked = root.querySelector<HTMLElement>("#mvcal-blocked")!;

  let meta: TransformMeta | null = null;
  // transforms.json が無く、ちょうど 4 枚のとき true。姿勢はサーバの canonical rig が
  // ファイル名の自然順で割り当てるので、カード順をアップロード名（view0..3）に写す。
  let canonical = false;
  let views: ViewEntry[] = [];
  let imageFiles: File[] = [];
  let guideReferenceScale: number | null = null;
  let dragging = -1;
  let activeAbort: AbortController | null = null;
  let timer: number | null = null;
  let generating = false;
  let lastPreflight: PreflightItem[] = [];
  const alphaCache = new WeakMap<File, AlphaCheckResult>();

  range.disabled = true;

  void listen<string>("server-log", (line) => {
    if (generating && String(line).trim()) stage.textContent = String(line).trim();
  });

  const cleanupViews = () => {
    views.forEach((v) => URL.revokeObjectURL(v.url));
    views = [];
  };

  const currentScale = (): number | null => {
    const raw = num.value.trim();
    if (!raw) return null;
    const value = Number(raw);
    return validScale(value) ? clamp(value, 0.001, 4) : null;
  };

  const orderedMeta = () => {
    const meshScale = currentScale();
    if (canonical) throw new Error("the canonical rig has no transforms.json to patch");
    if (!meta || meshScale == null) throw new Error("mesh_scale is required");
    return { ...meta, mesh_scale: meshScale, frames: views.map((v) => v.frame) };
  };

  const patchBlob = () =>
    new Blob([JSON.stringify(orderedMeta(), null, 2)], { type: "application/json" });

  const fmt = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  const alphaStatus = async (file: File) => {
    const cached = alphaCache.get(file);
    if (cached) return cached;
    const result = await imageAlphaStatus(file);
    alphaCache.set(file, result);
    return result;
  };

  const renderPreflight = (items: PreflightItem[]) => {
    lastPreflight = items;
    const errors = items.filter((i) => i.level === "error").length;
    const warnings = items.filter((i) => i.level === "warning").length;
    preflight.replaceChildren();
    const heading = document.createElement("b");
    heading.textContent = `${errors ? `${errors} error(s)` : "Preflight passed"}${warnings ? ` · ${warnings} warning(s)` : ""}`;
    preflight.appendChild(heading);
    const ul = document.createElement("ul");
    for (const item of items) {
      const li = document.createElement("li");
      li.className = item.level;
      li.textContent = `${item.level === "error" ? "✕" : item.level === "warning" ? "⚠" : "✓"} ${item.message}`;
      ul.appendChild(li);
    }
    preflight.appendChild(ul);
    applyGate();
  };

  // Generate の可否 = プリフライトに error 無し && 共有ゲート（サーバ online・他の生成が
  // 走っていない・Stop waiting 後の完了確認済み）。
  const applyGate = () => {
    const errors = lastPreflight.some((i) => i.level === "error");
    const why = blockedReason();
    gen.disabled = generating || errors || !canGenerate();
    blocked.hidden = generating || !why || !views.length;
    blocked.textContent = why ? `Generate is paused: ${why}.` : "";
  };
  subscribeGate(applyGate);

  const runPreflight = async () => {
    const meshScale = currentScale();
    let items: PreflightItem[];
    if (canonical) {
      items = validateCanonicalRig(imageFiles.length, meshScale, Number(res.value), orderOk.checked);
    } else {
      const metaForValidation = meta
        ? ({ ...meta, mesh_scale: meshScale ?? undefined } as TransformMeta)
        : null;
      items = validateMetadata(
        metaForValidation,
        views.map((v) => v.frame.file_path),
        Number(res.value),
      );
    }
    const alphaResults = await Promise.all(
      views.map(async (v) => ({ view: v, result: await alphaStatus(v.file) })),
    );
    for (const { view, result } of alphaResults) {
      if (result === "opaque") {
        items.push({
          level: "error",
          code: `alpha-${view.frame.file_path}`,
          message: `${view.frame.file_path}: image is fully opaque; an RGBA cutout matte is required`,
        });
      } else if (result === "decode_error") {
        items.push({
          level: "error",
          code: `decode-${view.frame.file_path}`,
          message: `${view.frame.file_path}: image could not be decoded for alpha validation`,
        });
      }
    }
    renderPreflight(items);
    return items;
  };

  const renderGuide = async () => {
    const scale = currentScale();
    const guides = [...grid.querySelectorAll<HTMLElement>(".mvcal-guide")];
    if (scale == null) {
      guides.forEach((g) => { g.hidden = true; });
      range.disabled = true;
      status.className = "mvcal-status mvcal-warn";
      status.textContent = "mesh_scale missing · enter a value manually; no default is assumed";
      return;
    }

    guides.forEach((g) => { g.hidden = false; });
    range.disabled = false;
    range.value = String(clamp(scale, 0.02, 2));
    if (guideReferenceScale == null) guideReferenceScale = scale;
    const ratio = clamp(guideReferenceScale / scale, 0.22, 2.2);
    guides.forEach((g) => {
      g.style.width = `${clamp(72 * ratio, 18, 96)}%`;
      g.style.height = `${clamp(72 * ratio, 18, 96)}%`;
    });

    const bounds = await Promise.all(views.slice(0, 4).map((v) => alphaBounds(v.file)));
    const cover = bounds.filter(Boolean).map((b) => Math.max(b!.w, b!.h));
    const spread = cover.length ? Math.max(...cover) - Math.min(...cover) : 0;
    status.className = `mvcal-status ${spread > 0.35 ? "mvcal-warn" : "mvcal-good"}`;
    const source = canonical
      ? "manual value for the canonical rig (not estimated)"
      : validScale(meta?.mesh_scale) && Math.abs(meta.mesh_scale - scale) < 1e-9
        ? "from transforms.json"
        : "manual value (not estimated)";
    status.textContent = `scale ${scale.toFixed(3)} · ${source} · ${views.length} views${spread > 0.35 ? " · view framing differs strongly; check cameras/FOV" : ""}`;
  };

  const renderCards = () => {
    grid.innerHTML = "";
    views.slice(0, 8).forEach((v, i) => {
      const card = document.createElement("div");
      card.className = "mvcal-view";
      card.draggable = true;
      card.dataset.index = String(i);
      card.innerHTML = `<img src="${v.url}" alt="view ${i}"><div class="mvcal-guide"></div><div class="mvcal-name"></div><div class="mvcal-order">${i + 1}</div>`;
      (card.querySelector(".mvcal-name") as HTMLElement).textContent = canonical
        ? `${CANONICAL_NAMES[i] ?? "?"} · ${v.file.name}`
        : v.frame.file_path;
      card.addEventListener("dragstart", () => { dragging = i; card.classList.add("dragging"); });
      card.addEventListener("dragend", () => { dragging = -1; card.classList.remove("dragging"); });
      card.addEventListener("dragover", (e) => e.preventDefault());
      card.addEventListener("drop", (e) => {
        e.preventDefault();
        const target = Number(card.dataset.index);
        if (dragging < 0 || dragging === target) return;
        const [moved] = views.splice(dragging, 1);
        views.splice(target, 0, moved);
        dragging = -1;
        renderCards();
        void renderGuide();
        void runPreflight();
      });
      grid.appendChild(card);
    });
  };

  const reconcile = async () => {
    cleanupViews();
    canonical = false;
    orderConfirm.hidden = true;
    orderHint.hidden = false;
    download.hidden = false;
    if (!meta) {
      // transforms.json 無し: ちょうど 4 枚なら canonical rig（サーバ側で合成）。
      // それ以外の枚数は受け付けない（4 視点でないと multiview モデルが破綻する）。
      if (imageFiles.length !== CANONICAL_RIG_VIEWS) {
        body.hidden = true;
        summary.textContent = imageFiles.length
          ? `${imageFiles.length} image(s) without transforms.json — exactly ${CANONICAL_RIG_VIEWS} turntable views (front, right, back, left) are required; add transforms.json for any other camera setup`
          : "no images";
        return;
      }
      canonical = true;
      orderConfirm.hidden = false;
      orderHint.hidden = true;
      download.hidden = true;   // 姿勢は C++ にしか無いので JSON は作れない
      orderOk.checked = false;
      const sorted = [...imageFiles].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      sorted.forEach((f, i) => {
        views.push({
          file: f,
          frame: { file_path: f.name, transform_matrix: [] },
          url: URL.createObjectURL(f),
        });
        void i;
      });
      summary.textContent = `${views.length} views · canonical turntable rig (no transforms.json)`;
      guideReferenceScale = null;
      num.value = "";
      range.value = "0.2";
      range.disabled = true;
      body.hidden = false;
      renderCards();
      await renderGuide();
      await runPreflight();
      return;
    }
    const byName = new Map<string, File>();
    imageFiles.forEach((f) => {
      byName.set(f.name, f);
      const rel = f.webkitRelativePath?.split("/").slice(1).join("/");
      if (rel) byName.set(rel, f);
    });
    for (const fr of meta.frames) {
      const f = byName.get(fr.file_path) ||
        byName.get(fr.file_path.replace(/^\.\//, "")) ||
        byName.get(basename(fr.file_path));
      if (f) views.push({ file: f, frame: fr, url: URL.createObjectURL(f) });
    }
    summary.textContent = `${imageFiles.length} image(s) · ${views.length}/${meta.frames.length} matched`;
    if (!views.length) {
      body.hidden = true;
      status.textContent = "No frame images matched transforms.json";
      return;
    }

    guideReferenceScale = validScale(meta.mesh_scale) ? meta.mesh_scale : null;
    if (guideReferenceScale != null) {
      num.value = guideReferenceScale.toFixed(3);
      range.value = String(clamp(guideReferenceScale, 0.02, 2));
      range.disabled = false;
    } else {
      num.value = "";
      range.value = "0.2";
      range.disabled = true;
    }
    body.hidden = false;
    renderCards();
    await renderGuide();
    await runPreflight();
  };

  const ingest = async (list: File[]) => {
    const tf = list.find((f) => f.name === "transforms.json" || f.name.toLowerCase().endsWith(".json"));
    if (!tf) meta = null;   // 新しいドロップに JSON が無ければ、前回の transforms は引き継がない
    if (tf) {
      try {
        meta = JSON.parse(await tf.text()) as TransformMeta;
      } catch {
        meta = null;
        summary.textContent = "Invalid transforms.json";
        body.hidden = true;
        return;
      }
    }
    const imgs = list.filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(f.name));
    if (imgs.length) imageFiles = imgs;
    await reconcile();
  };

  drop.addEventListener("click", () => picker.click());
  drop.addEventListener("keydown", (e) => { if ((e as KeyboardEvent).key === "Enter") picker.click(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => void ingest([...(e as DragEvent).dataTransfer?.files ?? []]));
  picker.addEventListener("change", () => void ingest([...(picker.files ?? [])]));

  range.addEventListener("input", () => {
    if (range.disabled) return;
    num.value = Number(range.value).toFixed(3);
    void renderGuide();
    void runPreflight();
  });
  num.addEventListener("input", () => {
    const scale = currentScale();
    if (scale != null && guideReferenceScale == null) guideReferenceScale = scale;
    range.disabled = scale == null;
    void renderGuide();
    void runPreflight();
  });
  res.addEventListener("change", () => void runPreflight());
  orderOk.addEventListener("change", () => void runPreflight());

  download.addEventListener("click", async () => {
    if (!meta || canonical) return;
    const items = await runPreflight();
    if (hasErrors(items)) {
      alert("Fix preflight errors before saving calibrated transforms.json");
      return;
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(patchBlob());
    a.download = "transforms.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  cancel.addEventListener("click", () => {
    activeAbort?.abort();
    stopWaiting();
  });
  gen.addEventListener("click", async () => {
    if ((!meta && !canonical) || !views.length || generating || !canGenerate()) return;
    const items = await runPreflight();
    if (hasErrors(items)) return;

    const meshScale = currentScale();
    if (meshScale == null) return;
    generating = true;
    setLocalGenerating(true);
    activeAbort = new AbortController();
    gen.disabled = true;
    cancel.disabled = false;
    progress.hidden = false;
    stage.textContent = "starting…";
    const started = Date.now();
    elapsed.textContent = "0:00";
    timer = window.setInterval(() => { elapsed.textContent = fmt(Date.now() - started); }, 1000);
    try {
      const mv: MultiviewFile[] = views.map((v) => ({ name: v.frame.file_path, blob: v.file }));
      const resolution = Number(res.value) === 1536 ? 1536 : 1024;
      const seedValue = Math.max(0, Number(seed.value) || 42);
      const { glb } = await generateMultiview(
        canonical ? null : patchBlob(),
        mv,
        { seed: seedValue, resolution, uv: "xatlas", numViews: views.length, autoMeshScale: false, meshScale },
        activeAbort.signal,
      );
      stage.textContent = "complete";
      const detail: MvResultDetail = {
        glb, input: views[0].file, name: canonical ? "canonical4" : "multiview",
        meshScale, resolution, seed: seedValue, canonical,
      };
      window.dispatchEvent(new CustomEvent<MvResultDetail>("pixal3d-mv-result", { detail }));
    } catch (e) {
      if (activeAbort.signal.aborted) {
        stage.textContent = "stopped waiting — the server keeps computing this generation";
      } else {
        stage.textContent = "failed";
        window.dispatchEvent(new CustomEvent("pixal3d-mv-error", {
          detail: String((e as Error).message || e),
        }));
      }
    } finally {
      generating = false;
      setLocalGenerating(false);
      activeAbort = null;
      if (timer != null) window.clearInterval(timer);
      timer = null;
      cancel.disabled = true;
      applyGate();
    }
  });
}
