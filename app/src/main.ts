import "./ui.css";
import { capabilities, generate, health } from "./api";
import { loadConfig } from "./config";
import { blockedReason, canGenerate, observeCapabilities, setLocalGenerating, setServerOnline, stopWaiting, subscribeGate } from "./gate";
import { mountMvCalibration, type MvResultDetail } from "./mv_calibration";
import { renderSettings } from "./settings";
import { all, clear as clearStore, del as removeRecord, get as getRecord, isEphemeral, newId, put } from "./store";
import { mountSvPanel, type SvResultDetail } from "./sv_panel";
import { isTauri, listen, saveBytes, saveToOutputDir } from "./tauri";
import { Viewer } from "./viewer";
import { DEFAULT_PARAMS, type GenMode, type GenParams, type GenRecord } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- element refs ----
const dropzone = $("dropzone");
const fileInput = $<HTMLInputElement>("file-input");
const inputPreview = $<HTMLImageElement>("input-preview");
const dropHint = $("dropzone-hint");
const generateBtn = $<HTMLButtonElement>("generate-btn");
const progress = $("progress");
const progressStage = $("progress-stage");
const progressElapsed = $("progress-elapsed");
const cancelBtn = $<HTMLButtonElement>("cancel-btn");
const resetViewBtn = $<HTMLButtonElement>("reset-view");
const saveGlbBtn = $<HTMLButtonElement>("save-glb");
const viewerCaption = $("viewer-caption");
const galleryEl = $("gallery");
const clearGalleryBtn = $("clear-gallery");
const backendBadge = $("backend-badge");
const serverDot = $("server-dot");
const serverLabel = $("server-label");
const setupBanner = $("setup-banner");

// ---- state ----
const viewer = new Viewer($("viewer-mount"));
let inputImage: Blob | null = null;
let inputName = "input.png";
let currentGlb: Blob | null = null;
let currentGlbName = "model.glb"; // default filename offered by "Save GLB…"
let activeId: string | null = null;
let generating = false;
let abort: AbortController | null = null;
let elapsedTimer: number | null = null;
let galleryUrls: string[] = [];
let warnedEphemeral = false;

// ---- controls -> params ----
function readParams(): GenParams {
  const res = parseInt(($("ctl-res") as HTMLSelectElement).value, 10);
  const seed = parseInt(($("ctl-seed") as HTMLInputElement).value, 10);
  return {
    resolution: (res === 512 || res === 1536 ? res : 1024) as GenParams["resolution"],
    seed: isNaN(seed) ? DEFAULT_PARAMS.seed : seed,
    bgRemoval: ($("ctl-bg") as HTMLSelectElement).value as GenParams["bgRemoval"],
    uv: ($("ctl-uv") as HTMLSelectElement).value as GenParams["uv"],
  };
}

function applyParams(p: GenParams): void {
  ($("ctl-res") as HTMLSelectElement).value = String(p.resolution);
  ($("ctl-seed") as HTMLInputElement).value = String(p.seed);
  ($("ctl-bg") as HTMLSelectElement).value = p.bgRemoval;
  ($("ctl-uv") as HTMLSelectElement).value = p.uv;
}

// ---- toasts ----
function toast(msg: string, kind: "" | "ok" | "err" = ""): void {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4000);
}

// ---- input image ----
function setInput(blob: Blob, name: string): void {
  inputImage = blob;
  inputName = name || "input.png";
  inputPreview.src = URL.createObjectURL(blob);
  inputPreview.classList.remove("hidden");
  dropHint.classList.add("hidden");
  updateGenerateEnabled();
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") fileInput.click();
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) setInput(f, f.name);
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }),
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  }),
);
dropzone.addEventListener("drop", (e) => {
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) setInput(f, f.name);
});
window.addEventListener("paste", async (e: ClipboardEvent) => {
  for (const item of e.clipboardData?.items ?? []) {
    const image = item.type.startsWith("image/") && item.getAsFile();
    if (image) return setInput(image, image.name || "pasted.png");
  }
  // WebKitGTK can omit image data from ClipboardEvent. Since paste was
  // user-triggered, try reading the clipboard item list for an image.
  const items = await navigator.clipboard.read().catch(() => []);
  for (const item of items) {
    const type = item.types.find((type) => type.startsWith("image/"));
    if (type) {
      const image = await item.getType(type).catch(() => null);
      if (image) return setInput(image, "pasted.png");
    }
  }
});

// ---- generate ----
function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function updateGenerateEnabled(): void {
  generateBtn.disabled = !(inputImage && canGenerate());
  const why = blockedReason();
  generateBtn.title = why && inputImage && !generating ? `Generate is paused: ${why}` : "";
}
subscribeGate(updateGenerateEnabled);

async function doGenerate(): Promise<void> {
  if (!inputImage || generating) return;
  if (!canGenerate()) return;
  const params = readParams();
  generating = true;
  setLocalGenerating(true);
  updateGenerateEnabled();
  progress.classList.remove("hidden");
  progressStage.textContent = "starting…";
  const started = Date.now();
  progressElapsed.textContent = "0:00";
  elapsedTimer = window.setInterval(() => {
    progressElapsed.textContent = fmtElapsed(Date.now() - started);
  }, 1000);
  abort = new AbortController();
  try {
    const { glb } = await generate(inputImage, params, abort.signal);
    currentGlb = glb;

    await recordResult({ mode: "trellis2", name: inputName, params, input: inputImage, glb });
  } catch (e) {
    if (abort?.signal.aborted) toast("Stopped waiting — the server keeps computing this generation");
    else toast((e as Error).message || "generation failed", "err");
  } finally {
    generating = false;
    setLocalGenerating(false);
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    progress.classList.add("hidden");
    updateGenerateEnabled();
  }
}

/**
 * 3 モード共通の結果処理: 出力フォルダへ保存 → gallery → viewer。順序は 0.9.0 と同じで、
 * ディスクへの保存を最優先にする（gallery DB や WebGL が死んでも成果物は残す）。
 */
async function recordResult(r: {
  mode: GenMode; name: string; params: GenParams; input: Blob; glb: Blob; mv?: GenRecord["mv"];
}): Promise<void> {
  const { glb, params } = r;
  const rec: GenRecord = {
    id: newId(),
    ts: Date.now(),
    name: r.name,
    params,
    mode: r.mode,
    input: r.input,
    glb,
    thumb: null,
    ...(r.mv ? { mv: r.mv } : {}),
  };
  currentGlb = glb;
  currentGlbName = `${glbStem(rec)}.glb`;
  {
    // 1) Write the GLB to the output folder FIRST. This on-disk file is the real
    // deliverable and must survive any later failure — a dead gallery DB (some
    // WebKitGTK builds can't open IndexedDB) or a WebGL/model-viewer crash must
    // never cost the user a successful generation.
    let savedPath: string | null = null;
    if (isTauri()) {
      try {
        const bytes = new Uint8Array(await glb.arrayBuffer());
        savedPath = await saveToOutputDir(`${glbStem(rec)}_${rec.id}.glb`, bytes);
      } catch (e) {
        toast(`Auto-save to output folder failed: ${(e as Error).message}`, "err");
      }
    }
    toast(savedPath ? `Saved to ${savedPath}` : "Generation complete", "ok");

    // 2) Add it to the gallery. The store falls back to in-memory if IndexedDB
    // is unavailable, so this never throws and never blocks the save above.
    await put(rec);
    if (isEphemeral() && !warnedEphemeral) {
      warnedEphemeral = true;
      toast(
        "Gallery won't persist across restarts (IndexedDB unavailable on this system) — " +
          "but every generation is still saved to your output folder.",
        "err",
      );
    }
    activeId = rec.id;
    setViewerTools(true);
    viewerCaption.textContent =
      `${modeLabel(r.mode)} · ${params.resolution} · seed ${params.seed} · ${(glb.size / 1e6).toFixed(1)} MB${mvCaption(rec)}`;
    await refreshGallery();

    // 3) Best-effort 3D preview + gallery thumbnail — never blocks the save above.
    try {
      await viewer.load(glb);
      const thumb = await viewer.thumbnail();
      if (thumb) {
        rec.thumb = thumb;
        await put(rec);
        await refreshGallery();
      }
    } catch (e) {
      toast(`3D preview couldn't render (your result is still saved): ${(e as Error).message}`, "err");
    }
  }
}

function modeLabel(mode: GenMode | undefined): string {
  return mode === "pixal3d-sv" ? "Pixal3D SV" : mode === "pixal3d-mv" ? "Pixal3D MV" : "TRELLIS.2";
}

/**
 * Stem shared by the auto-saved (`<stem>_<id>.glb`) and the "Save GLB…" (`<stem>.glb`)
 * filenames: `<image>_<res>_seed<seed>` for TRELLIS.2, `<image>_sv_…` for SV and
 * `multiview_…_scale<s>` / `canonical4_…_scale<s>` for MV.
 */
function glbStem(rec: GenRecord): string {
  const base = rec.name.replace(/\.[^.]+$/, "") || "model";
  const tag = rec.mode === "pixal3d-sv" ? "_sv" : "";
  const scale = rec.mv ? `_scale${rec.mv.meshScale.toFixed(3)}` : "";
  return `${base}${tag}_${rec.params.resolution}_seed${rec.params.seed}${scale}`;
}

/** " · mesh_scale 0.250" for multiview records, "" otherwise. */
function mvCaption(rec: GenRecord): string {
  return rec.mv ? ` · mesh_scale ${rec.mv.meshScale.toFixed(3)}` : "";
}

generateBtn.addEventListener("click", doGenerate);
// Stop waiting: 応答待ちだけをやめる。サーバはこの生成を最後まで計算するので、gate.ts が
// /capabilities の busy/completed で「終わった」と確認するまで全モードの Generate を止める。
cancelBtn.addEventListener("click", () => {
  abort?.abort();
  stopWaiting();
});

// ---- Pixal3D SV / MV panels ----
mountSvPanel($("sv-mount"));
mountMvCalibration($("mv-calibration-mount"));
window.addEventListener("pixal3d-sv-result", (event) => {
  const d = (event as CustomEvent<SvResultDetail>).detail;
  void recordResult({
    mode: "pixal3d-sv",
    name: d.name,
    params: { resolution: 1024, seed: d.seed, bgRemoval: "auto", uv: "xatlas" },
    input: d.input,
    glb: d.glb,
  });
});
window.addEventListener("pixal3d-sv-error", (event) => {
  toast(`Pixal3D single-view generation failed: ${(event as CustomEvent<string>).detail}`, "err");
});
window.addEventListener("pixal3d-mv-result", (event) => {
  const d = (event as CustomEvent<MvResultDetail>).detail;
  void recordResult({
    mode: "pixal3d-mv",
    name: d.name,
    params: { resolution: d.resolution, seed: d.seed, bgRemoval: "auto", uv: "xatlas" },
    input: d.input,
    glb: d.glb,
    mv: { meshScale: d.meshScale, numViews: d.numViews },
  });
});
window.addEventListener("pixal3d-mv-error", (event) => {
  toast(`Pixal3D multiview generation failed: ${(event as CustomEvent<string>).detail}`, "err");
});

// ---- mode switching ----
const modePanels: Record<GenMode, HTMLElement> = {
  trellis2: $("trellis2-panel"),
  "pixal3d-sv": $("sv-mount"),
  "pixal3d-mv": $("mv-calibration-mount"),
};
function setMode(mode: GenMode): void {
  for (const [m, el] of Object.entries(modePanels)) el.hidden = m !== mode;
  try { localStorage.setItem("trellis.mode", mode); } catch { /* private mode */ }
}
document.querySelectorAll<HTMLInputElement>('input[name="gen-mode"]').forEach((radio) => {
  radio.addEventListener("change", () => { if (radio.checked) setMode(radio.value as GenMode); });
});
{
  let saved: string | null = null;
  try { saved = localStorage.getItem("trellis.mode"); } catch { /* ignore */ }
  const initial: GenMode = saved === "pixal3d-sv" || saved === "pixal3d-mv" ? saved : "trellis2";
  const radio = document.querySelector<HTMLInputElement>(`input[name="gen-mode"][value="${initial}"]`);
  if (radio) radio.checked = true;
  setMode(initial);
}

// ---- viewer tools ----
function setViewerTools(on: boolean): void {
  resetViewBtn.disabled = !on;
  saveGlbBtn.disabled = !on;
}
resetViewBtn.addEventListener("click", () => viewer.resetView());
saveGlbBtn.addEventListener("click", async () => {
  if (!currentGlb) return;
  const bytes = new Uint8Array(await currentGlb.arrayBuffer());
  const ok = await saveBytes(currentGlbName, bytes);
  if (ok) toast("Saved", "ok");
});

// ---- gallery ----
async function loadRecord(id: string): Promise<void> {
  const rec = await getRecord(id);
  if (!rec) return;
  try {
    await viewer.load(rec.glb);
  } catch (e) {
    toast((e as Error).message, "err");
    return;
  }
  if (!rec.mode || rec.mode === "trellis2") {
    // SV/MV records have no TRELLIS.2 source image: leave the dropzone and the
    // single-image controls as they are.
    inputImage = rec.input;
    inputName = rec.name;
    inputPreview.src = URL.createObjectURL(rec.input);
    inputPreview.classList.remove("hidden");
    dropHint.classList.add("hidden");
    applyParams(rec.params);
  }
  currentGlb = rec.glb;
  currentGlbName = `${glbStem(rec)}.glb`;
  activeId = rec.id;
  setViewerTools(true);
  viewerCaption.textContent =
    `${modeLabel(rec.mode)} · ${rec.name} · ${rec.params.resolution} · seed ${rec.params.seed}${mvCaption(rec)}`;
  updateGenerateEnabled();
  await refreshGallery();
}

async function refreshGallery(): Promise<void> {
  galleryUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryUrls = [];
  const recs = await all();
  galleryEl.innerHTML = "";
  if (!recs.length) {
    galleryEl.innerHTML = `<div class="gallery-empty">No generations yet.</div>`;
    return;
  }
  for (const r of recs) {
    const item = document.createElement("div");
    item.className = "gitem" + (r.id === activeId ? " active" : "");
    item.title = `${r.name} · ${new Date(r.ts).toLocaleString()}`;

    const img = document.createElement("img");
    const src = r.thumb ?? r.input;
    const url = URL.createObjectURL(src);
    galleryUrls.push(url);
    img.src = url;
    item.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "gmeta";
    meta.textContent = r.mode && r.mode !== "trellis2" ? `${r.mode === "pixal3d-sv" ? "SV" : "MV"} ${r.params.resolution}` : `${r.params.resolution}`;
    item.appendChild(meta);

    const delBtn = document.createElement("button");
    delBtn.className = "gdel";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await removeRecord(r.id);
      if (activeId === r.id) activeId = null;
      await refreshGallery();
    });
    item.appendChild(delBtn);

    item.addEventListener("click", () => loadRecord(r.id));
    galleryEl.appendChild(item);
  }
}

clearGalleryBtn.addEventListener("click", async () => {
  if (!confirm("Delete all saved generations?")) return;
  await clearStore();
  activeId = null;
  await refreshGallery();
});

// ---- settings modal ----
const modal = $("settings-modal");
async function openSettings(): Promise<void> {
  await renderSettings($("settings-body"), () => {
    pollHealth();
    modal.classList.add("hidden");
    toast("Settings applied");
  });
  modal.classList.remove("hidden");
}
$("settings-btn").addEventListener("click", openSettings);
$("banner-settings").addEventListener("click", openSettings);
$("settings-close").addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.classList.add("hidden");
});

// ---- server status ----
async function pollHealth(): Promise<void> {
  const cfg = await loadConfig(true);
  backendBadge.textContent = cfg.backend !== "unknown" ? cfg.backend : "—";
  const ok = await health();
  setServerOnline(ok);
  // /capabilities は 0.10.0 で追加。旧サーバでは null になり、gate は 0.9.0 の挙動に戻る。
  observeCapabilities(ok ? await capabilities() : null);
  serverDot.className = "dot " + (ok ? "ok" : "err");
  serverLabel.textContent = ok ? "ready" : cfg.configured ? "offline" : "setup needed";
  const needSetup = !ok && !cfg.configured;
  setupBanner.classList.toggle("hidden", !needSetup);
  if (needSetup) {
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Trellis Studio isn't set up yet — run the installer or point it at your models directory.";
  } else if (!ok && cfg.configured) {
    setupBanner.classList.remove("hidden");
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Server is offline — it may still be loading, or check the models directory in settings.";
  }
  updateGenerateEnabled();
}

// ---- server log -> progress (Tauri only) ----
listen<string>("server-log", (line) => {
  const t = String(line).trim();
  if (generating && t) progressStage.textContent = t;
});

// ---- boot ----
async function boot(): Promise<void> {
  setViewerTools(false);
  await refreshGallery();
  await pollHealth();
  window.setInterval(pollHealth, 4000);
  if (!isTauri()) {
    // Browser mode: no shell to report a backend.
    backendBadge.textContent = "browser";
  }
}
boot();
