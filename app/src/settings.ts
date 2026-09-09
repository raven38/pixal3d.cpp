// Settings modal: shows the resolved config and lets the user adjust the bits
// that make sense per environment. In Tauri, saving hands the config to the shell
// (which restarts the server); in the browser we only expose host/port.

import { loadConfig, saveConfig } from "./config";
import { invoke, isTauri, logsDir, openLogsDir, openOutputDir, pickDirectory } from "./tauri";

interface ModelCacheInfo {
  managedRoot: string;
  activeModelsDir: string;
  activeIsManaged: boolean;
  exists: boolean;
  sizeBytes: number;
  files: number;
  manifestVersion?: string | null;
  deletable: boolean;
  note: string;
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function field(label: string, id: string, value: string, type = "text"): string {
  return `<label class="ctl"><span>${label}</span><input id="${id}" type="${type}" value="${escapeHtml(value)}" /></label>`;
}
function dirField(label: string, id: string, value: string): string {
  return `<label class="ctl"><span>${label}</span><div class="dir-row"><input id="${id}" type="text" value="${escapeHtml(value)}" /><button id="${id}-browse" class="tool-btn" type="button">Browse…</button><button id="${id}-open" class="tool-btn" type="button">Open</button></div></label>`;
}
function ro(label: string, value: string): string { return `<div class="kv">${label}: <b>${escapeHtml(value || "—")}</b></div>`; }
function fmtBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Math.max(0, n); let i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; }
  return `${value.toFixed(i < 2 ? 0 : 2)} ${units[i]}`;
}

export async function renderSettings(body: HTMLElement, onSaved: () => void): Promise<void> {
  const cfg = await loadConfig(true);
  if (isTauri()) {
    let outputDir = cfg.outputDir;
    if (!outputDir) { try { outputDir = await invoke<string>("default_output_dir"); } catch {} }
    const logDir = await logsDir();
    let cache: ModelCacheInfo | null = null;
    let cacheError = "";
    try { cache = await invoke<ModelCacheInfo>("model_cache_info"); } catch (e) { cacheError = String((e as Error).message ?? e); }
    const cacheHtml = cache ? `
      <div class="kv">Managed model cache (manifest-tracked files only): <b>${fmtBytes(cache.sizeBytes)}</b> · ${cache.files} file(s) · ${escapeHtml(cache.manifestVersion || "no manifest")}</div>
      <div class="kv muted">${escapeHtml(cache.managedRoot)}</div>
      <div class="kv muted">${escapeHtml(cache.note)}</div>
      ${cache.deletable ? `<button id="set-model-cache-delete" class="tool-btn" type="button">Delete cached Pixal3D models</button>` : ""}
      ${!cache.activeIsManaged && cfg.modelsDir ? `<div class="kv muted">External models are read-only to Studio. Changing this path only disconnects the reference; Studio never deletes that directory.</div>` : ""}`
      : `<div class="kv muted">Managed model cache unavailable${cacheError ? `: ${escapeHtml(cacheError)}` : ""}</div>`;

    body.innerHTML = `${ro("Backend", cfg.backend)}${ro("Server binary", cfg.serverBin)}${field("Models directory", "set-models", cfg.modelsDir)}${cacheHtml}${dirField("Output folder (generated GLBs are saved here)", "set-output", outputDir)}${field("GPU index (&lt;0 = CPU)", "set-gpu", String(cfg.gpu), "number")}${field("Port", "set-port", String(cfg.port), "number")}<label class="ctl"><span>Server logs</span><div class="dir-row"><input id="set-logs" type="text" value="${escapeHtml(logDir)}" readonly /><button id="set-logs-open" class="tool-btn" type="button">Open</button></div></label><div class="modal-actions"><button id="set-restart" class="tool-btn">Restart server</button><button id="set-save" class="primary">Save &amp; restart</button></div>`;

    const deleteBtn = body.querySelector("#set-model-cache-delete") as HTMLButtonElement | null;
    if (deleteBtn && cache) deleteBtn.onclick = async () => {
      if (!confirm(`Delete ${cache.files} cached Pixal3D model file(s), ${fmtBytes(cache.sizeBytes)}, from Studio's managed cache? External model folders will not be touched.`)) return;
      deleteBtn.disabled = true;
      try { const removed = await invoke<number>("delete_managed_model_cache"); alert(`Removed ${fmtBytes(removed)} from the managed Pixal3D cache.`); await renderSettings(body, onSaved); }
      catch (e) { deleteBtn.disabled = false; alert(`Could not delete managed models: ${(e as Error).message ?? e}`); }
    };
    (body.querySelector("#set-logs-open") as HTMLButtonElement).onclick = async () => { try { await openLogsDir(); } catch (e) { alert(`Could not open the logs folder: ${(e as Error).message ?? e}`); } };
    const outputInput = body.querySelector("#set-output") as HTMLInputElement;
    (body.querySelector("#set-output-browse") as HTMLButtonElement).onclick = async () => { const picked = await pickDirectory(outputInput.value.trim()); if (picked) outputInput.value = picked; };
    (body.querySelector("#set-output-open") as HTMLButtonElement).onclick = async () => { await saveConfig({ outputDir: outputInput.value.trim() }); try { await openOutputDir(); } catch (e) { alert(`Could not open the output folder: ${(e as Error).message ?? e}`); } };
    const save = async () => {
      const modelsDir = (body.querySelector("#set-models") as HTMLInputElement).value.trim();
      const gpu = parseInt((body.querySelector("#set-gpu") as HTMLInputElement).value, 10);
      const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10);
      await saveConfig({ modelsDir, gpu: isNaN(gpu) ? 0 : gpu, port: isNaN(port) ? 8080 : port, outputDir: outputInput.value.trim() });
      try { await invoke("restart_server"); } catch {}
      onSaved();
    };
    (body.querySelector("#set-save") as HTMLButtonElement).onclick = save;
    (body.querySelector("#set-restart") as HTMLButtonElement).onclick = async () => { try { await invoke("restart_server"); } catch {} onSaved(); };
  } else {
    body.innerHTML = `<div class="kv">Running in a browser — connecting to a trellis-server you launched.</div>${field("Host", "set-host", cfg.host)}${field("Port", "set-port", String(cfg.port), "number")}<div class="modal-actions"><button id="set-save" class="primary">Save</button></div>`;
    (body.querySelector("#set-save") as HTMLButtonElement).onclick = async () => { const host = (body.querySelector("#set-host") as HTMLInputElement).value.trim() || "127.0.0.1"; const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10); await saveConfig({ host, port: isNaN(port) ? 8080 : port }); onSaved(); };
  }
}
