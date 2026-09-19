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
    // MV と SV は別々の managed cache（models / models-sv）。表示と削除も別々に出す。
    const cacheBlock = async (cmd: string, label: string, deleteId: string, activeDir: string) => {
      let cache: ModelCacheInfo | null = null;
      let cacheError = "";
      try { cache = await invoke<ModelCacheInfo>(cmd); } catch (e) { cacheError = String((e as Error).message ?? e); }
      const html = cache ? `
      <div class="kv">${label} managed cache (manifest-tracked files only): <b>${fmtBytes(cache.sizeBytes)}</b> · ${cache.files} file(s) · ${escapeHtml(cache.manifestVersion || "no manifest")}</div>
      <div class="kv muted">${escapeHtml(cache.managedRoot)}</div>
      <div class="kv muted">${escapeHtml(cache.note)}</div>
      ${cache.deletable ? `<button id="${deleteId}" class="tool-btn" type="button">Delete cached ${label} models</button>` : ""}
      ${!cache.activeIsManaged && activeDir ? `<div class="kv muted">External models are read-only to Studio. Changing this path only disconnects the reference; Studio never deletes that directory.</div>` : ""}`
        : `<div class="kv muted">${label} managed cache unavailable${cacheError ? `: ${escapeHtml(cacheError)}` : ""}</div>`;
      return { cache, html };
    };
    const mvCache = await cacheBlock("model_cache_info", "Pixal3D MV", "set-model-cache-delete", cfg.modelsDir);
    const svCache = await cacheBlock("model_cache_info_sv", "Pixal3D SV", "set-model-cache-delete-sv", cfg.modelsDirSv);

    body.innerHTML = `${ro("Backend", cfg.backend)}${ro("Server binary", cfg.serverBin)}${field("Models directory (Pixal3D MV, f16)", "set-models", cfg.modelsDir)}${mvCache.html}${field("Single-view models directory (Pixal3D SV, optional — leave empty if not installed)", "set-models-sv", cfg.modelsDirSv)}${svCache.html}${dirField("Output folder (generated GLBs are saved here)", "set-output", outputDir)}${field("GPU index (&lt;0 = CPU)", "set-gpu", String(cfg.gpu), "number")}${field("Port", "set-port", String(cfg.port), "number")}<label class="ctl"><span>Server logs</span><div class="dir-row"><input id="set-logs" type="text" value="${escapeHtml(logDir)}" readonly /><button id="set-logs-open" class="tool-btn" type="button">Open</button></div></label><div class="modal-actions"><button id="set-restart" class="tool-btn">Restart server</button><button id="set-save" class="primary">Save &amp; restart</button></div>`;

    const wireDelete = (id: string, cmd: string, info: ModelCacheInfo | null, label: string) => {
      const btn = body.querySelector(`#${id}`) as HTMLButtonElement | null;
      if (!btn || !info) return;
      btn.onclick = async () => {
        if (!confirm(`Delete ${info.files} cached ${label} model file(s), ${fmtBytes(info.sizeBytes)}, from Studio's managed cache? External model folders will not be touched.`)) return;
        btn.disabled = true;
        try { const removed = await invoke<number>(cmd); alert(`Removed ${fmtBytes(removed)} from the managed ${label} cache.`); await renderSettings(body, onSaved); }
        catch (e) { btn.disabled = false; alert(`Could not delete managed models: ${(e as Error).message ?? e}`); }
      };
    };
    wireDelete("set-model-cache-delete", "delete_managed_model_cache", mvCache.cache, "Pixal3D MV");
    wireDelete("set-model-cache-delete-sv", "delete_managed_model_cache_sv", svCache.cache, "Pixal3D SV");
    (body.querySelector("#set-logs-open") as HTMLButtonElement).onclick = async () => { try { await openLogsDir(); } catch (e) { alert(`Could not open the logs folder: ${(e as Error).message ?? e}`); } };
    const outputInput = body.querySelector("#set-output") as HTMLInputElement;
    (body.querySelector("#set-output-browse") as HTMLButtonElement).onclick = async () => { const picked = await pickDirectory(outputInput.value.trim()); if (picked) outputInput.value = picked; };
    (body.querySelector("#set-output-open") as HTMLButtonElement).onclick = async () => { await saveConfig({ outputDir: outputInput.value.trim() }); try { await openOutputDir(); } catch (e) { alert(`Could not open the output folder: ${(e as Error).message ?? e}`); } };
    const save = async () => {
      const modelsDir = (body.querySelector("#set-models") as HTMLInputElement).value.trim();
      const modelsDirSv = (body.querySelector("#set-models-sv") as HTMLInputElement).value.trim();
      const gpu = parseInt((body.querySelector("#set-gpu") as HTMLInputElement).value, 10);
      const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10);
      await saveConfig({ modelsDir, modelsDirSv, gpu: isNaN(gpu) ? 0 : gpu, port: isNaN(port) ? 8080 : port, outputDir: outputInput.value.trim() });
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
