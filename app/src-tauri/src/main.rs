// Trellis Studio — Tauri v2 desktop shell around the trellis-server image→3D
// pipeline. Reads the installer-written config.json, launches & supervises the
// server, and exposes a few commands to the web UI.
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod config;
mod model_cache;
mod server;

use server::ServerState;
use tauri::Manager;

#[tauri::command]
fn get_config() -> Option<config::Config> {
    config::load().map(|mut c| {
        if c.output_dir.trim().is_empty() {
            c.output_dir = config::default_output_dir();
        }
        c
    })
}

#[tauri::command]
fn save_config(config: config::Config) -> Result<(), String> {
    config::save(&config)
}

#[tauri::command]
fn default_output_dir() -> String {
    config::default_output_dir()
}

#[tauri::command]
fn output_path(name: String) -> Result<String, String> {
    let dir = config::resolve_output_dir()?;
    Ok(dir.join(name).to_string_lossy().into_owned())
}

fn open_in_file_browser(path: &std::path::Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut cmd = std::process::Command::new("explorer");
    #[cfg(target_os = "macos")]
    let mut cmd = std::process::Command::new("open");
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = std::process::Command::new("xdg-open");
    cmd.arg(path);
    // explorer.exe returns a non-zero exit code even on success; spawn() ignores it.
    cmd.spawn().map(|_| ()).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_output_dir() -> Result<(), String> {
    let dir = config::resolve_output_dir()?;
    open_in_file_browser(&dir)
}

#[tauri::command]
fn logs_dir() -> String { config::logs_dir() }

#[tauri::command]
fn open_logs_dir() -> Result<(), String> {
    let dir = config::resolve_logs_dir()?;
    open_in_file_browser(&dir)
}

#[tauri::command]
fn current_log_path(state: tauri::State<ServerState>) -> Option<String> {
    server::log_path(state.inner())
}

#[tauri::command]
fn restart_server(app: tauri::AppHandle, state: tauri::State<ServerState>) -> Result<(), String> {
    let cfg = config::load().ok_or("no config.json found")?;
    server::start(&app, &cfg, state.inner(), false)
}

#[tauri::command]
fn server_running(state: tauri::State<ServerState>) -> bool {
    server::is_running(state.inner())
}

#[tauri::command]
fn model_cache_info() -> Result<model_cache::ModelCacheInfo, String> {
    let root = model_cache::managed_root()?;
    let active = config::load().map(|c| c.models_dir).unwrap_or_default();
    model_cache::inspect(&root, &active)
}

#[tauri::command]
fn model_cache_info_sv() -> Result<model_cache::ModelCacheInfo, String> {
    let root = model_cache::managed_root_sv()?;
    let active = config::load().map(|c| c.models_dir_sv).unwrap_or_default();
    model_cache::inspect(&root, &active)
}

#[tauri::command]
fn delete_managed_model_cache_sv(state: tauri::State<ServerState>) -> Result<u64, String> {
    let root = model_cache::managed_root_sv()?;
    match config::load() {
        Some(cfg) => {
            let info = model_cache::inspect(&root, &cfg.models_dir_sv)?;
            if info.active_is_managed {
                server::stop(state.inner());
            }
        }
        None => server::stop(state.inner()),
    }
    model_cache::delete_all(&root)
}

#[tauri::command]
fn delete_managed_model_cache(state: tauri::State<ServerState>) -> Result<u64, String> {
    let root = model_cache::managed_root()?;
    match config::load() {
        Some(cfg) => {
            let info = model_cache::inspect(&root, &cfg.models_dir)?;
            if info.active_is_managed {
                server::stop(state.inner());
            }
        }
        None => {
            // If config cannot be read, we cannot prove the running server is not
            // using the managed cache. Stop conservatively before destructive I/O.
            server::stop(state.inner());
        }
    }
    model_cache::delete_all(&root)
}

fn main() {
    // WebKitGTK ≥2.42 + the NVIDIA proprietary driver (and some other GPU/driver
    // combos) render a blank white window through the DMA-BUF path, and can even
    // crash the compositor on launch. Disabling that path fixes it with no
    // downside for this app's simple WebGL preview. Only set it if the user
    // hasn't already, so an explicit override still wins. (Linux only.)
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(ServerState::default())
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            default_output_dir,
            output_path,
            open_output_dir,
            logs_dir,
            open_logs_dir,
            current_log_path,
            restart_server,
            server_running,
            model_cache_info,
            delete_managed_model_cache,
            model_cache_info_sv,
            delete_managed_model_cache_sv
        ])
        .setup(|app| {
            if let Some(cfg) = config::load() {
                if !cfg.server_bin.is_empty() {
                    let state = app.state::<ServerState>();
                    // Autostart may adopt a server already on the port (manual
                    // launch / pre-fix orphan) rather than fail to bind.
                    if let Err(e) = server::start(app.handle(), &cfg, state.inner(), true) {
                        eprintln!("[studio] server autostart failed: {e}");
                    }
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                server::stop(window.state::<ServerState>().inner());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Trellis Studio")
        .run(|app, event| {
            // Every way out ends in server::stop. ExitRequested covers app.exit() and
            // the last window closing; Exit (LoopDestroyed) is the only event a macOS
            // `quit` (Cmd+Q, AppleScript, Dock) delivers — tao maps applicationWillTerminate
            // straight to it — which is how the server was orphaned on macOS (#24).
            if matches!(event, tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit) {
                server::stop(app.state::<ServerState>().inner());
            }
        });
}
