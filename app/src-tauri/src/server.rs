// Supervises the resident trellis-server child process: spawn it from the
// configured binary path, forward its stdout/stderr lines to the UI as
// `server-log` events (so the UI can show live stage progress), tee every line
// to a per-launch log file under the logs dir (so crashes/backend errors can be
// diagnosed after the fact), and make sure it dies with the app.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

use crate::config::{self, Config};

#[derive(Default)]
pub struct ServerState {
    child: Mutex<Option<Child>>,
    /// Path of the log file for the current/last launch, for the "open log" UI.
    log_path: Mutex<Option<PathBuf>>,
}

/// Is something already accepting connections on host:port? Used to detect a
/// server left running by a previously-crashed app instance (or a manually
/// launched one) so we don't spawn a duplicate that would fail to bind.
fn port_open(host: &str, port: u16) -> bool {
    match format!("{host}:{port}").to_socket_addrs() {
        Ok(mut addrs) => addrs
            .next()
            .map(|a| TcpStream::connect_timeout(&a, Duration::from_millis(300)).is_ok())
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Days-since-epoch -> (year, month, day). Howard Hinnant's civil-from-days.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn fmt_utc(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    let tod = secs % 86400;
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

fn fmt_hms(secs: u64) -> String {
    let tod = secs % 86400;
    format!("{:02}:{:02}:{:02}", tod / 3600, (tod % 3600) / 60, tod % 60)
}

/// Compact, sortable, filesystem-safe log filename for a launch at `secs`.
fn log_file_name(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    let tod = secs % 86400;
    format!(
        "trellis-server-{y:04}{m:02}{d:02}-{:02}{:02}{:02}.log",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

/// Keep the logs dir from growing forever: delete all but the newest `keep`
/// trellis-server-*.log files. Best-effort; failures are ignored.
fn prune_logs(dir: &std::path::Path, keep: usize) {
    let mut logs: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("trellis-server-") && n.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => return,
    };
    if logs.len() <= keep {
        return;
    }
    // Names are timestamp-sortable, so lexicographic == chronological.
    logs.sort();
    for p in &logs[..logs.len() - keep] {
        let _ = std::fs::remove_file(p);
    }
}

/// Shared handle to the current launch's log file. Both the stdout and stderr
/// reader threads write to it under one lock; `None` if the log couldn't open
/// (which never blocks generation — logging is best-effort).
type LogSink = Arc<Mutex<Option<File>>>;

fn write_log(sink: &LogSink, line: &str) {
    if let Ok(mut guard) = sink.lock() {
        if let Some(f) = guard.as_mut() {
            let _ = writeln!(f, "[{}] {}", fmt_hms(now_epoch()), line);
            let _ = f.flush();
        }
    }
}

fn pipe<R: Read + Send + 'static>(app: AppHandle, reader: R, sink: LogSink) {
    std::thread::spawn(move || {
        let buf = BufReader::new(reader);
        for line in buf.lines().map_while(Result::ok) {
            write_log(&sink, &line);
            let _ = app.emit("server-log", line);
        }
    });
}

/// Emit a `[studio]` diagnostic line to both the UI and the log file.
fn studio_log(app: &AppHandle, sink: &LogSink, msg: &str) {
    let line = format!("[studio] {msg}");
    write_log(sink, &line);
    let _ = app.emit("server-log", line);
}

/// (Re)start the server from the given config. Stops any child we own first.
///
/// `allow_reuse`: at autostart we adopt a server already bound to the port
/// (e.g. one a user launched by hand, or a pre-fix orphan) instead of spawning a
/// duplicate. On an explicit restart the user has just changed the config, so we
/// must NOT reuse a stale server — we spawn fresh so the new settings take
/// effect (surfacing a clear error if a foreign process still holds the port).
pub fn start(
    app: &AppHandle,
    cfg: &Config,
    state: &ServerState,
    allow_reuse: bool,
) -> Result<(), String> {
    stop(state);
    if cfg.server_bin.is_empty() {
        return Err("server binary is not configured".to_string());
    }

    // Open the per-launch log file (best-effort).
    let started = now_epoch();
    let sink: LogSink = Arc::new(Mutex::new(None));
    if let Ok(dir) = config::resolve_logs_dir() {
        prune_logs(&dir, 19); // keep 19 old + the one we're about to open = 20
        let path = dir.join(log_file_name(started));
        if let Ok(f) = File::create(&path) {
            *sink.lock().unwrap() = Some(f);
            *state.log_path.lock().unwrap() = Some(path);
        }
    }
    studio_log(
        app,
        &sink,
        &format!("Trellis Studio server log — {}", fmt_utc(started)),
    );
    studio_log(
        app,
        &sink,
        &format!(
            "config: bin={} models={} gpu={} backend={} host={} port={}",
            cfg.server_bin, cfg.models_dir, cfg.gpu, cfg.backend, cfg.host, cfg.port
        ),
    );

    if port_open(&cfg.host, cfg.port) {
        if allow_reuse {
            studio_log(
                app,
                &sink,
                &format!("reusing server already on {}:{}", cfg.host, cfg.port),
            );
            return Ok(());
        }
        // Explicit restart but a foreign process holds the port. Wait briefly in
        // case it's a socket we just released; if it persists, spawning will fail
        // to bind, so report it clearly rather than silently reusing stale config.
        let mut freed = false;
        for _ in 0..8 {
            std::thread::sleep(Duration::from_millis(200));
            if !port_open(&cfg.host, cfg.port) {
                freed = true;
                break;
            }
        }
        if !freed {
            let msg = format!(
                "another process is already using {}:{} — close it (or change the port) so the new settings can take effect",
                cfg.host, cfg.port
            );
            studio_log(app, &sink, &msg);
            return Err(msg);
        }
    }

    let mut cmd = Command::new(&cfg.server_bin);
    cmd.arg("--models")
        .arg(&cfg.models_dir)
        .arg("--gpu")
        .arg(cfg.gpu.to_string())
        .arg("--host")
        .arg(&cfg.host)
        .arg("--port")
        .arg(cfg.port.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Don't flash a console window on Windows.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    // On Linux, ask the kernel to SIGKILL the server if we (the parent) die, so a
    // hard crash of the app can't orphan a server on the port — which would then
    // get reused with stale config and make Settings changes appear ignored.
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0);
            Ok(())
        });
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch {}: {e}", cfg.server_bin))?;

    if let Some(out) = child.stdout.take() {
        pipe(app.clone(), out, sink.clone());
    }
    if let Some(err) = child.stderr.take() {
        pipe(app.clone(), err, sink.clone());
    }

    *state.child.lock().unwrap() = Some(child);
    studio_log(
        app,
        &sink,
        &format!("launched {} on {}:{}", cfg.server_bin, cfg.host, cfg.port),
    );
    Ok(())
}

pub fn stop(state: &ServerState) {
    if let Some(mut child) = state.child.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

pub fn is_running(state: &ServerState) -> bool {
    let mut guard = state.child.lock().unwrap();
    match guard.as_mut() {
        Some(child) => matches!(child.try_wait(), Ok(None)),
        None => false,
    }
}

/// Path of the current/last launch's log file, if one was opened.
pub fn log_path(state: &ServerState) -> Option<String> {
    state
        .log_path
        .lock()
        .unwrap()
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
}
