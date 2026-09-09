use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Serialize)]
pub struct ModelCacheInfo {
    #[serde(rename = "managedRoot")]
    pub managed_root: String,
    #[serde(rename = "activeModelsDir")]
    pub active_models_dir: String,
    #[serde(rename = "activeIsManaged")]
    pub active_is_managed: bool,
    pub exists: bool,
    #[serde(rename = "sizeBytes")]
    pub size_bytes: u64,
    pub files: usize,
    #[serde(rename = "manifestVersion")]
    pub manifest_version: Option<String>,
    pub deletable: bool,
    pub note: String,
}

pub fn managed_root() -> Result<PathBuf, String> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join("portable.dat").exists() {
                return Ok(dir.join("models"));
            }
        }
    }
    dirs::data_local_dir()
        .map(|d| d.join("trellis-studio").join("models"))
        .ok_or_else(|| "could not determine managed model cache directory".to_string())
}

fn normalize_abs(path: &Path) -> Result<PathBuf, String> {
    if path.exists() {
        return path.canonicalize().map_err(|e| e.to_string());
    }
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_err(|e| e.to_string())?.join(path)
    };
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => { out.pop(); }
            other => out.push(other.as_os_str()),
        }
    }
    Ok(out)
}

fn paths_equivalent(a: &Path, b: &Path) -> Result<bool, String> {
    if a.exists() && b.exists() {
        return Ok(a.canonicalize().map_err(|e| e.to_string())?
            == b.canonicalize().map_err(|e| e.to_string())?);
    }
    Ok(normalize_abs(a)? == normalize_abs(b)?)
}

fn simple_manifest_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.contains('\\') && name != "." && name != ".."
}

fn reject_symlink_root(root: &Path) -> Result<(), String> {
    match fs::symlink_metadata(root) {
        Ok(meta) if meta.file_type().is_symlink() => Err("managed model root must not be a symlink".into()),
        Ok(meta) if !meta.is_dir() => Err("managed model root must be a directory".into()),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

fn manifest_entries(root: &Path) -> Result<(Option<String>, Vec<String>), String> {
    let manifest = root.join("pixal3d-models.json");
    let meta = match fs::symlink_metadata(&manifest) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((None, Vec::new())),
        Err(e) => return Err(e.to_string()),
    };
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err("managed model manifest must be a regular file, not a symlink".into());
    }
    let v: Value = serde_json::from_str(&fs::read_to_string(&manifest).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid pixal3d-models.json: {e}"))?;
    let version = v.get("version").and_then(Value::as_str).map(str::to_owned);
    let files = v.get("files").and_then(Value::as_array).ok_or("manifest files[] missing")?;
    let mut names = Vec::with_capacity(files.len());
    for ent in files {
        let name = ent.get("name").and_then(Value::as_str).ok_or("manifest file name missing")?;
        if !simple_manifest_name(name) { return Err(format!("unsafe manifest file name: {name:?}")); }
        if !names.iter().any(|n| n == name) { names.push(name.to_owned()); }
    }
    Ok((version, names))
}

pub fn inspect(managed_root: &Path, active_models_dir: &str) -> Result<ModelCacheInfo, String> {
    reject_symlink_root(managed_root)?;
    let managed = normalize_abs(managed_root)?;
    let active_is_managed = if active_models_dir.trim().is_empty() {
        false
    } else {
        paths_equivalent(managed_root, Path::new(active_models_dir))?
    };
    let exists = managed.exists();
    let (manifest_version, names) = if exists { manifest_entries(&managed)? } else { (None, Vec::new()) };
    let mut size_bytes = 0u64;
    let mut files = 0usize;
    for name in &names {
        let p = managed.join(name);
        let meta = match fs::symlink_metadata(&p) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.to_string()),
        };
        if meta.file_type().is_symlink() || !meta.is_file() { continue; }
        size_bytes = size_bytes.saturating_add(meta.len());
        files += 1;
    }
    Ok(ModelCacheInfo {
        managed_root: managed.to_string_lossy().into_owned(),
        active_models_dir: active_models_dir.to_owned(),
        active_is_managed,
        exists,
        size_bytes,
        files,
        manifest_version,
        deletable: exists && files > 0,
        note: if active_is_managed {
            "Studio-managed model cache. Only manifest-tracked regular files are counted and deletable; untracked files are left untouched.".into()
        } else {
            "The active models directory is external. Studio will never delete files from it.".into()
        },
    })
}

pub fn delete_all(managed_root: &Path) -> Result<u64, String> {
    reject_symlink_root(managed_root)?;
    let managed = normalize_abs(managed_root)?;
    if !managed.exists() { return Ok(0); }
    let (_version, names) = manifest_entries(&managed)?;
    if names.is_empty() {
        return Err("refusing to delete: no valid pixal3d-models.json manifest is present".into());
    }

    // Validate every target before deleting anything. This avoids partial deletion
    // for traversal/symlink/non-file errors discovered late in the list.
    let mut targets = Vec::new();
    for name in names {
        if !simple_manifest_name(&name) { return Err("unsafe manifest path".into()); }
        let p = managed.join(&name);
        if p.parent() != Some(managed.as_path()) { return Err("delete target escaped managed root".into()); }
        let meta = match fs::symlink_metadata(&p) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.to_string()),
        };
        if meta.file_type().is_symlink() { return Err(format!("refusing to delete symlinked model: {name}")); }
        if !meta.is_file() { return Err(format!("refusing to delete non-file model: {name}")); }
        targets.push((p, meta.len()));
    }

    let mut removed = 0u64;
    for (p, len) in targets {
        if let Err(e) = fs::remove_file(&p) {
            return Err(format!("deleted {removed} bytes before failure removing {}: {e}", p.display()));
        }
        removed = removed.saturating_add(len);
    }
    let manifest = managed.join("pixal3d-models.json");
    if manifest.exists() {
        let meta = fs::symlink_metadata(&manifest).map_err(|e| e.to_string())?;
        if meta.file_type().is_symlink() || !meta.is_file() {
            return Err("refusing to remove non-regular manifest".into());
        }
        fs::remove_file(manifest).map_err(|e| format!("deleted {removed} bytes but could not remove manifest: {e}"))?;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("pixal3d-cache-test-{}-{n}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }
    fn write_manifest(root: &Path, names: &[&str]) {
        let files: Vec<Value> = names.iter().map(|n| serde_json::json!({"name": n})).collect();
        fs::write(root.join("pixal3d-models.json"), serde_json::json!({"version":"test","files":files}).to_string()).unwrap();
    }

    #[test]
    fn deletes_only_manifest_listed_files() {
        let root = temp_root();
        fs::write(root.join("a.gguf"), b"aaa").unwrap();
        fs::write(root.join("keep.txt"), b"keep").unwrap();
        write_manifest(&root, &["a.gguf"]);
        assert_eq!(delete_all(&root).unwrap(), 3);
        assert!(!root.join("a.gguf").exists());
        assert!(root.join("keep.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_traversal_in_manifest() {
        let root = temp_root();
        write_manifest(&root, &["../outside.txt"]);
        assert!(delete_all(&root).unwrap_err().contains("unsafe"));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_model() {
        use std::os::unix::fs::symlink;
        let root = temp_root();
        let outside = root.parent().unwrap().join(format!("outside-{}", std::process::id()));
        fs::write(&outside, b"secret").unwrap();
        symlink(&outside, root.join("a.gguf")).unwrap();
        write_manifest(&root, &["a.gguf"]);
        assert!(delete_all(&root).unwrap_err().contains("symlink"));
        assert!(outside.exists());
        let _ = fs::remove_file(outside);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_managed_root() {
        use std::os::unix::fs::symlink;
        let target = temp_root();
        write_manifest(&target, &["a.gguf"]);
        fs::write(target.join("a.gguf"), b"aaa").unwrap();
        let alias = target.parent().unwrap().join(format!("pixal3d-cache-root-alias-{}", std::process::id()));
        symlink(&target, &alias).unwrap();
        assert!(delete_all(&alias).unwrap_err().contains("must not be a symlink"));
        assert!(target.join("a.gguf").exists());
        let _ = fs::remove_file(alias);
        let _ = fs::remove_dir_all(target);
    }
}
