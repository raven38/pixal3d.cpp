#!/usr/bin/env python3
"""Generate/verify Pixal3D GGUF model-set manifests.

The manifest is deliberately frontend-agnostic so Desktop and Web can share the
same version/size/SHA256 contract. It never downloads or deletes files.

Schema v1 intentionally describes the full textured production set. Semantic
constraints that JSON Schema cannot express for an array (unique names and the
complete required-role set) are enforced by validate_manifest_shape().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

MODEL_FILES = {
    "dinov3.gguf": {"role": "image_encoder", "required": True},
    "pixal3d_naf.gguf": {"role": "naf", "required": True},
    "pixal3d_ss_flow_mv.gguf": {"role": "ss_flow", "required": True},
    "ss_dec.gguf": {"role": "ss_decoder", "required": True},
    "pixal3d_shape_flow_512_mv.gguf": {"role": "shape_flow_512", "required": True},
    "shape_dec.gguf": {"role": "shape_decoder", "required": True},
    "pixal3d_shape_flow_1024_mv.gguf": {"role": "shape_flow_1024", "required": True},
    "pixal3d_tex_flow_1024_mv.gguf": {"role": "texture_flow_1024", "required": True},
    "tex_dec.gguf": {"role": "texture_decoder", "required": True},
}

ROOT_KEYS = {"schema_version", "model_set", "version", "source", "files"}
FILE_KEYS = {"name", "role", "required", "size_bytes", "sha256"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest_shape(m: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(m, dict):
        return ["manifest must be an object"]

    unknown_root = sorted(set(m) - ROOT_KEYS)
    if unknown_root:
        errors.append("unknown top-level keys: " + ", ".join(unknown_root))
    if m.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(m.get("model_set"), str) or not m["model_set"]:
        errors.append("model_set missing")
    if not isinstance(m.get("version"), str) or not m["version"]:
        errors.append("version missing")
    if "source" in m and not isinstance(m["source"], str):
        errors.append("source must be a string")

    files = m.get("files")
    if not isinstance(files, list) or not files:
        return errors + ["files must be a non-empty array"]

    seen_names: set[str] = set()
    seen_roles: set[str] = set()
    expected_roles = {spec["role"] for spec in MODEL_FILES.values() if spec["required"]}
    expected_names = {name for name, spec in MODEL_FILES.items() if spec["required"]}
    for i, ent in enumerate(files):
        p = f"files[{i}]"
        if not isinstance(ent, dict):
            errors.append(f"{p} must be object")
            continue
        unknown_file = sorted(set(ent) - FILE_KEYS)
        if unknown_file:
            errors.append(f"{p} has unknown keys: " + ", ".join(unknown_file))
        name = ent.get("name")
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
            errors.append(f"{p}.name invalid")
        elif name in seen_names:
            errors.append(f"duplicate file name: {name}")
        else:
            seen_names.add(name)
        role = ent.get("role")
        if not isinstance(role, str) or not role:
            errors.append(f"{p}.role invalid")
        elif role in seen_roles:
            errors.append(f"duplicate role: {role}")
        else:
            seen_roles.add(role)
        if not isinstance(ent.get("required"), bool):
            errors.append(f"{p}.required must be bool")
        if not isinstance(ent.get("size_bytes"), int) or ent["size_bytes"] <= 0:
            errors.append(f"{p}.size_bytes invalid")
        digest = ent.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(f"{p}.sha256 invalid")

    missing_names = sorted(expected_names - seen_names)
    missing_roles = sorted(expected_roles - seen_roles)
    if missing_names:
        errors.append("manifest missing required model files: " + ", ".join(missing_names))
    if missing_roles:
        errors.append("manifest missing required roles: " + ", ".join(missing_roles))
    return errors


def generate(root: Path, version: str, model_set: str, source: str | None) -> dict:
    files = []
    missing = []
    for name, spec in MODEL_FILES.items():
        p = root / name
        if not p.is_file():
            if spec["required"]:
                missing.append(name)
            continue
        files.append({
            "name": name,
            "role": spec["role"],
            "required": spec["required"],
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    if missing:
        raise SystemExit("missing required model files: " + ", ".join(missing))
    out = {"schema_version": 1, "model_set": model_set, "version": version, "files": files}
    if source:
        out["source"] = source
    return out


def verify(root: Path, manifest: dict) -> list[str]:
    errors = validate_manifest_shape(manifest)
    if errors:
        return errors
    for ent in manifest["files"]:
        p = root / ent["name"]
        if not p.is_file():
            if ent["required"]:
                errors.append(f"missing: {ent['name']}")
            continue
        size = p.stat().st_size
        if size != ent["size_bytes"]:
            errors.append(f"size mismatch {ent['name']}: {size} != {ent['size_bytes']}")
            continue
        digest = sha256_file(p)
        if digest != ent["sha256"]:
            errors.append(f"sha256 mismatch {ent['name']}")
    return errors


def self_test() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, name in enumerate(MODEL_FILES):
            (root / name).write_bytes((name + str(i)).encode())
        m = generate(root, "test", "pixal3d-test", None)
        assert verify(root, m) == []

        bad = json.loads(json.dumps(m))
        bad["unexpected"] = True
        assert any("unknown top-level" in e for e in validate_manifest_shape(bad))
        bad = json.loads(json.dumps(m))
        bad["files"][0]["unexpected"] = True
        assert any("unknown keys" in e for e in validate_manifest_shape(bad))
        bad = json.loads(json.dumps(m))
        bad["files"].pop()
        assert any("missing required" in e for e in verify(root, bad))
        bad = json.loads(json.dumps(m))
        bad["files"][1]["name"] = bad["files"][0]["name"]
        assert any("duplicate file name" in e for e in validate_manifest_shape(bad))
        bad = json.loads(json.dumps(m))
        bad["files"][0]["name"] = ".."
        assert any("name invalid" in e for e in validate_manifest_shape(bad))

        (root / "dinov3.gguf").write_bytes(b"corrupt")
        errs = verify(root, m)
        assert any("size mismatch dinov3.gguf" in e or "sha256 mismatch dinov3.gguf" in e for e in errs)
    print("MODEL_MANIFEST_SELF_TEST_OK")
    return m


def schema_test(schema_path: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise SystemExit("schema-test requires the 'jsonschema' package") from exc
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest = self_test()
    jsonschema.validate(instance=manifest, schema=schema)

    bad = json.loads(json.dumps(manifest))
    bad["unexpected"] = True
    try:
        jsonschema.validate(instance=bad, schema=schema)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("schema unexpectedly accepted an unknown top-level property")
    for bad_name in (".", ".."):
        bad = json.loads(json.dumps(manifest))
        bad["files"][0]["name"] = bad_name
        try:
            jsonschema.validate(instance=bad, schema=schema)
        except jsonschema.ValidationError:
            pass
        else:
            raise AssertionError(f"schema unexpectedly accepted unsafe name {bad_name!r}")
    print("MODEL_MANIFEST_SCHEMA_TEST_OK")


def check_committed(models_root: Path, doc: Path) -> int:
    """Validate every committed release manifest and the digests recorded in the docs.

    実重みを持たない CI 用のゲート。manifest の形状・schema・9 role の充足と、
    release checklist に書かれた SHA256 が実ファイルと一致することだけを見る。
    """
    manifests = sorted(models_root.glob("*/pixal3d-models.json"))
    if not manifests:
        print(f"no committed manifest under {models_root}/*/pixal3d-models.json", file=sys.stderr)
        return 2
    failed = False
    doc_text = doc.read_text() if doc.is_file() else ""
    for mp in manifests:
        m = json.loads(mp.read_text())
        errors = validate_manifest_shape(m)
        expected_dir = f"{m.get('model_set')}-{m.get('version')}"
        if mp.parent.name != expected_dir:
            errors.append(f"directory {mp.parent.name} does not match {expected_dir}")
        digest = sha256_file(mp)
        if doc_text and digest not in doc_text:
            errors.append(f"sha256 {digest} is not recorded in {doc}")
        if errors:
            failed = True
            for e in errors:
                print(f"{mp}: {e}", file=sys.stderr)
        else:
            print(f"MODEL_MANIFEST_COMMITTED_OK {mp} {m['model_set']} {m['version']} {digest}")
    return 2 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("models_dir", type=Path)
    g.add_argument("--version", required=True)
    g.add_argument("--model-set", default="pixal3d-q8_0")
    g.add_argument("--source")
    g.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("models_dir", type=Path)
    v.add_argument("manifest", type=Path)
    c = sub.add_parser("check-committed")
    c.add_argument("models_root", type=Path, nargs="?", default=Path("models"))
    c.add_argument("--doc", type=Path, default=Path("docs/PIXAL3D_RELEASE_CHECKLIST.md"))
    sub.add_parser("self-test")
    s = sub.add_parser("schema-test")
    s.add_argument("schema", type=Path, nargs="?", default=Path("models/pixal3d-model-set.schema.json"))
    a = ap.parse_args()

    if a.cmd == "self-test":
        self_test()
        return 0
    if a.cmd == "check-committed":
        return check_committed(a.models_root, a.doc)
    if a.cmd == "schema-test":
        schema_test(a.schema)
        return 0
    if a.cmd == "generate":
        m = generate(a.models_dir, a.version, a.model_set, a.source)
        a.output.write_text(json.dumps(m, indent=2) + "\n")
        print(a.output)
        return 0

    m = json.loads(a.manifest.read_text())
    errors = verify(a.models_dir, m)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 2
    print(f"MODEL_MANIFEST_OK {m['model_set']} {m['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
