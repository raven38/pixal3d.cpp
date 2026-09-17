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

# flow 重みは model_family（mv = multiview, sv = single-view）ごとに別ファイル。
# decoder / image_encoder / naf は両 family で共通（同じバイト列を共有する）。
MODEL_FAMILIES = ("mv", "sv")
DEFAULT_MODEL_FAMILY = "mv"


def model_files(family: str = DEFAULT_MODEL_FAMILY) -> dict:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model_family: {family!r}")
    return {
        "dinov3.gguf": {"role": "image_encoder", "required": True},
        "pixal3d_naf.gguf": {"role": "naf", "required": True},
        f"pixal3d_ss_flow_{family}.gguf": {"role": "ss_flow", "required": True},
        "ss_dec.gguf": {"role": "ss_decoder", "required": True},
        f"pixal3d_shape_flow_512_{family}.gguf": {"role": "shape_flow_512", "required": True},
        "shape_dec.gguf": {"role": "shape_decoder", "required": True},
        f"pixal3d_shape_flow_1024_{family}.gguf": {"role": "shape_flow_1024", "required": True},
        f"pixal3d_tex_flow_1024_{family}.gguf": {"role": "texture_flow_1024", "required": True},
        "tex_dec.gguf": {"role": "texture_decoder", "required": True},
    }


# 既存の MV manifest との互換のため MODEL_FILES は mv 集合を指したままにする。
MODEL_FILES = model_files(DEFAULT_MODEL_FAMILY)

ROOT_KEYS = {"schema_version", "model_set", "version", "model_family", "source", "files"}
FILE_KEYS = {"name", "role", "required", "size_bytes", "sha256"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_family(files: list) -> str | None:
    """flow 4 role のファイル名末尾から family を推定する。混在・不明は None。"""
    flow_roles = {"ss_flow", "shape_flow_512", "shape_flow_1024", "texture_flow_1024"}
    names = [ent.get("name") for ent in files
             if isinstance(ent, dict) and ent.get("role") in flow_roles and isinstance(ent.get("name"), str)]
    if not names:
        return None
    for fam in MODEL_FAMILIES:
        if all(n.endswith(f"_{fam}.gguf") for n in names):
            return fam
    return None


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

    # family の決定は web/app/single_view.js modelFamilyForManifest と同じ規則:
    # flow のファイル名 (*_mv.gguf / *_sv.gguf) から推定し、model_family が明示されて
    # いればそれと一致しなければならない。混在・不明・不一致は拒否する。
    explicit = m.get("model_family")
    if explicit is not None and explicit not in MODEL_FAMILIES:
        errors.append("model_family must be one of: " + ", ".join(MODEL_FAMILIES))
        explicit = None
    inferred = infer_family(files)
    if explicit is not None and inferred is not None and explicit != inferred:
        errors.append(f"model_family {explicit} does not match the flow file names ({inferred})")
    family = explicit or inferred or DEFAULT_MODEL_FAMILY

    expected = model_files(family)
    seen_names: set[str] = set()
    seen_roles: set[str] = set()
    expected_roles = {spec["role"] for spec in expected.values() if spec["required"]}
    expected_names = {name for name, spec in expected.items() if spec["required"]}
    role_of_name = {name: spec["role"] for name, spec in expected.items()}
    name_of_role = {spec["role"]: name for name, spec in expected.items()}
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
        # name と role の対応を固定する（role の入れ替えで別種の GGUF を別段へ渡させない）。
        if isinstance(name, str) and isinstance(role, str):
            if name in role_of_name and role_of_name[name] != role:
                errors.append(f"{p}: {name} must have role {role_of_name[name]}, not {role}")
            elif name not in role_of_name and role in name_of_role:
                errors.append(f"{p}: role {role} must be {name_of_role[role]}, not {name}")
            if name in role_of_name and ent.get("required") is not True:
                errors.append(f"{p}: {name} must be required")
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


def generate(root: Path, version: str, model_set: str, source: str | None,
             family: str = DEFAULT_MODEL_FAMILY) -> dict:
    files = []
    missing = []
    for name, spec in model_files(family).items():
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
    if family != DEFAULT_MODEL_FAMILY:
        # 既存の mv manifest を変えないよう、mv 以外のときだけ明示する（省略時はファイル名から推定）。
        out["model_family"] = family
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

    # sv family: _sv の flow 名が要求され、mv の名前は満たさない。family の混在は拒否する。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, name in enumerate(model_files("sv")):
            (root / name).write_bytes((name + str(i)).encode())
        sv = generate(root, "test", "pixal3d-sv-test", None, family="sv")
        assert sv["model_family"] == "sv"
        assert verify(root, sv) == []
        assert all(not n.endswith("_mv.gguf") for n in (f["name"] for f in sv["files"]))
        bad = json.loads(json.dumps(sv))
        del bad["model_family"]  # 省略時はブラウザと同じくファイル名から sv と推定される
        assert validate_manifest_shape(bad) == []
        bad = json.loads(json.dumps(sv))
        bad["model_family"] = "xx"
        assert any("model_family must be one of" in e for e in validate_manifest_shape(bad))
        bad = json.loads(json.dumps(m))
        bad["model_family"] = "sv"  # mv のファイル群に sv と書いても通らない
        assert any("does not match the flow file names" in e for e in validate_manifest_shape(bad))
        bad = json.loads(json.dumps(sv))
        bad["model_family"] = "mv"
        assert any("does not match the flow file names" in e for e in validate_manifest_shape(bad))
        # 混在（ss_flow だけ mv）は family 不明 → 既定 mv の集合を満たさず拒否
        bad = json.loads(json.dumps(sv))
        bad["files"][2]["name"] = "pixal3d_ss_flow_mv.gguf"
        assert any("missing required model files" in e for e in validate_manifest_shape(bad))
        # role の入れ替え（dinov3 <-> ss_flow）は名前と role の対応違反として拒否
        bad = json.loads(json.dumps(sv))
        bad["files"][0]["role"], bad["files"][2]["role"] = bad["files"][2]["role"], bad["files"][0]["role"]
        assert any("must have role" in e for e in validate_manifest_shape(bad))
        bad = json.loads(json.dumps(sv))
        bad["files"][0]["required"] = False
        assert any("must be required" in e for e in validate_manifest_shape(bad))
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
    bad = json.loads(json.dumps(manifest))
    bad["model_family"] = "xx"
    try:
        jsonschema.validate(instance=bad, schema=schema)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("schema unexpectedly accepted an unknown model_family")
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
    g.add_argument("--model-family", choices=MODEL_FAMILIES, default=DEFAULT_MODEL_FAMILY)
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
        m = generate(a.models_dir, a.version, a.model_set, a.source, family=a.model_family)
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
