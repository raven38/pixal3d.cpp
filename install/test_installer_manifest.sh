#!/usr/bin/env bash
# install/install.sh の model-set 契約テスト（0.10.0: MV + SV の 2 ディレクトリ）。
#
#   bash install/test_installer_manifest.sh
#
# 実モデルも GitHub も不要。小さな 9 ファイルの MV / SV セットを作り、
#   1. --verify-models が無傷を受理し、破損・欠損・サイズ不一致を拒否する
#   2. web/app/manifest_conformance.json の全ケースで family 判定が JS/Python/C++ と一致する
#      （拒否ケースはファイルに触る前に落ちる。受理ケースは family の合う側だけ通る）
#   3. トラバーサル名・混在 manifest の拒否
#   4. --asset-base-url の実インストール経路: config.json に modelsDirSv が書かれる、
#      SV のインストール失敗で MV ディレクトリが無傷のまま・config は書かれない・.part が残らない、
#      空き容量不足はダウンロード前に落ちる
#   5. WSL2 のユーザーモード CUDA ドライバ判定（#35、nvidia-smi / uname をスタブ）:
#      580 系 → cuda12、590 以上 → cuda、非 WSL の 580 → cuda、--backend cuda 明示 + 580 → 停止
# を確認する。最後に INSTALLER_MANIFEST_OK を出す。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL="$ROOT/install/install.sh"; BASH_BIN="${BASH_BIN:-bash}"
CONF="$ROOT/web/app/manifest_conformance.json"
WORK="$(mktemp -d)"
trap 'kill "${HTTP_PID:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok   $*"; }

# --- fixtures ---------------------------------------------------------------
python3 - "$WORK" <<'PY'
import hashlib, json, os, sys
work = sys.argv[1]
def roles(fam):
    return [('dinov3.gguf','image_encoder'), ('pixal3d_naf.gguf','naf'),
            (f'pixal3d_ss_flow_{fam}.gguf','ss_flow'), ('ss_dec.gguf','ss_decoder'),
            (f'pixal3d_shape_flow_512_{fam}.gguf','shape_flow_512'), ('shape_dec.gguf','shape_decoder'),
            (f'pixal3d_shape_flow_1024_{fam}.gguf','shape_flow_1024'),
            (f'pixal3d_tex_flow_1024_{fam}.gguf','texture_flow_1024'), ('tex_dec.gguf','texture_decoder')]
def make(fam, explicit):
    d = os.path.join(work, f'src-{fam}'); os.makedirs(d, exist_ok=True)
    files = []
    for i, (name, role) in enumerate(roles(fam)):
        data = f'installer-test-{fam}-{i}-{role}'.encode()   # 同名でも MV/SV で中身が違う
        open(os.path.join(d, name), 'wb').write(data)
        files.append({'name': name, 'role': role, 'required': True,
                      'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    m = {'schema_version': 1, 'model_set': f'pixal3d-test-{fam}', 'version': 'v1', 'files': files}
    if explicit: m['model_family'] = fam
    json.dump(m, open(os.path.join(d, 'pixal3d-models.json'), 'w'), indent=2)
make('mv', False)   # 公開 MV manifest と同じく model_family 省略
make('sv', True)
PY
MV_SRC="$WORK/src-mv"; SV_SRC="$WORK/src-sv"
cp -R "$MV_SRC" "$WORK/mv"; cp -R "$SV_SRC" "$WORK/sv"
MV="$WORK/mv"; SV="$WORK/sv"
EMPTY="$WORK/empty"; mkdir -p "$EMPTY"

run() {  # 期待 rc、grep パターン（stderr+stdout）、引数...
  local want_rc="$1" pattern="$2"; shift 2
  local out rc=0
  out="$("$BASH_BIN" "$INSTALL" "$@" 2>&1)" || rc=$?
  if [ "$rc" != "$want_rc" ]; then echo "$out" >&2; fail "rc=$rc want=$want_rc: install.sh $*"; fi
  if [ -n "$pattern" ] && ! grep -q -- "$pattern" <<<"$out"; then echo "$out" >&2; fail "output lacks '$pattern': install.sh $*"; fi
  LAST_OUT="$out"
}

# --- 1. verify-only ---------------------------------------------------------
run 0 "model set OK" --verify-models --models-dir "$MV"
pass "MV verify-only accepts an intact set"
run 0 "pixal3d-test-sv v1 (sv)" --verify-models --models-dir "$MV" --models-dir-sv "$SV"
grep -q "pixal3d-test-mv v1 (mv)" <<<"$LAST_OUT" || fail "MV was not verified alongside SV"
pass "verify-only checks both directories"

printf 'x' >> "$SV/tex_dec.gguf"
run 1 "does not verify: tex_dec.gguf" --verify-models --models-dir "$MV" --models-dir-sv "$SV"
grep -q "pixal3d-test-mv v1 (mv)" <<<"$LAST_OUT" || fail "MV should still verify before the SV failure"
pass "SV corruption is rejected while MV still verifies"
cp "$SV_SRC/tex_dec.gguf" "$SV/tex_dec.gguf"

rm "$SV/pixal3d_naf.gguf"
run 1 "does not verify: pixal3d_naf.gguf" --verify-models --models-dir "$MV" --models-dir-sv "$SV"
pass "SV missing file is rejected"
cp "$SV_SRC/pixal3d_naf.gguf" "$SV/pixal3d_naf.gguf"

# サイズ一致・内容違い（SHA だけが違う）
python3 - "$SV/ss_dec.gguf" <<'PY'
import sys; p=sys.argv[1]; b=bytearray(open(p,'rb').read()); b[0]^=1; open(p,'wb').write(b)
PY
run 1 "does not verify: ss_dec.gguf" --verify-models --models-dir "$MV" --models-dir-sv "$SV"
pass "same-size content change is rejected"
cp "$SV_SRC/ss_dec.gguf" "$SV/ss_dec.gguf"

# MV セットの manifest を SV ディレクトリに置く（family 取り違え）
run 1 "expects the sv set" --verify-models --models-dir "$MV" --models-dir-sv "$SV" --model-manifest-sv "$MV/pixal3d-models.json"
pass "an MV manifest is rejected for the SV directory"
run 1 "expects the mv set" --verify-models --models-dir "$MV" --model-manifest "$SV/pixal3d-models.json"
pass "an SV manifest is rejected for the MV directory"

# トラバーサル名
python3 - "$MV/pixal3d-models.json" "$WORK/traversal.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m['files'][0]['name'] = '../evil.gguf'; json.dump(m, open(sys.argv[2], 'w'))
PY
run 1 "" --verify-models --models-dir "$MV" --model-manifest "$WORK/traversal.json"
grep -q "does not verify" <<<"$LAST_OUT" && fail "traversal name reached the file stage"
pass "traversal name is rejected before any file is read"

# 混在 manifest（ss_flow だけ _mv、他 _sv）
python3 - "$SV/pixal3d-models.json" "$WORK/mixed.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m.pop('model_family', None)
for f in m['files']:
    if f['role'] == 'ss_flow': f['name'] = 'pixal3d_ss_flow_mv.gguf'
json.dump(m, open(sys.argv[2], 'w'))
PY
run 1 "" --verify-models --models-dir "$MV" --models-dir-sv "$SV" --model-manifest-sv "$WORK/mixed.json"
grep -q "does not verify" <<<"$LAST_OUT" && fail "mixed manifest reached the file stage"
pass "mixed-family manifest is rejected"

# --- 2. conformance vectors (JS/Python/C++ と同じ判定) -------------------------
n_cases=0
while IFS=$'\t' read -r id family; do
  n_cases=$((n_cases+1))
  python3 - "$CONF" "$id" "$WORK/case.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c = [x for x in d['cases'] if x['id'] == sys.argv[2]][0]
json.dump(c['manifest'], open(sys.argv[3], 'w'))
PY
  # MV 側として渡す
  rc=0; out="$("$BASH_BIN" "$INSTALL" --verify-models --models-dir "$EMPTY" --model-manifest "$WORK/case.json" 2>&1)" || rc=$?
  # SV 側として渡す（MV は無傷のセットで通す）
  rc_sv=0; out_sv="$("$BASH_BIN" "$INSTALL" --verify-models --models-dir "$MV" --models-dir-sv "$EMPTY" --model-manifest-sv "$WORK/case.json" 2>&1)" || rc_sv=$?
  case "$family" in
    null)
      [ "$rc" != 0 ] && [ "$rc_sv" != 0 ] || fail "$id: expected rejection (rc=$rc rc_sv=$rc_sv)"
      grep -q "does not verify" <<<"$out$out_sv" && { echo "$out"; echo "$out_sv"; fail "$id: rejected at the file stage, not the manifest stage"; }
      ;;
    mv)
      # manifest は通り、ファイルが無いので file stage で落ちる
      grep -q "does not verify" <<<"$out" || { echo "$out"; fail "$id: MV manifest should pass the manifest stage"; }
      grep -q "expects the sv set" <<<"$out_sv" || { echo "$out_sv"; fail "$id: MV manifest should be refused as SV"; }
      ;;
    sv)
      grep -q "does not verify" <<<"$out_sv" || { echo "$out_sv"; fail "$id: SV manifest should pass the manifest stage"; }
      grep -q "expects the mv set" <<<"$out" || { echo "$out"; fail "$id: SV manifest should be refused as MV"; }
      ;;
    *) fail "$id: unknown expected family '$family'";;
  esac
done < <(python3 -c "
import json; d=json.load(open('$CONF'))
for c in d['cases']: print(c['id'] + '\t' + (c['family'] or 'null'))
")
[ "$n_cases" -ge 18 ] || fail "only $n_cases conformance cases ran"
pass "conformance vectors: $n_cases cases agree with the shared contract"

# --- 3. install path over --asset-base-url ------------------------------------
ASSETS="$WORK/assets"; mkdir -p "$ASSETS/bundle"
case "$(uname -s)" in
  Darwin) BUNDLE=trellis-metal-macos-arm64.tar.gz; BACKEND_ARGS=();;
  *)      BUNDLE=trellis-vulkan-linux-x64.tar.gz; BACKEND_ARGS=(--backend vulkan);;
esac
printf '#!/bin/sh\necho stub\n' > "$ASSETS/bundle/trellis-server"; chmod +x "$ASSETS/bundle/trellis-server"
tar -C "$ASSETS/bundle" -czf "$ASSETS/$BUNDLE" trellis-server
cp -R "$MV_SRC" "$ASSETS/mv"; cp -R "$SV_SRC" "$ASSETS/sv"
PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
(cd "$ASSETS" && python3 -m http.server --bind 127.0.0.1 "$PORT" >/dev/null 2>&1) &
HTTP_PID=$!
for i in $(seq 1 40); do curl -fsS "http://127.0.0.1:$PORT/$BUNDLE" -o /dev/null 2>/dev/null && break; sleep 0.25; done
BASE="http://127.0.0.1:$PORT"
DEST="$WORK/dest"; CFG="$WORK/cfg"
COMMON=(--asset-base-url "$BASE" "${BACKEND_ARGS[@]}" --skip-app -y --dest "$DEST" --config-dir "$CFG"
        --model-manifest "$ASSETS/mv/pixal3d-models.json" --model-base-url "$BASE/mv")

# 3a. MV + SV の両方をインストール
run 0 "model set verified" "${COMMON[@]}" --model-manifest-sv "$ASSETS/sv/pixal3d-models.json" --model-base-url-sv "$BASE/sv"
python3 - "$CFG/config.json" "$CFG/release.json" "$DEST" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1])); rec = json.load(open(sys.argv[2])); dest = sys.argv[3]
assert cfg['modelsDir'] == os.path.join(dest, 'models'), cfg
assert cfg['modelsDirSv'] == os.path.join(dest, 'models-sv'), cfg
assert rec['model_set']['model_set'] == 'pixal3d-test-mv' and rec['model_set']['verified'] is True, rec
assert rec['model_set_sv']['model_set'] == 'pixal3d-test-sv' and rec['model_set_sv']['verified'] is True, rec
assert rec['models_dir_sv'] == cfg['modelsDirSv'], rec
for d in ('models', 'models-sv'):
    assert os.path.exists(os.path.join(dest, d, 'pixal3d-models.json')), d
assert open(os.path.join(dest, 'models', 'dinov3.gguf'), 'rb').read() != open(os.path.join(dest, 'models-sv', 'dinov3.gguf'), 'rb').read()
print('CONFIG_OK', cfg['modelsDirSv'])
PY
run 0 "pixal3d-test-sv v1 (sv)" --verify-models --models-dir "$DEST/models" --models-dir-sv "$DEST/models-sv"
pass "install writes modelsDirSv and both sets verify afterwards"

# 3b. 同じディレクトリを両方に指定するのは拒否
run 1 "must differ from the MV models dir" "${COMMON[@]}" --models-dir-sv "$DEST/models" --model-manifest-sv "$ASSETS/sv/pixal3d-models.json" --model-base-url-sv "$BASE/sv"
pass "SV into the MV directory is refused"

# 3c. SV の失敗が MV を壊さない: SV ソースを破損させ、config を消してから再実行
rm -rf "$DEST/models-sv" "$CFG"
MV_SHA_BEFORE="$(cat "$DEST/models/"*.gguf | shasum -a 256 | cut -d' ' -f1)"
printf 'broken' > "$ASSETS/sv/tex_dec.gguf"
run 1 "model file does not match the manifest: tex_dec.gguf" "${COMMON[@]}" --model-manifest-sv "$ASSETS/sv/pixal3d-models.json" --model-base-url-sv "$BASE/sv"
[ ! -e "$CFG/config.json" ] || fail "config.json was written although the SV install failed"
[ "$(cat "$DEST/models/"*.gguf | shasum -a 256 | cut -d' ' -f1)" = "$MV_SHA_BEFORE" ] || fail "MV files changed after the SV failure"
[ -z "$(find "$DEST" -name '*.part')" ] || fail ".part files left behind: $(find "$DEST" -name '*.part')"
[ ! -e "$DEST/models-sv/tex_dec.gguf" ] || fail "the mismatching SV file was kept"
[ ! -e "$DEST/models-sv/pixal3d-models.json" ] || fail "SV manifest was copied although the set is incomplete"
run 0 "model set OK" --verify-models --models-dir "$DEST/models"
pass "a failing SV install leaves the MV set intact, writes no config and no .part"
cp "$SV_SRC/tex_dec.gguf" "$ASSETS/sv/tex_dec.gguf"

# 3d. SV 無しの再実行では modelsDirSv が空（未導入）になる
rm -rf "$DEST/models-sv" "$CFG"
run 0 "model set verified" "${COMMON[@]}"
python3 -c "import json,sys; c=json.load(open('$CFG/config.json')); assert c['modelsDirSv']=='', c; print('SV_EMPTY_OK')"
pass "without an SV set modelsDirSv is empty"

# 3e. 空き容量不足はダウンロード前に落ちる（size_bytes 1e15 の manifest）
python3 - "$ASSETS/sv/pixal3d-models.json" "$WORK/huge.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m['files'][0]['size_bytes'] = 10**15; json.dump(m, open(sys.argv[2], 'w'))
PY
rm -rf "$DEST/models-sv" "$CFG"
run 1 "not enough free space" "${COMMON[@]}" --model-manifest-sv "$WORK/huge.json" --model-base-url-sv "$BASE/sv"
[ -z "$(find "$DEST" -name '*.part')" ] || fail ".part left after the free-space refusal"
[ ! -e "$DEST/models-sv/pixal3d_naf.gguf" ] || fail "a download happened despite the free-space refusal"
pass "free-space precheck refuses before downloading"

# --- 5. WSL2 user-mode CUDA driver (#35) ---------------------------------------
# uname / nvidia-smi を PATH の先頭でスタブする。uname は -s/-r/-m だけ差し替え、他は本物へ。
STUB="$WORK/stubbin"; mkdir -p "$STUB"
REAL_UNAME="$(command -v uname)"
cat > "$STUB/uname" <<EOF
#!/bin/sh
case "\$1" in
  -s) echo Linux;;
  -r) echo "\${STUB_KERNEL:-6.6.87.1-microsoft-standard-WSL2}";;
  -m) echo x86_64;;
  *) exec "$REAL_UNAME" "\$@";;
esac
EOF
cat > "$STUB/nvidia-smi" <<'EOF'
#!/bin/sh
case "$1" in
  -L) echo "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-stub)";;
  --query-gpu=*) echo "8.9";;
  *) printf '%s\n' "Sun Sep 20 13:35:50 2026" \
       "+-----------------------------------------------------------------------------------------+" \
       "| NVIDIA-SMI ${STUB_SMI:-580.178.04}             Driver Version: 591.86         CUDA Version: 13.1     |";;
esac
EOF
chmod +x "$STUB/uname" "$STUB/nvidia-smi"
# cuda / cuda12 の Linux バンドル（スタブ）を asset サーバに置く
for b in trellis-cuda-linux-x64.tar.gz trellis-cuda12-linux-x64.tar.gz; do
  tar -C "$ASSETS/bundle" -czf "$ASSETS/$b" trellis-server
done
run_stub() {  # 期待 rc、grep パターン、STUB_SMI、STUB_KERNEL、引数...
  local want_rc="$1" pattern="$2" smi="$3" kernel="$4"; shift 4
  local out rc=0
  out="$(PATH="$STUB:$PATH" STUB_SMI="$smi" STUB_KERNEL="$kernel" "$BASH_BIN" "$INSTALL" "$@" 2>&1)" || rc=$?
  if [ "$rc" != "$want_rc" ]; then echo "$out" >&2; fail "rc=$rc want=$want_rc: [smi=$smi kernel=$kernel] install.sh $*"; fi
  if [ -n "$pattern" ] && ! grep -q -- "$pattern" <<<"$out"; then echo "$out" >&2; fail "output lacks '$pattern': [smi=$smi] install.sh $*"; fi
  LAST_OUT="$out"
}
WSL_KERNEL="6.6.87.1-microsoft-standard-WSL2"; NATIVE_KERNEL="6.8.0-45-generic"
WSL_COMMON=(--asset-base-url "$BASE" --skip-app --skip-models -y --dest "$WORK/wsl-dest" --config-dir "$WORK/wsl-cfg")
backend_in_config() { python3 -c "import json; print(json.load(open('$WORK/wsl-cfg/config.json'))['backend'])"; }

rm -rf "$WORK/wsl-dest" "$WORK/wsl-cfg"
run_stub 0 "auto-detected backend: .*cuda12" 580.178.04 "$WSL_KERNEL" "${WSL_COMMON[@]}"
grep -q "R590" <<<"$LAST_OUT" || fail "cuda12 fallback must name the required driver branch"
[ "$(backend_in_config)" = cuda12 ] || fail "config.json backend should be cuda12, got $(backend_in_config)"
pass "WSL2 + user-mode driver 580.x auto-selects cuda12 (with the reason)"

rm -rf "$WORK/wsl-dest" "$WORK/wsl-cfg"
run_stub 0 "auto-detected backend: .*cuda" 590.44.01 "$WSL_KERNEL" "${WSL_COMMON[@]}"
[ "$(backend_in_config)" = cuda ] || fail "config.json backend should be cuda for a 590 driver, got $(backend_in_config)"
pass "WSL2 + user-mode driver 590.x keeps cuda"

rm -rf "$WORK/wsl-dest" "$WORK/wsl-cfg"
run_stub 0 "auto-detected backend: .*cuda" 580.178.04 "$NATIVE_KERNEL" "${WSL_COMMON[@]}"
[ "$(backend_in_config)" = cuda ] || fail "a non-WSL 580 driver must not be downgraded, got $(backend_in_config)"
pass "native Linux + 580.x is left on cuda (WSL2-only rule)"

rm -rf "$WORK/wsl-dest" "$WORK/wsl-cfg"
run_stub 1 "Use --backend cuda12" 580.178.04 "$WSL_KERNEL" "${WSL_COMMON[@]}" --backend cuda
[ ! -e "$WORK/wsl-cfg/config.json" ] || fail "config.json must not be written when --backend cuda is refused"
pass "WSL2 + 580.x + explicit --backend cuda stops before installing"

rm -rf "$WORK/wsl-dest" "$WORK/wsl-cfg"
run_stub 0 "backend (forced): .*cuda12" 580.178.04 "$WSL_KERNEL" "${WSL_COMMON[@]}" --backend cuda12
pass "WSL2 + 580.x + explicit --backend cuda12 installs"

echo INSTALLER_MANIFEST_OK
