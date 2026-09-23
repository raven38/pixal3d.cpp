#!/usr/bin/env bash
# trellis-server の HTTP 契約テスト（設計書 A7）。モデル重みは要らない。
#
#   tests/server_contract.sh ./build/trellis-server
#
# 検証すること:
#   - GET /health は "ok" のまま（Studio が文字列比較している。設計書 D6）
#   - GET /capabilities: configured / available / reason の判定（未設定・manifest 契約違反・
#     ファイル欠損・サイズ不一致・family 取り違え）と busy / completed
#   - POST /generate-sv: whitelist、strict な数値、PNG のみ、1024 のみ、SV 未導入は 503
#   - POST /generate-mv: seed=abc が 400 になる（0.10.0 の挙動変更。以前は黙って 0）
#   - POST /generate-trellis2-mv: 2..8 image contract / fusion / camera-field separation /
#     decompression-bomb IHDR preflight（/generate-sv と同じ 64 Mpixel 上限、全 num_images 枚）/
#     duplicate multipart field rejection
#   - 生成が走ると busy=true → 終了で busy=false かつ completed が 1 増える
#     （manifest どおりのサイズの sparse ファイルを置いて available=true にし、
#      GGUF ロードで失敗させる。500 で返り、それでも completed は増えること）
set -u

SERVER="${1:?usage: server_contract.sh <trellis-server binary>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${SERVER_CONTRACT_PORT:-18431}"
U="http://127.0.0.1:${PORT}"
WORK="$(mktemp -d)"
SRV_PID=""
# kill 0 はプロセスグループ全体（呼び出し元の shell も）を殺すので、PID が空のときは呼ばない。
cleanup() { [ -n "${SRV_PID}" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT

failures=0
pass() { printf 'ok   %s\n' "$1"; }
fail() { printf 'FAIL %s\n' "$1"; failures=$((failures + 1)); }
expect_status() {  # expect_status <label> <want> <curl args...>
    local label="$1" want="$2"; shift 2
    local body code
    body="$(curl -s -o "$WORK/body" -w '%{http_code}' "$@")"
    code="$body"
    body="$(cat "$WORK/body")"
    if [ "$code" = "$want" ]; then pass "$label [$code] ${body:0:90}"; else fail "$label: want $want got $code: ${body:0:120}"; fi
}
cap() { curl -s "$U/capabilities"; }
jq_py() {  # jq_py <json> <python expr over d>
    # eval の対象はこのスクリプトに書いた式だけ（サーバ応答は json.loads でデータとして読む）。
    python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2"
}

# ---- fixtures ------------------------------------------------------------
MV="$WORK/mv"; SV="$WORK/sv"; BAD="$WORK/bad"
mkdir -p "$MV" "$SV" "$BAD"
cp "$ROOT/models/pixal3d-f16-v1/pixal3d-models.json" "$MV/"
cp "$ROOT/models/pixal3d-sv-q8_0-v1/pixal3d-models.json" "$SV/"

# 小さな pre-matted PNG（Python 標準ライブラリだけで作る）
python3 - "$WORK/prematted.png" <<'PY'
import struct, sys, zlib
w, h = 64, 64
rows = []
for y in range(h):
    row = bytearray([0])
    for x in range(w):
        inside = 16 <= x < 48 and 16 <= y < 48
        row += bytes([200, 120, 80, 255]) if inside else bytes([0, 0, 0, 0])
    rows.append(bytes(row))
raw = b"".join(rows)
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
IMG="$WORK/prematted.png"

# decompression-bomb 再現（issue #66 反証レビュー）: IHDR は 40000x40000
# （16 億画素超）を主張するが、ファイル自体は数百バイトしかない。
python3 - "$WORK/bomb.png" <<'PY'
import struct, sys, zlib
w, h = 40000, 40000
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"\x00" * 100) + chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
BOMB="$WORK/bomb.png"

# ---- 1. MV/SV とも manifest だけ（ファイル欠損）で起動 ----------------------
"$SERVER" --models "$MV" --models-sv "$SV" --gpu -1 --port "$PORT" >"$WORK/server1.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 50); do curl -s "$U/health" >/dev/null 2>&1 && break; sleep 0.1; done

[ "$(curl -s "$U/health")" = "ok" ] && pass "/health body is still the literal ok" || fail "/health body changed"

C="$(cap)"
[ "$(jq_py "$C" 'd["mv"]["configured"] and not d["mv"]["available"]')" = "True" ] && pass "mv configured but unavailable" || fail "mv status: $C"
[ "$(jq_py "$C" '"missing model file" in d["mv"]["reason"]')" = "True" ] && pass "mv reason names the missing file" || fail "mv reason: $C"
[ "$(jq_py "$C" 'd["sv"]["model_set"] == "pixal3d-sv-q8_0" and d["sv"]["model_family"] == "sv"')" = "True" ] && pass "sv identity is read from the manifest" || fail "sv identity: $C"
[ "$(jq_py "$C" 'd["busy"] is False and d["completed"] == 0')" = "True" ] && pass "idle: busy=false completed=0" || fail "idle state: $C"
[ "$(jq_py "$C" 'd.get("trellis2_mv",{}).get("available") is False and d["trellis2_mv"]["max_images"] == 8')" = "True" ] && pass "trellis2_mv capability reports missing TRELLIS.2 flows" || fail "trellis2_mv capability: $C"
[ "$(jq_py "$C" '"missing TRELLIS.2 model file" in d["trellis2_mv"].get("reason","")')" = "True" ] && pass "trellis2_mv capability names the missing model file" || fail "trellis2_mv reason: $C"

expect_status "generate-sv without an available SV set -> 503" 503 -X POST "$U/generate-sv" -F "image=@$IMG"
expect_status "generate-sv unknown field -> 400"             400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "band=2"
expect_status "generate-sv mesh_scale -> 400"                400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "mesh_scale=1.0"
expect_status "generate-sv fov=4 -> 400"                     400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "fov=4"
expect_status "generate-sv fov=nan -> 400"                   400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "fov=nan"
expect_status "generate-sv fov=inf -> 400"                   400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "fov=inf"
expect_status "generate-sv fov=0.5x -> 400"                  400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "fov=0.5x"
expect_status "generate-sv resolution=1536 -> 400"           400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "resolution=1536"
expect_status "generate-sv resolution=1024x -> 400"          400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "resolution=1024x"
expect_status "generate-sv seed=abc -> 400"                  400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "seed=abc"
expect_status "generate-sv seed=-1 -> 400"                   400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "seed=-1"
expect_status "generate-sv seed=4294967296 -> 400"           400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "seed=4294967296"
expect_status "generate-sv uv=typo -> 400"                   400 -X POST "$U/generate-sv" -F "image=@$IMG" -F "uv=typo"
expect_status "generate-sv non-PNG -> 400"                   400 -X POST "$U/generate-sv" -F "image=@$ROOT/CMakeLists.txt"
expect_status "generate-sv missing image -> 400"             400 -X POST "$U/generate-sv" -F "fov=0.5"
expect_status "trellis2-mv missing num_images -> 400"          400 -X POST "$U/generate-trellis2-mv" -F "image0=@$IMG" -F "image1=@$IMG"
expect_status "trellis2-mv one image -> 400"                    400 -X POST "$U/generate-trellis2-mv" -F "num_images=1" -F "image0=@$IMG"
expect_status "trellis2-mv nine images -> 400"                  400 -X POST "$U/generate-trellis2-mv" -F "num_images=9"
expect_status "trellis2-mv invalid fusion -> 400"               400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "fusion=wat" -F "image0=@$IMG" -F "image1=@$IMG"
expect_status "trellis2-mv sparse image indices -> 400"         400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$IMG" -F "image2=@$IMG"
expect_status "trellis2-mv duplicate image0 part -> 400"        400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$IMG" -F "image0=@$IMG" -F "image1=@$IMG"
expect_status "trellis2-mv rejects mesh_scale -> 400"           400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$IMG" -F "image1=@$IMG" -F "mesh_scale=1"
expect_status "trellis2-mv seed=abc -> 400"                     400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$IMG" -F "image1=@$IMG" -F "seed=abc"
expect_status "trellis2-mv valid input but no TRELLIS.2 set -> 503" 503 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$IMG" -F "image1=@$IMG"
expect_status "generate-sv decompression-bomb IHDR -> 400 (sanity)"    400 -X POST "$U/generate-sv" -F "image=@$BOMB"
expect_status "trellis2-mv decompression-bomb IHDR in image0 -> 400" 400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$BOMB" -F "image1=@$IMG"
expect_status "trellis2-mv decompression-bomb IHDR in image1 -> 400" 400 -X POST "$U/generate-trellis2-mv" -F "num_images=2" -F "image0=@$IMG" -F "image1=@$BOMB"
expect_status "generate-mv seed=abc -> 400 (was silently 0)" 400 -X POST "$U/generate-mv" -F "view0=@$IMG" -F "mesh_scale=1" -F "seed=abc"
expect_status "generate seed=abc -> 400 (was silently 0)"    400 -X POST "$U/generate" -F "image=@$IMG" -F "seed=abc"

kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null || true

# ---- 2. --models-sv 無し / family 取り違え / 契約違反 / サイズ不一致 -----------
# family 取り違え: --models に SV セットを渡す
"$SERVER" --models "$SV" --gpu -1 --port "$PORT" >"$WORK/server2.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 50); do curl -s "$U/health" >/dev/null 2>&1 && break; sleep 0.1; done
C="$(cap)"
[ "$(jq_py "$C" 'd["sv"]["configured"] is False and d["sv"]["available"] is False')" = "True" ] && pass "sv unconfigured without --models-sv" || fail "sv unconfigured: $C"
[ "$(jq_py "$C" '"model_family mismatch" in d["mv"]["reason"]')" = "True" ] && pass "an SV set under --models is reported as a family mismatch (server still runs)" || fail "family mismatch: $C"
expect_status "generate-sv unconfigured -> 503" 503 -X POST "$U/generate-sv" -F "image=@$IMG"
kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null || true

# 契約違反: model_family=sv なのに flow 名が _mv
python3 - "$ROOT/models/pixal3d-f16-v1/pixal3d-models.json" "$BAD/pixal3d-models.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m["model_family"] = "sv"
json.dump(m, open(sys.argv[2], "w"))
PY
"$SERVER" --models "$MV" --models-sv "$BAD" --gpu -1 --port "$PORT" >"$WORK/server3.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 50); do curl -s "$U/health" >/dev/null 2>&1 && break; sleep 0.1; done
C="$(cap)"
[ "$(jq_py "$C" '"does not match the flow file names" in d["sv"]["reason"]')" = "True" ] && pass "explicit family vs flow names mismatch is rejected" || fail "contract violation: $C"
kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null || true

# ---- 3. サイズ一致の sparse ファイルで available=true にし、busy/completed を見る -----
python3 - "$SV/pixal3d-models.json" "$SV" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
for f in m["files"]:
    p = os.path.join(sys.argv[2], f["name"])
    with open(p, "wb") as fh:
        fh.truncate(f["size_bytes"])   # sparse: ディスクを消費しない
PY
"$SERVER" --models "$MV" --models-sv "$SV" --gpu -1 --port "$PORT" >"$WORK/server4.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 50); do curl -s "$U/health" >/dev/null 2>&1 && break; sleep 0.1; done
C="$(cap)"
[ "$(jq_py "$C" 'd["sv"]["available"] is True')" = "True" ] && pass "sv available with manifest + 9 files of the right size" || fail "sv available: $C"

# サイズ不一致を 1 本作ると unavailable に戻る
truncate -s 1000 "$SV/pixal3d_naf.gguf"
C="$(cap)"
[ "$(jq_py "$C" '"size mismatch pixal3d_naf.gguf" in d["sv"]["reason"]')" = "True" ] && pass "a size mismatch makes sv unavailable again" || fail "size mismatch: $C"
python3 - "$SV/pixal3d-models.json" "$SV" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
for f in m["files"]:
    if f["name"] == "pixal3d_naf.gguf":
        with open(os.path.join(sys.argv[2], f["name"]), "wb") as fh: fh.truncate(f["size_bytes"])
PY

# 生成を投げる: available なのでモデルロードまで進み、中身が空なので失敗して 500 になる。
# その間 busy=true、終わると busy=false・completed=1。
# 空 GGUF は即座に失敗するので、busy=true を観測できる時間を staging 側で作る:
# 6000x6000（36 Mpixel、上限 64 Mpixel 未満）の PNG はデコード + クロップに 1 秒前後かかり、
# それは BusyScope の内側で走る。
python3 - "$WORK/big.png" <<'BIGPNG'
import struct, sys, zlib
w = h = 6000
row_bg = bytes([0]) + bytes([0, 0, 0, 0]) * w
row_fg = bytes([0]) + bytes([0, 0, 0, 0]) * 2000 + bytes([200, 120, 80, 255]) * 2000 + bytes([0, 0, 0, 0]) * 2000
raw = b"".join(row_fg if 2000 <= y < 4000 else row_bg for y in range(h))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
BIGPNG
curl -s -o "$WORK/gen.body" -w '%{http_code}' -X POST "$U/generate-sv" -F "image=@$WORK/big.png" >"$WORK/gen.code" &
GEN_PID=$!
saw_busy=0
for _ in $(seq 1 400); do
    C="$(cap)"
    if [ "$(jq_py "$C" 'd["busy"]')" = "True" ]; then saw_busy=1; break; fi
    sleep 0.02
done
wait "$GEN_PID"
[ "$saw_busy" = 1 ] && pass "busy=true while a generation runs" || fail "never saw busy=true (generation may have failed before BusyScope)"
code="$(cat "$WORK/gen.code")"
[ "$code" = "500" ] && pass "generation on empty GGUFs fails with 500 [$(head -c 80 "$WORK/gen.body")]" || fail "expected 500 from empty GGUFs, got $code: $(head -c 120 "$WORK/gen.body")"
C="$(cap)"
[ "$(jq_py "$C" 'd["busy"] is False and d["completed"] == 1')" = "True" ] && pass "after it: busy=false completed=1 (failures count too)" || fail "post state: $C"

kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null || true
SRV_PID=""

if [ "$failures" = 0 ]; then echo; echo "SERVER_CONTRACT_OK"; exit 0; fi
echo; echo "SERVER_CONTRACT_FAIL ($failures)"; exit 1
