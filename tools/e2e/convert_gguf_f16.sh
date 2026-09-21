#!/usr/bin/env bash
# TRELLIS.2 flow モデルを f16 GGUF へ変換する（#65 E2E の前提資産準備）。
# 常駐 CPU pod（例: <cpu-pod>, GPU無し）で実行する想定。torch 不要、numpy + gguf のみ。
#
# 使い方（pod 内）:
#   TRELLIS_MODELS=/nfs/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/<pin> \
#   TRELLIS_GGUF_OUT=/nfs/trellis2-mv/gguf-f16 \
#   CONVERT_PY=/path/to/tools/convert.py \
#   DECODER_SRC=/nfs/pixal3d_weights/gguf-f16 \
#   bash convert_gguf_f16.sh
set -euo pipefail

TRELLIS_MODELS="${TRELLIS_MODELS:?TRELLIS_MODELS (HF snapshot dir, ckpts/ 直下必須) を指定してください}"
TRELLIS_GGUF_OUT="${TRELLIS_GGUF_OUT:?TRELLIS_GGUF_OUT (出力先ディレクトリ) を指定してください}"
CONVERT_PY="${CONVERT_PY:?CONVERT_PY (tools/convert.py のパス) を指定してください}"
DECODER_SRC="${DECODER_SRC:-/nfs/pixal3d_weights/gguf-f16}"
VENV_DIR="${VENV_DIR:-/nfs/trellis2-mv/convert-venv}"

export TRELLIS_MODELS TRELLIS_GGUF_OUT

echo "=== venv 準備 (${VENV_DIR}) ==="
if [ ! -x "${VENV_DIR}/bin/python3" ]; then
    python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install -q --upgrade pip
"${VENV_DIR}/bin/pip" install -q numpy gguf

echo "=== TRELLIS.2 flow -> f16 GGUF 変換 ==="
"${VENV_DIR}/bin/python3" "${CONVERT_PY}" ss_flow shape_flow_512 shape_flow_1024 tex_flow_1024

echo "=== decoder 3本 + dinov3 を共有f16から再利用（コピー） ==="
mkdir -p "${TRELLIS_GGUF_OUT}"
for f in ss_dec shape_dec tex_dec dinov3; do
    src="${DECODER_SRC}/${f}.gguf"
    if [ ! -f "${src}" ]; then
        echo "FATAL: ${src} が見つかりません" >&2
        exit 1
    fi
    cp -v "${src}" "${TRELLIS_GGUF_OUT}/${f}.gguf"
done

echo "=== 完了。出力一覧 ==="
ls -la "${TRELLIS_GGUF_OUT}"
sha256sum "${TRELLIS_GGUF_OUT}"/*.gguf
