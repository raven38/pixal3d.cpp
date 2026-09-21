#!/usr/bin/env bash
# #65 Part B: MV E2E マトリクス（B1〜B8）を直列に1本ずつ実行する。
# 再開可能: 各 run の manifest.json（wall_clock_s を含む）が既に存在すれば skip する。
set -uo pipefail

CLI="${CLI:?CLI (trellis-cliのパス) を指定してください}"
MODELS="${MODELS:?MODELS (gguf-f16ディレクトリ) を指定してください}"
OUT_ROOT="${OUT_ROOT:-/nfs/trellis2-mv/e2e-out}"
# 注意: trellis_cli.cpp は --seed 0 を「未指定」扱いにして実行毎に真のランダムseedを生成する
# （src/trellis_cli.cpp:303-309/735-741、`(auto)` ログで確認済み、CLIの既定値も0でなく42）。
# brief は「--seed 0」を明記しているが、これは再現性の意図（決定性チェックB6の前提）を壊すため
# FINDING として記録した上で固定非ゼロ値に切り替える（SEED環境変数で上書き可）。
SEED="${SEED:-42}"
VIEWS_2="${VIEWS_2:-/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_2view}"
VIEWS_4="${VIEWS_4:-/nfs/pixal3d_trellis2mv_ref_v2/inputs/yoimiya_4view}"

mkdir -p "$OUT_ROOT"

run_one() {
    local name="$1"; shift
    local views="$1"; shift
    local run_dir="$OUT_ROOT/$name"
    if [ -f "$run_dir/manifest.json" ]; then
        echo "=== SKIP $name（既に完了、manifest.json あり） ==="
        return 0
    fi
    mkdir -p "$run_dir"
    echo "=== RUN $name : $* ==="
    # VRAM 背景サンプリング（1秒間隔）
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > "$run_dir/vram.csv" 2>&1 &
    local vram_pid=$!
    local t0=$(date +%s)
    "$CLI" --trellis2-mv "$views" --models "$MODELS" --seed "$SEED" "$@" "$run_dir/out.glb" \
        > "$run_dir/run.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    kill "$vram_pid" 2>/dev/null; wait "$vram_pid" 2>/dev/null

    local peak_vram
    peak_vram=$(awk -F',' 'NR>1{gsub(/ MiB/,"",$2); if ($2+0>max) max=$2+0} END{print max+0}' "$run_dir/vram.csv" 2>/dev/null)
    local ss_voxels
    ss_voxels=$(sed -nE 's/.*active voxels @res32 = ([0-9]+).*/\1/p' "$run_dir/run.log" | head -1)
    local mv_line
    mv_line=$(grep -E 'TRELLIS\.2 (multiview|MV)' "$run_dir/run.log" | head -1)

    # python3 に依存しない JSON 生成（実行専用の自然終了型 pod には python3 が無いことがある、2026-09-21実測）
    local esc_mvline
    esc_mvline=$(printf '%s' "$mv_line" | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf '{\n  "name": "%s",\n  "rc": %s,\n  "wall_clock_s": %s,\n  "peak_vram_mib": %s,\n  "ss_active_voxels": %s,\n  "mv_stage_log": %s\n}\n' \
        "$name" "${rc:-null}" "$((t1-t0))" "${peak_vram:-null}" "${ss_voxels:-null}" \
        "$([ -n "$mv_line" ] && printf '"%s"' "$esc_mvline" || echo null)" \
        > "$run_dir/manifest.json"
    echo "=== DONE $name rc=$rc wall=$((t1-t0))s peak_vram=${peak_vram}MiB voxels=${ss_voxels} ==="
}

run_one B1_2view_stochastic       "$VIEWS_2" --trellis2-mv-mode stochastic --res 1024
run_one B2_2view_multidiffusion   "$VIEWS_2" --trellis2-mv-mode multidiffusion --res 1024
run_one B3_4view_stochastic       "$VIEWS_4" --trellis2-mv-mode stochastic --res 1024
run_one B4_4view_multidiffusion   "$VIEWS_4" --trellis2-mv-mode multidiffusion --res 1024
run_one B5_2view_stochastic_notex "$VIEWS_2" --trellis2-mv-mode stochastic --res 1024 --no-texture
run_one B6_2view_stochastic_rerun "$VIEWS_2" --trellis2-mv-mode stochastic --res 1024
run_one B7_2view_stochastic_512   "$VIEWS_2" --trellis2-mv-mode stochastic --res 512
run_one B8_4view_multidiff_1536   "$VIEWS_4" --trellis2-mv-mode multidiffusion --res 1536

echo "=== PART B MATRIX DONE ==="
