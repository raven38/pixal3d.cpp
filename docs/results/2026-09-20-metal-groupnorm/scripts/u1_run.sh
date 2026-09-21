#!/bin/zsh
# U1 の計測（build-metal-nth = nth パッチ入り ggml）: (1) --gn-oracle で native group_norm の精度、(2) --naf-ops --naf-native-gn T=1024 で
# GROUP_NORM の op 時間と whole graph（各 2 本）。ロック + ゲート（40 %/30 s）を通し、run 中の GPU 使用率を .gpu に残す。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
D=$REPO/docs/results/2026-09-20-metal-groupnorm; OUT=$D/nth; mkdir -p $OUT
W=$HOME/data/weights/pixal3d/gguf-q8_0; VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
SNAP=$OUT/ps_snapshots.log; SELFKEY=pixal3d-metal-groupnorm
GPU_QUIET=${GPU_QUIET:-40}; QUIET_SECS=${QUIET_SECS:-30}
source $D/scripts/gate_lib.sh
BIN=$REPO/build-metal-nth
go() {  # go <log> <cmd...>
  local log=$1; shift
  gate "nth $(basename $log) ~30s"; snapshot "before $(basename $log)"; gpu_sampler_start $log.gpu
  run_guarded $log "$@"; local rc=$?
  kill $GPU_SP 2>/dev/null; snapshot "after $(basename $log)"; lock_release; echo "$(basename $log) exit=$rc"
}
go $OUT/gn_oracle_nth.log $BIN/trellis-test-naf --gn-oracle $W/pixal3d_naf.gguf 0
for i in 1 2; do
  go $OUT/nafops_T1024_nth_native_$i.log $BIN/trellis-test-pixal3d-cond-tex $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --naf-ops --views 1 --S 1024 --R 64 --naf-t 1024 --naf-native-gn
done
go $OUT/nafops_T1024_nth_lowering_1.log $BIN/trellis-test-pixal3d-cond-tex $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --naf-ops --views 1 --S 1024 --R 64 --naf-t 1024
echo "U1 DONE $(date)"
