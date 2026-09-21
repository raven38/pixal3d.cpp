#!/bin/zsh
# issue #55 の P1 / P2 を凍結ゲート（他 trellis 無し かつ GPU < 25 % が 90 s 連続、E1 と同条件）で 2 block ずつ再確認する。
# codex 反証レビューの指摘: ab2 は 40 %/30 s に緩めたので、設計 doc の凍結条件（90 s ゲート）での値を別に持つ。ab_gn2.sh から派生。
# 3 行目以降の run / 構成は ab_gn2.sh と同一（resume 付き）。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$REPO/docs/results/2026-09-20-metal-groupnorm/ab3_strict; mkdir -p $OUT
COORDS=$HOME/data/weights/pixal3d/synth_tex_fixture/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
SELFKEY=pixal3d-metal-groupnorm
GPU_QUIET=${GPU_QUIET:-25}; QUIET_SECS=${QUIET_SECS:-90}   # 凍結ゲート（90 s 連続の静穏、25 % は E1 と同条件）での確認用
source $REPO/docs/results/2026-09-20-metal-groupnorm/scripts/gate_lib.sh
typeset -A SEEN   # resume: 既に $OUT に同じ (arm, name) のログが k 本あれば、その key の最初の k 回は飛ばす（driver 再起動用）
run() {  # run <arm: new|old> <name> <args...>
  local arm=$1; shift; local name=$1; shift
  local key=${arm}_${name}; SEEN[$key]=$(( ${SEEN[$key]:-0} + 1 ))
  if [ ${SEEN[$key]} -le $(ls $OUT/${key}_*.log 2>/dev/null | wc -l) ]; then echo "--- skip $key #${SEEN[$key]} (already measured)"; return 0; fi
  gate "$name ~1-2min"; echo "--- $arm $name $(date '+%H:%M:%S')"; snapshot "before $arm $name"
  ( while true; do sleep 30; snapshot "periodic"; done ) & local SP=$!
  local log=$OUT/${arm}_${name}_$(date '+%H%M%S').log
  gpu_sampler_start $log.gpu
  if [ "$arm" = "old" ]; then run_guarded $log $BIN "$@" --naf-native-gn; else run_guarded $log $BIN "$@"; fi
  local rc=$?
  kill $SP $GPU_SP 2>/dev/null; snapshot "after $arm $name"; lock_release; echo "exit=$rc"
}
NAF=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --naf-ops --views 1)
COMMON=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --profile-cond --mode gpu)
T1024=(--S 1024 --R 64 --naf-t 1024 --coords $COORDS)
T512=(--S 1024 --R 64 --naf-t 512 --coords $COORDS)
LR=(--S 512 --R 32 --naf-t 512 --stride 3)
echo "P1 START $(date)"
for i in 1 2; do
  if [ $((i % 2)) -eq 1 ]; then A=new; B=old; else A=old; B=new; fi
  run $A "nafops_T1024" $NAF --S 1024 --R 64 --naf-t 1024
  run $B "nafops_T1024" $NAF --S 1024 --R 64 --naf-t 1024
  run $A "nafops_T512"  $NAF --S 1024 --R 64 --naf-t 512
  run $B "nafops_T512"  $NAF --S 1024 --R 64 --naf-t 512
  run $A "nafops_LR"    $NAF --S 512 --R 32 --naf-t 512
  run $B "nafops_LR"    $NAF --S 512 --R 32 --naf-t 512
done
echo "P1 DONE $(date)"
echo "P2 START $(date)"
for i in 1 2; do
  if [ $((i % 2)) -eq 1 ]; then A=new; B=old; else A=old; B=new; fi
  run $A "tex" $COMMON $T1024
  run $B "tex" $COMMON $T1024
  run $A "hr"  $COMMON $T512
  run $B "hr"  $COMMON $T512
  run $A "lr"  $COMMON $LR
  run $B "lr"  $COMMON $LR
done
echo "P2 DONE $(date)"
echo "STRICT DONE $(date)"
