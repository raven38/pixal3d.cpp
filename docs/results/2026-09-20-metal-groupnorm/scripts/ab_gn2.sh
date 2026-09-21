#!/bin/zsh
# issue #55 の P1 / P2 再計測（GPU 使用率ゲート版、gate_lib.sh）。同一バイナリで 新（既定）vs 旧（--naf-native-gn）、
# P1 = --naf-ops（T1024 / T512 / LR、各 arm n=6 の paired block）、P2 = --mode gpu 4 view（tex / hr / lr、各 arm n=6、ABBA 交互）、
# P2-num = 同じ coords で新旧の cond 出力を保存して比較。issue #55 コメント 5748387443 の C（6 paired blocks）に合わせた。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$REPO/docs/results/2026-09-20-metal-groupnorm/ab2; mkdir -p $OUT
COORDS=$HOME/data/weights/pixal3d/synth_tex_fixture/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
SELFKEY=pixal3d-metal-groupnorm
# 18:30 再開: 25 %/90 s のゲートは Chrome / WindowServer（操作中 11〜65 %）で開かなかったので 40 %/30 s に緩め、
# run 中の GPU 使用率を 5 s おきに <log>.gpu へ残す（注意: 自分の compute 中は 96〜100 % になるので他者の利用とは区別できない。
# 汚染判定は ps の他 trellis と、6 block の paired 差のばらつき（min / median / max）で行う）。
GPU_QUIET=${GPU_QUIET:-40}; QUIET_SECS=${QUIET_SECS:-30}
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
for i in 1 2 3 4 5 6; do
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
for i in 1 2 3 4 5 6; do
  if [ $((i % 2)) -eq 1 ]; then A=new; B=old; else A=old; B=new; fi
  run $A "tex" $COMMON $T1024
  run $B "tex" $COMMON $T1024
  run $A "hr"  $COMMON $T512
  run $B "hr"  $COMMON $T512
  run $A "lr"  $COMMON $LR
  run $B "lr"  $COMMON $LR
done
echo "P2 DONE $(date)"
# P2-num: 同じ coords で 新 / 旧 の conditioning 出力（global / proj）を保存し、数値差を取る（計時には使わない）。
NUM=$OUT/num; mkdir -p $NUM
echo "P2NUM START $(date)"
for cfg in "tex ${T1024[*]}" "hr ${T512[*]}" "lr ${LR[*]}"; do
  set -- ${=cfg}; name=$1; shift
  gate "num $name"; run_guarded $NUM/new_$name.log $BIN $COMMON "$@" --save-prefix $NUM/new_$name; echo "num new $name exit=$?"; lock_release
  gate "num $name"; run_guarded $NUM/old_$name.log $BIN $COMMON "$@" --naf-native-gn --save-prefix $NUM/old_$name; echo "num old $name exit=$?"; lock_release
done
python3 $REPO/docs/results/2026-09-20-metal-groupnorm/scripts/compare_cond.py $NUM | tee $NUM/compare.txt
echo "P2NUM DONE $(date)"
