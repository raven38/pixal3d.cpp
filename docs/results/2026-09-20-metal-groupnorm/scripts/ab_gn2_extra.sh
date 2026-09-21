#!/bin/zsh
# ab_gn2.sh の追加 block: classify_runs.py で汚染と判定された cell を補うために、指定 config の (new, old) を 1 pair ずつ追加で回す。
# 使い方: ab_gn2_extra.sh <block#> <config>...   config = nafops_T1024 | nafops_T512 | nafops_LR | tex | hr | lr
set -u
REPO=/Users/<redacted-user>/Downloads/pixal3d-metal-groupnorm
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/nfs/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$REPO/docs/results/2026-09-20-metal-groupnorm/ab2
COORDS=$HOME/nfs/weights/pixal3d/synth_tex_fixture/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
SELFKEY=pixal3d-metal-groupnorm
GPU_QUIET=${GPU_QUIET:-40}; QUIET_SECS=${QUIET_SECS:-30}
source $REPO/docs/results/2026-09-20-metal-groupnorm/scripts/gate_lib.sh
run() {
  local arm=$1; shift; local name=$1; shift
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
i=$1; shift
if [ $((i % 2)) -eq 1 ]; then A=new; B=old; else A=old; B=new; fi
for cfg in "$@"; do
  case $cfg in
    nafops_T1024) args=($NAF --S 1024 --R 64 --naf-t 1024);;
    nafops_T512)  args=($NAF --S 1024 --R 64 --naf-t 512);;
    nafops_LR)    args=($NAF --S 512 --R 32 --naf-t 512);;
    tex) args=($COMMON --S 1024 --R 64 --naf-t 1024 --coords $COORDS);;
    hr)  args=($COMMON --S 1024 --R 64 --naf-t 512 --coords $COORDS);;
    lr)  args=($COMMON --S 512 --R 32 --naf-t 512 --stride 3);;
    *) echo "unknown config $cfg"; exit 1;;
  esac
  echo "EXTRA block $i $cfg START $(date)"
  run $A $cfg $args; run $B $cfg $args
  echo "EXTRA block $i $cfg DONE $(date)"
done
