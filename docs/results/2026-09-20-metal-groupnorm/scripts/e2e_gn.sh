#!/bin/zsh
# issue #55 E1: 同一バイナリで E2E 2 本（旧 = --naf-native-gn の native GroupNorm → 新 = 既定の lowering）。
# cyclops MV 1024、seed 1、q8_0 MV v1、--profile-cond。実行ごとに 90 s 静穏ゲート（gate_lib.sh: 他 trellis 無し + GPU 使用率 < 25%）、30 s おきに ps を記録
# （テンプレ: docs/results/2026-09-20-cond-profile/scripts/e2e_profile_cond.sh）。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
OUT=$REPO/docs/results/2026-09-20-metal-groupnorm/e2e; mkdir -p $OUT
SNAP=$OUT/ps_snapshots.log
# 実行ファイル名（comm）で判定する: args だと他セッションの git/エディタが src/trellis_*.cpp を触るたびにゲートが戻る
SELFKEY=pixal3d-metal-groupnorm
source $REPO/docs/results/2026-09-20-metal-groupnorm/scripts/gate_lib.sh
run() {  # run <arm>
  local arm=$1
  gate "E2E $arm ~35min"; echo "--- $arm GATE PASSED $(date)"; snapshot "before $arm"
  ( while true; do sleep 30; snapshot "periodic"; done ) & local SP=$!
  local extra=(); [ "$arm" = "old" ] && extra=(--naf-native-gn)
  run_guarded $OUT/e2e_$arm.log $REPO/build-metal/trellis-cli --views $REPO/docker/linux-webgpu-gate/e2e/views -m $HOME/data/weights/pixal3d/gguf-q8_0 --res 1024 --seed 1 --profile-cond $extra -o $OUT/$arm.glb
  local rc=$?; kill $SP 2>/dev/null; snapshot "after $arm"; lock_release; echo "$arm exit=$rc $(date)"
}
run old
run new
echo "E2E DONE $(date)"
