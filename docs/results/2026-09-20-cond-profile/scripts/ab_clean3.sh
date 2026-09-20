#!/bin/zsh
# 再計測（codex 反証レビュー A1〜A3 対応）: 実行ごとに 90 s 静穏ゲート、前・中（30 s）・後に ps を記録。
# 追加サンプル: device T1024 ×3 / T512 ×2 / LR ×1、host T512 ×2、naf-ops（im2col / generic / generic-gn / direct、compare 付き）。
set -u
REPO=/Users/s25705/Downloads/pixal3d-cond-profile
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$(dirname $0)/clean3; mkdir -p $OUT
COORDS=$(dirname $0)/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
others() { ps -Ao pid,etime,args -r | grep -E "trellis|post-replay" | grep -v grep | grep -v "pixal3d-cond-profile/build-metal" | grep -v "ab_clean3" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep -E "trellis|post-replay" | grep -v grep | cut -c1-140; } >> $SNAP; }
gate() { local quiet=0; while [ $quiet -lt 90 ]; do if [ -z "$(others)" ]; then quiet=$((quiet+10)); else quiet=0; fi; sleep 10; done; }
run() {  # run <name> <args...>
  local name=$1; shift
  gate; echo "--- $name $(date '+%H:%M:%S')"; snapshot "before $name"
  ( while true; do sleep 30; snapshot "periodic"; done ) & local SP=$!
  $BIN "$@" > $OUT/${name}_$(date '+%H%M%S').log 2>&1; local rc=$?
  kill $SP 2>/dev/null; snapshot "after $name"; echo "exit=$rc"
}
COMMON=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --profile-cond)
T1024=(--S 1024 --R 64 --naf-t 1024 --coords $COORDS)
T512=(--S 1024 --R 64 --naf-t 512 --coords $COORDS)
LR=(--S 512 --R 32 --naf-t 512 --stride 3)
# 前窓 90 s に他の trellis バイナリが居た first-round の採用分（T1024 host / LR gpu / LR host / T512 gpu 各 1 本）の置き換え
run "T1024 host" $COMMON --mode host $T1024
run "LR host"    $COMMON --mode host $LR
run "LR gpu"     $COMMON --mode gpu  $LR
run "T512 gpu"   $COMMON --mode gpu  $T512
run "T1024 host" $COMMON --mode host $T1024
run "LR host"    $COMMON --mode host $LR
echo "AB3 DONE $(date)"
