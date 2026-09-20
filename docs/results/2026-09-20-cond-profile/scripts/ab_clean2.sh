#!/bin/zsh
# 再計測（codex 反証レビュー A1〜A3 対応）: 実行ごとに 90 s 静穏ゲート、前・中（30 s）・後に ps を記録。
# 追加サンプル: device T1024 ×3 / T512 ×2 / LR ×1、host T512 ×2、naf-ops（im2col / generic / generic-gn / direct、compare 付き）。
set -u
REPO=/Users/s25705/Downloads/pixal3d-cond-profile
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$(dirname $0)/clean2; mkdir -p $OUT
COORDS=$(dirname $0)/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
others() { ps -Ao pid,etime,args -r | grep -E "trellis|post-replay" | grep -v grep | grep -v "pixal3d-cond-profile/build-metal" | grep -v "ab_clean2" ; }
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
run "T1024 gpu"  $COMMON --mode gpu  $T1024
run "T512 host"  $COMMON --mode host $T512
run "T512 gpu"   $COMMON --mode gpu  $T512
run "T1024 gpu"  $COMMON --mode gpu  $T1024
run "T512 gpu"   $COMMON --mode gpu  $T512
run "T1024 gpu"  $COMMON --mode gpu  $T1024
run "LR gpu"     $COMMON --mode gpu  $LR
run "T512 host"  $COMMON --mode host $T512
echo "AB2 DONE $(date)"
NAF=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --naf-ops --views 1)
run "nafops T1024 im2col"      $NAF --S 1024 --R 64 --naf-t 1024
run "nafops T1024 generic"     $NAF --S 1024 --R 64 --naf-t 1024 --naf-generic
run "nafops T1024 gn"          $NAF --S 1024 --R 64 --naf-t 1024 --naf-generic-gn
run "nafops T1024 im2col"      $NAF --S 1024 --R 64 --naf-t 1024
run "nafops T1024 generic cmp" $NAF --S 1024 --R 64 --naf-t 1024 --naf-generic --naf-compare
run "nafops T1024 gn cmp"      $NAF --S 1024 --R 64 --naf-t 1024 --naf-generic-gn --naf-compare
run "nafops T1024 direct cmp"  $NAF --S 1024 --R 64 --naf-t 1024 --naf-direct-conv --naf-compare
run "nafops T512 im2col"       $NAF --S 1024 --R 64 --naf-t 512
run "nafops T512 generic"      $NAF --S 1024 --R 64 --naf-t 512 --naf-generic
run "nafops T512 gn cmp"       $NAF --S 1024 --R 64 --naf-t 512 --naf-generic-gn --naf-compare
echo "NAFOPS DONE $(date)"
