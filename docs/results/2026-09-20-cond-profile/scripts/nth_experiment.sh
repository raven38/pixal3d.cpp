#!/bin/zsh
# 診断: ggml-metal group_norm の nth を 32 → 1024 にして NAF グラフを測る（submodule の一時パッチ。終了後に復元）。
set -u
REPO=/Users/s25705/Downloads/pixal3d-cond-profile
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$(dirname $0)/nth; mkdir -p $OUT
SNAP=$OUT/ps_snapshots.log
others() { ps -Ao pid,etime,args -r | grep -E "trellis|post-replay" | grep -v grep | grep -v "pixal3d-cond-profile/build-metal" | grep -v "nth_experiment" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep -E "trellis|post-replay" | grep -v grep | cut -c1-140; } >> $SNAP; }
gate() { local quiet=0; while [ $quiet -lt 90 ]; do if [ -z "$(others)" ]; then quiet=$((quiet+10)); else quiet=0; fi; sleep 10; done; }
run() { local name=$1; shift; gate; echo "--- $name $(date '+%H:%M:%S')"; snapshot "before $name"
  ( while true; do sleep 30; snapshot "periodic"; done ) & local SP=$!
  $BIN "$@" > $OUT/${name}_$(date '+%H%M%S').log 2>&1; local rc=$?; kill $SP 2>/dev/null; snapshot "after $name"; echo "exit=$rc"; }
python3 $(dirname $0)/nth_patch.py apply || exit 1
cmake --build $REPO/build-metal --target trellis-test-pixal3d-cond-tex -j > $OUT/build_patched.log 2>&1 || { echo "build failed"; python3 $(dirname $0)/nth_patch.py revert; exit 1; }
echo "BUILD PATCHED $(date)"
NAF=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --naf-ops --views 1)
run "nth1024 T1024 im2col" $NAF --S 1024 --R 64 --naf-t 1024
run "nth1024 T1024 im2col" $NAF --S 1024 --R 64 --naf-t 1024
run "nth1024 T1024 gn cmp" $NAF --S 1024 --R 64 --naf-t 1024 --naf-generic-gn --naf-compare
run "nth1024 T512 im2col"  $NAF --S 1024 --R 64 --naf-t 512
python3 $(dirname $0)/nth_patch.py revert
cmake --build $REPO/build-metal --target trellis-test-pixal3d-cond-tex -j > $OUT/build_reverted.log 2>&1; echo "BUILD REVERTED $(date) exit=$?"
echo "NTH DONE $(date)"
