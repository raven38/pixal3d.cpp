#!/bin/zsh
# segfault（use-after-free、db0623b で修正）で落ちた LR gpu 2 本の再実行と、T512 gpu 1 本（単一グラフ経路の内訳取得用）。
set -u
REPO=/Users/s25705/Downloads/pixal3d-cond-profile
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$(dirname $0)/clean
COORDS=$(dirname $0)/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
others() { ps -Ao pid,etime,args -r | grep "trellis" | grep -v grep | grep -v "pixal3d-cond-profile/build-metal" | grep -v "ab_rerun" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep "trellis" | grep -v grep | cut -c1-140; } >> $SNAP; }
quiet=0
while [ $quiet -lt 90 ]; do
  if [ -z "$(others)" ]; then quiet=$((quiet+10)); else quiet=0; fi
  sleep 10
done
echo "GATE PASSED $(date)"; snapshot "gate rerun"
for i in 1 2; do
  echo "--- LR gpu $(date '+%H:%M:%S')"; snapshot "before LR gpu rerun"
  $BIN $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --mode gpu --profile-cond --S 512 --R 32 --naf-t 512 --stride 3 > $OUT/LR_gpu_$(date '+%H%M%S').log 2>&1; echo "exit=$?"; snapshot "after LR gpu rerun"; sleep 60
done
echo "--- T512 gpu (internals) $(date '+%H:%M:%S')"; snapshot "before T512 gpu rerun"
$BIN $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --mode gpu --profile-cond --S 1024 --R 64 --naf-t 512 --coords $COORDS > $OUT/T512_gpu_$(date '+%H%M%S').log 2>&1; echo "exit=$?"; snapshot "after T512 gpu rerun"
echo "RERUN DONE $(date)"
