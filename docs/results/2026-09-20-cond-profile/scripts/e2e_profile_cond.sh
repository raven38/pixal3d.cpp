#!/bin/zsh
# E2E 1 本（設計 §5-3）: cyclops MV 1024、seed 1、q8_0 MV v1、--profile-cond（--profile は付けない）。
# ゲート: 他の trellis プロセスが 90 s 無い。実行中は 30 s おきに ps を記録。
set -u
REPO=/Users/s25705/Downloads/pixal3d-cond-profile
OUT=$(dirname $0)/e2e
mkdir -p $OUT
SNAP=$OUT/ps_snapshots.log
others() { ps -Ao pid,etime,args -r | grep "trellis" | grep -v grep | grep -v "pixal3d-cond-profile/build-metal" | grep -v "e2e_profile_cond" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep "trellis" | grep -v grep | cut -c1-140; } >> $SNAP; }
quiet=0
while [ $quiet -lt 90 ]; do
  if [ -z "$(others)" ]; then quiet=$((quiet+10)); else quiet=0; fi
  sleep 10
done
echo "GATE PASSED $(date)"; snapshot "gate"
( while true; do snapshot "periodic"; sleep 30; done ) &
SNAP_PID=$!
$REPO/build-metal/trellis-cli --views $REPO/docker/linux-webgpu-gate/e2e/views -m $HOME/data/weights/pixal3d/gguf-q8_0 --res 1024 --seed 1 --profile-cond -o $OUT/cond.glb > $OUT/e2e_cond.log 2>&1
echo "exit=$?"
kill $SNAP_PID 2>/dev/null
snapshot "end"
echo "E2E DONE $(date)"
