#!/bin/zsh
# クリーンな A/B（設計 §5-1）: GPU 版 / host 版を別プロセス、ABBAAB（各 3 回）、条件間 60 s 冷却、
# 開始前に他の trellis プロセスが 90 s 無いことを待ち、実行中は 30 s おきに ps を記録する。
set -u
REPO=/Users/s25705/Downloads/pixal3d-cond-profile
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$(dirname $0)/clean
mkdir -p $OUT
COORDS=$(dirname $0)/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log

others() { ps -Ao pid,etime,args -r | grep "trellis" | grep -v grep | grep -v "pixal3d-cond-profile/build-metal" | grep -v "ab_clean" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep "trellis" | grep -v grep | cut -c1-140; } >> $SNAP; }

# gate: 他の trellis プロセスが 90 s 連続で居ない
quiet=0
while [ $quiet -lt 90 ]; do
  if [ -z "$(others)" ]; then quiet=$((quiet+10)); else quiet=0; fi
  sleep 10
done
echo "GATE PASSED $(date)"; snapshot "gate"

# バックグラウンドの ps 記録（30 s おき）
( while true; do snapshot "periodic"; sleep 30; done ) &
SNAP_PID=$!

run_one() {  # $1=tag $2=mode $3.. args
  local tag=$1 mode=$2; shift 2
  echo "--- $tag $mode $(date '+%H:%M:%S')"
  snapshot "before $tag $mode"
  $BIN $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --mode $mode --profile-cond "$@" > $OUT/${tag}_${mode}_$(date '+%H%M%S').log 2>&1
  echo "exit=$?"
  snapshot "after $tag $mode"
  sleep 60
}

for cond in "T1024 --S 1024 --R 64 --naf-t 1024 --coords $COORDS" "T512 --S 1024 --R 64 --naf-t 512 --coords $COORDS" "LR --S 512 --R 32 --naf-t 512 --stride 3"; do
  set -- ${=cond}; tag=$1; shift
  for mode in gpu host host gpu gpu host; do
    run_one $tag $mode "$@"
  done
done

# L2 診断（別プロセス、fresh）: NAF encoder の op 表、im2col / generic lowering、T=1024 / T=512
for T in 1024 512; do
  echo "--- naf-ops T=$T im2col $(date '+%H:%M:%S')"; snapshot "before naf-ops T=$T"
  $BIN $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --S 1024 --R 64 --naf-t $T --naf-ops --views 1 > $OUT/nafops_T${T}_im2col.log 2>&1; echo "exit=$?"; sleep 60
  echo "--- naf-ops T=$T generic $(date '+%H:%M:%S')"
  TRELLIS_DBG_NAF_GENERIC=1 $BIN $W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --S 1024 --R 64 --naf-t $T --naf-ops --views 1 > $OUT/nafops_T${T}_generic.log 2>&1; echo "exit=$?"; sleep 60
done
kill $SNAP_PID 2>/dev/null
echo "ALL DONE $(date)"
