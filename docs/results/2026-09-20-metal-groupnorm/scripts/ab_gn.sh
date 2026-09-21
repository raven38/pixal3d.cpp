#!/bin/zsh
# issue #55 の P1 / P2（docs/design/2026-09-20-metal-groupnorm.md §3）: 同一バイナリで 新（既定 = Metal GroupNorm lowering）vs
# 旧（--naf-native-gn = native GroupNorm）。実行ごとに 90 s 静穏ゲート、前・中（30 s）・後に ps を記録
# （テンプレ: docs/results/2026-09-20-cond-profile/scripts/ab_clean2.sh）。採否は classify_runs.py で機械判定する。
set -u
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
BIN=$REPO/build-metal/trellis-test-pixal3d-cond-tex
W=$HOME/data/weights/pixal3d/gguf-q8_0
VIEWS=$REPO/docker/linux-webgpu-gate/e2e/views
OUT=$REPO/docs/results/2026-09-20-metal-groupnorm/ab; mkdir -p $OUT
COORDS=$HOME/data/weights/pixal3d/synth_tex_fixture/hr_coords_f32.npy
SNAP=$OUT/ps_snapshots.log
# 実行ファイル名（comm）で判定する: args だと他セッションの git/エディタが src/trellis_*.cpp を触るたびにゲートが戻る
others() { ps -Ao pid,etime,comm -r | grep -E "trellis-(cli|server|test|smoke)|post-replay" | grep -v "pixal3d-metal-groupnorm/build-metal" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep -E "trellis|post-replay" | grep -v grep | cut -c1-140; } >> $SNAP; }
gate() { local quiet=0; local n=0; while [ $quiet -lt 90 ]; do if [ -z "$(others)" ]; then quiet=$((quiet+10)); else quiet=0; n=$((n+1)); [ $((n % 30)) -eq 1 ] && echo "gate: waiting ($(date '+%H:%M:%S')): $(others | head -2 | cut -c1-100)"; fi; sleep 10; done; }
run() {  # run <arm: new|old> <name> <args...>
  local arm=$1; shift; local name=$1; shift
  gate; echo "--- $arm $name $(date '+%H:%M:%S')"; snapshot "before $arm $name"
  ( while true; do sleep 30; snapshot "periodic"; done ) & local SP=$!
  if [ "$arm" = "old" ]; then $BIN "$@" --naf-native-gn > $OUT/${arm}_${name}_$(date '+%H%M%S').log 2>&1; else $BIN "$@" > $OUT/${arm}_${name}_$(date '+%H%M%S').log 2>&1; fi
  local rc=$?
  kill $SP 2>/dev/null; snapshot "after $arm $name"; echo "exit=$rc"
}
NAF=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --naf-ops --views 1)
COMMON=($W/dinov3.gguf $W/pixal3d_naf.gguf $VIEWS 0 --profile-cond --mode gpu)
T1024=(--S 1024 --R 64 --naf-t 1024 --coords $COORDS)
T512=(--S 1024 --R 64 --naf-t 512 --coords $COORDS)
LR=(--S 512 --R 32 --naf-t 512 --stride 3)
echo "P1 START $(date)"
run new "nafops_T1024" $NAF --S 1024 --R 64 --naf-t 1024
run old "nafops_T1024" $NAF --S 1024 --R 64 --naf-t 1024
run new "nafops_T512"  $NAF --S 1024 --R 64 --naf-t 512
run old "nafops_T512"  $NAF --S 1024 --R 64 --naf-t 512
run old "nafops_T1024" $NAF --S 1024 --R 64 --naf-t 1024
run new "nafops_T1024" $NAF --S 1024 --R 64 --naf-t 1024
run old "nafops_T512"  $NAF --S 1024 --R 64 --naf-t 512
run new "nafops_T512"  $NAF --S 1024 --R 64 --naf-t 512
run new "nafops_LR"    $NAF --S 512 --R 32 --naf-t 512
run old "nafops_LR"    $NAF --S 512 --R 32 --naf-t 512
echo "P1 DONE $(date)"
echo "P2 START $(date)"
for i in 1 2 3; do
  if [ $((i % 2)) -eq 1 ]; then A=new; B=old; else A=old; B=new; fi
  run $A "tex" $COMMON $T1024
  run $B "tex" $COMMON $T1024
  run $A "hr"  $COMMON $T512
  run $B "hr"  $COMMON $T512
  run $A "lr"  $COMMON $LR
  run $B "lr"  $COMMON $LR
done
echo "P2 DONE $(date)"
# P2-num（設計 §3、追加）: 同じ coords で 新 / 旧 の conditioning 出力（global / proj）を保存し、数値差を取る。
# 計時には使わない（ゲートは掛けるがスナップショット不要）。
NUM=$OUT/num; mkdir -p $NUM
echo "P2NUM START $(date)"
for cfg in "tex ${T1024[*]}" "hr ${T512[*]}" "lr ${LR[*]}"; do
  set -- ${=cfg}; name=$1; shift
  gate; $BIN $COMMON "$@" --save-prefix $NUM/new_$name > $NUM/new_$name.log 2>&1; echo "num new $name exit=$?"
  gate; $BIN $COMMON "$@" --naf-native-gn --save-prefix $NUM/old_$name > $NUM/old_$name.log 2>&1; echo "num old $name exit=$?"
done
python3 $REPO/docs/results/2026-09-20-metal-groupnorm/scripts/compare_cond.py $NUM | tee $NUM/compare.txt
echo "P2NUM DONE $(date)"
