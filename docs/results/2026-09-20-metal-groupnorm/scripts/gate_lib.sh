# 静穏ゲート（issue #55 で改訂）: 他の trellis バイナリが居ない **かつ** GPU の Device Utilization（ioreg、root 不要）が
# 90 s 連続で GPU_QUIET% 未満（5 s おきにサンプル）。WindowServer / Chrome / Claude app など「trellis 以外の GPU 利用」を
# ps では捕まえられなかった（#55 の 1 回目の A/B は GPU 100% 状態で回っていた）ので、GPU 使用率を直接見る。
# 2026-09-20 17:19 に別セッション（pixal3d-cond-sparse）の同型ゲートと同時に開いて衝突したので、
#   (1) 静穏後に 3〜15 s の乱数待ち → 再確認（両者が同じ静穏窓の終わりで同時起動するのを避ける）、
#   (2) 協調ロック /tmp/pixal3d-metal-gpu.lock（mkdir で原子的に取得、owner / 開始時刻 / 予定終了を書く。相手も同じ
#       パスを使えば順番待ちになる。90 分より古いロックは放置とみなして奪う）、
#   (3) run_guarded: 起動後 25 s 以内に他の trellis を見たら自分の実行だけ止めて再試行（最大 3 回）、
# を足した。使い方: source gate_lib.sh; SELFKEY=<自分の worktree 名>; SNAP=<ps ログ>; gate; run_guarded <logfile> <cmd...>
GPU_QUIET=${GPU_QUIET:-25}
LOCK=${LOCK:-/tmp/pixal3d-metal-gpu.lock}
WAITDIR=${WAITDIR:-/tmp/pixal3d-metal-gpu.waiting}
QUIET_SECS=${QUIET_SECS:-90}   # 静穏が続く必要秒数（ab_gn2 の再開では 30）
gpu_util() { ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
others() { ps -Ao pid,etime,comm -r | grep -E "trellis-(cli|server|test|smoke)|post-replay" | grep -v "$SELFKEY/build-metal" ; }
snapshot() { { echo "== $(date '+%H:%M:%S') $1 gpu_util=$(gpu_util) load=$(sysctl -n vm.loadavg | tr -d '{}')"; ps -Ao pid,pcpu,etime,args -r | head -8 | cut -c1-140; echo "-- trellis:"; ps -Ao pid,pcpu,etime,args -r | grep -E "trellis-|post-replay" | grep -v grep | cut -c1-140; } >> $SNAP; }
quiet_now() { local u=$(gpu_util); [ -z "$u" ] && u=0; [ -z "$(others)" ] && [ "$u" -lt "$GPU_QUIET" ]; }
lock_acquire() {  # 協調ロック。相手が同じパスを使わなければ単に自分の目印
  # 2026-09-20 18:30 公平化（pixal3d-cond-sparse からの提案）: 解放直後に取り直すと 10 s ポーリングの相手が飢餓するので、
  # 待つ側は $WAITDIR/<SELFKEY> を置き、取る側は自分以外の待機者（30 分より新しいもの）が居れば 20 s 譲ってから試す。ポーリングは 1 s。
  mkdir -p "$WAITDIR" 2>/dev/null
  if [ -n "$(find "$WAITDIR" -type f -mmin -30 ! -name "$SELFKEY" 2>/dev/null)" ]; then echo "gate: yielding 20 s to other waiter(s): $(ls "$WAITDIR" | tr '\n' ' ')"; sleep 20; fi
  while ! mkdir "$LOCK" 2>/dev/null; do
    touch "$WAITDIR/$SELFKEY"
    # mkdir 失敗と stat の間に相手が解放するとロックが消えている（18:35 に age=1.7e9 s で「放置」と誤判定した実績）。
    # その場合は何も消さずに取り直す（相手が直後に再取得していたら rm -rf で相手のロックを消してしまう）。
    local mt=$(stat -f %m "$LOCK" 2>/dev/null)
    if [ -z "$mt" ]; then sleep 1; continue; fi
    local age=$(( $(date +%s) - mt ))
    if [ $age -gt 5400 ]; then echo "gate: stale lock ($age s) -> taking over"; rm -rf "$LOCK"; continue; fi
    sleep 1
  done
  rm -f "$WAITDIR/$SELFKEY"
  echo "owner=$SELFKEY pid=$$ since=$(date '+%H:%M:%S') expect=$1" > "$LOCK/owner"
}
lock_release() { [ -f "$LOCK/owner" ] && grep -q "pid=$$" "$LOCK/owner" && rm -rf "$LOCK"; }
gate() {  # gate [expected-duration-note]
  local quiet=0 n=0 u
  lock_acquire "${1:-?}"
  while true; do
    quiet=0
    while [ $quiet -lt $QUIET_SECS ]; do
      if quiet_now; then quiet=$((quiet+5)); else quiet=0; n=$((n+1)); [ $((n % 60)) -eq 1 ] && echo "gate: waiting ($(date '+%H:%M:%S')) gpu=$(gpu_util)% others=$(others | wc -l | tr -d ' ')"; fi
      sleep 5
    done
    sleep $(( 3 + RANDOM % 13 ))       # ジッタ 3〜15 s
    if quiet_now; then break; fi        # 再確認
    echo "gate: re-check failed after jitter ($(date '+%H:%M:%S')), waiting again"
  done
  echo "gate: passed $(date '+%H:%M:%S') gpu=$(gpu_util)%"
}
gpu_sampler_start() {  # gpu_sampler_start <file>: 5 s おきに "HH:MM:SS util" を追記（run 中の他者の GPU 利用を残す）。停止は kill $GPU_SP
  ( while true; do echo "$(date '+%H:%M:%S') $(gpu_util)"; sleep 5; done ) >> "$1" & GPU_SP=$!
}
run_guarded() {  # run_guarded <logfile> <cmd...>: 起動後 25 s 以内に他 trellis を見たら止めて再ゲート（最大 3 回）
  local log=$1; shift
  local attempt=1 rc
  while true; do
    "$@" > "$log" 2>&1 & local P=$!
    local collided=0
    for i in 1 2 3 4 5; do sleep 5; if [ -n "$(others)" ]; then collided=1; break; fi; kill -0 $P 2>/dev/null || break; done
    if [ $collided -eq 1 ] && [ $attempt -lt 4 ]; then
      echo "run: collision within 25 s (attempt $attempt), stopping my run and re-gating"; kill $P 2>/dev/null; wait $P 2>/dev/null
      attempt=$((attempt+1)); lock_release; gate; continue
    fi
    wait $P; rc=$?; return $rc
  done
}
