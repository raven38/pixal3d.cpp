#!/bin/bash
# B3-REF R1: 参照(PyTorch)側 stochastic mode 黒化率シード掃引の自然終了型pod driver。
# 設計: docs/design/2026-09-22-trellis2-mv-b3-ref-seed-sweep.md
# lessons.md「数分以上かかる計算をsleep Pod+kubectl execで回さない」に従い、
# このスクリプト自体をpod commandとして直接実行し、終了と同時にpodも終了させる。
set -uo pipefail
OUT=/nfs/trellis2-mv/b3-ref-sweep
SCRIPTS=$OUT/scripts
mkdir -p "$OUT"

export HF_HOME=/nfs/.cache/huggingface HF_HUB_OFFLINE=1
PY=/nfs/envs/miniforge3/envs/trellis2/bin/python

echo "=== [$(date '+%F %T')] selftest ==="
$PY "$SCRIPTS/ref_trellis2_mv_seed_sweep.py" --selftest 2>&1 | tee "$OUT/selftest.log"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "=== [$(date '+%F %T')] SELFTEST_FAIL rc=$rc ==="
  exit "$rc"
fi

echo "=== [$(date '+%F %T')] sweep start ==="
$PY "$SCRIPTS/ref_trellis2_mv_seed_sweep.py" --out "$OUT" 2>&1 | tee "$OUT/sweep.log"
rc=${PIPESTATUS[0]}
echo "=== [$(date '+%F %T')] sweep finished rc=$rc ==="
echo "$rc" > "$OUT/sweep.rc"
exit "$rc"
