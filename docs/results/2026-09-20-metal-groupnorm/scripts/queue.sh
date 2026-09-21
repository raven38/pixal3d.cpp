#!/bin/zsh
# #55 の計測キュー: E1（e2e_gn.sh）→ P1/P2/P2-num 再計測（ab_gn2.sh）を直列に回す。どちらも gate_lib.sh の
# GPU 静穏ゲート付き（他 trellis バイナリ無し + GPU 使用率 < 25% が 90 s）。
REPO=/Users/s25705/Downloads/pixal3d-metal-groupnorm
D=$REPO/docs/results/2026-09-20-metal-groupnorm
echo "QUEUE: start $(date)"
zsh $D/scripts/e2e_gn.sh > $D/e2e_gn_driver.log 2>&1
echo "QUEUE: e2e_gn.sh finished $(date)"
zsh $D/scripts/ab_gn2.sh > $D/ab_gn2_driver.log 2>&1
echo "QUEUE: ab_gn2.sh finished $(date)"
