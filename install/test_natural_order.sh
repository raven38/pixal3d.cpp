#!/usr/bin/env bash
# install/generate.sh の natural_sort が C++ 側 natural_name_less と同じ順序を返すことを確かめる。
# 引数に trellis-test-transforms-synth を渡すとその出力と突き合わせ、渡さなければ
# 下の EXPECTED（同じ規則を人手で書き下したもの）と比較する。
#
#   install/test_natural_order.sh [path/to/trellis-test-transforms-synth]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# generate.sh から natural_sort 定義だけを取り込む（実行はしない）
source <(sed -n '/^natural_sort()/,/^}/p' "$here/generate.sh")

NAMES=(view10.png view2.png view02.png view1.png view00_azim000.png view03_azim270.png
       view01_azim090.png view02_azim180.png a.png A.png v1x10.png v1x2.png img.png
       0.png 00.png 1.png 010.png 10.png)

# 期待順。規則は natural_name_less（数字列は桁数→辞書順、それ以外はバイト比較、数値が
# 同値なら元のバイト列）。読み方の例:
#   "010.png" < "10.png"  — 数値は同値なので元のバイト列で決まる（'0' < '1'）
#   "view1.png" < "view2.png" < "view02_azim180.png" — 02 と 2 は同値なので続きを見て
#                                                      '.'(0x2E) < '_'(0x5F) で view2 が先
#   "A.png" < "a.png"     — ASCII の大文字が先（locale に依存しない）
EXPECTED="0.png
00.png
1.png
010.png
10.png
A.png
a.png
img.png
v1x2.png
v1x10.png
view00_azim000.png
view1.png
view01_azim090.png
view02.png
view2.png
view02_azim180.png
view03_azim270.png
view10.png"

shell_out="$(printf '%s\n' "${NAMES[@]}" | natural_sort)"

if [ $# -ge 1 ]; then
  cpp_out="$("$1" --sort "${NAMES[@]}")"
  if [ "$shell_out" != "$cpp_out" ]; then
    echo "FAIL: shell natural_sort differs from natural_name_less" >&2
    diff <(printf '%s\n' "$shell_out") <(printf '%s\n' "$cpp_out") >&2 || true
    exit 1
  fi
  echo "ok   shell natural_sort matches natural_name_less (${#NAMES[@]} names)"
fi

if [ "$shell_out" != "$EXPECTED" ]; then
  echo "FAIL: shell natural_sort differs from the expected order" >&2
  diff <(printf '%s\n' "$shell_out") <(printf '%s\n' "$EXPECTED") >&2 || true
  exit 1
fi
echo "ok   shell natural_sort matches the expected order"
echo "PASSED"
