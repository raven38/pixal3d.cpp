#!/usr/bin/env python3
"""ps スナップショットから各実行の汚染を設計 §5 の事前基準で分類する。
基準: (a) 自分以外の trellis バイナリ（trellis-*, post-replay）が 1 つでも居た → 汚染（GPU/CPU を使う）。
      (b) trellis 以外で CPU 50% 超のプロセスが「継続」（その実行の before〜after のスナップショットのうち 2 つ以上）→ 汚染。
      (c) fresh の前窓: before の 90 s 前までのスナップショット（前の実行の after / periodic）に他の trellis バイナリが居たら
          「pre-window」として別途表示する（設計の fresh = 直前 90 s 以上 GPU を使う trellis 無し）。
      suggestd（macOS CoreSuggestions、2 日以上 93〜98% で常駐、全実行・全 arm に共通）は別枠で数える。
使い方: classify_runs.py <ps_snapshots.log>  （自分のバイナリは pixal3d-cond-profile/ 配下）"""
import re, sys
from collections import defaultdict
log = sys.argv[1]; selfkey = "pixal3d-cond-profile"
blocks = []
for ln in open(log, errors="replace"):
    m = re.match(r"^== (\d\d:\d\d:\d\d) (.*)", ln)
    if m: blocks.append([m.group(1), m.group(2).strip(), []]); continue
    if not blocks: continue
    m = re.match(r"^\s*(\d+)\s+([\d.]+)\s+([\d:-]+)\s+(.*)", ln)
    if m: blocks[-1][2].append((float(m.group(2)), m.group(4).strip()))
runs = []; i = 0
while i < len(blocks):
    if blocks[i][1].startswith("before"):
        # 対になる after（同名）まで。無ければ次の before / gate まで（naf-ops は after を取っていない）
        j = i + 1
        while j < len(blocks) and not (blocks[j][1].startswith("after") or blocks[j][1].startswith("before") or blocks[j][1].startswith("gate")): j += 1
        end = j if j < len(blocks) and blocks[j][1].startswith("after") else j - 1
        # 前窓: before の 90 s 前以降のブロック（同じ実行の before は含めない）
        def secs(t): h, m, s_ = t.split(":"); return int(h) * 3600 + int(m) * 60 + int(s_)
        pre = [b for b in blocks[:i] if 0 <= secs(blocks[i][0]) - secs(b[0]) <= 90]
        runs.append((blocks[i][1][7:], blocks[i][0], blocks[end][0], blocks[i:end + 1], pre)); i = end + 1
    else: i += 1
is_trellis = lambda a: ("trellis" in a or "post-replay" in a) and selfkey not in a and "classify" not in a
for name, t0, t1, bl, pre in runs:
    other = set(); heavy = defaultdict(int); sugg = 0
    pre_trellis = set()
    for b in pre:
        for cpu, a in b[2]:
            if selfkey in a: continue
            if is_trellis(a): pre_trellis.add(re.sub(r"\s+/Users.*|\s+/private.*|\s+dist.*", "", a)[:60])
    for b in bl:
        seen = set()
        for cpu, a in b[2]:
            if selfkey in a: continue
            if is_trellis(a): other.add(re.sub(r"\s+/Users.*|\s+/private.*|\s+dist.*", "", a)[:60]); continue
            if "suggestd" in a or "CoreSuggestions" in a: sugg += 1 if cpu > 50 and "s" not in seen else 0; seen.add("s"); continue
            if cpu > 50:
                k = a[:50]
                if k not in seen: heavy[k] += 1; seen.add(k)
    sustained = {k: v for k, v in heavy.items() if v >= 2}
    verdict = "CONTAMINATED" if (other or sustained) else "clean"
    pw = f"  pre-window(90s, {len(pre)} snaps): {'OTHER TRELLIS' if pre_trellis else 'ok'}"
    print(f"{name:14s} {t0}-{t1} {len(bl):2d} snaps  {verdict:12s} suggestd>50% in {sugg}/{len(bl)}{pw}")
    for a in sorted(pre_trellis): print("    pre-window trellis:", a)
    for a in sorted(other): print("    other trellis:", a)
    for k, v in sustained.items(): print(f"    cpu>50% x{v}: {k}")
