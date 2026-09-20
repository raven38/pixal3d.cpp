#!/usr/bin/env python3
"""E2E ログの検証: (1) [cond] accounted と段合計の一致、(2) 参照ログとの行接頭辞集合の差、(3) [cond]/[cond-v] 抽出。
使い方: verify_e2e.py <e2e_cond.log> <ref_e2e.log>"""
import re, sys
log, ref = sys.argv[1], sys.argv[2]
L = open(log, errors="replace").read().splitlines()
Rf = open(ref, errors="replace").read().splitlines()

# (1) accounted vs 段合計
stage_total = {
    "SS": re.compile(r"active voxels @res32 = \d+\s+\(([\d.]+)s\)"),
    "LR shape SLAT": re.compile(r"^\s+LR shape SLAT \(([\d.]+)s\)"),
    "HR shape SLAT": re.compile(r"^\s+HR shape SLAT \(([\d.]+)s\)"),
    "shape decode": re.compile(r"^\s+shape decode \(([\d.]+)s\)"),
    "texture SLAT + decode": re.compile(r"^\s+texture SLAT \+ decode \(([\d.]+)s\)"),
}
acc = re.compile(r"\[cond\] (.+?) accounted \(([\d.]+)s\)")
laps = re.compile(r"\[cond\] (?!.*accounted)(.+?) \(([\d.]+)s\)$")
print("## (1) [cond] accounted vs stage total")
cur_sum = 0.0; pending_acc = None
for ln in L:
    m = laps.search(ln)
    if m: cur_sum += float(m.group(2)); continue
    m = acc.search(ln)
    if m: pending_acc = (m.group(1), float(m.group(2)), cur_sum); cur_sum = 0.0; continue
    for st, rx in stage_total.items():
        m = rx.search(ln)
        if m and pending_acc:
            name, a, s = pending_acc; tot = float(m.group(1))
            ok = abs(a - tot) <= 0.1 and abs(s - a) <= 0.1 + 0.05 * (s > 0)
            print(f"  {st:24s} total {tot:7.1f}  accounted {a:7.1f}  sum(laps) {s:7.1f}  diff {a-tot:+.1f}  {'OK' if ok else 'MISMATCH'}")
            pending_acc = None

# (2) 接頭辞集合（数値・パスを正規化してから）
def norm(s):
    s = re.sub(r"/[^\s]+", "<path>", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "<hex>", s)
    s = re.sub(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
SL = {norm(x) for x in L}; SR = {norm(x) for x in Rf}
only_new = sorted(SL - SR); only_ref = sorted(SR - SL)
print(f"\n## (2) prefix set: only in --profile-cond log ({len(only_new)}), only in ref ({len(only_ref)})")
non_cond = [x for x in only_new if not x.startswith("[cond")]
print(f"  new lines not starting with [cond]/[cond-v]: {len(non_cond)}")
for x in non_cond[:40]: print("   +", x)
print(f"  ref-only lines: {len(only_ref)}")
for x in only_ref[:40]: print("   -", x)

# (3) 抽出
print("\n## (3) [cond] lines")
for ln in L:
    if "[cond]" in ln or ln.startswith("[") and "/6]" in ln or re.search(r"\((\d+\.\d)s\)$", ln) and "[cond" not in ln:
        print(ln.rstrip())
