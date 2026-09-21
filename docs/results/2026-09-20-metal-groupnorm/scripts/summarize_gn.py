#!/usr/bin/env python3
"""issue #55 の P1 / P2 ログ集計。ab/ の <arm>_<name>_<HHMMSS>.log を読み、
P1（nafops_*）: whole graph 秒、GROUP_NORM / NORM の op 合計、compute ms、（あれば）native 比の L2rel。
P2（tex / hr / lr）: [gpu] total ms と slowest view。arm（new / old）× name ごとに個別値と中央値を出す。
使い方: summarize_gn.py <ab_dir> [--accept accepted.txt]（accepted.txt = 採用するログのファイル名、1 行 1 つ。無ければ全部）
<log>.gpu（run 中の GPU 使用率、5 s おき）は自分の run を含む合計なので汚染基準には使わない（最初 / 最後のサンプルだけ参考表示）。"""
import re, sys, os, glob, statistics
d = sys.argv[1]; accept = None
if "--accept" in sys.argv:
    accept = set(l.strip() for l in open(sys.argv[sys.argv.index("--accept") + 1]) if l.strip())
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    b = os.path.basename(f)
    if accept is not None and b not in accept: continue
    m = re.match(r"(new|old)_(\S+)_(\d{6})\.log", b)
    if not m: continue
    arm, name, t = m.groups(); s = open(f, errors="replace").read()
    r = {"file": b, "time": t}
    # run 中の GPU 使用率（gate_lib.sh gpu_sampler_start、5 s おき）。ioreg の Device Utilization は自分の run を含む合計なので
    # （自分の compute 中は 96〜100 %）他者の利用と区別できず、汚染基準には使えない。参考として最初と最後のサンプル（自分の compute 前後）だけ出す。
    gpu = [int(l.split()[1]) for l in open(f + ".gpu", errors="replace") if len(l.split()) == 2 and l.split()[1].isdigit()] if os.path.exists(f + ".gpu") else []
    r["gpu_edge"] = (gpu[0], gpu[-1]) if len(gpu) >= 2 else None
    if name.startswith("nafops"):
        w = re.search(r"whole graph ([\d.]+)s", s); r["whole"] = float(w.group(1)) if w else None
        gn = 0.0
        for op in ("GROUP_NORM", "NORM"):
            for mm in re.finditer(r"\[prof\]\s+%s\s+([\d.]+)s\s+[\d.]+%%\s+\(n=(\d+), ([\d.]+) ms/op\)" % op, s):
                gn += float(mm.group(1)); r[op] = "%s s (n=%s, %s ms/op)" % (mm.group(1), mm.group(2), mm.group(3))
        r["gn_total"] = gn
        c = re.search(r"naf-ops: .*compute=(\d+) ms", s); r["compute_ms"] = int(c.group(1)) if c else None
        l2 = re.search(r"L2rel=([\d.e+-]+)", s); r["L2rel_vs_native"] = l2.group(1) if l2 else None
        g = re.search(r"generic_gn=(\d)", s); r["generic_gn"] = g.group(1) if g else "?"
    else:
        t2 = re.search(r"\[gpu\].*total=(\d+) ms\s+slowest view=(\d+) ms", s)
        r["total_s"] = int(t2.group(1)) / 1000.0 if t2 else None; r["slowest_view_s"] = int(t2.group(2)) / 1000.0 if t2 else None
        # per-view NAF encoder compute (chunked tex path: "naf_enc x2: ... compute X"), or the whole single-graph compute
        enc = [float(m.group(1)) for m in re.finditer(r"naf_enc x\d+: build [\d.]+ up [\d.]+ compute ([\d.]+)", s)]
        single = [float(m.group(1)) for m in re.finditer(r"cond_slat_gpu_single[^\n]*\| compute ([\d.]+) \|", s)]
        r["naf_enc_view"] = statistics.median(enc) if enc else None
        r["single_compute_view"] = statistics.median(single) if single else None
    rows.setdefault((name, arm), []).append(r)
for (name, arm) in sorted(rows):
    rs = rows[(name, arm)]
    if name.startswith("nafops"):
        vals = [r["whole"] for r in rs if r["whole"] is not None]
        print(f"{name:14s} {arm:4s} n={len(vals)} whole graph: " + " / ".join(f"{v:.2f}" for v in vals) +
              (f"  median {statistics.median(vals):.2f} / min {min(vals):.2f} s" if vals else "") +
              "  gn_total: " + " / ".join(f"{r['gn_total']:.2f}" for r in rs) +
              "  generic_gn=" + ",".join(r["generic_gn"] for r in rs) +
              "  L2rel_vs_native=" + ",".join(str(r["L2rel_vs_native"]) for r in rs))
    else:
        vals = [r["total_s"] for r in rs if r["total_s"] is not None]
        print(f"{name:14s} {arm:4s} n={len(vals)} total: " + " / ".join(f"{v:.1f}" for v in vals) +
              (f"  median {statistics.median(vals):.1f} s (min {min(vals):.1f} / max {max(vals):.1f})" if vals else "") +
              "  slowest view: " + " / ".join(f"{r['slowest_view_s']:.2f}" for r in rs if r["slowest_view_s"] is not None))
        enc = [r["naf_enc_view"] for r in rs if r["naf_enc_view"] is not None]
        sg = [r["single_compute_view"] for r in rs if r["single_compute_view"] is not None]
        if enc: print(f"{'':19s} naf_enc compute/view (median of 4 views): " + " / ".join(f"{v:.2f}" for v in enc) + f"  median {statistics.median(enc):.2f} / min {min(enc):.2f}")
        if sg: print(f"{'':19s} single-graph compute/view (DINO+NAF+taps): " + " / ".join(f"{v:.2f}" for v in sg) + f"  median {statistics.median(sg):.2f} / min {min(sg):.2f}")
    for r in rs: print("    ", r["file"], f"gpu first/last sample={r['gpu_edge'][0]}/{r['gpu_edge'][1]}%" if r["gpu_edge"] else "gpu samples: n/a")
