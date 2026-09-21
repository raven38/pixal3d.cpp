#!/usr/bin/env python3
"""クリーン A/B のログ（clean/<tag>_<mode>_<hhmmss>.log）を集計して Markdown 表を出す。"""
import glob, os, re, statistics as st, sys
# 使い方: summarize_ab.py [dir ...] [--accept <list>]  （--accept: 採用するログの basename を 1 行 1 つ。無ければ全部）
args = sys.argv[1:]; accept = None
if '--accept' in args:
    i = args.index('--accept'); accept = set(l.strip() for l in open(args[i + 1]) if l.strip() and not l.startswith('#')); del args[i:i + 2]
dirs = args or [os.path.join(os.path.dirname(__file__), 'clean')]
runs = {}
files = sorted(f for D in dirs for f in glob.glob(os.path.join(D, '*.log')))
for f in files:
    b = os.path.basename(f)
    m = re.match(r'(T1024|T512|LR)[ _](gpu|host)_(\d{6})\.log', b)
    if not m: continue
    if accept is not None and b not in accept: continue
    tag, mode, hms = m.groups()
    txt = open(f, errors='replace').read()
    tot = None
    mm = re.search(r'^gpu path: .*?  ([\d.]+) s$', txt, re.M) if mode == 'gpu' else re.search(r'^host path: ([\d.]+) s$', txt, re.M)
    if mm: tot = float(mm.group(1))
    views = []
    if mode == 'host':
        for v in re.finditer(r'cond_slat view (\d+): normalize ([\d.]+) \| dino ([\d.]+) \| chw ([\d.]+) \| lr_proj ([\d.]+) \| naf ([\d.]+) \| hr_proj ([\d.]+) \| accum ([\d.]+)  \(view ([\d.]+)s\)', txt):
            views.append(dict(zip(['v','normalize','dino','chw','lr_proj','naf','hr_proj','accum','view'], [float(x) for x in v.groups()])))
        nafl = {}
        for k in ['graph alloc','inputs','graph compute','readback','unpermute','free']:
            vals = [float(x) for x in re.findall(r'naf_ggml S=\d+ T=\d+ ' + re.escape(k) + r'\s+\+\s*([\d.]+) ms', txt)]
            if vals: nafl[k] = vals
        dino = [float(x) for x in re.findall(r'dinov3 S=\d+ ntok=\d+: build\+alloc [\d.]+ \| upload [\d.]+ \| compute ([\d.]+)', txt)]
    else:
        for v in re.finditer(r'cond_slat_gpu S=\d+ R=\d+ T=\d+ view (\d+): host taps ([\d.]+) \| chunk_prep ([\d.]+)(.*?)\(attn chunks skipped (\d+); laps ([\d.]+) of view ([\d.]+)s\)', txt):
            d = {'v': int(v.group(1)), 'taps': float(v.group(2)), 'chunk_prep': float(v.group(3)), 'skipped': int(v.group(5)), 'laps': float(v.group(6)), 'view': float(v.group(7))}
            for g in re.finditer(r'\| (\w+) x(\d+): build ([\d.]+) up ([\d.]+) compute ([\d.]+) free ([\d.]+)', v.group(4)):
                d[g.group(1)] = dict(n=int(g.group(2)), build=float(g.group(3)), up=float(g.group(4)), compute=float(g.group(5)), free=float(g.group(6)))
            views.append(d)
        nafl, dino = {}, []
    runs.setdefault((tag, mode), []).append(dict(file=b, total=tot, views=views, nafl=nafl, dino=dino, hms=hms))

def med(xs): return st.median(xs) if xs else float('nan')
print('## 合計（別プロセス、ABBAAB、秒）\n')
print('| 条件 | 経路 | n | 中央値 | min | max | 個別（実行順） |')
print('|---|---|---:|---:|---:|---:|---|')
for (tag, mode), rs in sorted(runs.items()):
    tots = [r['total'] for r in rs if r['total'] is not None]
    print(f"| {tag} | {mode} | {len(tots)} | {med(tots):.1f} | {min(tots) if tots else float('nan'):.1f} | {max(tots) if tots else float('nan'):.1f} | {' / '.join(f'{t:.1f}' for t in tots)} |")
print('\n## host 版 view ごとの sub-stage（全実行・全 view の中央値、秒）\n')
print('| 条件 | normalize | dino | chw | lr_proj | naf | hr_proj | accum | view 合計 |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for (tag, mode), rs in sorted(runs.items()):
    if mode != 'host': continue
    vs = [v for r in rs for v in r['views']]
    if not vs: continue
    print(f"| {tag} | " + ' | '.join(f"{med([v[k] for v in vs]):.2f}" for k in ['normalize','dino','chw','lr_proj','naf','hr_proj','accum','view']) + ' |')
print('\n## host 版 naf_upsample_ggml の内訳（中央値、ms）\n')
print('| 条件 | graph alloc | inputs | graph compute | readback | unpermute | free |')
print('|---|---:|---:|---:|---:|---:|---:|')
for (tag, mode), rs in sorted(runs.items()):
    if mode != 'host': continue
    agg = {}
    for r in rs:
        for k, vals in r['nafl'].items(): agg.setdefault(k, []).extend(vals)
    if not agg: continue
    print(f"| {tag} | " + ' | '.join(f"{med(agg.get(k, [])):.0f}" for k in ['graph alloc','inputs','graph compute','readback','unpermute','free']) + ' |')
print('\n## GPU 版（分割グラフ・sparse）view ごとの graph 種別（中央値、秒）\n')
print('| 条件 | host taps | chunk_prep | dino compute | naf_enc compute | naf_qk compute (n) | naf_attn compute (n, skipped) | build+up+free 合計 | view 合計 |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for (tag, mode), rs in sorted(runs.items()):
    if mode != 'gpu': continue
    vs = [v for r in rs for v in r['views']]
    if not vs: continue
    def gm(kind, f): return med([v[kind][f] for v in vs if kind in v])
    def gn(kind): return med([v[kind]['n'] for v in vs if kind in v])
    ovh = med([sum(v[k]['build'] + v[k]['up'] + v[k]['free'] for k in ['dino','naf_enc','naf_qk','naf_attn'] if k in v) for v in vs])
    print(f"| {tag} | {med([v['taps'] for v in vs]):.2f} | {med([v['chunk_prep'] for v in vs]):.2f} | {gm('dino','compute'):.2f} | {gm('naf_enc','compute'):.2f} | {gm('naf_qk','compute'):.2f} ({gn('naf_qk'):.0f}) | {gm('naf_attn','compute'):.2f} ({gn('naf_attn'):.0f}, {med([v['skipped'] for v in vs]):.0f}) | {ovh:.2f} | {med([v['view'] for v in vs]):.2f} |")
