# codex 設計+実装レビュー（issue #55、`docs/design/2026-09-20-metal-groupnorm.md`）— 2026-09-20

実行: `codex exec -m gpt-5.6-luna -s read-only`（1 回目は既定モデル gpt-5.6-sol で、私の並行編集を検知して最終回答なしで終了 →
2 回目は「Selected model is at capacity」→ 3 回目 luna で完走）。プロンプトは設計書 + `git diff HEAD`（実装差分）+ 反証指向の 8 論点。
判定: **差し戻し**。以下、原文 → 対応。

## 対応（作者）

| 指摘 | 判断 | 対応 |
|---|---|---|
| N1 の合否が CPU backend で必ず FAIL | 正しい（実測で再現） | 合否を「その backend で `naf_ggml_opts_for()` が選ぶ構成」に対して判定する形に変更。Metal / CPU とも PASS を確認 |
| `TRELLIS_DBG_NAF_GENERIC=0` の拡張は README の env 契約と矛盾 | 正しい（既存の env 自体も矛盾しているが、広げない） | env 拡張を撤回し、`--no-fa` と同型の CLI フラグ `--naf-native-gn`（`trellis_args` → `g_naf_native_gn`）に変更。README のフラグ表に追加 |
| E1 は noise 差の別サンプルを V/F/bbox/目視で「非破壊」と判定していて甘い | 正しい | E1 を「段の時間 + 壊れていないスモーク」に格下げし、数値の非回帰は N1 / N2 / 新設の P2-num（flow に入る cond テンソルの新旧比較）で示す。設計書に「品質非回帰の証明ではない」と明記 |
| upstream の barrier はレース修正になっていない | **不同意**（SIMD group 間の順序は barrier でしか保証されない。上流自身の `kernel_norm_fuse_impl` L3011 が同じ位置に barrier を置いている前例あり） | 提案は維持。PR 本文では「観測された故障は無い、形式的な欠落、`kernel_norm_fuse_impl` と同じ守り」と書く |
| N2 の CPU 比較は GroupNorm の切り分けにならない | 正しい（切り分けは N1 の役目） | N2 に `enc_cat`（encoder 出力、attention 前）の比較を追加し、設計書の主張を「方向の確認」に限定 |
| `--gn-oracle` の引数無検査 / `--naf-compare-cpu` のサイズ・finite 未確認 | 正しい | 両方直した |
| backend 名の接頭辞判定より `ggml_backend_is_metal()` | 正しい | `GGML_USE_METAL` 下で `ggml_backend_is_metal()` に変更 |
| kernel の `int start/end`・`ne02 % ngrp` の除数 | 正しいが既存の問題 | 範囲外。PR 本文で触れる |
| `nm` が未使用 | 誤り（L2rel の分子に使っている） | 変更なし |

## 原文

### レビュー結果サマリー
- **判定**: 差し戻し
- **重大な問題**:
  - N1 の合否判定が CPU backend でも Metal 用条件を適用し、提示済みの CPU 実測で必ず FAIL する（`src/test_naf.cpp:461-466`）。
  - `TRELLIS_DBG_NAF_GENERIC=0` を挙動変更用に拡張しており、README の「挙動を変える環境変数は無い」と矛盾する（`src/naf_gpu.cpp:180-182`, `README.md:144-145`）。
  - E1 は noise 差による別サンプルを許容したまま、V/F・bbox ±5% と目視だけで非破壊と判定しており、条件差による大きな品質劣化を通過させる。
  - upstream 提案の追加 barrier は、現カーネルの同期関係を見る限りレース修正になっていない（`ggml-metal.metal:3218-3249`）。
  - N2 の CPU 比較は NAF 全体の比較であり、差分が GroupNorm 由来か f16 GEMM 由来かを証明できない。
- **軽微な指摘**:
  - `--gn-oracle` はオプションだけを渡すと `p[0]` を無検査で参照する（`src/test_naf.cpp:475-487`）。
  - backend 名の接頭辞判定は現 submodule では動くが、公開 API の `ggml_backend_is_metal()` より脆い。
  - kernel の `start/end` と `gs` は 32-bit 演算・`int` に依存しており、大きなテンソルで範囲外になる。
  - `--naf-compare-cpu` の比較ループにサイズ・finite チェックがない。

### 詳細（上の 1〜8 に番号で対応。各項目に根拠のファイル:行）

1. backend 判定

`strncmp(..., "MTL", 3)` は、この submodule の実装では `MTL0`、`MTL1` という backend 名が生成されるため動作します（`thirdparty/ggml/src/ggml-metal/ggml-metal.cpp:293-298`）。

ただし、Metal 判定の意味で backend 名文字列を利用する必要はありません。公開 API に `ggml_backend_is_metal()` があり、GUID で判定します（`thirdparty/ggml/include/ggml-metal.h:42-45`, `ggml-metal.cpp:616-618`）。

したがって現状は「直ちに壊れる」ものではありませんが、backend 名の命名変更やラッパー backend で誤判定する設計です。`ggml_backend_is_metal(naf.backend)` を使う方が妥当です。

`bname` を空文字列にして null backend を安全に扱っている点は問題ありません（`src/naf_gpu.cpp:173-174`）。

2. kill-switch の環境変数拡張

実装は `TRELLIS_DBG_NAF_GENERIC=0` のとき Metal の `generic_groupnorm` も無効化します（`src/naf_gpu.cpp:180-182`）。

しかし README は、`TRELLIS_DBG_*` はデバッグログだけであり、挙動変更用の環境変数は残していないと明記しています（`README.md:144-145`）。設計書の「診断専用だから問題ない」という説明だけでは、この既存契約を上書きできません（`docs/design/2026-09-20-metal-groupnorm.md:37-42`）。

さらにこれは `naf_ggml_opts_for()` を通る production/device 経路にも作用します（`src/pixal3d_cond_gpu.cpp:197`, `src/pixal3d_cond_gpu.cpp:525`）。環境変数が残ったプロセスやサービス環境により、意図せず計算経路が変わります。

CLI フラグを追加しない方針を維持するなら、少なくともこの環境変数を production 経路から隔離する明示的な条件が必要です。現状のままなら設計・README・実装の契約不一致です。

3. N1 の Metal 限定性

N1 の実装は backend に関係なく以下を評価します（`src/test_naf.cpp:461-466`）。

```cpp
rows[1].maxd <= 1e-3 &&
rows[1].maxd <= 1.5 * rows[0].maxd
```

提示された CPU 実測では、

- CPU native: `9.5e-7`
- CPU lowering: `2.17e-3`

なので絶対誤差条件と倍率条件の両方で FAIL します。これは実装の偶然の失敗ではなく、設計書自身が述べる backend 依存の数値特性です（`docs/design/2026-09-20-metal-groupnorm.md:67-68`）。

N1 は次のいずれかに直すべきです。

- Metal backend のときだけ N1 を合否判定する。
- CPU は `REFERENCE_ONLY` として値を印字し、exit code には反映しない。
- backend ごとに別基準を定義する。

少なくとも `gpu=-1` を「参考値」と説明しながら、同じ関数で失敗終了させる現状は受け入れ基準として破綻しています。

4. N2 の CPU 比較と f16 GEMM 仮説

CPU 比較は全 NAF 出力を対象にしています（`src/test_pixal3d_cond_tex.cpp:189-205`）。この差分には少なくとも以下が混在します。

- encoder の f16-staged GEMM
- 畳み込みの reduction order
- GroupNorm
- RoPE
- attention の matmul / softmax
- GPU と CPU の浮動小数点演算順序

したがって、差が約 `1e-4` だから「GroupNorm ではなく Metal の f16 GEMM」と結論することはできません。f16 GEMM は有力な仮説ですが、現在の比較はその切り分けになっていません。

N1 の op 単体比較は GroupNorm の切り分けになっていますが、N2 の全体比較については、

- encoder 出力直後
- GroupNorm 後
- attention 入力
- 最終 NAF 出力

の中間テンソルを同一入力で比較する必要があります。

既存の GPU encoder tolerance `1e-2` と整合することは、「この差を許容してよい」根拠にはなりますが、「差の原因が f16 GEMM」と証明する根拠にはなりません。

5. E1 の非破壊判定

E1 の条件（`docs/design/2026-09-20-metal-groupnorm.md:91`）には大きな穴があります。

まず、LR conditioning の差により後続 HR の noise 行番号結合が変わり、別サンプルになることを設計書自身が認めています。その状態で、

- active voxel 数一致
- GLB の V/F と bbox が ±5%
- 5 視点目視

だけでは、意味的な品質劣化や局所的な形状崩壊を検出できません。

特に V/F と bbox は、以下の失敗を通過します。

- 頂点位置が大きく変わっても同じ bbox に収まる
- 左右非対称化しても V/F が同じ
- 部分的な欠損・突起・穴があっても全体 bbox が ±5% に収まる
- テクスチャが別物でも GLB 構造値だけ一致する

5 視点目視は必要ですが、noise が異なる別サンプルの比較では「非破壊」の判定になりません。少なくとも以下のどちらかが必要です。

- HR/texture noise と中間 SLAT を固定して、同一サンプルの数値比較を行う。
- noise 差を認めるなら、複数 seed・複数入力で品質分布を比較し、単一 E2E の before/after 判定にしない。

`active voxels @res32` は NAF 非依存の SS 経路なので、同一 seed なら回帰検出として有効です。しかし、それは NAF 変更の非破壊性を保証しません。

6. upstream 提案 §2.4

(a) `nth` を group size 基準にする点は方向として正しいです。カーネルは実際に、

```metal
gs = ne00 * ne01 * ceil(ne02 / ngrp)
```

で group の要素数を計算し、`tgpig` ごとにその範囲を処理しています（`thirdparty/ggml/src/ggml-metal/ggml-metal.metal:3187-3201`）。したがって `ne00/4` を基準にするより group size 基準の方が意味が合います。

ただし実装提案には次の不足があります。

- `ne00 * ne01` が 32-bit のまま評価される可能性がある。
- kernel 側の `start` / `end` は `int`（`thirdparty/ggml/src/ggml-metal/ggml-metal.metal:3190-3191`）。
- `gs` は `int64_t` に代入される前の乗算で overflow し得る。
- `ne02` が `ngrp` で割り切れない場合、最後の group は実要素数が `gs` 未満なのに `mean` と `variance` を `gs` で割る（`thirdparty/ggml/src/ggml-metal/ggml-metal.metal:3224`, `3251`）。GGML API 側に除数条件の検証もありません（`thirdparty/ggml/src/ggml.c:3168-3182`）。

(b) zero-fill 前の barrier については、提示されたレース説明を支持できません。

最初の縮約は、

- `buf` 初期化後に barrier（`3212`）
- `buf[sgitg]` 書き込み後に barrier（`3218`）
- 全 thread が `buf` を読み取り（`3220`）

という順序です。

その後、mean を計算して variance ループへ進み、zero-fill は `3235-3237` で行われます。variance 側の `buf` 読み取りは `3247` であり、その前に既存の barrier `3239` があります。

したがって、variance 側 zero-fill の前に barrier を追加しても、現在のコード上は新たな同期保証になりません。zero-fill と前段 `buf` 読み取りの間には、すでに mean reduction 完了後の計算区間があり、前段の `buf` 読み取りは `3220` で終わっています。

upstream 提案に含めるなら、まず race を再現する最小 kernel または Metal の race 検証結果が必要です。現状の説明は「形式的なレース」としては根拠不足です。

7. `--naf-compare-cpu` とレイアウト

レイアウト自体は一致しています。

CPU 側は `[C,T,T]` の channel-major で出力します（`src/naf.cpp:405-411`）。GGML 側も最終的に block-major から `[C,T,T]` raster に戻しています（`src/naf_gpu.cpp:439-446`）。したがって、

```cpp
naf_upsample(...)
naf_upsample_ggml(...)
```

の比較対象レイアウトに明白な取り違えはありません。

ただし比較コードには問題があります。

- `cpu.size() == a.size()` を確認していない。
- finite 値を確認していない。
- CPU 比較は GroupNorm 単体ではなく全 NAF 経路である。
- `nm` が計算されるが使われていない（`src/test_pixal3d_cond_tex.cpp:194-201`）。

また、`--naf-compare-cpu` は `naf_compare=true` にしてから、lowering/native 比較ブロックの中でのみ実行されます（`src/test_pixal3d_cond_tex.cpp:170-186`）。`o` が native のままなら CPU 比較を実行しない仕様ですが、CLI の説明だけではこの条件が明確ではありません。

8. 影響範囲

Metal では `naf_ggml_opts_for()` が `generic_groupnorm=true` を返すため、以下の経路に影響します。

- host の `naf_upsample_ggml`（`src/naf_gpu.cpp:364-369`）
- device sparse 経路（`src/pixal3d_cond_gpu.cpp:197`, `525`）
- `test_naf --ggml`（`src/test_naf.cpp:289-297`）
- Metal の `naf_upsample()` 自動 dispatch（`src/naf.cpp:441-448`）

CUDA では `bname` が `CUDA...` のため `generic_groupnorm=false` です。通常の `naf_upsample()` は CUDA 専用経路に分岐し、そこでは native 設定が明示されています（`src/naf.cpp:427-448`, `src/naf_gpu.cpp:543-545`）。この点は意図した範囲です。

WebGPU では `generic_groupnorm` は false のままで、既存の `generic_lowering` 判定が維持されます（`src/naf_gpu.cpp:168`, `184-187`）。ただし `TRELLIS_DBG_NAF_GENERIC=0` は従来どおり WebGPU の generic lowering を無効化し、今回さらに Metal の GroupNorm も無効化します。つまりこの環境変数の影響範囲は「Metal だけ」ではありません。

総じて、実装の主経路そのものは Metal に限定されますが、受け入れ判定、診断用環境変数、E2E 比較、upstream race 説明が未分離です。特に N1 の backend 非依存判定と E1 の noise 非固定は、反例が既に存在するため、差し戻しが妥当です。