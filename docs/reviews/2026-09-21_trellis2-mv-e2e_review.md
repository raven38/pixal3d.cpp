# 設計レビュー: trellis2-mv 実重み E2E（#65）2026-09-21

対象: `docs/design/2026-09-21-trellis2-mv-e2e.md`

## 経緯

`codex exec` を2回試したが、いずれも MCP (context7) の認証エラー
（`AuthRequired ... Transport channel closed`）でレビュー本文を出力せずに終了した
（環境的制約と判断、`~/.claude/instructions/design_review.md` の代替手順に従う）。
独立 subagent（general-purpose、フレッシュコンテキスト）に反証指向レビューを委任した。

## 判定

**条件付きで進めてよい。条件: A1（Part A の入力 fixture を両 SHA 共通で固定する設計に修正）。**
それ以外（Part B/C の計画、pod 運用、共有資源の扱い）は許容範囲。

## A: 止めるべき重大な指摘（対応済み）

- **A1**: Part A の数値比較（f04b33c vs 0d4cfc5）を、両 SHA が個別に `trellis-cli --image` を実行して
  得た dump 同士で比較する設計だと、SS 段の active voxel 集合が SHA 間で変わり得るため
  （t-schedule 修正の副作用そのもの）、「rel 誤差の差」が t-schedule 修正の効果なのか別 voxel 集合の
  難易度差なのか区別できない。→ **対応**: 数値比較は `tools/ref_slat_shape.py` で1回だけ生成した
  画像非依存の固定 `ref_dir` を両 SHA で使い回す設計に修正（設計ドキュメント §2.3 手順2）。
  GLB 見た目比較は手順3として分離し、精度の証拠ではなく定性チェックと明記。

## B: 軽微な指摘（対応済み）

- **B1**: Part A+B 同居 pod は NFS 永続化があるため必須の分離理由はないが、Part B の run driver に
  再開可能性（NFS 出力の有無で完了済み run をスキップ）を持たせるべき。→ 設計 §3 に追記。
- **B2**: 共有 pod `<cpu-pod>` での GGUF 変換前に軽く負荷確認すべき。→ 設計 §3 に追記。
- **B3**: B6（決定性チェック）は GLB バイト一致だけでなく、SS/shape/tex SLat 中間 dump の sha256 も
  比較し、どの段から分岐したか切り分けられるようにすべき。→ 設計 §3 に追記。
- **B4**: 予算・pod 削除・ディスク監視は既存設計で手当て済み（追加なし）。

## 確認された事実（設計の主張どおり）

- `tools/convert.py` の MANIFEST は `ss_flow`/`shape_flow_512`/`shape_flow_1024`/`tex_flow_1024` を
  既に含み、`TRELLIS_MODELS`/`TRELLIS_GGUF_OUT` だけで動作する（torch 不要、numpy+gguf のみ）。
- f04b33c→0d4cfc5 の差分は `flow_runner.cpp`（t-schedule 二重精度化そのもの）・`trellis_cli.cpp`
  （リテラル変更のみ、単一画像パスに機能的混入なし）・MV系テスト新規追加・server専用 fix に整理できる。
