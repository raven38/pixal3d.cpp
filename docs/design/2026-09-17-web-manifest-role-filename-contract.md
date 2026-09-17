# Web 側 manifest 検証を Python の name↔role / family 契約と同一規則にする

- 状態: **実装済み（2026-09-17）**。PR #13 レビュー指摘 2 への対応。codex 設計レビュー 1 周目（差し戻し）の
  A 項目 5 件・B 項目 6 件を改訂 2 で反映し、2 周目で承認
  （`docs/reviews/2026-09-17_web-manifest-role-filename-contract_review.md`）
- 実測（この Mac、Node 26.8.2 / Playwright 1.63.0 Chromium）: `npm run test:sv` OK（vector 18 case +
  committed 3 manifest + release/local validator）、`python tools/model_manifest.py self-test` →
  `MODEL_MANIFEST_CONFORMANCE_OK 18 cases`、`schema-test` / `check-committed` OK、
  `test_headless.mjs`（production bundle、9 本固定名モック + 契約違反キャッシュ注入）OK、
  `test_cache_failure_modes.mjs` 16/16 ok（case 3 は「manifest 段階で拒否・GGUF 転送 0」）
- 後送り: `.github/workflows/model-manifest.yml` の paths に `web/app/manifest_conformance.json` を足す
  （push token に workflow scope が無い）
- 対象: `web/app/single_view.js::modelFamilyForManifest` / `web/app/release_store.js::validateReleaseManifest` /
  `web/app/model_store.js::validateManifest` と、それらを叩くテスト、`scripts/build_web_dist.sh`
- 変更しないもの: C++ ランタイム、`tools/model_manifest.py` の規則そのもの、committed 3 manifest、
  `manifestIdentity`、`.github/workflows/*`（push token に workflow scope が無い）

## 概要

PR #13 は `tools/model_manifest.py` に「既知ファイル名 ↔ role の固定」「`model_family` と flow ファイル名の
整合」を入れたが、ブラウザ側 `modelFamilyForManifest()` は flow 名の `_sv.gguf` / `_mv.gguf` 部分一致で
family を分類するだけで、name↔role の束縛を見ていない。`validateReleaseManifest()` / `validateManifest()`
も同様。レビュー指摘: `model_family: "sv"` を明示した manifest なら `ss_flow` role に `foo.gguf` を置いても
SV と分類され、Python と規則が違う（PR コメントの「Python はブラウザと同じ規則」は不正確）。

本変更で、ブラウザ側にも Python と同じ name↔role / family 契約テーブルを持たせ、release / local /
family 判定の 3 箇所が同じ関数で判定するようにする。**同一にするのはこの部分契約（下記 1〜7）であり、
Python `validate_manifest_shape()` の全規則（未知キー拒否・`source` 型・非空文字列・数値型判定）ではない。**

## 背景

### 何を揃えるか（Python `validate_manifest_shape` の family / name↔role 部分）

1. `model_family` が明示されていれば `mv|sv` のいずれか
2. flow 4 role の name が **全て** `_mv.gguf` 末尾なら mv、全て `_sv.gguf` 末尾なら sv、混在・不明は None
3. 明示値と推定値が両方あって不一致なら拒否
4. family = 明示 || 推定 || 既定 `mv`
5. 既知 name は固定 role・`required: true` 必須。未知 name に既知 role を付けるのも拒否
   （未知 name + 未知 role の余剰ファイルは許容）
6. name 重複・role 重複を拒否
7. family の required 9 name / 9 role が全て揃うこと

### なぜ揃えるか（根拠の訂正）

改訂 1 は「ブラウザ worker も固定名で開くので name がずれると実行時に落ちる」を根拠にしたが、これは誤り。
`web/real_e2e/worker.js::filesFromManifest` は manifest の role → name を引いて WASM に**引数として**渡すので、
ブラウザでは `foo.gguf`/`ss_flow` のようなカスタム名でも現在は実行できる。固定名を開くのは native CLI
（`src/trellis_cli.cpp::mv_weight_paths`）だけ。

したがって本変更は「実行不能 manifest の早期拒否」ではなく、**配布契約を native CLI・Python 検証・ブラウザで
統一する破壊的変更**である。採用理由:

- レビュー指摘そのものが「同じ role→filename 契約をブラウザにも」であり、Python と規則が違う状態を
  残すと `check-committed` が通った manifest とブラウザが受理する manifest の集合が一致しない
- 公開 manifest は 3 本とも固定名で、カスタム名で配備された manifest は存在しない（配備元は
  `models/*/pixal3d-models.json` と HF の byte 一致コピーのみ）
- role の誤配線（`dinov3.gguf` を `ss_flow` に）を manifest 作成時点で止められる

### ブラウザ側の現状（訂正込み）

- `modelFamilyForManifest`: 上記 2〜4 相当を `includes('_sv.gguf')` の部分一致でやる。5・6・7 は見ない。
  flow 無し・`model_family` 無しは `null`
- `validateReleaseManifest`（公開 URL から取る manifest）: 安全名・name/role 重複・required・size/sha・
  **ちょうど 9 件**。5 は見ない
- `model_store.validateManifest`（「Install local verified set…」と OPFS 読み戻し）: 安全名・name 重複・
  role 型・size/sha。role 重複も 5 も 7 も見ない
- **`cacheStatus()` は manifest に書かれた `required:true` エントリの存在しか見ない**（canonical 9 role は
  見ない）。`main.js` は `s.ready && family === inputMode` で Generate を有効にするので、2 ファイルの
  部分セット manifest でも Generate まで進み、worker の `model manifest missing required role` で落ちる。
  改訂 1 の「完備は cacheStatus が別途見る」は誤りだった

## 提案

`web/app/model_family.js` を新設し、Python `model_files(family)` と同じテーブルと判定関数を置く。

```js
export const MODEL_FAMILIES = ['mv', 'sv'];
export const DEFAULT_MODEL_FAMILY = 'mv';
export function modelFilesForFamily(family)              // { name: { role, required } }  Python model_files と同一
export function inferModelFamily(files)                  // Python infer_family と同一（endsWith、全一致のみ）
export function manifestContractErrors(manifest)         // string[]（空なら合格）。規則 1〜7 を Python と同じ順で
export function modelFamilyForManifest(manifest)         // 契約違反ゼロなら family、それ以外 null
```

- `single_view.js` の `modelFamilyForManifest` は `model_family.js` からの re-export に置き換える
  （`main.js` / テストの import パスを変えない）
- `release_store.validateReleaseManifest` は既存チェックの後に `manifestContractErrors(m)` を呼び、
  1 件でもあれば throw。「ちょうど 9 件」は既存どおり残す（下表）
- `model_store.validateManifest` も既存チェックの後に `manifestContractErrors(m)` を呼び、1 件でもあれば
  throw。**local manifest も規則 7（9 本完備）を要求する**（部分セットは `cacheStatus` → Generate 有効に
  到達してしまうため）。`validateManifest` を export して Node テストから直接叩けるようにする
- `scripts/build_web_dist.sh` のコピー列挙に `model_family.js` を追加する。dist に対して走る
  `test_headless.mjs` が import 失敗で検出する（workflow の `test -f` 追加は scope の都合で後送り）

### Python との意図的な差（文書化して残す）

| 項目 | Python | ブラウザ | 理由 |
|---|---|---|---|
| 余剰ファイル（未知 name + 未知 role） | 許容 | release manifest では「ちょうど 9 件」で拒否。local / family 判定では許容 | release manifest の余剰は無駄なダウンロードになるので既存の 9 件固定を残す |
| flow 無し・`model_family` 無し | 既定 `mv` として規則 7 で拒否 | `modelFamilyForManifest` は `null`（規則 7 違反なので Python と同じく不合格。ブラウザの「不合格」は `null`） | 同じ結果 |
| `files` が空配列 | 拒否（非空必須） | `validateManifest` / `validateReleaseManifest` は非空を要求。`modelFamilyForManifest` 単体は規則 7 で `null` | 同じ結果。既存テストの `{ model_family:'sv', files:[] }` → `'sv'` は **期待値を `null` に変える** |
| 未知 top-level / file キー、`source` 型、空文字列、数値型 | Python が拒否 | ブラウザは従来どおり（一部のみ） | 本変更の範囲外。「同一」と言うのは規則 1〜7 だけ |
| `_sv.gguf` の判定 | `endswith` | `includes` → `endsWith` に変更 | `pixal3d_ss_flow_sv.gguf.bak` を sv と誤分類しない |

### 配備済み OPFS キャッシュへの影響

- 公式 3 manifest（`pixal3d-f16-v1` / `pixal3d-q8_0-v1` / `pixal3d-sv-q8_0-v1`）は固定名・9 本完備なので
  読み戻しで拒否されない（テストで確認）。`manifestIdentity` は変えないので再ダウンロードも起きない
- カスタム名・部分セットの manifest が OPFS にある場合、新 `validateManifest` が拒否 → `cacheStatus` は
  catch して `ready:false, manifest:null` → UI は「未インストール」として再インストールを求める。
  これは意図した挙動（そのキャッシュは新契約では無効）。headless テストで「拒否される manifest を
  OPFS に注入 → reload → `ready:false` かつ Generate 無効」を 1 ケース足す

## 設計

```
manifest JSON
   │
   ├─ release_store.validateReleaseManifest  ── 既存チェック（9 件固定含む）─┐
   ├─ model_store.validateManifest           ── 既存チェック ──────────────┼─▶ model_family.manifestContractErrors ─▶ throw / pass
   └─ main.js (Generate 可否)                                               │
                                             modelFamilyForManifest ─────────┘  (errors.length ? null : family)
```

テーブルは JS 側 1 箇所（`model_family.js`）。Python 側 `model_files()` との同一性は
**共有 conformance vector**（`web/app/manifest_conformance.json`）で担保する: 各 case に manifest と
期待値（`family: 'sv'|'mv'|null`）を書き、Python `self-test` と JS `test_single_view.mjs` の両方が同じ
ファイルを読んで判定を照合する。Python 側は `validate_manifest_shape` のエラー有無（family 部分）と
`infer_family`/明示値から得た family を期待値と比較する（Python 固有の shape 規則に掛からないよう
vector の manifest は Python の shape 要件も満たす形で書く）。

## 実装計画

1. `web/app/manifest_conformance.json` を作る（下記テスト計画の case 群）
2. `web/app/model_family.js` 新設（テーブル + 関数）
3. `web/app/single_view.js`: `modelFamilyForManifest` の本体を削除し re-export
4. `web/app/release_store.js`: `validateReleaseManifest` 末尾で契約チェック
5. `web/app/model_store.js`: `validateManifest` を export し、末尾で契約チェック（9 本完備込み）
6. `scripts/build_web_dist.sh`: `model_family.js` をコピー列挙に追加
7. `tools/model_manifest.py`: `self-test` に conformance vector の照合を追加
8. `web/app/test_single_view.mjs`: 既存 fixture に `required:true` と 9 本を揃える／`files:[]` の期待値を
   `null` に／vector 照合／committed 3 manifest 読み込み／`validateReleaseManifest`・`validateManifest` の
   直接テスト
9. `web/app/test_headless.mjs`: モック manifest を **9 本の固定名・小さなダミーバイト列**に変更
   （`payload.models.length` の期待値も 9 に）。無効 manifest 注入 → reload → `ready:false` ケースを追加
10. `web/app/test_cache_failure_modes.mjs`: remote 名を固定名に変更。case 3（role 付け替え）は
    「manifest 取得時点で拒否され GGUF 転送 0 本」に期待値を変更
11. `web/app/README.md`: 契約の統一と上表の意図的差異、カスタム名 manifest が無効になった旨
12. PR #13 コメントの「Python follows exactly the same rule」を訂正する文言を PR に追記

## 影響範囲

- 新規: `web/app/model_family.js`、`web/app/manifest_conformance.json`
- 変更: `single_view.js`、`release_store.js`、`model_store.js`、`scripts/build_web_dist.sh`、
  `tools/model_manifest.py`（self-test のみ）、`test_single_view.mjs`、`test_headless.mjs`、
  `test_cache_failure_modes.mjs`、`web/app/README.md`
- CI: `web-app-headless.yml`（paths に `web/app/**` と `scripts/build_web_dist.sh` を含む）は本変更で走る。
  `model-manifest.yml` は `tools/model_manifest.py` 変更で走る。**vector だけを変えたときに Python CI が
  走らない**（paths に `web/app/manifest_conformance.json` を足すには workflow scope が要る）→ 後送り項目
- 破壊的変更: 固定名以外の name を既知 role に付けた manifest、および 9 本未満の manifest は、以後
  ブラウザでも拒否される（release / local とも）。公開 manifest は全て適合

## テスト計画

conformance vector（Python・JS 共通、期待は `family` の値）:

| case | 期待 |
|---|---|
| committed 3 manifest（ファイルから読む。vector には含めずテスト側で読む） | sv / mv / mv、errors 空 |
| SV 9 本・`model_family` 省略 | `'sv'` |
| SV 9 本・`model_family:'sv'` | `'sv'` |
| MV 9 本・`model_family` 省略 | `'mv'` |
| `model_family:'sv'` + `ss_flow` が `foo.gguf` | `null` |
| `dinov3.gguf` と `pixal3d_ss_flow_sv.gguf` の role 入れ替え | `null` |
| MV 名の集合に `model_family:'sv'` | `null` |
| ss_flow だけ `_mv`、他 `_sv` | `null` |
| `model_family:'xx'` | `null` |
| 既知 name に `required:false` | `null` |
| role 重複 | `null` |
| name 重複 | `null` |
| SV 9 本 + 未知 name + 未知 role の余剰 1 件 | `'sv'`（family 判定・local では許容。release は 9 件固定で別途拒否） |
| `pixal3d_ss_flow_sv.gguf.bak` を ss_flow に | `null` |
| 8 本（`tex_dec.gguf` 欠落） | `null` |
| flow 無し・`model_family` 無し | `null` |

JS 単体（`test_single_view.mjs`）で追加: `validateReleaseManifest` が `foo.gguf`/ss_flow で throw、
10 件目の余剰で throw（9 件固定）。`validateManifest` が 8 本で throw、9 本固定名ダミーで通る。

既存: `npm run test:sv` / `npm run test:rig` / `test_headless.mjs`（playwright）/ `test_cache_failure_modes.mjs`
（ローカル）/ `python tools/model_manifest.py self-test schema-test check-committed`。

## 代替案

- **A. ブラウザは分類のみ、差異を文書化するだけ**: レビューが許容している選択肢。却下理由は
  「Python `check-committed` が受理する集合とブラウザが受理する集合を一致させたい」という契約統一の
  ポリシー（実行可否ではない）。ブラウザ側の実行経路は role→name 解決なのでカスタム名も動くが、
  それを配布契約として認めると native CLI とブラウザで有効な manifest が食い違う。
- **B. Python と完全同一（local manifest も 9 本完備必須）**: 改訂 1 では「weightless CI のモックが
  通らない」として却下したが、モックを 9 本の小さなダミーにすれば weightless のまま通る。
  部分セットが Generate 有効まで到達する現状の穴も塞げるので、**改訂 2 で採用**。
- **C. テーブルを JSON にして Python と JS で共有**: Python 側 `model_files()` の構造変更が PR #13 の
  範囲を超える。代わりに conformance vector を共有して同一性をテストで担保する（採用）。
