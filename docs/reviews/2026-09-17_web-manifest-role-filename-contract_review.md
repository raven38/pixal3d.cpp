# 設計レビュー（codex exec、1 周目）: Web 側 manifest 検証を Python の name↔role 契約と同一規則にする

- 日付: 2026-09-17
- 対象: `docs/design/2026-09-17-web-manifest-role-filename-contract.md`（改訂 1）
- 実行: `codex exec -s read-only`（反証指向、A: 実装を止める項目 / B: 後から直せる項目 の区分）
- 判定: **差し戻し** → 改訂 2 で A 5 件・B 6 件を反映。A 項目はいずれもコードで裏取り済み
  （`web/real_e2e/worker.js::filesFromManifest` の role→name 解決、`model_store.js::cacheStatus` が
  manifest 内 `required` しか見ないこと、`scripts/build_web_dist.sh` の明示列挙、
  `test_single_view.mjs` fixture に `required` が無いこと、`test_cache_failure_modes.mjs` case 3）
- 2 周目（再確認）: 改訂 2 に対し「1 周目の指摘を直したか」のみを確認（本ファイル末尾に追記）

## codex 出力（1 周目、サマリーと詳細）

（本文は下に続く）
### レビュー結果サマリー

- 判定: 差し戻し
- A 項目:
  - Web ランタイムも固定ファイル名しか扱えない、という設計根拠が事実と異なる。Worker は manifest の role→name を引数として WASM に渡しており、カスタム名でも現在は実行可能。
  - local manifest の部分セット許容は、`cacheStatus()` を通じて `ready=true`、さらに Generate 有効まで到達する。設計書の「完備は cacheStatus が別途見る」は誤り。
  - 新設する `model_family.js` が production bundle のコピー対象に含まれていない。このままでは built `web/dist` が import 失敗し、CI を破壊する。
  - 既存 `test_single_view.mjs` の family fixture は `required` を持たないため、新しい規則では既存陽性テストが `null` になり `npm run test:sv` が落ちる。
  - `test_cache_failure_modes.mjs` は単なる実名への変更では足りない。「role 付け替え manifest のインストール成功」を期待しており、新契約と正面衝突する。
- B 項目:
  - Python は未知 name＋未知 role の余剰ファイルを許す一方、release validator は「ちょうど9件」で拒否する。この差が意図的差異表にない。
  - Python の全 shape 規則と同一であるかのような表現は狭めるべき。未知キー、`source` 型、空文字、数値型などは依然異なる。
  - committed manifest と mutation のJSテストだけではPython/JS同一性を証明できない。共有conformance vectorか相互実行が必要。
  - Python self-test とブラウザテストは対称ではない。
  - 配備済みOPFSについて「キャッシュ無効化なし」とする表現が不正確。identityが同じでも新validatorで読み戻し不能になり得る。
  - `model_store.validateManifest` は非exportなので、テスト計画は間接テストかexport方針を明記する必要がある。

### 詳細

#### 1) Python側規則との同一性

name↔roleテーブル、family推定の `endsWith()`、明示familyとの不一致拒否という狭い範囲は、提案どおり実装すればPythonに合わせられます。

ただし「Webランタイムも固定名しか開かない」という根拠は成立しません。[worker.js](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/real_e2e/worker.js:27) は manifest から role→name を作り、[worker.js](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/real_e2e/worker.js:57) で得た名前をそのままWASM引数に渡しています。固定名配列は manifest がない旧ハーネス用fallbackです。

したがって `foo.gguf/ss_flow` はネイティブCLIでは非互換でも、Web経路では現在実行可能です。今回の変更は「実行不能manifestを早期拒否する修正」ではなく、「WebもネイティブCLIの命名ポリシーに制限する破壊的変更」です。採用自体は可能ですが、理由・互換性・移行方針をその前提で書き直す必要があります。

また、Pythonの `validate_manifest_shape()` 全体との同一性ではありません。Pythonには以下もあります。

- 未知top-level/file keyの拒否
- `source` の型検査
- `model_set` / `version` の非空検査
- 非空files
- Python固有の数値型判定

タイトルとREADMEは「Pythonのname↔role/family部分契約と同じ」に限定すべきです。

#### 2) 意図的差異

最大の破綻は部分セットです。[model_store.js](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/app/model_store.js:160) の `cacheStatus()` は、canonicalな9 roleを検査せず、manifestに書かれた `required:true` のエントリだけ確認します。よって2ファイルmanifestを保存すると、その2本が存在するだけで `ready=true` になります。

その後、[main.js](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/app/main.js:196) は `s.ready && family === inputMode` だけで `modelsReady=true` にします。つまり設計書の主張と逆に、部分セットはGenerate有効まで進み、実Workerで不足roleにより失敗します。

対処は次のどちらかが必要です。

- local install・OPFS読み戻しにも `requireComplete:true` を適用する。
- 部分セットを許すテスト専用経路をproduction validatorから分離する。

weightless CIでも、9本の小さなダミーバイト列を使えば実重みは不要です。したがって代替案Bの「9本にするとweightless CIを壊す」は成立しません。

もう一つの漏れは余剰ファイルです。Pythonはrequired 9本に加えて未知name＋未知roleを許しますが、[release_store.js](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/app/release_store.js:26) は件数が9以外なら拒否します。「完備」と「ちょうど9件」は異なるため、残すなら意図的差異表へ追加すべきです。

#### 3) 影響範囲・CI・配備済みキャッシュ

production bundleの組み立ては明示的なファイル列挙です。[build_web_dist.sh](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/scripts/build_web_dist.sh:17) に `model_family.js` がありません。このまま `single_view.js` 等からimportすると、source treeでは通っても `web/dist` では404になります。実装計画と影響範囲にビルドスクリプト、distファイル存在検査を追加する必要があります。

既存テストにも未整理の破壊があります。

- [test_single_view.mjs](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/app/test_single_view.mjs:24) の陽性fixtureには `required:true` がない。新しい規則5により陽性ではなくなる。
- [test_cache_failure_modes.mjs](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/app/test_cache_failure_modes.mjs:141) はrole付け替えmanifestについて、インストール成功と再取得を期待している。新設validatorならmanifest取得時点で拒否されるため、期待値そのものを「ダウンロード開始前に拒否、GGUF転送0本」へ変更する必要がある。
- 同テストの全remote名も固定契約外なので、名前の変更は必要だが、それだけでは上記期待値衝突は解消しない。

配備済みキャッシュについても、`manifestIdentity` を変えないことは「無効化なし」を意味しません。新しい `validateManifest()` が保存済みmanifestを拒否すると、`cacheStatus()` はcatchしてmanifestなしとして返します。特に現在動作可能なWebカスタム名manifestが対象です。公式3 manifestの確認に加え、旧形式キャッシュを注入したreloadテストと、拒否時のユーザー向け再インストール表示が必要です。

#### 4) テスト計画とPython self-testの対称性

「対称」という主張は現状では誤りです。Python self-testには次がありますが、提示されたWeb計画にはありません。

- 未知top-level key
- 未知file key
- required file欠落
- unsafe name `..`
- size/SHA破損

逆にWeb計画のduplicate role、未知余剰、`.bak` suffixはPython self-testに直接対応ケースがありません。

また、同じhappy-path manifestをPythonとJSがそれぞれ受理するだけでは、複製テーブルの同一性を保証できません。将来Pythonだけ変更されても、現在のworkflow triggerはWeb CIとPython manifest CIが分離しているためドリフトを検出できません。

少なくとも次のいずれかが必要です。

- JSONの共有conformance vectorsをPythonとJSの両方で実行する。
- Pythonが正誤ケースと期待family/errorsを生成し、Nodeテストが同じfixtureを読む。
- workflowのpathsを交差させ、`tools/model_manifest.py` と `web/app/model_family.js` のどちらの変更でも両テストを走らせる。

なお `validateManifest()` は現在非exportです。[model_store.js](/Users/<redacted-user>/Downloads/pixal3d-sv-model-set/web/app/model_store.js:16) を直接テストする計画ならexportが必要で、exportしないなら `installFiles()` または保存済みOPFS読み戻しを通す統合テストとして書くべきです。

#### 5) 代替案の却下理由

- 代替案A: 却下理由は不成立です。Web Workerはroleから名前を解決して渡すため、カスタム名はWebでは実行可能です。却下するなら「ネイティブCLIとの配布契約統一」「誤role配線をmanifest作成時に防ぐ」といったポリシー上の理由へ変更してください。
- 代替案B: 却下理由は不成立です。9本すべてを小さなモックBufferにすればweightlessのままです。むしろproduction validatorとGenerate可否を安全に保つ最も単純な案です。
- 代替案C: PRスコープを抑える判断自体は妥当です。ただし、別々のコードに同じテーブルを書き、3 manifestと個別mutationを通すだけでは同一性を担保できません。Python実装をJSON化しなくても、共有conformance fixtureだけなら小さい変更で済みます。

実装再開条件は、少なくとも「localは9本必須」「Webカスタム名を意図的に廃止するか再検討」「distへ新moduleを収録」「既存2テストの期待値を契約に合わせて更新」の4点を設計へ反映することです。

## 2 周目（再確認、改訂 2 に対して「1 周目の指摘を直したか」のみ）

### 再確認結果

- 判定: 承認
- A1: 直った — Web worker が role→name を解決して WASM に渡す事実を明記し、契約統一を変更理由に訂正している。
- A2: 直った — local manifest にも9本完備を要求し、部分セットから Generate が有効になる経路を遮断している。
- A3: 直った — `scripts/build_web_dist.sh` のコピー対象へ `model_family.js` を追加する計画が明記されている。
- A4: 直った — fixture を `required:true` かつ9本完備に直し、空配列の期待値も `null` に変更している。
- A5: 直った — role付け替えケースをインストール成功ではなく、manifest取得時点での拒否・GGUF転送0本へ変更している。
- B1: 直った — 未知name＋未知roleの余剰について、Python・release・local間の差異を表に明記している。
- B2: 直った — 「同一」の対象を family/name↔role の規則1〜7に限定し、Pythonの全shape規則とは区別している。
- B3: 直った — PythonとJavaScriptが共通の conformance vector を読み、同じ期待値と照合する計画になっている。
- B4: 直った — 共通vectorをPython self-testとJavaScriptテストの双方で実行し、同じ部分契約を対称に検証する設計になっている。
- B5: 直った — 新契約に違反する既存OPFSキャッシュは無効化され、`ready:false` になる影響を明記している。
- B6: 直った — `model_store.validateManifest` をexportし、Nodeテストから直接検証する方針が明記されている。
- 実装再開条件: なし。改訂2の記載どおり実装へ進めてよい。
