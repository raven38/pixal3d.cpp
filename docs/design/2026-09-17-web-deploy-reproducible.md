# Web 本番デプロイの再現性と SV UI の配備（issue #10）

- ステータス: 実施済み（2026-09-17。codex 設計レビュー 10 件を反映した改訂 1。配備 version `3933e60f…`、SV/MV gate とも新ランタイムで PASS — `docs/PIXAL3D_RELEASE_CHECKLIST.md`）
- 関連: #8（SV UI、マージ済み）、#9 / PR #13（SV モデルセット公開）、#11（実ブラウザ SV gate）

## 背景と問題

本番 URL（`pixal3d-web` Worker、Cloudflare static assets）は 2026-09-11 の配備のまま MV-only UI を
配信している。PR #8 が main に入っても再デプロイされていない理由は 2 つある。

1. `npm run deploy` は `scripts/build_web_dist.sh` を経由し、`web/real_e2e/pixal3d_real_geometry.{js,wasm}`
   が無いと fail-closed で止まる。この 2 ファイルは gitignore 済みで、クリーンなチェックアウトから
   作る手順が CI に無い（手元の Homebrew emcc で 2026-09-09 03:25 に組んだものが本番に載っている）。
2. Cloudflare の配備権限（アカウント・トークン名・コマンド）が文書化されておらず、誰がどう更新
   しているか追えない。

## 決定

### D1. WASM ランタイムは「再ビルド」であり、ビルド元を固定して出所を配信物に同梱する（追跡可能・再ビルド可能。ビット再現は主張しない）

選択肢は (a) 手元の既存 2 ファイルをそのまま使い続ける、(b) release asset として保存し hash 照合して
取得する、(c) CI で毎回クリーンビルドする。**(c) を採り、(b) は artifact 保持（90 日）で代替する。**

- (a) は「開発者ローカルの未追跡ファイル」に依存し続けるので却下（#10 の要求そのもの）。
- ビルドは手元で **47 s**（M4 Max、fresh）なので CI コストは小さい。
- `scripts/build_wasm_real_geometry.sh` が docs/PIXAL3D_E2E_STATUS.md §6 の emcmake 手順を固定し、
  `web/real_e2e/build-info.json`（source commit / ggml commit / 適用 patch 一覧 / emcc 版 / 両ファイルの
  SHA256）を書く。`build_web_dist.sh` はこれを `dist/real_e2e/build-info.json` として同梱するので、
  本番で配信されている wasm の出所を URL から確認できる。
- `.github/workflows/web-wasm-runtime.yml` は emsdk **6.0.9** をピン留めしてこのスクリプトを走らせ、
  成果物 + digest を artifact にする。デプロイ時はこの artifact を `web/real_e2e/` に置く
  （手元ビルドでなく CI 成果物を配信する）。PR でも main push でも同じビルド入力
  （`src/**`, `include/**`, `web/ss/CMakeLists.txt`, `CMakeLists.txt`, patch, submodule）を監視する。
- 「決定的」はここでは **追跡可能・同じ手順で再ビルド可能** の意味に限定する。同じ emsdk 6.0.9 でも
  macOS Homebrew 版と Linux 公式版で wasm は byte 一致しなかった（js は一致）。CI run 同士の一致は
  再実行で確認する（結果はチェックリストに記録）。artifact の保持は 90 日なので、本番に載せた
  ランタイムは `v0.9.0-web-alpha` の GitHub Release asset にも添付する（タグ付け作業の一部）。
- `build_web_dist.sh` は本番ビルドで `build-info.json` を必須にし、js / wasm の両方が記録された
  size + SHA256 と一致しなければ止める（CI の UI smoke だけが `PIXAL3D_WEB_ALLOW_MISSING_WASM=1` で
  wasm 無しを許す）。

**副作用（重要）**: 本番の wasm が 2026-09-09 のバイナリから main 現在の C++ へ更新される。
MV の release gate（IoU 0.9795）は旧バイナリで測ったものなので、**新ランタイムは SV gate（#11）と
MV gate の両方を同じ wasm SHA256 で PASS してはじめて受け入れ**とする。順序は、配備予定の dist と
同一バイト列に対して SV gate を先に回し（配備の前提）、配備後に MV gate を回す。MV gate が落ちたら
`wrangler rollback` で直前 version に戻す（手順は runbook）。旧→新の差分は
`git log 2026-09-09..` の `src/dit.cpp`（MLP 分割は opt-in で既定経路は不変）、`naf.cpp`（MSVC 対応）、
`pixal3d_input.cpp` / `transforms_json.cpp`（#3/#5/#6 の入力経路）と ggml patch 0006。

### D2. デプロイは main の配信物、モデル配信元は HF の commit SHA で固定、SV URL は明示注入のみ

- 配信物のアプリ側ファイル（`build_web_dist.sh` が `dist/` にコピーする全ファイル）は `origin/main`
  と一致していなければならない。配備ツール自体（スクリプト・runbook）が未マージの間は、
  配備コミットと「アプリ側ファイルが一致する main のコミット」を両方チェックリストに残す。
  照合は `git diff origin/main -- <コピー対象の全パス>` と `git status --porcelain` が空であること、
  `build-info.source_commit` が artifact を作った workflow run のコミットと一致すること、
  `dist/` 全ファイルの SHA256 一覧（配備 receipt）で行う。
- モデル配信元は `resolve/main` でなく **HF の commit SHA**（`resolve/<sha>`）で固定する。HF 側の
  main が動いても同じ Worker version は同じバイト列を配る。MV `pixal3d-q8_0-v1` も同様に固定する。
- `build_web_dist.sh` の既定は変えない（CI が「明示設定なしに SV manifest を注入しない」ことを
  assert している）。本番は `PIXAL3D_SV_MODEL_MANIFEST_URL` / `PIXAL3D_SV_MODEL_BASE_URL` を
  HF の `raven38/pixal3d-sv-q8_0-v1` に向けて注入する。
- 既定の入力モードは PR #8 の実装どおり **SV が設定されていれば SV、無ければ MV**（`main.js`）。
  issue #10 の「MV remains the default while no public SV manifest is configured」はこの条件付きの
  文で、本番では SV が既定になる。スモークもこの規則を検査する。

### D3. 配備権限の文書化（`docs/runbooks/web-deploy.md`）

- 配備先の同一性は「Worker 名 `pixal3d-web` を所有する Cloudflare アカウント」で表し、
  `wrangler whoami` がそのアカウントを示すことを前提条件にする。メールアドレス・アカウント ID は
  書かない（公開リポジトリ）。
- 将来 CI から配備するときの secret 名は `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID`
  （`cloudflare/wrangler-action`）。secret の作成はユーザーが行い、リポジトリには置かない。
- 配備先の同一性は、`wrangler deployments list` を配備前後で取り同じ Worker の version が増えたこと、
  配信 URL が `https://pixal3d-web.raven38.workers.dev/` のままであること、その URL から取得した
  `index.html` / `real_e2e/*` の SHA256 が配備 receipt（`dist/` の一覧）と一致すること、の 3 点で
  確認する（Worker 名だけでは別アカウントの同名 Worker と区別できない）。

### D4. デプロイ後スモークは専用スクリプト

`web/app/test_headless.mjs` は「SV manifest が未設定であること」を assert する（L98/L139）ので
本番には使えない。`web/app/smoke_production.mjs`（Playwright）で次を見る:
`#input-mode` が見える／既定が設定規則どおり（SV 設定時は `sv`）／SV・MV の manifest がそれぞれの
family として解決し `#download-models` が有効／index.html に `PIXAL3D_SV_MODEL_*` の両 URL がある／
配信されている wasm の SHA256 が `real_e2e/build-info.json` と一致する。**スモークは「SV が選べて
manifest が解決する」ことまでしか証明しない**。GGUF 取得・推論・GLB までは #11 の実ブラウザ gate
が担い、それが受け入れ条件（下の 6）になる。

## 受け入れ基準（凍結・改訂 1）

1. `web-wasm-runtime` workflow がクリーンなチェックアウトで成功し、artifact に js/wasm/build-info/digest がある。
2. 本番 `index.html` に `input-mode` と SV の 2 URL（HF commit SHA 固定）があり、`smoke_production.mjs` が全項目 OK。
3. 本番の `real_e2e/pixal3d_real_geometry.js` と `.wasm` の SHA256 が CI artifact の値と一致する。
4. `wrangler deployments list` で同一 Worker の version が 1 つ増え、配信 URL が変わっていない。
5. 配備 receipt（配備コミット、アプリ側ファイルが一致する main コミット、artifact run、`dist/` 全ファイルの
   SHA256、Cloudflare version ID、直前 version ID）がチェックリストに残っている。
6. **同じ wasm SHA256 で SV gate（#11）と MV gate が PASS している**。SV gate は配備前に配備予定 dist と
   同一バイト列で、MV gate は配備後に本番 URL で回す。MV gate が落ちたら直前 version へ rollback する。
