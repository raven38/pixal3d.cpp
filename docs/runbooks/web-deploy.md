# Runbook: Pixal3D Web の本番デプロイ（Cloudflare Workers static assets）

対象: `https://pixal3d-web.raven38.workers.dev/`（Worker 名 `pixal3d-web`、`wrangler.jsonc`、
静的配信のみ・推論はブラウザ内）。設計は `docs/design/2026-09-17-web-deploy-reproducible.md`。

## 0. 配備権限（identity）

- 配備先は **Worker `pixal3d-web` を所有する Cloudflare アカウント**。`cd web && npx wrangler whoami`
  がそのアカウントを示すこと（OAuth ログイン。別アカウントだと別 Worker が新規作成されるだけで
  本番は更新されない）。メールアドレス・アカウント ID はこのリポジトリに書かない。
- CI から配備する場合（未構成）: `cloudflare/wrangler-action` に repository secret
  `CLOUDFLARE_API_TOKEN`（Workers Scripts: Edit）と `CLOUDFLARE_ACCOUNT_ID` を渡す。secret の作成は
  アカウント所有者が行う。トークン値を YAML・ログ・PR に出さない。
- 「同じ Worker を更新した」ことの確認は §4 の `wrangler deployments list` の前後比較で行う。

## 1. 配備するコミット

`origin/main` のコミットから組む（未マージの PR を本番に混ぜない）。作業ツリーで次が全て空であること:

```sh
git fetch origin && git status --porcelain            # 未追跡・未コミットが無い（gitignore 済み wasm は除く）
git diff origin/main -- web/app/index.html web/app/main.js web/app/model_store.js web/app/release_store.js \
  web/app/preflight.js web/app/sha256.js web/app/single_view.js web/real_e2e/calibration.js web/real_e2e/worker.js \
  models/pixal3d-q8_0-v1/pixal3d-models.json app/public/vendor/model-viewer.min.js scripts/build_web_dist.sh wrangler.jsonc
```

配備ツール（このスクリプト群）が未マージの間は、配備コミットと「アプリ側ファイルが一致する
main のコミット」を両方 receipt に書く。

## 2. WASM ランタイム（クリーンビルド成果物を使う）

`web/real_e2e/pixal3d_real_geometry.{js,wasm}` は gitignore 済み。**CI artifact を使う**:

```sh
gh run list --workflow=web-wasm-runtime.yml --branch main --limit 1     # 配備するコミットの run を選ぶ
gh run download <run-id> --dir /tmp/wasm-artifact
(cd /tmp/wasm-artifact/* && sed 's|web/real_e2e/||' SHA256SUMS | sha256sum -c)   # macOS: shasum -a 256 -c
cp /tmp/wasm-artifact/*/{pixal3d_real_geometry.js,pixal3d_real_geometry.wasm,build-info.json} web/real_e2e/
```

`build-info.json` の `source_commit` が artifact を作った workflow run の `head_sha`（PR run は merge ref）と
一致すること（`gh run view <run-id> --json headSha`）。
手元でビルドする場合は `scripts/build_wasm_real_geometry.sh --fresh`（emsdk 6.0.9 を有効化してから）。
ホストが違うと wasm は byte 一致しない（同じ 6.0.9 でも macOS Homebrew 版と Linux 公式版で異なる）ので、
本番に載せるのは CI artifact に統一する。

## 3. 配信物の組み立てと配備

モデル配信元は HF の **commit SHA** で固定する（`resolve/main` は使わない。HF 側が動いても同じ
Worker version が同じバイト列を配るため）。SHA は `curl -s https://huggingface.co/api/models/raven38/<repo> | jq -r .sha`。
SV の配信元は既定では注入されない（CI の assert）ので本番は明示する:

```sh
MV_SHA=$(curl -s https://huggingface.co/api/models/raven38/pixal3d-q8_0-v1 | jq -r .sha)
SV_SHA=$(curl -s https://huggingface.co/api/models/raven38/pixal3d-sv-q8_0-v1 | jq -r .sha)
bash scripts/build_web_dist.sh \
  https://huggingface.co/raven38/pixal3d-q8_0-v1/resolve/$MV_SHA \
  https://huggingface.co/raven38/pixal3d-sv-q8_0-v1/resolve/$SV_SHA/pixal3d-models.json \
  https://huggingface.co/raven38/pixal3d-sv-q8_0-v1/resolve/$SV_SHA
(cd web/dist && find . -type f | sort | xargs shasum -a 256) > /tmp/dist-receipt.txt   # 配備 receipt
```

`build_web_dist.sh` は `web/real_e2e/build-info.json` の size + SHA256 と同梱する js / wasm を照合してから
`dist/real_e2e/build-info.json` として同梱する（無い・不一致なら失敗）。

**配備前ゲート**: 配備予定の `web/dist` を `python3 -m http.server` で配信し、SV の release gate
（`web/app/run_release_gate.mjs --mode sv …`、#11）を回して PASS させてから配備する。

```sh
cd web && npx wrangler deploy --config ../wrangler.jsonc     # dist を作り直さず、gate を通した dist をそのまま配る
```

（`npm run deploy` は `predeploy` で dist を作り直すので、gate 済み dist を配るときは wrangler を直接呼ぶ。）

## 4. 配備後の確認

```sh
cd web && npx wrangler deployments list --config ../wrangler.jsonc | head   # 配備前後で同じ Worker の version が 1 つ増える
node app/smoke_production.mjs https://pixal3d-web.raven38.workers.dev/      # WEB_PRODUCTION_SMOKE_OK
curl -s https://pixal3d-web.raven38.workers.dev/real_e2e/build-info.json     # 配信中ランタイムの出所
# 配信物 = receipt の照合（index.html と real_e2e/* を取得して SHA256 を比べる）
for f in index.html real_e2e/pixal3d_real_geometry.js real_e2e/pixal3d_real_geometry.wasm real_e2e/worker.js main.js; do
  printf '%s  %s\n' "$(curl -s https://pixal3d-web.raven38.workers.dev/$f | shasum -a 256 | cut -d' ' -f1)" "$f"
done | diff - <(grep -E ' \./(index.html|real_e2e/pixal3d_real_geometry.js|real_e2e/pixal3d_real_geometry.wasm|real_e2e/worker.js|main.js)$' /tmp/dist-receipt.txt | sed 's| \./|  |')
```

**rollback**: `cd web && npx wrangler rollback --config ../wrangler.jsonc`（直前 version を選ぶ）。
rollback 後も `smoke_production.mjs` を回し、receipt に rollback 先 version ID を追記する。

スモークは MV/SV の両 manifest が正しい family に解決し `Download release models` が有効になることまで
見る。実 GPU での生成はスモークに含まない（release gate: `web/app/run_release_gate.mjs`）。
**ランタイム（wasm）を更新した配備の後は、MV / SV の release gate を新ランタイムで再実行し、
`docs/PIXAL3D_RELEASE_CHECKLIST.md` に runtime SHA256 と併記する。**

## 5. 記録

チェックリストの Web alpha 節に、配備日・配備コミット・runtime SHA256（`build-info.json`）・
SV/MV URL・スモーク結果を 1 行で残す。生成物やログはリポジトリに入れない。
