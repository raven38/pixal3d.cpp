# transforms.json 無しで multiview 生成を通す（canonical rig）

2026-09-15 / 対象: `trellis-cli` / `trellis-server` / `install/generate.sh`

## 背景

ブラウザ版は既に transforms.json 無しの入力に対応している（`web/real_e2e/calibration.js` の
`CANONICAL_RIG`、回帰テスト `web/app/test_canonical_rig.mjs`、commit 4b66676）。
native 側（CLI・server・generate.sh）は `dir/transforms.json` が必須で、同じ 4 枚を渡しても動かない。
本変更はブラウザが既に決めた規約を native へ移植するだけで、新しい規約は作らない。

## 規約（calibration.js から逐語で移植）

- transforms.json が無いときに限り合成する。ある場合の挙動は一切変えない。
- 画像は **ちょうど 4 枚**（png/jpg/jpeg/webp）。ファイル名の自然順（数字を数値として比較。
  `view2` < `view10`）で並べ、順に front / right / back / left と解釈する。
- `camera_angle_x = 0.3490658503988659`（20°）、仰角 0、距離 3.1192049980163574 の
  固定 4 姿勢（値は calibration.js と同一）。
- `mesh_scale` は **明示必須**。既定値 1.0 を仮定しない（誤ったスケールは multiview 形状を
  静かに壊すため。`transforms_json.cpp` の既存エラーメッセージとリリースノートの約束を維持）。
- 4 枚以外・`mesh_scale` 未指定/非正は明示エラーで拒否（fail closed）。

## 変更点

1. `src/transforms_json.{cpp,h}`: `synthesize_canonical_rig(dir, mesh_scale, out, err)` を追加。
   姿勢テーブルと自然順ソートはここ 1 箇所に置く。
2. 読み込み経路を 1 本化: `trellis_cli.cpp` の `mv_load_views` と `pixal3d_input.cpp` の
   `pixal3d_load_input_views` が別々に `load_transforms_json` を呼んでいるので、
   「transforms.json があれば parse、無ければ合成」を共通関数にして両方から呼ぶ。
3. `trellis_args.cpp`: `--mesh-scale <f>` を追加。transforms.json が無い場合は必須。
   ある場合は上書き（ブラウザの calibrator と同じ挙動）。合成モードでは `--num-views` は 4 のみ許可。
4. `trellis-server.cpp` `/generate-mv`: `transforms` パートを任意にし、`mesh_scale` フォーム
   フィールドを受ける。view は multipart filename で staging し、共通ローダに合成させる。
   ヘッダコメント（L10-14）も更新。
5. `install/generate.sh`: `transforms.json` 必須チェックを外し `--mesh-scale` を追加。
   remote モードは transforms.json が無ければ画像 4 枚を自然順で `view0..3` として送り、
   `-F mesh_scale=…` を付ける。
6. ドキュメント: README のフラグ表、`docs/spec/30-pixal3d-cond.md` の入力契約。

## 検証（GLB だけで判定しない）

1. **決定的比較**: `docker/linux-webgpu-gate/e2e/views/` の同梱 transforms.json を parse した結果と、
   同じ 4 枚から `mesh_scale=1.0` で合成した結果を全フィールド比較（4 frame × 16 float + fov +
   mesh_scale がビット一致）。`test_canonical_rig.mjs` の項目 1 の C++ 版。
2. **段階一致**: transforms.json を除いたコピーに対し Mac Metal で 1 回実行し、SS 段の
   `ss_coords@64 N=4438 bbox x[3..28] y[2..29] z[0..31]`（既知値）が一致することを
   フル生成の前に確認する。
3. **異常系**: 画像 3 枚 → 明示エラー、`--mesh-scale` 未指定 → 明示エラー（1.0 に落ちない）。

## 対象外

- alpha チャンネルを持たない画像は従来どおりローダで拒否する（本変更では触らない）。
- ブラウザ UI は変更不要。
- ベースブランチは `linux-webgpu-gate`（`install/generate.sh` は PR #1 にしか無いため）。

## codex 設計レビュー（2026-09-15）で確定した点

1. **mesh_scale は「値」と「明示された事実」を分離する**。共通ローダは `(float value, bool was_set)`
   を受ける。`pixal3d_load_input_views` は既定引数で従来の JSON 経路のまま（既存呼び出し側は無変更）。
2. **transforms.json がある場合の挙動は変えない**。JSON には有効な `mesh_scale` が引き続き必須で、
   `--mesh-scale` は parse 成功後の上書きだけに効く。欠落・不正 JSON を CLI 値で救済しない。
3. **fallback は「transforms.json が存在しない」場合のみ**。空ファイル・壊れた JSON・ディレクトリ・
   permission error は従来どおりエラー（`load_transforms_json` の false を fallback 条件にしない）。
4. **server はディレクトリ列挙を入力の正本にしない**。衝突しない一時ディレクトリを作り、
   その request で受理したファイルの明示リストを合成ローダへ渡す（残骸が 4 枚判定に混ざるのを防ぐ）。
5. **server はモデル実行前に検証して 400 + 具体的な JSON エラーを返す**（従来は generic 500）。
6. **`--num-views` / `--mesh-scale` は厳密パース**（`atoi` をやめ、全体消費・範囲・有限性を検査）。
   未指定と `0` / 負値 / 非数値を区別し、合成モードでは 4 以外を拒否。
7. **自然順は C++ 側で一意に定義する**: ASCII、大小文字を区別、数字列は整数として比較、
   数値が同値なら元のバイト列で tie-break（locale 非依存）。
8. **検証は合成関数単体で終えない**: CLI・server の両経路と異常系、および
   「JSON 版」と「合成版」を同一バイナリ・同一 seed で対にして `ss_coords` のハッシュまで比較する。
   最終受け入れでは両者の GLB を生成してレンダ比較する。
