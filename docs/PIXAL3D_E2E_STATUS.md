# Pixal3D browser E2E status

`feat/browser-partial-e2e` の現状。数値は測ったものだけを書き、未実施は未実施と書く。

## 0. 何が blocker だったか（解決済み）

`pixal3d_cond_slat_gpu(... {S=1024, R=64, naf_T=1024})`（texture 段）は、単一グラフのままだと
ggml gallocr が **1 本 11 344 MB** のバッファを要求する。主犯は NAF 出力 `[1024, 1024²]` f32
= 4 294 967 296 B で、実測済みの WebGPU `maxBufferSize` 4 294 967 292 B を **4 バイト**超える。

解決:

1. **sparse coords**: `pixal3d_cond_slat_gpu` に active voxel の coords を渡せるようにし、
   dense `R³` accumulator（2 GiB）を作らない。dense + `pixal3d_gather_proj` と全成分ビット一致。
2. **分割グラフ**: 1 view を 4 種のグラフに割る（DINOv3 / NAF encoder 2 枝 / block 行 stripe の
   RoPE / block chunk の neighborhood attention）。NAF 出力は一度も materialize せず、
   chunk ごとに projection tap だけ回収して accumulator へ足す。

単一バッファ最大 **11 344 MB → 3 332 MB**。数値差は hr 側だけで L2 rel 8.5e-09 / cos 1.0000000000
（chunk をまたぐ tap 加算の順序差）、global と lr はビット一致。詳細と内訳表は
`docs/PIXAL3D_WEBGPU_MEMORY.md` §11。

## 1. Real-input full E2E（本命）

Browser target: `pixal3d-webgpu-real-geometry-wasm`
Harness: `web/real_e2e/`（`run_playwright.js` あり）
C ABI: `pixal3d_real_full_run()`（geometry のみの `pixal3d_real_geometry_run()` も残置）

```text
RGBA images + transforms.json
 -> live SS conditioning / SS Flow / SS decode
 -> live Shape-512 conditioning / Flow
 -> shape decoder upsample + Pixal3D grid64 quantization
 -> live Shape-1024 conditioning (NAF T=512) / Flow
 -> live Texture-1024 conditioning (NAF T=1024, 分割グラフ)
 -> Texture Flow
 -> Shape Decode / Texture Decode
 -> production postprocess (weld / hole fill / narrow-band remesh / cleanup / QEM / UV / PBR bake)
 -> textured GLB
```

fixture 由来の neural 出力の注入は無い。report は
`REAL_INPUT_LIVE_TEXTURE_COND=1` / `REAL_INPUT_TEXTURE_COND_BLOCKED=0` を出す。

## 2. Fixture-input full model E2E

Browser target: `pixal3d-webgpu-full-e2e-wasm`
Harness: `web/full_e2e/`（`run_playwright.js` あり）
C ABI: `pixal3d_full_fixture_run()`

Texture conditioning は live に切り替え済み（`FULL_E2E_LIVE_TEXTURE_COND=1`）。
旧 fixture 経路は debug チェックボックス（C ABI の `tex_cond_fixture=1`）のときだけ通り、
そのとき report は `FULL_E2E_LIVE_TEXTURE_COND=0` を出す。

このターゲットは `s1024_images.npy` / `images_512.npy` / `camera_angle_x.npy` /
`transform_matrix.npy` / `mesh_scale.npy` と各段の noise を持つ fixture bundle を要求する。
bundle は `tools/make_fixture_bundle.py <views_dir> <out_dir>` で作れる（前処理は
`inference_mv.py` の `to_cond_tensor` と同じ PIL LANCZOS + アルファ乗算）。pod で PyTorch が
吐いた同名テンソル（views4）と **全項目 bit 一致**するので、bundle 作成に GPU は要らない。
noise は書かず `deterministic_noise()` の mt19937 に任せる（native と揃うため）。

なお C++ の real-input 経路（`src/pixal3d_input.cpp` の `resize_premult`）は
`stbir_resize_uint8` を使っていて **PIL LANCZOS とは実装が違う**。fixture 経路を通すと
前処理が PyTorch 側に揃うので、両経路の差はこの resize 実装差を含む。

### 実行結果（2026-09-08）

**公式サンプル（cyclops）の bundle: SS / Shape-512 / Shape-1024 / Texture の 4 flow を
すべて完走した後、decode・postprocess の段で `std::bad_alloc`**
（`FULL_FIXTURE_RESULT: EXCEPTION std::bad_alloc`）。所要は SS 194.9 s / Shape-512 165.7 s /
Shape-1024 1476.3 s（17 660 token, native は 17 690）/ live tex cond 54.2 s
（graph peak 1 538 MB）/ Texture Flow 988.2 s。

同じ入力の native は decode で 4 730 295 voxel / raw mesh V=4 725 397 F=9 510 803 を作る。
ブラウザは `remesh_res=512` に落としてもこの規模の入力メッシュに対して
TriBvh（950 万三角形）と QEM を 4 GiB のヒープ内で持てない。**flow 段ではなく tail が原因**で、
fixture 経路そのものの問題ではない。§4 の「後回しでよい」判断はこの結果で見直しが要る:
**予算差は品質の問題にとどまらず、被写体が大きいと browser では完走しない**。

**より小さい被写体では完走する（= fixture 経路は動く）**。`views4 + mesh_scale=0.206` の
bundle（decode 175 万 voxel / 369 万三角形）で同じ harness を回すと:

```
FULL_E2E_LIVE_SS=1 / LIVE_SHAPE512=1 / LIVE_SHAPE1024=1 / LIVE_TEXTURE_COND=1
FULL_E2E_TEXTURE_FLOW=1 / FULL_E2E_GLB=1
FULL_MODEL_E2E_RESULT: LIVE_ALL_STAGES V=1754158 F=3675898
FULL_FIXTURE_RESULT: OK_LIVE_TEXTURE_COND   (GLB 21 665 332 B)
```

所要は SS 180.6 s / Shape-512 25.5 s / Shape-1024 328.0 s（4 794 token, native 4 750）/
live tex cond 65.9 s（graph peak 1 538 MB）/ Texture Flow 165.0 s /
postprocess remesh 1 353 592 面 → QEM 447 556 面。

| | browser fixture 経路 | native real-input 経路 |
|---|---|---|
| 前処理 | PIL LANCZOS（bundle） | `stbir_resize_uint8` |
| Shape-1024 tokens | 4 794 | 4 750 |
| 最終 GLB | V=336 487 F=447 556 | V=694 278 F=945 776 |
| bbox | (0.6296, 0.9989, 0.3126) | (0.6320, 0.9858, 0.3115) |
| 入力に対する mean silhouette IoU | **0.9044** | **0.9089** |

**fixture-input full E2E は解決済み**。前処理も seed も違う 2 経路が IoU 0.904 対 0.909 で
一致しており、bundle 経路が real-input 経路と同じ形を出すことが確認できた。
残る制約は §4 の tail のサイズ上限だけである。

## 3. Partial E2E（最小の integration gate）

Browser target: `pixal3d-webgpu-partial-e2e-wasm`、harness `web/partial_e2e/`

```text
Shape-1024 fixture -> Texture Flow -> Shape Decode -> Texture Decode -> production postprocess -> textured GLB
```

fixture は `PIXAL3D_DUMP_FIXTURE=<dir>` を付けた native の
`trellis-test-pixal3d-real-e2e` が実データから書き出せる
（`hr_coords` / `f32_shape_slat` / `f32_tex_concat_cond` / `tex_cond_global` /
`tex_cond_proj` / `tex_noise` / `tex_norm_{mean,std}`）。

## 4. Production postprocess

共通実装 `include/pixal3d_postprocess.h` / `src/pixal3d_postprocess.cpp`。
partial / fixture full / real full の 3 経路が同じ tail を使う。

**現状（2026-09-08）: 正しさの blocker ではないが、予算の食い違いが残っている。**
browser も weld → hole fill → narrow-band DC remesh → cleanup → QEM → UV atlas →
voxel PBR bake → textured GLB の全段を通り `textured_glb=OK` を出す。違うのは予算だけ:

| | remesh_res | target_faces | 実測 remesh | 最終 F |
|---|---|---|---|---|
| native | 1024 | 1 000 000 | 15 422 816 面 | 955 372 |
| browser | 512 | 500 000 | 3 319 396 面 | 490 292 |

**予算差の影響（同一入力・同一 seed の 2 つの GLB の相互シルエット IoU）**:

| 解像度 | IoU |
|---|---|
| 256 px | 0.9223 |
| 512 px | 0.8863 |
| 1024 px | **0.7192** |

粗い形は一致する（入力に対する IoU も browser 0.3696 / native 0.3674 とほぼ同じで、
予算差はシルエットを動かさない）が、**細部は 1024 px で見ると明確に違う**。
backend 差も混じっているが conditioning の parity が cos 0.99999 なので、
差の主因は remesh グリッド（512 対 1024）と面数（490k 対 955k）である。

**原因は wasm32 の構造的な上限**: `web/ss/CMakeLists.txt` の
`-sMAXIMUM_MEMORY=4294967296` は wasm32 の 4 GiB そのもの（MEMORY64 は使っていない）。
res=1024 の narrow-band remesh の出力は被写体で大きく変わる:

| 入力 | remesh @1024 |
|---|---|
| views4（`mesh_scale` 欠落, 壊れた形） | 7 676 815 V / 15 422 816 F ← ブラウザで `std::bad_alloc` |
| 公式サンプル（cyclops） | 9 158 222 V / 18 318 052 F |
| views4 + `mesh_scale=0.206`（正しい形） | **2 934 931 V / 5 890 304 F** |

**注意**: browser を 512/500k に落とす判断は、**壊れた入力で測った 15.4M 面**を根拠にして
いる（§4a）。正しい入力なら同じ被写体で 5.89M 面 = 2.6 分の 1 で、4 GiB に収まる可能性がある。
**未検証**。検証するには postprocess 予算を C ABI から渡せるようにして、
正しい入力で browser 側を 1024/1M で回す必要がある（約 40 分）。

**実測（2026-09-09、native Metal の最大 RSS）**: `pixal3d-partial-run` に
`PIXAL3D_REMESH_RES` / `PIXAL3D_TARGET_FACES` を足して cyclops（decode 4 730 295 voxel /
9 485 720 面）で測った。

| `remesh_res` | remesh 出力 | QEM 後 | **native の最大 RSS** |
|---|---|---|---|
| 1024 / 1M | 9 158 222 V / 18 318 052 F | 952 316 F | **7.30 GB** |
| 512 / 500k | 2 267 769 V / 4 536 476 F | 482 520 F | **5.39 GB** |

**ブラウザ実測（2026-09-09、同じ cyclops）**: native の RSS から wasm を推定するのは
誤りだった。wasm の live ピークは native の最大 RSS の約 **1/3** になる
（ポインタ幅が半分・`std::vector` の容量確保の差）。段別に測ると:

| 段 | 512（完走） | 1024 |
|---|---|---|
| postprocess 開始 | live 918 MB | live 918 MB |
| weld done（4 725 404 V / 9 510 791 F） | 1 027 MB | 1 027 MB |
| tri_bvh done（950 万三角形） | 1 644 MB | 1 644 MB |
| remesh done | 1 723 MB（4 536 496 面） | **std::bad_alloc**（18 318 052 面を作ろうとして） |
| qem done | 1 731 MB | — |
| uv done（ピーク） | **1 869 MB** | — |
| 結果 | `PARTIAL_E2E_RESULT: OK` | `FAIL rc=1` |

**512 は host live ピーク 1 869 MB / 4 GiB で余裕がある。1024 は remesh の中で落ちる。**
つまりブラウザの 512 は妥当な設定で、`target_faces` を上げる余地はあっても
`remesh_res` を 1024 にはできない。落ちる場所が「remesh の中」と特定できたのは、
今日入れた段別ログ（`plog` が live/free を即時 flush する）による。

小さい被写体（views4_fixed: decode 1 673 286 voxel / 3 409 360 面）はもともと 512 で完走する。
入力規模に応じて予算を自動で決めるなら、判定に使うべきは decode の面数（BVH がその上に
乗る）と remesh の予測面数の両方。

**優先度の判断（2026-09-08、§2 の結果を受けて改訂）: 後回しにできるのは小さい被写体だけ。**
当初は「正しさの blocker ではない」と判断したが、公式サンプル（cyclops、decode で
950 万三角形）の fixture-input full E2E は `remesh_res=512` でも `std::bad_alloc` で落ちた
（§2）。**被写体が大きいと browser では tail が完走しない**ので、これは品質だけの問題ではない。
差の定量化（下表）は有効で、公式サンプルは正しい入力でも remesh 18.3M 面になるので、
検証しても結論は「被写体次第」になる公算が高く、その場合の本当の対処は
「1024 に上げる」ではなく「入力サイズに応じた自動フォールバック」という別作業になる。
**後回しをやめる条件**: browser の出力が native と同等の細かさであることを主張したく
なったとき。1024 px の相互 IoU 0.7192 が直接の反証になるので、先に予算を可変にして測ること。

## 4a. 正しい入力での検証（2026-09-08）

これまでの real-input E2E は `~/Downloads/yoimiya_blender_1024` 由来の 4 view（以降 views4）で
回しており、「`mesh_scale` がどの手元データにも無い」「view 0 が canonical front view である
保証が無い」ため**形状品質を語れない**と書いてきた。Pixal3D 公式の multiview サンプル
（`assets/mv_images/example`: azim 0/90/180/270、view00 が front、`mesh_scale: 1.0` を明示、
FOV 0.349 rad、距離 3.119）で同じ native Metal の full E2E を回して切り分けた。

判定は着手前に凍結した基準:
(1) 4 視点で頭部欠損が無く、単眼・鼻輪・歯・耳・台座が出る、
(2) **入力カメラでの シルエット IoU ≥ 0.85**（`tools/silhouette_iou.py`、新規）。

### 結果: 公式サンプルは合格

| view | IoU | 生成の被覆率 | 入力の被覆率 |
|---|---|---|---|
| azim000 | 0.9805 | 0.453 | 0.455 |
| azim090 | 0.9772 | 0.615 | 0.622 |
| azim180 | 0.9796 | 0.528 | 0.533 |
| azim270 | 0.9768 | 0.615 | 0.622 |

**mean silhouette IoU 0.9785**。投影サイズ比 mine/ref = 0.997（= 入力と整合する
`mesh_scale` の推定値 0.997、宣言値 1.0）。テクスチャ付きレンダでも単眼・鼻輪・歯・唇・耳・
台座がすべて再現され、背面が滑らかなところまで入力と一致する。
`REAL_FULL_E2E_RESULT: OK V=4730295 F=9485720 glb_bytes=45113764 atlas=4096`、
bbox (0.7566, 0.8451, 0.9869)。

### views4 が不合格だった理由は入力側にある

同じ `tools/silhouette_iou.py` を views4 の結果に当てると:

| | 公式サンプル | views4 |
|---|---|---|
| `mesh_scale` の宣言 | 1.0（明示） | **無し（既定 1.0）** |
| mean silhouette IoU | **0.9785** | **0.1071** |
| 投影サイズ比 mine/ref | 0.997 | **0.206**（view ごと 0.203〜0.208） |
| 整合する `mesh_scale` の推定 | 0.997 | **0.206** |
| Shape-1024 tokens | 17 690 | 12 083 |

views4 は 4 view すべてで一貫して線形 4.85 倍ぶんスケールが食い違っている。既定の
`mesh_scale = 1.0` で走らせると、ProjGrid は被写体が画面の約 20%（面積で約 4%）しか占めない
前提で DINOv3/NAF 特徴をサンプリングすることになる。頭部が欠けるといった破綻はここから来る。
**軸の対応は導出せず符号付き軸置換 48 通りの総当たりで決めており**（公式サンプルでは
perm=(0,2,1) sign=(1,1,1) が 0.9647、次点の別対応が 0.8568 と明確に分離）、
views4 ではどの置換を選んでも 0.13 を超えなかったので、置換の選び方で結論は変わらない。

### 実証: views4 に `mesh_scale = 0.206` を入れるだけで直る

推定値をそのまま `transforms.json` に書いて（画像も重みもコード も一切変えずに）同じ
native Metal の full E2E を回した:

| | views4（`mesh_scale` 無し） | views4 + `mesh_scale=0.206` |
|---|---|---|
| mean silhouette IoU | 0.1071 | **0.9089** |
| 投影サイズ比 mine/ref | 0.206 | **1.003** |
| Shape-1024 tokens | 12 083 | 4 750 |
| raw mesh bbox | (0.9652, 0.9692, 0.5763) | (0.6303, 0.9843, 0.3098) |
| 最終 GLB bbox | (0.9571, 0.9815, 0.5965) | (0.6320, 0.9858, 0.3115) |
| 見た目 | **頭部が欠けた塊** | 頭・髪・顔・腕・脚・ブーツ・衣装が揃った全身 |

view ごとの IoU は 0.9014 / 0.9060 / 0.9102 / 0.9180。補正後の推定 `mesh_scale` は 0.207
（入れた 0.206 と一致）。bbox が「大きい塊」から「細長い全身」に変わっているのは、
補正前は被写体が grid のごく一部にしか投影されず、SS がその周辺を丸ごと占有と判定していた
ためで、補正後は人物の縦横比（0.63 : 0.99 : 0.31）が正しく出ている。

**結論**: 正しい入力（canonical front view + 正しい `mesh_scale`）を与えれば、
共通 C++ パイプラインは入力と IoU 0.91〜0.98 で一致する形状とテクスチャを出す。
これまでの real-input E2E で見えていた形状の破綻は**入力メタデータの不備**であって、
pipeline / backend の問題ではない。`mesh_scale` が transforms.json に無いデータを
そのまま流すと既定 1.0 で静かに壊れるので、`tools/silhouette_iou.py` の推定値で
先に確認すること。

## 5. 検証の位置づけ（重要）

- 分割グラフ版の conditioning は、**同一 backend の単一グラフ版**（PyTorch 参照で検証済みの経路）
  との同値性で確認している。
- **PyTorch 参照との突き合わせは実施済み（2026-09-08）**。結果は §5a。

## 5a. Texture-1024 conditioning の PyTorch 参照突き合わせ（2026-09-08）

参照: `tools/ref_pixal3d_cond_tex.py`（新規）を A100 80GB の pod で実行。
`DinoV3ProjMultiViewFeatureExtractor(**IMAGE_COND_CONFIGS["tex_1024"])`（image_size 1024,
grid_resolution 64, use_naf_upsample, **naf_target_size 1024**, multiview_fusion average）を
f32 で回し、dense な `z_proj [1, 64^3, 2048]`（2.1 GB）は保存せず、C++ が実際に使った
active voxel（`hr_coords.npy`, N=12 083）で gather した `[N, 2048]` だけを保存する。

C++ 側: `trellis-test-pixal3d-cond-tex --coords hr_coords.npy`（Metal、**分割グラフ**の
`cond_slat_gpu_chunked` 経路、view_alloc 3 332 MB）。入力 view・coords・`mesh_scale` は同一。
`PIXAL3D_DUMP_FIXTURE` を付けた `trellis-test-pixal3d-real-e2e` の dump とこのテストの出力は
**bit 一致**（`max|d| = 0`）なので、どちらを比較に使っても同じ。

比較は `tools/compare_cond_ref.py <ref_dir> <cpp_dir_or_prefix>` で再現できる。判定は既存の
`src/test_pixal3d_cond_slat.cpp` の `compare()` と同じ定義:
**`rel = max|d| / max|ref|`、tol 2e-2**（絶対値の max|d| ではない）。

| tensor | max abs | mean abs | **rel (tol 2e-2)** | L2 rel | cosine | bit identical |
|---|---|---|---|---|---|---|
| `global` [1,5,1024] | 3.631e-02 | 3.144e-03 | **1.238e-03 PASS** | 5.011e-03 | 0.9999874425 | no |
| `proj` lr `[:1024]` | 5.689e-02 | 3.944e-03 | **2.327e-03 PASS** | 5.402e-03 | 0.9999854080 | no |
| `proj` hr `[1024:]` | 4.753e-02 | 2.910e-03 | **1.950e-03 PASS** | 3.996e-03 | 0.9999922445 | no |
| `proj` 全体 | 5.689e-02 | 3.427e-03 | **2.327e-03 PASS** | 4.757e-03 | 0.9999887460 | no |

非有限値は両側とも 0。**全項目が基準を約 10 倍の余裕で満たしている**。

**texture 段固有の実装は誤差を足していない**: `z_global` は DINOv3 の最終 LN 後の先頭 5 トークン
（cls + register 4本）を view 平均しただけで、NAF も ProjGrid も chunk 分割も通らない。その
`global` が既に L2 rel 5.011e-03 出ており、NAF@1024 + ProjGrid HR + block-chunk 累積を通した
`proj hr` はむしろ小さい（3.996e-03 / cos 0.9999922）。naf_target_size=1024 の neighborhood
attention・sparse coords・4分割グラフは測れるほどの誤差を持ち込んでいない。

### 残差の出どころを 2 つ潰した（どちらも否定）

1. **f16 重みの精度ではない**。`FORCE_F32=1 tools/convert.py dinov3` で f32 の
   `dinov3.gguf`（1.21 GB, f16=0 / f32=318）を作って同じ比較を回したところ、
   `global` rel 1.238e-03 → 1.246e-03、`proj` 2.327e-03 → 2.328e-03 と**ほぼ変化なし**
   （4 桁目まで同じ）。f16 の丸めが原因という仮説は反証された。
2. **checkpoint の違いでもない**。C++ は timm の
   `vit_large_patch16_dinov3.lvd1689m`、参照は `camenduru/dinov3-vitl16-pretrain-lvd1689m`
   を使っているが、`patch_embed.proj.weight` / `embeddings.patch_embeddings.weight`
   （どちらも F32 [1024,3,16,16]）は **sha256 が byte 一致**
   （`ce6713297defe2507b1374abcd80f4fe9a2b8920ee1ec79d0747e66602f6d095`）。同じ重みである。

したがって残差は **C++ と参照の DINOv3 forward の差**である。**2026-09-09 に層別で特定した（§5b）。**これは texture 段の問題ではなく、
ss / shape_512 / shape_1024 / tex_1024 の全段が等しく持っている既存の差で、
`trellis-test-dinov3` 自身の基準（`max|d|/gmax < 5e-2`）の内側にある。DINOv3 段そのものの
parity 調査は本作業の範囲外。

**参照テンソルの所在**: pod の PVC `/nfs/pixal3d_condtex/out/`
（`tex_cond_global.npy` sha256 `312e5875…`、`tex_cond_proj.npy` sha256 `d6b07833…`、
`meta.json`、および `s1024_images.npy` / `camera_angle_x.npy` / `transform_matrix.npy` /
`mesh_scale.npy`）。入力の coords は `/nfs/pixal3d_condtex/podsend/hr_coords.npy`。
`kubectl cp` は 99 MB の `tex_cond_proj.npy` を 1 度無言で切り詰めたので、取り出しは毎回
sha256 で照合すること。

**環境の注記**: 参照 pod は natten **0.17.5**（shi-labs.com の証明書が期限切れで pip の
wheel index が引けないため、手元で取得して sha256 照合した wheel を PVC 経由で入れた）、
transformers は Pixal3D の requirements が pin する **4.57.3**（image の 5.8.0 は
`DINOv3ViTModel.layer` が無く `extract_features` が動かない）。`ref_pixal3d_cond_slat.py` の
メタは natten 0.21.0 で取られており、**natten の版は一致していない**（NAF は新旧どちらの
API にも対応しているが、版差の影響は測っていない）。

## 5b. DINOv3 の残差の内訳（2026-09-09、特定済み）

`tools/ref_dinov3_layers.py`（参照、CPU pod で可・natten 不要）と
`TRELLIS_DBG_DINOV3_LAYERS=<dir>`（C++）で、各ブロック出力の先頭 5 トークン
（cls + register 4 本 = z_global そのもの）を突き合わせた。入力は同じ CHW
（PIL LANCZOS + アルファ乗算 + ImageNet 正規化）を両側に与え、conditioning 経路にあった
resize 実装差（C++ は `stbir_resize_uint8`、参照は PIL LANCZOS）を排除している。

### 見つかった実バグ: 埋め込みトークンが f16 に丸められていた

| 段 | 修正前 | 埋め込みだけ f32 | 全部 f32 |
|---|---|---|---|
| 埋め込み（ブロック前） | 2.128e-04 | **0.000e+00** | 0.000e+00 |
| layer00 | 7.402e-05 | 7.413e-05 | 7.413e-05 |
| layer23 | 7.410e-05 | 7.427e-05 | 7.427e-05 |
| final（非アフィン LN 後） | 6.234e-04 | 6.400e-04 | 6.400e-04 |

原因は完全に確定した（3 つの数字が一致する）:

```
cpp_embed vs GGUF の値   L2 rel = 0.000e+00   （C++ は GGUF の値をそのまま使う）
ref_embed vs GGUF(f16)   L2 rel = 2.128e-04
ref を f16 に丸めた場合  L2 rel = 2.128e-04
```

`cls_token` は `[1024,1,1]`、`reg_token` は `[1024,4,1]` なので `tools/convert.py` の
「2D 以上は f16」に引っかかっていた。これらは行列積の重みではなく残差ストリームへ直接
足される値なので、丸めがそのままトークンの誤差になる。f32 に戻して 10 KB 増。

### 残る 6.4e-04 は浮動小数の加算順序（構造的）

埋め込みを直しても layer00 以降は変わらない。**全部 f32 の GGUF でも 4 桁まで同一**
なので、重み精度でもない。入力はビット完全、埋め込みもビット完全、重みも f32 で、
なお layer 0 の出力が 1e-4 台ずれる。

| layer00 | L2 rel | 値の桁 |
|---|---|---|
| token2（massive activation） | 7.41e-05 | \|max\| = 137 683 |
| 他 4 トークン | 2.27e-04 | \|max\| = 24 |

DINOv3 の layer 0 は埋め込み O(0.5) を **137 683 まで増幅する（利得 2.9×10^5）**。
この massive activation を作る過程で桁落ちが起きるため、加算順序の違いがそのまま
可視の差になる。相対誤差が 24 層を通じて 7.4e-05 でほぼ一定なのは、layer 0 で一度
入った差が残差接続をそのまま流れているためで、累積していない。

PyTorch のカーネルと加算順序を一致させない限り消えない差であり、DINOv3 段自身の
判定基準（`src/test_dinov3.cpp` の `max|d|/gmax < 5e-2`）に対して約 700 倍の余裕がある。
**実害のある差ではないと判断する。**

## 5c. CUDA での `check_graph_supported`（2026-09-09、静的検証）

`src/ss_decoder.cpp` に足した guard が CUDA を壊さないことを、vendored ggml の
ソースに対して静的に確認した。SS decoder が組む op は
ADD / CONT / MUL / NORM / PERMUTE / RESHAPE / MUL_MAT / UNARY(SILU) と、
非 Apple では `ggml_conv_3d` が展開する IM2COL_3D。
`thirdparty/ggml/src/ggml-cuda/ggml-cuda.cu` の `supports_op` はこのすべてに true を返す
（IM2COL_3D は 3078 行と 5435 行、SILU は 2900 行）。

**CUDA 実機での実行ではない。** 手元に CUDA が無く、`#ifdef __APPLE__` は
コンパイル時分岐なので macOS では im2col 経路をそもそも踏めない。

## 7. 実測（2026-09-08, M4 Max / Chrome WebGPU vs Metal）

入力: `transforms.json` + pre-matted RGBA 4 view（1024²）。`mesh_scale` はどの手元データにも
無いので既定 1.0。view 0 が canonical front view である保証も無い —— **以下は統合とメモリの
gate であって、形状品質の評価ではない**。

以下は §7b の SS decoder 修正**後**の true full E2E（browser: `web/real_e2e/`、seed 1、
native: `trellis-test-pixal3d-real-e2e`、同じ 4 view）。

| stage | browser (Chrome/WebGPU) | native (Metal) |
|---|---|---|
| SS Flow 12 step | 115.3 s (22 forwards) | 144.4 s (FA) / 89.3 s (NOFA) |
| Shape-512 Flow | 64.2 s | 102.2 s / 63.5 s |
| Shape-1024 tokens | 11 907 | 11 964 / 12 083 |
| Shape-1024 Flow | 755.7 s | 600.9 s |
| **live Texture-1024 cond** | **41.4 s, graph peak 1 538 MB, resident 2 741 MB** | 37.9 s, 3 332 MB, 2 741 MB |
| Texture Flow 12 step | 489.3 s (12 forwards) | 331.8 s |
| production postprocess | remesh_res=512 → remesh 3 319 396 面 → QEM 490 292 面, atlas 4096 | remesh_res=1024 → remesh 15 422 816 面 → QEM 955 372 面 |
| textured GLB | V=399 782 F=490 292 29 035 652 B | V=831 007 F=955 372 47 992 476 B |
| **textured GLB bbox** | **(0.9560, 0.9781, 0.5833)** | (0.9571, 0.9815, 0.5965) |

postprocess の予算がブラウザだけ違う（remesh_res 512 / target_faces 500 000 対 1024 / 1 000 000）
のは wasm32 の 4 GiB ヒープ制約によるもので、face 数とメッシュの粗さの差はここから来ている。
神経回路の段も**完全に同条件ではない**: browser は `g_no_fa`（plain softmax）で回り SS decoder
だけ CPU backend、native は FlashAttention + 全段 Metal。したがって上の表は backend 統合と
メモリの gate であって、backend 間の数値 parity の測定ではない（parity は §7a の同一 fixture
比較で測っている）。

token 数がばらつくのは、SS decode の閾値が離散判定で、backend 間（および FA / SDPA 間）の
わずかな数値差がそのまま active voxel 数を変えるため。

ブラウザで踏んだ 3 つの落とし穴（いずれも「これらのターゲットが一度もブラウザで実行されて
いなかった」ことの現れ）:

1. `dit_N4096_dcond5_proj1: 184 unsupported node(s)` —— WebGPU backend に BF16 K/V の
   FLASH_ATTN_EXT が無い。full / real の E2E が `g_no_fa` を立てていなかった。
2. `std::bad_alloc` —— res=1024 の narrow-band remesh が 7.8M 頂点 / 15.6M 面を作り、
   wasm32 の 4 GiB ヒープに収まらない。`remesh_res` を足してブラウザは 512 で回す。
3. `thread constructor failed: Not supported` —— wasm ターゲットは pthread 無しでリンク
   しているので `std::thread` が生成できない。remesh の並列化を直列に落とした
   （候補ビットセットがスレッドごとに res³/8 バイト要るので、メモリ的にも効く）。

## 7a. Texture Flow の native vs browser per-step parity（同一 fixture）

fixture は native の real-input run が吐いた Shape-1024 fixture（N=12 083）。native は
`pixal3d-texture-run --texture ... <dump_prefix>`（Metal, `g_no_fa` は texture 経路が常に立てる）、
browser は `web/texture/run_playwright.js`（Chrome/WebGPU）。同じ条件・同じ noise。

| step | max abs | mean abs | L2 rel | cosine | bit identical |
|---|---|---|---|---|---|
| 1 | 3.184e-04 | 4.963e-06 | 6.708e-06 | 1.000000000 | no |
| 4 | 4.334e-04 | 1.433e-05 | 2.134e-05 | 1.000000000 | no |
| 8 | 1.572e-03 | 3.910e-05 | 7.608e-05 | 0.999999997 | no |
| 12 / x_final | 3.258e-03 | 1.394e-04 | 2.546e-04 | 0.999999968 | no |

非有限値は両側とも 0。誤差は step とともに単調に積み上がるだけで、跳ぶ step は無い
（spec 32 の step-jump 破損の兆候は出ていない）。実時間は browser 496.0 s / native 12 forwards。

## 7b. 解決済み: ブラウザの live geometry が X 軸に潰れていた（2026-09-08）

**症状**: browser の real-input E2E は完走して GLB を書くのに、形状が X 軸に潰れていた
（最終 GLB の bbox [0.101, 0.996, 0.980]。native は [0.905, 0.971, 0.578]）。

**切り分け**: 実入力パイプラインの各段に計測（座標の bbox / 重心、テンソルの
mean/std/min/max/非有限数）を入れて browser と native を並べた。

| stage | native (Metal) | browser 修正前 | browser 修正後 |
|---|---|---|---|
| `ss_cond_proj` | std 0.966560 | 0.966559 | 0.966559 |
| `ss_latent` | mean +0.007359 std 0.532235 | +0.007341 / 0.532413 | 同左 |
| **`ss_logits`** | mean −146.74 std 56.01 **nonfinite 0** | mean −27.60 std **12518** **nonfinite 21143** | mean −146.78 std 56.07 **nonfinite 0** |
| `ss_coords@64` | x[0..31] y[0..31] z[5..24] centroid(15.2,18.4,15.3) | **x[2..4]** y[0..31] z[0..31] | x[0..31] y[0..31] z[5..24] centroid(15.2,18.4,15.3) |
| `hr_coords@64` | N=12 083 x[1..63] z[11..48] | N=10 901 **x[4..10]** | N=11 907 x[1..62] z[11..48] |
| raw mesh bbox | (0.9652, 0.9692, 0.5763) | — | **(0.9537, 0.9750, 0.5799)** |
| 最終 textured GLB bbox | (0.9571, 0.9815, 0.5965) | **(0.1006, 0.9961, 0.9801)** | **(0.9560, 0.9781, 0.5833)** |

**根本原因**: ggml WebGPU backend は `GGML_OP_IM2COL_3D` も `GGML_OP_CONV_3D` も実装して
いない（`docs/PIXAL3D_WEBGPU_OP_GAP.md`: "every Conv3d in this file is unsupported
regardless of which branch compiles"）。SS decoder (`src/ss_decoder.cpp`) は全段が Conv3d で、
かつ `ss_decode` は `trellis_graph_dump` は呼ぶが **`check_graph_supported` を呼んでいなかった**
ため、FlashAttention のときのように実行前に弾かれず、黙って NaN 混じりの occupancy logits を
返していた。その logits から取った active voxel が薄いスラブになり、以降の Shape-512 /
Shape-1024 / decode が全部その上に積み上がっていた。

**修正**:

1. wasm ビルドでは **SS decoder だけ CPU backend に載せる**（`Model::load(path, -1)`。
   `trellis-test-pixal3d-ss-sample` の `dec_gpu=-1` と同じ扱い）。他の段は WebGPU のまま。
2. `ss_decode` に `check_graph_supported` を追加し、同種の「黙って壊れる」を実行前のエラーにする。

**native への回帰が無いことの確認**: guard を足した `src/ss_decoder.cpp` でリビルドした
`build-metal/trellis-test-pixal3d-real-e2e` は Metal で `REAL_GEOMETRY_RESULT: OK`
（V=4 098 362 F=8 244 512、`ss_logits nonfinite=0`）。Metal backend は Conv3D を持つので
guard は素通りする。CUDA は手元に無いので未確認（`ggml-cuda` は IM2COL_3D を実装しているので
通るはずだが実測していない）。`ss_decode` を呼ぶ翻訳単位は
`src/pixal3d_real_geometry_wasm.cpp` と `src/pixal3d_full_e2e_wasm.cpp` の 2 つだけで、
どちらも `SS_DEC_BACKEND` 対応済み（`grep -ln ss_decode src/*.cpp` で確認）。

**修正後の true full E2E**: browser が textured GLB を bbox (0.9560, 0.9781, 0.5833) で出力
（native (0.9571, 0.9815, 0.5965)）。4 視点レンダも native と同じシルエット・同じ色・
同じ破綻の仕方（頭部が欠ける = 入力ビュー由来で backend 由来ではない）になった。

**教訓**: `check_graph_supported` を通っているグラフ（DiT / conditioning / NAF）は未対応 op を
実行前に弾くが、通っていないグラフは黙って壊れる。新しいグラフを足したら必ず通すこと。

## 6. 走らせ方（このリポジトリでの実行手順）

WASM ビルド:

```sh
emcmake cmake -S . -B build-wasm-ss -DGGML_WEBGPU=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DTRELLIS_WASM_SS=ON
cmake --build build-wasm-ss --target pixal3d-webgpu-ss-wasm pixal3d-webgpu-texture-wasm \
  pixal3d-webgpu-partial-e2e-wasm pixal3d-webgpu-full-e2e-wasm pixal3d-webgpu-real-geometry-wasm -j
```

必要な GGUF は 9 本: `dinov3` / `pixal3d_naf` / `pixal3d_ss_flow_mv` / `ss_dec` /
`pixal3d_shape_flow_512_mv` / `shape_dec` / `pixal3d_shape_flow_1024_mv` /
`pixal3d_tex_flow_1024_mv` / `tex_dec`。NAF だけ HF に無く、`valeoai/NAF` の release
`naf_release.pth` を safetensors 化してから `tools/convert.py pixal3d_naf`。

ブラウザ（リポジトリルートを配信してから）:

```sh
python3 -m http.server 8200 --directory .   # 別シェル
cd web
PIXAL3D_BASE_URL=http://127.0.0.1:8200/web/real_e2e/ \
  node real_e2e/run_playwright.js <gguf_dir> <views_dir> 1        # real-input full E2E
PIXAL3D_BASE_URL=http://127.0.0.1:8200/web/partial_e2e/ \
  node partial_e2e/run_playwright.js <tex_flow.gguf> <shape_dec.gguf> <tex_dec.gguf> <fixture_dir>
```

native の同一 C++ 経路（ブラウザへ投げる前の dry-run。browser と同じ SDPA 経路を踏むなら
`TRELLIS_NOFA=1`）:

```sh
cmake --build build-metal --target trellis-test-pixal3d-real-e2e
PIXAL3D_DUMP_FIXTURE=<dir> ./build-metal/trellis-test-pixal3d-real-e2e \
  <dinov3> <naf> <ss_flow> <ss_dec> <shape512> <shape_dec> <shape1024> \
  <views_dir> <out.glb> 1 <tex_flow> <tex_dec>
```

conditioning のメモリと数値だけを見るとき:

```sh
cmake --build build-metal --target trellis-test-pixal3d-cond-tex
TRELLIS_DBG_COND=1 ./build-metal/trellis-test-pixal3d-cond-tex <dinov3> <naf> <views_dir> 0 \
  --S 1024 --R 64 --naf-t 1024 --views 4 --stride 4 --save-prefix /tmp/tex1024
```
