# Texture Flow: MLP 中間のトークン分割で比例項を下げる

対象 issue: #12
状態: 設計（実装前）— codex による反証レビュー（2026-09-09）を反映した第 2 版
日付: 2026-09-09

第 1 版には 3 つの裏付け不足があり、レビューで指摘された。訂正内容は末尾の「第 1 版からの訂正」を見ること。

## 背景

Texture Flow の GPU 常駐量は N（grid-64 のアクティブボクセル数 = DiT のトークン数）に
比例する。実測 2 点（native Metal, `TRELLIS_ATTN_CHUNK_MB=128`、
`docs/PIXAL3D_WEBGPU_MEMORY.md` §12）:

```
活性化 = 59.7 KiB/token x N + 0.2 MiB    （N=12083 -> 705.1 MiB / N=17690 -> 1032.2 MiB）
cond   = 16.0 KiB/token x N              （proj [2048,N] を positive/negative 2 本）
合計   = 重み（固定）+ 75.7 KiB/token x N
```

（単位は KiB / MiB である。実測は 1048576 で除している —
`src/test_pixal3d_slat_sample.cpp:401-403`。）

固定項（重み）は #8 の Q8_0 化で 2647 → 1408 MiB まで下げた。活性化の attention 分は
#8 のクエリ分割で 1888 → 1032 MiB まで下げた。次に狙うのは比例項である。

## 比例項に効きそうな項（あくまで内訳の目安）

d_model = 1536, d_proj = 2048:

```
qkv        3 x 1536 x 4B = 18 KiB/token
MLP 中間   4 x 1536 x 4B = 24 KiB/token   <- 単体では最大
proj       2048 x 4B     =  8 KiB/token
cond       2048 x 4B x 2 = 16 KiB/token
```

**これらは足し算できない。** ピークは「区間ごとの生存量の最大値」であって項の総和ではなく、
qkv と MLP 中間は別の演算区間にある。したがって「MLP を k 分割すれば 75.7 が
75.7 − 24 + 24/k になる」という第 1 版の式は成立しない。**MLP がいま本当にピークを
決めているかは、実装前にアロケータの区間別生存量で確認する**（下の「事前確認」）。

## 設計

`src/dit.cpp` の DiT ブロック内、

```cpp
hh = layernorm(c, h, p.ln_eps);
hh = modulate(c, hh, scale_mlp, shift_mlp);
hh = lin(c, m, b + ".mlp.mlp.0", hh);   // [4*d_model, N]
hh = ggml_gelu(c, hh);
hh = lin(c, m, b + ".mlp.mlp.2", hh);   // [d_model, N]
```

の `fc1 -> gelu -> fc2` をトークン方向に分割する。形は `sdpa` のクエリ分割
（`src/dit.cpp:206-218`）に合わせる:

```cpp
T* out = nullptr;
for (int64_t t0 = 0; t0 < N; t0 += nt) {
    const int64_t n = std::min(nt, N - t0);
    T* xc = (n == N) ? hh : ggml_cont(c, ggml_view_2d(c, hh, d_model, n, hh->nb[1],
                                                      (size_t)t0 * hh->nb[1]));
    T* y = lin(c, m, b + ".mlp.mlp.0", xc);   // [4*d_model, n]
    y = ggml_gelu(c, y);
    y = lin(c, m, b + ".mlp.mlp.2", y);       // [d_model, n]
    out = out ? ggml_concat(c, out, y, 1) : y;
}
```

環境変数は `TRELLIS_MLP_CHUNK_MB` と `TRELLIS_MLP_MAX_CHUNKS` を置く。

### 数学的同値性とビット一致は別物

**数学的には同値である。** LayerNorm は特徴軸 `ne[0]` の縮約（`src/dit.cpp:47-51`）、
modulate は特徴ごとの broadcast（`:264-266`）、gate も要素積（`:316`）で、いずれも
トークンを混ぜない。`fc1` / `fc2` は列（トークン）方向に独立、`gelu` は要素ごと。

**しかしビット一致は保証されない。** 列数を変えると行列積のカーネル選択が変わる:

- Metal は列数 2〜8 で専用カーネル、9 以上で matrix-matrix 経路
  （`thirdparty/ggml/src/ggml-metal/ggml-metal-ops.cpp:2057-2078, 2164-2167`）
- WebGPU は列数 1 で別経路（`thirdparty/ggml/src/ggml-webgpu/ggml-webgpu.cpp:1478-1511`）

大半のチャンクが大きくても、**末尾チャンクが 1〜8 トークンになると**この分岐を踏む。
したがってビット一致は「証明」ではなく **backend・重み型・末尾チャンク幅ごとに検証する
事項**として扱う。

### 逐次 concat について（第 1 版の説明は誤り）

第 1 版は「逐次 concat にすると allocator がチャンク中間を解放できる」と書いたが誤り。
ggml はノード出力を確保したあと親の残り参照数を減らし、最後の利用で領域を再利用可能に
する（`thirdparty/ggml/src/ggml-alloc.c:776-816`）。`[4*d_model, n]` の中間を最後に読むのは
**fc2 の mul_mat** であって concat ではない。concat まで保持されるわけではない。

またグラフ構築の順序と実行順序は別で、実行順は依存関係からの深さ優先走査で決まる
（`thirdparty/ggml/src/ggml.c:6928-6961, 7168`）。同じ concat 木なら構築ループの書き方は
影響しない。

第 1 版が根拠にした 1888 → 1032 MiB は、**attention のチャンク予算 1024 → 128 MiB と
チャンク上限 32 → 256 を変えた比較**であって（`docs/PIXAL3D_WEBGPU_MEMORY.md:669-678`）、
concat の形を切り分けた実験ではない。

**逐次 concat にはコストがある。** concat は in-place 対象ではなく
（`ggml-alloc.c:22-48`）、最終 concat では旧出力・末尾出力・新しい全長出力が同時に要る。
また等分 k = 8 のとき concat 出力への書き込みだけで全長出力の約 4.4 倍になる
（WebGPU は concat 出力全体を dispatch する —
`thirdparty/ggml/src/ggml-webgpu/ggml-webgpu.cpp:2330, 2385-2387`）。
平衡木にすれば書き込みは約 3 倍に減るがピークは上がる。まず `sdpa` と同じ逐次で実装し、
測ってから木の形を選ぶ。

## 事前確認（実装前にやる）

1. **いまのピークが本当に MLP 由来か。** アロケータの区間別生存量を見て、ピークを作って
   いる区間を特定する。MLP でなければこの issue の前提が崩れる。
2. **既定のバイト予算だと、そもそも分割されない。** 中間 1 本の予算を 128 MiB にすると
   `nt = 5461` で、

   | N | MLP チャンク数 | attention スコア 1 チャンク |
   |---:|---:|---:|
   | 12 083 | 3 | 127.77 MiB |
   | 17 690 | 4 | 127.95 MiB |
   | 27 357 | 6 | 134.00 MiB |
   | 50 770 | 10 | 462.49 MiB |

   **native 既定の 1024 MiB では N = 12 083 / 17 690 で MLP は 1 チャンクのまま**である。
   効くのは wasm 既定（128 MiB）の経路。さらに N = 50 770 では attention 側の 256 チャンク
   上限が効いてスコアが再び増える（`src/dit.cpp:187-205`）。低い N の 2 点から得た比例係数を
   上限 N へ外挿してはいけない。

## 期待効果

**第 1 版の「1.40 倍」の表は撤回する。** 上の理由（ピークは区間の最大値、チャンク数は
バイト予算に対して段差、attention 側の上限が高 N で効く）から、係数の外挿で上限 N を
出すことはできない。効果は実測で示す。

## リスク

- **末尾チャンクでカーネル選択が変わる**（上記）。末尾 1 / 2 / 8 / 9 トークンを狙って
  検証する。
- **concat のコピー量**（上記）。速度が落ちうる。
- **グラフのノード数。** 1 チャンクあたり view / cont / mul_mat x2 / add x2 / GELU / concat の
  **8 ノード**（bias の有無で増減）。positive / negative は **1 グラフに 2 本載らない** ——
  同じ runner へ順に forward する（`src/flow_runner.cpp:254-255`）。`gproj_` も 1 本で
  （`:103-105`）毎回そこへアップロードする（`:139-141`）。枠は #8 でテンソルメタ 131072 /
  グラフノード 262144 まで広げてある。
- **予算ゲートの既存問題（この issue とは独立）。** `check_device_budget()` の
  `alloc_bytes_` は `ggml_gallocr_get_buffer_size()` の実測値なので
  （`src/flow_runner.cpp:118-121`）、MLP 分割の効果は自動的に反映され、見積もり式の更新は
  要らない。一方 `need` が `size_t` なので（`:62`）**wasm32 では合計 4 GiB 以上で加算が
  周回して過小判定になる**。上限 N を広げる検証をするならこれを先に直す。
  超過時のエラーメッセージも `TRELLIS_ATTN_CHUNK_MB` しか案内していない（`:80-86`）。

## 検証計画（受け入れ条件）

1. **数値一致。** `trellis-test-pixal3d-slat-sample --stage tex --dump` で 12 step の latent を
   分割前後で出す。**ビット一致は backend / 重み型 / 末尾チャンク幅ごとに確認する**:
   Metal と WebGPU、f16 と Q8_0、末尾チャンク 1 / 2 / 8 / 9 トークン。
   ビット一致しない組があるなら、その組と差の大きさを明記する（隠さない）。
   ベースラインは同一ビルド・同一マシンで取り直す。
2. **ピークが下がる。** N を 2 点（12083 と 17489）で `alloc_bytes_` を測る。
   区間別生存量も併せて記録し、ピークを決めている区間が変わったことを示す。
3. **チャンク境界と上限付近の確保量。** チャンク数が変わる N の前後、および予測上限
   付近で実確保量を測る。係数の外挿はしない。
4. **速度。** 12 step の所要時間を分割前後で比較する。concat のコピー増を含めて悪化しない
   ことを示す。悪化するなら平衡木を試す。
5. **チャンク数の掃引。** k = 2 / 4 / 8 / 16 で確保量と所要時間を測り、既定値の根拠を残す。

## 第 1 版からの訂正

| 第 1 版の主張 | 訂正 |
|---|---|
| トークン独立なのでビット一致する | 数学的同値は成立するが、末尾チャンクで行列積のカーネル選択が変わるためビット一致は保証されない。検証事項に降格 |
| 逐次 concat だから allocator が中間を解放できる。1888→1032 MiB はその効果 | 中間を最後に読むのは fc2 であって concat ではない。1888→1032 MiB は attention のチャンク予算とチャンク上限を変えた比較で、concat の形とは無関係 |
| 75.7 − 24.6 + 24.6/8 = 54.2 KiB/token、上限 N が 1.40 倍 | ピークは区間の最大値で項の総和ではない。単位も MiB/KiB。式も 54.7 で 54.2 ではない。表は撤回し実測に置き換える |
| `check_device_budget()` の見積もり式を更新しないとゲートが過大判定する | `alloc_bytes_` は実測値なので式の更新は不要（これは第 1 版の途中で自分でも訂正済み）。ただし `need` の `size_t` が wasm32 で周回する既存問題がある |
| 1 チャンク約 6 ノード、positive/negative が 1 グラフに 2 本 | 8 ノード。positive/negative は同じ runner へ順に forward するので 1 グラフに 2 本載らない |

## 次に効く項（この issue の範囲外）

MLP を分割したあと何が効くかは、事前確認 1 の区間別生存量が出てから決める。
cond（`proj [2048,N]` の GPU 入力領域）については、ホスト側の positive/negative 配列と
GPU 入力領域（`gproj_` 1 本）を区別して議論する必要がある。
