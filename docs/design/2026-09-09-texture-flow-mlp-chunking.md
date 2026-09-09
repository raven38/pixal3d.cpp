# Texture Flow: MLP 中間のトークン分割で比例項を下げる

対象 issue: #12
状態: 設計（実装前）
日付: 2026-09-09

## 背景

Texture Flow の GPU 常駐量は N（grid-64 のアクティブボクセル数 = DiT のトークン数）に
比例する。実測 2 点（native Metal, `TRELLIS_ATTN_CHUNK_MB=128`、
`docs/PIXAL3D_WEBGPU_MEMORY.md` §12）:

```
活性化 = 59.7 KB/token x N + 0.2 MB      （N=12083 -> 705.1 MB / N=17690 -> 1032.2 MB）
cond   = 16.0 KB/token x N               （proj [2048,N] を positive/negative 2 本）
合計   = 重み（固定）+ 75.7 KB/token x N
```

WebGPU の予算 4095 MB に対する現状の上限 N:

| 重み | 上限 N |
|---|---|
| f16 2647 MB | 19 588 |
| Q8_0 1408 MB | 36 348 |

固定項（重み）は #8 の Q8_0 化で 2647 → 1408 MB まで下げた。活性化の attention 分は
#8 のクエリ分割で 1888 → 1032 MB まで下げた。**次に効くのは比例項 75.7 KB/token** である。

## 比例項の内訳（d_model = 1536, d_proj = 2048）

```
qkv        3 x 1536 x 4B = 18.4 KB/token
MLP 中間   4 x 1536 x 4B = 24.6 KB/token   <- 最大
proj       2048 x 4B     =  8.0 KB/token
cond       2048 x 4B x 2 = 16.0 KB/token
                         ≒ 67 KB/token（実測 75.7 との差は残差・一時テンソル）
```

最大項は MLP 中間 24.6 KB/token である。

## 設計

`src/dit.cpp` の DiT ブロック内、

```cpp
hh = layernorm(c, h, p.ln_eps);
hh = modulate(c, hh, scale_mlp, shift_mlp);
hh = lin(c, m, b + ".mlp.mlp.0", hh);   // [4*d_model, N]  <- ここが 24.6 KB/token
hh = ggml_gelu(c, hh);
hh = lin(c, m, b + ".mlp.mlp.2", hh);   // [d_model, N]
```

の `fc1 -> gelu -> fc2` をトークン方向に分割する。既に同じ技法が 3 箇所にあり、型も
揃っているので流用できる:

- `src/dit.cpp` の `sdpa` クエリ分割（`kAttnChunkBytes`）
- `src/sparse.cpp` の `sparse_c2s` の出力ボクセル分割（`chunk2`）
- `src/pixal3d_cond_gpu.cpp` の `naf_block_chunk`

形は `sdpa` のクエリ分割と同じにする:

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

チャンク数は attention と同じく「1 チャンクの中間テンソルのバイト数」で決める。
環境変数は `TRELLIS_MLP_CHUNK_MB`（既定は attention と揃えて native 1024 / wasm 128）と
`TRELLIS_MLP_MAX_CHUNKS` を置く。

### なぜビット一致するか

MLP はトークンごとに完全に独立である。`fc1` / `fc2` は `ggml_mul_mat` で列（トークン）方向に
独立、`gelu` は要素ごと。トークンをまたぐ縮約が一つも無いので、分割しても各要素の
演算順序が変わらない。attention のクエリ分割が softmax の行が閉じているためビット一致
だったのと同じ理屈で、MLP はさらに単純である。

### 逐次 concat にする理由

`out = out ? ggml_concat(c, out, y, 1) : y;` と逐次に畳むと、ggml の allocator が各チャンクの
`[4*d_model, n]` 中間をその concat の後で解放できる。全チャンクを作ってから一度に concat
すると全部が同時に生存してピークが下がらない。attention の分割で活性化が 1888 → 1032 MB に
下がったのはこの形だったからで、同じ形にする。

## 期待効果

チャンク数 k のとき MLP 中間の寄与は 24.6 / k KB/token になる。k = 8 とすると:

```
75.7 - 24.6 + 24.6/8 = 54.2 KB/token
```

上限 N は次のようになる（予算 4095 MB）:

| 重み | 現状の上限 N | 分割後の上限 N | 倍率 |
|---|---|---|---|
| f16 2647 MB | 19 588 | 27 357 | 1.40 |
| Q8_0 1408 MB | 36 348 | 50 770 | 1.40 |

**この表は設計上の予測であり、実測で置き換えること。** attention のときも予測と実測は
ずれた（上限 32 チャンクだと 1 チャンク 553 クエリで頭打ちになり、活性化は 1316 MB より
下がらなかった）。

## リスク

- **グラフのノード数・テンソルメタデータ枠が先に枯れる。** 1 チャンクあたり約 6 ノード
  （view / cont / mul_mat x2 / add x2 / gelu / concat）で、30 ブロック x 2（positive/negative）分が
  1 グラフに載る。k = 8 なら 30 x 8 x 6 ≒ 1440 ノード増。`src/flow_runner.cpp` の枠は #8 で
  テンソルメタ 131072・グラフノード 262144 まで広げてあるので余裕はあるはずだが、
  実測で確認する。1 テンソル 368 B なので広げるのは安い。
- **速度。** attention の分割はむしろ速くなった（644.4 → 602.5 秒）。MLP も mul_mat の
  タイル効率が落ちない限り悪化しないはずだが、チャンクが小さすぎると起動オーバーヘッドが
  効く。k を振って測る。
- **予算ゲートの式がずれる。** `DitRunner::check_device_budget()` は活性化を
  `TRELLIS_ATTN_CHUNK_MB` から見積もっている。MLP 分割を入れたら見積もり式も更新しないと、
  ゲートが過大に見積もって通るはずの N を弾く。

## 検証計画（受け入れ条件）

1. **ビット一致。** `trellis-test-pixal3d-slat-sample --stage tex --dump` で 12 step の
   latent を分割前後で出し、`max|d| = 0` を確認する（attention の分割と同じ基準）。
   分割前の dump は `~/data/weights/pixal3d/tex_run_A/cpp_tex_x_step*.npy` を使うのではなく、
   同一ビルド・同一マシンで取り直す（コンパイラや重みが変わっていない保証のため）。
2. **比例項が下がる。** N を 2 点（例 12083 と 17489）で測り、
   `活性化 = a x N + b` の a が 59.7 KB/token から有意に下がることを数値で示す。
   1 点だけでは比例項と固定項を分離できない。
3. **上限 N が上がる。** `check_device_budget()` の見積もりを更新したうえで、
   予算 4095 MB に対する上限 N を f16 / Q8_0 の両方で再計算して表を更新する。
4. **速度が悪化しない。** 12 step の所要時間を分割前後で比較する。
5. **チャンク数の掃引。** k = 2 / 4 / 8 / 16 で活性化と所要時間を測り、選んだ既定値の
   根拠を残す。

## 次に効く項（この issue の範囲外）

MLP を分割すると最大項は qkv 18.4 KB/token と cond 16.0 KB/token になる。cond は活性化では
なく入力（`proj [2048,N]` を positive/negative 2 本、forward ごとに再アップロード）なので、
分割ではなく「negative をゼロテンソルとして共有する」「forward 間で再アップロードしない」
といった別の手が要る。qkv は attention のクエリ分割と同じ形で分割できる可能性がある。
