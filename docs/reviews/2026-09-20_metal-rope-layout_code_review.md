### 判定: 承認

差し戻し相当の実装不具合は見つかりませんでした。指定された6主張は、現在のコードと ggml 実装に照らして成立しています。

### 重大な問題（file:line 付き、優先度順）

なし。

特に懸念されていた RoPE の consumer については、次を確認しました。

- self-attention の q/k は RMSNorm 後に両方とも `apply_rope` を通り、その直後の唯一の consumer は `sdpa` です。[src/dit.cpp:314](src/dit.cpp:314)
- RMSNorm は RoPE より前なので、de-interleave 後の channel 順序を学習済み gamma が参照することはありません。[src/dit.cpp:325](src/dit.cpp:325)
- cross-attention は別に q/k を生成し、どちらも RoPE を通りません。cond 由来 k と latent 由来 q の片側だけが置換される反例はありません。[src/dit.cpp:332](src/dit.cpp:332)
- exact SDPA は dim 0 を `ggml_mul_mat(k, q)` で縮約し、FA も `[head_dim, token, head]` へ並べ替えた後に head_dim を内積化します。mask は生成済み score の key/query 軸に作用するため、head_dim 内の共通置換とは独立です。[src/dit.cpp:238](src/dit.cpp:238) [src/dit.cpp:265](src/dit.cpp:265)

stride も整合しています。`cont` 後の `xp=[half,2,nh*L]` は `nb[1]=half*sizeof(float)`、`nb[2]=hd*sizeof(float)` です。したがって、`nb1=xp->nb[2]` は head stride、`nb2=xp->nb[2]*nh` は token stride、`offset=xp->nb[1]` は odd 面の先頭を正しく表します。[src/dit.cpp:83](src/dit.cpp:83)

cos/sin の `[half,1,L]` は ggml の `ggml_can_repeat()` 条件を満たし、head 軸 `1 → nh` に反復されます。WebGPU shader も各軸を `% b_ne*` で反復しています。[ggml.c:1551](thirdparty/ggml/src/ggml.c:1551)

`concat(dim=1)` は `[half,2,nh*L]` の新規連続 tensor を生成します。これを `[hd,nh,L]` に reshape する flat-order は、各 head ごとに `[ev0..evHalf-1, od0..odHalf-1]` となり、意図通りです。[src/dit.cpp:90](src/dit.cpp:90)

`--profile` も以下の理由で妥当です。

- `ggml_graph_clear`、`ggml_graph_add_node`、graph/node accessor はすべて `GGML_API` の公開 API です。[ggml.h:2729](thirdparty/ggml/include/ggml.h:2729)
- スライスは元グラフのトポロジカル順を維持して逐次再実行するため、gallocr の lifetime reuse と矛盾しません。
- サンプラへ返す `outv` は profiling pass より前にホストへコピーされています。[src/flow_runner.cpp:154](src/flow_runner.cpp:154)
- 次 forward では入力、cos/sin、cond、proj が再アップロードされるため、profiling pass が共有バッファを上書きしても次のサンプル入力には残りません。

### 軽微な指摘

- WebGPU の運用文書が旧 graph のままです。`SET_ROWS 120`、`rope_idx`、ARANGE 回避を現在も使用すると記載されています。[docs/PIXAL3D_WEBGPU_OP_GAP.md:461](docs/PIXAL3D_WEBGPU_OP_GAP.md:461) [docs/spec/31-webgpu-bringup.md:851](docs/spec/31-webgpu-bringup.md:851)  
  現在の `CONT/MUL/SUB/ADD/CONCAT` graph に更新すべきです。

- `rope_idx` は実質 dead API ですが、引数と `dit_rope_index()` が全階層に残っています。[include/dit.h:48](include/dit.h:48) [src/dit.cpp:422](src/dit.cpp:422)  
  外部 ABI 互換が不要になった段階で削除すれば、将来「渡せば使われる」と誤解される余地をなくせます。少なくとも「まだ upload する caller がある」というコメントは、リポジトリ内検索結果とは一致しません。

- 新しい layout テストは executable を追加しただけで、自動テストには登録されていません。[CMakeLists.txt:464](CMakeLists.txt:464)  
  CI が全 executable を明示実行しない構成なら退行を検出できません。

- テストコメントの「bit-exact fp32 ops」は実測結果および `1e-6` 許容と矛盾します。[src/test_rope_layout.cpp:5](src/test_rope_layout.cpp:5)  
  「backend-dependent rounding を許容」とする方が正確です。

- q·k テストはホスト側 double accumulation であり、FA/exact-SDPA カーネル自体は通していません。[src/test_rope_layout.cpp:76](src/test_rope_layout.cpp:76)  
  レイアウト証明としては十分ですが、カーネル統合テストとは区別すべきです。

### 検証できなかった点

- CUDA、Vulkan、実ブラウザ WebGPU 上での新 RoPE の実行結果。WebGPU shader の stride/broadcast/concat 実装は静的確認しましたが、ブラウザ実行はしていません。
- 実 GGUF を使った `--profile` 有無の同一 forward 出力比較。作業ツリー内に GGUF がなかったため、今回その場では再実行できませんでした。
- CUDA/Metal FA と exact SDPA の新旧レイアウト出力差の直接比較。数学的には共通置換ですが、K の BF16/F16 cast と backend 固有の縮約順による丸め差は残ります。

今回実行した検証では、CPU・M4 Max Metal の `trellis-test-rope-layout` がともに PASSしました。最大 layout 誤差は `4.77e-07`、全 head/query/key の q·k 最大差は `2.6e-06` でした。Native の RoPE/bench/CLI と、DiT を含む主要 WASM 5ターゲットも再コンパイル・リンクに成功しています。
