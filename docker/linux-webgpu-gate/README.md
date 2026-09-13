# linux-webgpu-gate — Linux / GPU pod で Web 版を検証するためのイメージ（M12）

macOS 以外の WebGPU 実装（NVIDIA / Vulkan / Dawn）で Web 版を回すための headless Chrome 環境。
共有 GPU クラスタの L4 pod は egress が HTTPS のみで apt が使えないため、必要物は全て Mac 上で焼き込む。
レジストリ・namespace 等の実値と実 pod YAML はリポジトリ外（作業ノート側）に置く。

| 層 | Dockerfile | タグ（`$REGISTRY:`） | 中身 |
|---|---|---|---|
| Vulkan | `Dockerfile.vulkan` | `vulkan-noble` | ubuntu:24.04 + libglvnd/libvulkan1 + NVIDIA ICD json |
| Chrome | `Dockerfile.chrome` | `chrome-153` | + Node 22.14 + Playwright 1.63（Chromium 153.0.8010.12、macOS ゲートと同じ）+ xvfb |
| gate | `Dockerfile.gate` | `gate-20260911`（digest `sha256:7cfb45fa…`） | + `probe_webgpu.mjs` |

```sh
export DOCKER_BUILDKIT=1 REGISTRY=<registry>/webgpu-gate
docker buildx build --platform linux/amd64 --provenance=false --sbom=false -f Dockerfile.vulkan -t $REGISTRY:vulkan-noble --load .
docker push $REGISTRY:vulkan-noble
docker buildx build ... --build-arg REGISTRY=$REGISTRY -f Dockerfile.chrome -t $REGISTRY:chrome-153 --load . && docker push $REGISTRY:chrome-153
docker buildx build ... --build-arg REGISTRY=$REGISTRY -f Dockerfile.gate   -t $REGISTRY:gate-20260911 --load . && docker push $REGISTRY:gate-20260911
```

pod 側（L4 1 枚、cpu 8–16 / mem 32–64 Gi で足りる）:

```yaml
env:
- {name: NVIDIA_DRIVER_CAPABILITIES, value: all}
- {name: XDG_RUNTIME_DIR, value: /tmp}
- {name: EXTRA_ARGS, value: "--enable-dawn-features=vulkan_enable_f16_on_nvidia"}
command: ["bash","-lc"]
args:
- |
  cd /opt/gate && timeout 300 xvfb-run -a node probe_webgpu.mjs --headless xvfb --out /tmp/webgpu_probe.json
  echo "probe rc=$?"     # xvfb-run を最終コマンドにしない（PID 1 化で固まる）
```

## 実 GPU で WebGPU が出るための 4 条件（2026-09-11 L4 / driver 580.178.04 で実測）

1. pod env `NVIDIA_DRIVER_CAPABILITIES=all`（`graphics` 無しでは ICD 自体が注入されない）
2. イメージに **libglvnd0 / libgl1 / libglx0** があること。`nvidia/cuda:*` で ICD が
   `Could not get 'vkCreateInstance'` でスキップされたのはこれが無く `libGLX_nvidia.so.0` の dlopen が
   落ちていたため。直接確認済み: `nvidia/cuda:12.8.0-base-ubuntu24.04` に同じ apt 行を足すだけで
   `deviceName = NVIDIA L4`（`2026-09-11-l4-vulkaninfo-cuda-base.log`）。CUDA と Vulkan/Chrome を
   同居させたい（M11 の CUDA 実測）ときはこのベースを使う
3. **xvfb + headed**（`xvfb-run -a` + `headless: false`）で確認済み。Playwright の `headless: true` は
   `chromium-headless-shell` になり、同じ pod でも `google/swiftshader`・maxBufferSize 1 GiB しか出ない
   （`probe-noflag.log` の `headless=new` ラベルはこの headless-shell 実行。`ignoreDefaultArgs` を付けても
   バイナリは headless-shell のまま）。フル Chromium の new headless（`channel: 'chromium'`）は未検証
4. `--enable-dawn-features=vulkan_enable_f16_on_nvidia`。Dawn は NVIDIA+Vulkan で `shader-f16` を
   明示的に塞いでいる（`PhysicalDeviceVk.cpp` の `crbug.com/42251215`、toggle `VulkanEnableF16OnNvidia`）。
   （Chromium 153.0.8010.12 / Dawn main 2026-09 時点。TODO 扱いなので将来外れうる）Vulkan 側は `shaderFloat16 / storageBuffer16BitAccess / VK_KHR_shader_float16_int8` を全て持っているのに
   features に出ず、ggml-webgpu と同じ `requestDevice({requiredFeatures:['shader-f16', …]})` が
   `Unsupported feature: shader-f16` で落ちる。`allow_unsafe_apis` / `WebGPUExperimentalFeatures` /
   `--use-angle=gl` では変わらない（`docs/results/linux-webgpu-gate/2026-09-11-l4-probe-noflag.log`）

`probe_webgpu.mjs` の判定は Mac で両方向を確認済み: `--angle vulkan`（Mac に Vulkan 無し）→ SwiftShader を
検出して `FAIL`、`--angle metal` → `REAL_GPU_OK`。

## 実測値（`docs/results/linux-webgpu-gate/`）

| | NVIDIA L4 (Vulkan/Dawn, Chromium 153) | M4 Max (Metal, 同 Chromium) |
|---|---:|---:|
| adapter | `nvidia` / `lovelace` | `apple` / `metal-3` |
| shader-f16 | ✓（toggle 必須） | ✓ |
| maxBufferSize | 4,294,967,292 | 4,294,967,292 |
| maxStorageBufferBindingSize | **2,147,483,644** | 4,294,967,292 |
| requestDevice（ggml 相当） | ✓ | ✓ |
| 3,332 MB 単一バッファ確保 | ✓ | ✓ |
| f16 compute dispatch | ✓ | ✓ |

`maxStorageBufferBindingSize` が M4 Max の半分（2 GiB）。`ggml-alloc` はグラフバッファを
`maxStorageBufferBindingSize` 以下のチャンクに分割する（`ggml_backend_webgpu_buffer_type_get_max_size`）ので、
browser 実測の per-view graph buffer 2837.8 MB は 2 チャンクに割れる。問題になるのは **単一テンソルが 2 GiB を
超える場合**だけで、現行 browser 経路の最大単一テンソルは 1 GB（`docs/spec/31-webgpu-bringup.md` §10.4）。
`PIXAL3D_WEBGPU_OP_GAP.md` B6 の im2col 2.416 GB は L4 では確保不能なので、この経路が復活したら L4 で落ちる。

## 落とし穴

- `xvfb-run` を bash の最終コマンドにしない（exec されて PID 1 になり、node 終了後に Xvfb の後始末で
  無限に固まる。テンプレートは `timeout 300` と後続 `echo` で回避）
- Mac 上の `docker build` でも `http://archive.ubuntu.com:80` は接続失敗が頻発する。sources を https へ倒す

## フル生成（release gate）を pod で回す — `Dockerfile.e2e`

gate 層に `run_release_gate.mjs` + release manifest + 入力 4 view を焼く。モデル本体は Hugging Face から
`--models-url` で直接、app は公開配信元（`WEB_APP_URL`）から取るので pod 側の準備は不要。
起動は `xvfb-run -a node run_release_gate.mjs /opt/gate/models /opt/gate/views --models-url … --out … --glb …` に
`GATE_CHROMIUM_ARGS="--use-angle=vulkan --enable-features=Vulkan,WebGPU --ignore-gpu-blocklist --enable-dawn-features=vulkan_enable_f16_on_nvidia --no-sandbox --disable-dev-shm-usage"`
（`--use-angle=vulkan` 等を落とすと SwiftShader・1 GiB になる。1 回目はこれで落とした）。

2026-09-11 L4 実測（`docs/results/linux-webgpu-gate/l4-20260911/`）: 取得 437 s、生成 3209 s
（SS 169.6 / Shape-512 179.1 / Shape-1024 1279.0 / Texture 768.2）、GLB 32,050,460 B、再転送 0、削除 3.80 GB。
