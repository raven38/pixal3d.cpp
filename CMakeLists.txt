cmake_minimum_required(VERSION 3.18)
project(trellis_cpp VERSION 0.6.0 LANGUAGES CXX C)

# Browser WebGPU smoke test (docs/spec/31-webgpu-bringup.md §8): scripts/build_wasm_smoke.sh
# configures this whole repo with -DTRELLIS_WASM_SMOKE=ON under Emscripten, but only needs
# ggml + src/test_webgpu_smoke.cpp -- trellis_core/the CLI aren't buildable under Emscripten
# yet and pulling them in isn't this test's job. web/smoke/CMakeLists.txt is a small standalone
# project that vendors thirdparty/ggml itself; skip the rest of this file in this mode.
# Local patches to the vendored ggml (patches/ggml-webgpu/*.patch, see
# docs/GGML_FORK_DIFF.md "Local patches"). Applied to the submodule working tree at configure
# time, idempotently: a patch that already applies in reverse is skipped. The submodule pin is
# unchanged; `git -C thirdparty/ggml checkout .` reverts them.
option(PIXAL3D_GGML_PATCHES "Apply patches/ggml-webgpu/*.patch to thirdparty/ggml at configure time" ON)
if(PIXAL3D_GGML_PATCHES AND GGML_WEBGPU)
  file(GLOB _pixal3d_ggml_patches ${CMAKE_SOURCE_DIR}/patches/ggml-webgpu/*.patch)
  foreach(_p ${_pixal3d_ggml_patches})
    execute_process(COMMAND git apply --check --reverse ${_p} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/thirdparty/ggml
                    RESULT_VARIABLE _rev OUTPUT_QUIET ERROR_QUIET)
    if(_rev EQUAL 0)
      message(STATUS "ggml patch already applied: ${_p}")
    else()
      execute_process(COMMAND git apply ${_p} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/thirdparty/ggml RESULT_VARIABLE _fwd)
      if(NOT _fwd EQUAL 0)
        message(FATAL_ERROR "failed to apply ggml patch ${_p} (set -DPIXAL3D_GGML_PATCHES=OFF to skip)")
      endif()
      message(STATUS "ggml patch applied: ${_p}")
    endif()
  endforeach()
endif()
option(TRELLIS_WASM_SMOKE "Build only the browser WebGPU smoke test (Emscripten only)" OFF)
if(EMSCRIPTEN AND TRELLIS_WASM_SMOKE)
  add_subdirectory(web/smoke)
  return()
endif()
# Browser Pixal3D SS path (docs/spec/31-webgpu-bringup.md §9): scripts/build_wasm_ss.sh.
# web/ss/CMakeLists.txt compiles only the SS-path sources of trellis_core on top of ggml.
option(TRELLIS_WASM_SS "Build only the browser Pixal3D SS path (Emscripten only)" OFF)
if(EMSCRIPTEN AND TRELLIS_WASM_SS)
  add_subdirectory(web/ss)
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Git QUIET)
set(TRELLIS_GIT_COMMIT "")
set(TRELLIS_GIT_DIRTY 0)
set(TRELLIS_APP_VERSION "v${PROJECT_VERSION}")
string(TIMESTAMP TRELLIS_BUILD_TIMESTAMP_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE TRELLIS_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TRELLIS_GIT_COMMIT_RESULT)
  if(TRELLIS_GIT_COMMIT_RESULT EQUAL 0)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE TRELLIS_GIT_STATUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(NOT "${TRELLIS_GIT_STATUS}" STREQUAL "")
      set(TRELLIS_GIT_DIRTY 1)
    endif()
  else()
    set(TRELLIS_GIT_COMMIT "")
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" tag --sort=-v:refname
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE TRELLIS_GIT_TAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT "${TRELLIS_GIT_TAGS}" STREQUAL "")
    string(REPLACE "\n" ";" TRELLIS_GIT_TAG_LIST "${TRELLIS_GIT_TAGS}")
    list(GET TRELLIS_GIT_TAG_LIST 0 TRELLIS_LATEST_TAG)
    if(NOT "${TRELLIS_LATEST_TAG}" STREQUAL "")
      set(TRELLIS_APP_VERSION "${TRELLIS_LATEST_TAG}")
    endif()
  endif()
endif()

if(NOT TRELLIS_APP_VERSION MATCHES "^v")
  set(TRELLIS_APP_VERSION "v${TRELLIS_APP_VERSION}")
endif()

set(TRELLIS_BUILD_BACKEND "cpu")

# --- vendored ggml submodule, built in-tree --------------------------------
# Backend is selected at configure time with the upstream flags
# (-DGGML_CUDA=ON / -DGGML_VULKAN=ON); the matching ggml-<backend> target then
# exists and we link it. DiT/sparse tensor names exceed the default 64 chars,
# so raise GGML_MAX_NAME for both ggml and our own sources (the header guards
# it with #ifndef, so a compile definition is enough — no submodule patch).
set(GGML_DIR ${CMAKE_SOURCE_DIR}/thirdparty/ggml)
add_compile_definitions(GGML_MAX_NAME=128)
# Colocate ggml's shared libs with our executables so they're found via $ORIGIN
# both in the build tree and once packaged.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(thirdparty/ggml)
set(GGML_RPATH "$ORIGIN")
# ggml-webgpu's subgroup-matrix mul_mat/flash_attn shaders accumulate in f16 and overflow on
# DINOv3's attention scores (NaN) -- docs/spec/31-webgpu-bringup.md §9. Pixal3D builds take the
# f32-accumulating reg_tile path unless this is switched on for an A/B measurement.
option(PIXAL3D_WEBGPU_SUBGROUP_MATRIX "Let ggml-webgpu use the f16-accumulating subgroup-matrix shaders" OFF)
if(TARGET ggml-webgpu AND NOT PIXAL3D_WEBGPU_SUBGROUP_MATRIX)
  target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_SUBGROUP_MATRIX=0)
endif()
# One DiT forward is a single graph submission; upstream's 30 s queue-wait ceiling aborts it on
# slow adapters (patches/ggml-webgpu/0002). 10 minutes.
if(TARGET ggml-webgpu)
  target_compile_definitions(ggml-webgpu PRIVATE WEBGPU_RUNTIME_WAIT_TIMEOUT_MS=600000u)
endif()
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# --- CUDA language (for the custom deformable-conv kernel) ------------------
if(TARGET ggml-cuda)
  set(TRELLIS_BUILD_BACKEND "cuda")
  if(EXISTS /usr/local/cuda-13.3/bin/nvcc AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.3/bin/nvcc)
  endif()
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 120)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)   # match ggml's shared cudart (avoid 2 runtimes -> invalid device ordinal)
  find_package(CUDAToolkit)
endif()

# --- core library ----------------------------------------------------------
add_library(trellis_core STATIC
  src/trellis_model.cpp
  src/dit.cpp
  src/flow_runner.cpp
  src/ss_decoder.cpp
  src/mesh_glb.cpp
  src/dinov3.cpp
  src/sparse.cpp
  src/shape_decoder.cpp
  src/dual_grid.cpp
  src/preprocess.cpp
  src/uv_bake.cpp
  src/birefnet.cpp
  thirdparty/xatlas/xatlas.cpp
  thirdparty/meshoptimizer/simplifier.cpp
  src/tri_bvh.cpp
  src/remesh_dc.cpp
  src/decimate_qem.cpp
  src/proj_grid.cpp
  src/pixal3d_cond.cpp
  src/pixal3d_cond_gpu.cpp
  src/naf.cpp
  src/naf_gpu.cpp
  src/transforms_json.cpp
  src/pixal3d_input.cpp
  src/image_preprocess.cpp
  src/model_manifest.cpp
  src/pixal3d_postprocess.cpp
  src/graph_dump.cpp
)
target_include_directories(trellis_core PUBLIC include ${GGML_DIR}/include ${CMAKE_SOURCE_DIR}/thirdparty/stb ${CMAKE_SOURCE_DIR}/thirdparty/xatlas ${CMAKE_SOURCE_DIR}/thirdparty/meshoptimizer ${CMAKE_SOURCE_DIR}/thirdparty/fqms)
target_compile_definitions(trellis_core PRIVATE TRELLIS_GIT_COMMIT="${TRELLIS_GIT_COMMIT}" TRELLIS_GIT_DIRTY=${TRELLIS_GIT_DIRTY} TRELLIS_BUILD_TIMESTAMP="${TRELLIS_BUILD_TIMESTAMP_UTC}" TRELLIS_APP_VERSION="${TRELLIS_APP_VERSION}")
target_link_libraries(trellis_core PUBLIC ggml ggml-base ggml-cpu)
# flow_t_schedule()/the guidance-interval decision (src/flow_runner.cpp) must reproduce numpy
# float64 linspace + rescale bit-for-bit: a fused multiply-add collapses the two roundings numpy's
# separate multiply and add ufuncs perform into one, which is enough to flip an exact-decimal
# guidance_interval boundary (e.g. t==0.6) the wrong way. Clang defaults to -ffp-contract=on and
# GCC to -ffp-contract=fast; disable it for this one file (the sampler loop is not perf-sensitive).
# MSVC doesn't recognize this GCC/Clang flag (harmless D9002 warning only) -- guard it.
if(NOT MSVC)
  set_source_files_properties(src/flow_runner.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# --- WebP textures (reference-parity GLB export) ---------------------------
# Optional lossy-WebP textures via EXT_texture_webp. The exporter only advertises
# the extension when WebP encoding is actually used for the embedded textures.
option(TRELLIS_WEBP "Encode GLB textures as WebP (EXT_texture_webp)" ON)
if(TRELLIS_WEBP)
  include(FetchContent)
  set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_Declare(libwebp
    URL https://github.com/webmproject/libwebp/archive/refs/tags/v1.5.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(libwebp)
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})
  target_compile_definitions(trellis_core PRIVATE TRELLIS_HAVE_WEBP)
  target_link_libraries(trellis_core PUBLIC webp)
  message(STATUS "trellis.cpp: WebP texture encoding enabled")
endif()
if(TARGET ggml-cuda)
  target_compile_definitions(trellis_core PUBLIC TRELLIS_USE_CUDA)
  target_link_libraries(trellis_core PUBLIC ggml-cuda)
  target_sources(trellis_core PRIVATE src/deform_conv.cu)
  # Self-contained CUDA QEM decimation (mirrors the CPU port; uses cub for scan/sort/RLE).
  target_sources(trellis_core PRIVATE src/decimate_qem.cu)
  # NAF (valeoai/NAF feature upsampler) CUDA path: the ggml-graph encoder in
  # naf_gpu.cpp + a custom neighborhood-attention kernel (naf_attn.cu).
  # naf_upsample() in naf.cpp dispatches to it only when compiled here
  # (TRELLIS_USE_CUDA, set below) and the loaded Model is on a CUDA backend; other
  # GPU backends take naf_gpu.cpp's all-ggml graph path (naf_upsample_ggml).
  target_sources(trellis_core PRIVATE src/naf_attn.cu)
  if(MSVC)
    # CUDA 13.3 CCCL requires MSVC's conforming preprocessor for host compilation.
    target_compile_options(trellis_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
  endif()
  target_compile_definitions(trellis_core PUBLIC TRELLIS_HAVE_GPU_DECIMATE)
  set_target_properties(trellis_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(trellis_core PUBLIC CUDA::cudart)
  message(STATUS "trellis.cpp: CUDA backend enabled (${GGML_CUDA_LIB})")
else()
  # No CUDA kernel in this build. Run BiRefNet's deformable conv on the GPU via a
  # Vulkan compute shader when ggml is using Vulkan, or a HIP kernel when ggml is
  # using HIP (both are minutes on the CPU at the larger decoder scales); otherwise
  # on the CPU, std::thread-parallel over output channels (deform_conv_cpu.cpp) --
  # no OpenMP/libomp dependency.
  # HIP compiles deform_conv.cu below (issue #20); mirror the CUDA branch and skip
  # the CPU source there so deform_conv2d_run has a single definition.
  if(NOT TARGET ggml-hip)
    target_sources(trellis_core PRIVATE src/deform_conv_cpu.cpp)
  endif()
  if(TARGET ggml-vulkan)
    set(TRELLIS_BUILD_BACKEND "vulkan")
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(DEFORM_SPV_H ${CMAKE_CURRENT_BINARY_DIR}/deform_conv_comp.h)
    add_custom_command(
      OUTPUT ${DEFORM_SPV_H}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O -mfmt=c --target-env=vulkan1.2
              -o ${DEFORM_SPV_H} ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_conv.comp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_conv.comp
      COMMENT "glslc deform_conv.comp -> SPIR-V (C array)")
    add_custom_target(trellis_deform_spv DEPENDS ${DEFORM_SPV_H})
    target_sources(trellis_core PRIVATE src/deform_conv_vk.cpp)
    add_dependencies(trellis_core trellis_deform_spv)
    target_include_directories(trellis_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(trellis_core PUBLIC Vulkan::Vulkan)
    set_source_files_properties(src/deform_conv_cpu.cpp PROPERTIES
      COMPILE_DEFINITIONS TRELLIS_DEFORM_VULKAN)
    set_source_files_properties(src/deform_conv_vk.cpp PROPERTIES
      COMPILE_DEFINITIONS "TRELLIS_DEFORM_SPV_HEADER=\"deform_conv_comp.h\"")
    # Self-contained Vulkan QEM decimation: same four per-round compute kernels as the CUDA/HIP
    # port, driven from the host (adjacency/edges/compaction). The shader needs 64-bit atomics
    # (GL_EXT_shader_atomic_int64), so compile against vulkan1.2 like deform_conv.
    set(DECIMATE_SPV_H ${CMAKE_CURRENT_BINARY_DIR}/decimate_qem_comp.h)
    add_custom_command(
      OUTPUT ${DECIMATE_SPV_H}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O -mfmt=c --target-env=vulkan1.2
              -o ${DECIMATE_SPV_H} ${CMAKE_CURRENT_SOURCE_DIR}/src/decimate_qem.comp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/decimate_qem.comp
      COMMENT "glslc decimate_qem.comp -> SPIR-V (C array)")
    add_custom_target(trellis_decimate_spv DEPENDS ${DECIMATE_SPV_H})
    target_sources(trellis_core PRIVATE src/decimate_qem_vk.cpp)
    add_dependencies(trellis_core trellis_decimate_spv)
    # Target-level so the dispatch block in decimate_qem.cpp is active too (mirrors how the CUDA
    # path defines TRELLIS_HAVE_GPU_DECIMATE for the whole library).
    target_compile_definitions(trellis_core PRIVATE TRELLIS_HAVE_VK_DECIMATE)
    set_source_files_properties(src/decimate_qem_vk.cpp PROPERTIES
      COMPILE_DEFINITIONS "TRELLIS_DECIMATE_SPV_HEADER=\"decimate_qem_comp.h\"")
  endif()
endif()
if(TARGET ggml-vulkan)
  set(TRELLIS_BUILD_BACKEND "vulkan")
  target_link_libraries(trellis_core PUBLIC ggml-vulkan)
  message(STATUS "trellis.cpp: Vulkan backend enabled")
endif()
if(TARGET ggml-metal)
  # macOS/Apple Silicon: ggml's Metal backend (GGML_METAL defaults ON for
  # Apple builds). The generic device-enumeration path in make_backend() picks
  # the Metal GPU automatically; the two custom kernels (deform_conv,
  # decimate_qem) run their CPU fallbacks on this backend.
  set(TRELLIS_BUILD_BACKEND "metal")
  target_link_libraries(trellis_core PUBLIC ggml-metal)
  message(STATUS "trellis.cpp: Metal backend enabled")
endif()
if(TARGET ggml-hip)
  set(TRELLIS_BUILD_BACKEND "hip")
  target_link_libraries(trellis_core PUBLIC ggml-hip)
  # Compile the self-contained QEM decimation kernel as HIP (ggml enabled the HIP
  # language via its subdirectory). hipcub headers live under the ROCm include root.
  enable_language(HIP)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH})
    else()
      set(ROCM_PATH /opt/rocm)
    endif()
  endif()
  # Enabling HIP makes every target that links trellis_core (which now carries a HIP object)
  # require HIP_ARCHITECTURES. Set the default globally so all subsequently-defined targets
  # (the executables below) and trellis_core itself inherit the configured GPU target(s).
  # The GPU decimation kernel must be built for the same arch(es) as the rest of
  # the backend. Prefer the user-provided GPU_TARGETS / AMDGPU_TARGETS (exactly
  # what ggml itself compiles for) over an auto-detected CMAKE_HIP_ARCHITECTURES:
  # on some toolchains CMAKE_HIP_ARCHITECTURES is pre-set to a value that omits the
  # user's GPU, so the old `if(NOT CMAKE_HIP_ARCHITECTURES)` guard skipped the
  # GPU_TARGETS the user passed and built the kernel for the wrong arch — hence the
  # runtime "device kernel image is invalid" launch error (issue #14), after which
  # decimation silently fell back to the slow CPU path.
  if(DEFINED GPU_TARGETS AND NOT "${GPU_TARGETS}" STREQUAL "")
    set(TRELLIS_HIP_ARCH "${GPU_TARGETS}")
  elseif(DEFINED AMDGPU_TARGETS AND NOT "${AMDGPU_TARGETS}" STREQUAL "")
    set(TRELLIS_HIP_ARCH "${AMDGPU_TARGETS}")
  elseif(CMAKE_HIP_ARCHITECTURES)
    set(TRELLIS_HIP_ARCH "${CMAKE_HIP_ARCHITECTURES}")
  else()
    set(TRELLIS_HIP_ARCH "gfx1151")
  endif()
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
  endif()
  # Both the QEM decimation kernel and BiRefNet's deformable conv (issue #20) build
  # as HIP so ROCm users get GPU BiRefNet instead of the minutes-long CPU fallback.
  set_source_files_properties(src/decimate_qem.cu src/deform_conv.cu PROPERTIES LANGUAGE HIP)
  target_sources(trellis_core PRIVATE src/decimate_qem.cu src/deform_conv.cu)
  target_compile_definitions(trellis_core PUBLIC TRELLIS_HAVE_GPU_DECIMATE)
  target_include_directories(trellis_core PRIVATE ${ROCM_PATH}/include)
  # Force the resolved arch(es) onto the HIP objects (and everything linking
  # trellis_core), overriding any auto-detected default.
  set_target_properties(trellis_core PROPERTIES HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
  set_source_files_properties(src/decimate_qem.cu src/deform_conv.cu PROPERTIES HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
  message(STATUS "trellis.cpp: HIP/ROCm backend enabled (GPU decimation + BiRefNet deform via HIP, arch ${TRELLIS_HIP_ARCH})")
endif()

target_compile_definitions(trellis_core PRIVATE TRELLIS_BUILD_BACKEND="${TRELLIS_BUILD_BACKEND}")

# --- executables -----------------------------------------------------------
add_executable(trellis-smoke src/smoke_test.cpp)
target_link_libraries(trellis-smoke PRIVATE trellis_core)
set_target_properties(trellis-smoke PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-decode-replay src/decode_replay.cpp)
target_link_libraries(trellis-decode-replay PRIVATE trellis_core)
set_target_properties(trellis-decode-replay PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Debug-only allocation trace of the sparse shape decoder (docs/PIXAL3D_WEBGPU_MEMORY.md):
# runs shape_decode on a coords/latent pair with TRELLIS_DBG_ALLOC_TRACE set.
add_executable(trellis-shape-dec-alloc-trace src/shape_dec_alloc_trace.cpp)
target_link_libraries(trellis-shape-dec-alloc-trace PRIVATE trellis_core)
set_target_properties(trellis-shape-dec-alloc-trace PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-transforms-synth src/test_transforms_synth.cpp)
target_link_libraries(trellis-test-transforms-synth PRIVATE trellis_core)
set_target_properties(trellis-test-transforms-synth PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# TASK-PORT P5: list_view_images() coverage gaps not already covered by trellis-test-transforms-synth.
add_executable(trellis-test-list-view-images src/test_list_view_images.cpp)
target_link_libraries(trellis-test-list-view-images PRIVATE trellis_core)
set_target_properties(trellis-test-list-view-images PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# TASK-PORT P4: SS MV stage tests (weight-driven, needs ss_dec.gguf; see the file's header comment
# for why the 4 helper functions it needs are file-local static rather than a new shared module).
add_executable(trellis-test-trellis2-mv-ss src/test_trellis2_mv_ss.cpp)
target_link_libraries(trellis-test-trellis2-mv-ss PRIVATE trellis_core)
set_target_properties(trellis-test-trellis2-mv-ss PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-flow src/test_ss_flow.cpp)
target_link_libraries(trellis-test-ss-flow PRIVATE trellis_core)
set_target_properties(trellis-test-ss-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-shape-flow src/test_shape_flow.cpp)
target_link_libraries(trellis-test-shape-flow PRIVATE trellis_core)
set_target_properties(trellis-test-shape-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Manual CUDA regression for the >2 Gi half2 mask-stride overflow in
# high-token-count 1024-cascade FlashAttention. The default case needs
# about 8.1 GiB of VRAM, so keep it out of the routine test suite.
if(TARGET ggml-cuda)
  add_executable(trellis-test-fa-mask-overflow src/test_fa_mask_overflow.cpp)
  target_link_libraries(trellis-test-fa-mask-overflow PRIVATE trellis_core)
  set_target_properties(trellis-test-fa-mask-overflow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

  # Small CUDA regression for trellis's V-range protection around the tensor-core
  # FlashAttention kernels. The optional `raw` mode demonstrates the backend bug.
  add_executable(trellis-test-fa-bf16-range src/test_fa_bf16_range.cpp)
  target_link_libraries(trellis-test-fa-bf16-range PRIVATE trellis_core)
  set_target_properties(trellis-test-fa-bf16-range PROPERTIES BUILD_RPATH "${GGML_RPATH}")
endif()

add_executable(trellis-test-ss-sample src/test_ss_sample.cpp)
target_link_libraries(trellis-test-ss-sample PRIVATE trellis_core)
set_target_properties(trellis-test-ss-sample PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-birefnet src/test_birefnet.cpp)
target_link_libraries(trellis-test-birefnet PRIVATE trellis_core)
set_target_properties(trellis-test-birefnet PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-preprocess src/test_preprocess.cpp)
target_link_libraries(trellis-test-preprocess PRIVATE trellis_core)
set_target_properties(trellis-test-preprocess PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# SV 入力経路（gauge カメラ + pre-matted RGBA クロップ）の契約テスト。外部ファイル不要。
add_executable(trellis-test-sv-input src/test_sv_input.cpp)
target_link_libraries(trellis-test-sv-input PRIVATE trellis_core)
set_target_properties(trellis-test-sv-input PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-deform src/test_deform.cpp)
target_link_libraries(trellis-test-deform PRIVATE trellis_core)
set_target_properties(trellis-test-deform PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-dec src/test_ss_dec.cpp)
target_link_libraries(trellis-test-ss-dec PRIVATE trellis_core)
set_target_properties(trellis-test-ss-dec PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-slat-shape src/test_slat_shape.cpp)
target_link_libraries(trellis-test-slat-shape PRIVATE trellis_core)
set_target_properties(trellis-test-slat-shape PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-trellis2-mv-sampler src/test_trellis2_mv_sampler.cpp)
target_link_libraries(trellis-test-trellis2-mv-sampler PRIVATE trellis_core)
set_target_properties(trellis-test-trellis2-mv-sampler PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# TASK-PORT P3: synthetic sampler oracle (weightless, CPU-only), ported from the old track's
# src/test_flow_multi.cpp. Kept alongside (not replacing) trellis-test-trellis2-mv-sampler above.
add_executable(trellis-test-flow-multi src/test_flow_multi.cpp)
target_link_libraries(trellis-test-flow-multi PRIVATE trellis_core)
set_target_properties(trellis-test-flow-multi PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-dinov3 src/test_dinov3.cpp)
target_link_libraries(trellis-test-dinov3 PRIVATE trellis_core)
set_target_properties(trellis-test-dinov3 PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-sparse-conv src/test_sparse_conv.cpp)
target_link_libraries(trellis-test-sparse-conv PRIVATE trellis_core)
set_target_properties(trellis-test-sparse-conv PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-bigop src/test_bigop.cpp)
target_link_libraries(trellis-test-bigop PRIVATE trellis_core)
set_target_properties(trellis-test-bigop PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-c2s src/test_c2s.cpp)
target_link_libraries(trellis-test-c2s PRIVATE trellis_core)
set_target_properties(trellis-test-c2s PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-sparse-degenerate src/test_sparse_degenerate.cpp)
target_link_libraries(trellis-test-sparse-degenerate PRIVATE trellis_core)
set_target_properties(trellis-test-sparse-degenerate PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-shape-dec src/test_shape_dec.cpp)
target_link_libraries(trellis-test-shape-dec PRIVATE trellis_core)
set_target_properties(trellis-test-shape-dec PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# decimate_qem のベンチ / A-B ハーネス（issue #29）: torus か --dump-post を読んで時間と V/F/開放辺/成分を印字
add_executable(trellis-test-decimate-bench src/test_decimate_bench.cpp)
target_link_libraries(trellis-test-decimate-bench PRIVATE trellis_core)
set_target_properties(trellis-test-decimate-bench PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# CPU decimate_qem の回帰テスト: 変更前実装（エッジ順固定）との bit 一致と幾何の健全性
add_executable(trellis-test-decimate-cpu src/test_decimate_cpu.cpp)
target_link_libraries(trellis-test-decimate-cpu PRIVATE trellis_core)
set_target_properties(trellis-test-decimate-cpu PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-cli src/trellis_cli_main.cpp src/trellis_cli.cpp src/trellis_args.cpp)
target_link_libraries(trellis-cli PRIVATE trellis_core)
set_target_properties(trellis-cli PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Resident HTTP server (keeps the process warm; wraps trellis_run).
find_package(Threads REQUIRED)
add_executable(post-replay src/post_replay.cpp)
target_link_libraries(post-replay PRIVATE trellis_core Threads::Threads)
set_target_properties(post-replay PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-server src/trellis-server.cpp src/trellis_cli.cpp src/trellis_args.cpp)
target_include_directories(trellis-server PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/cpp-httplib)
target_link_libraries(trellis-server PRIVATE trellis_core Threads::Threads)
set_target_properties(trellis-server PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-full src/test_ss_full.cpp)
target_link_libraries(trellis-test-ss-full PRIVATE trellis_core)
set_target_properties(trellis-test-ss-full PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-ss-flow src/test_pixal3d_ss_flow.cpp)
target_link_libraries(trellis-test-pixal3d-ss-flow PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-ss-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-slat-flow src/test_pixal3d_slat_flow.cpp)
target_link_libraries(trellis-test-pixal3d-slat-flow PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-slat-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-proj-grid src/test_proj_grid.cpp)
target_link_libraries(trellis-test-proj-grid PRIVATE trellis_core)
set_target_properties(trellis-test-proj-grid PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-rope-layout src/test_rope_layout.cpp)
target_link_libraries(trellis-test-rope-layout PRIVATE trellis_core)
set_target_properties(trellis-test-rope-layout PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-dit-bench src/test_dit_bench.cpp)
target_link_libraries(trellis-test-dit-bench PRIVATE trellis_core)
set_target_properties(trellis-test-dit-bench PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-cond-ss src/test_pixal3d_cond_ss.cpp)
target_link_libraries(trellis-test-pixal3d-cond-ss PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-cond-ss PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-naf src/test_naf.cpp)
target_link_libraries(trellis-test-naf PRIVATE trellis_core)
set_target_properties(trellis-test-naf PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-cond-slat src/test_pixal3d_cond_slat.cpp)
target_link_libraries(trellis-test-pixal3d-cond-slat PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-cond-slat PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-real-e2e src/pixal3d_real_geometry_wasm.cpp)
target_link_libraries(trellis-test-pixal3d-real-e2e PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-real-e2e PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-cond-tex src/test_pixal3d_cond_tex.cpp)
target_link_libraries(trellis-test-pixal3d-cond-tex PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-cond-tex PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Native build of the browser SS entry point (src/pixal3d_wasm.cpp) -- same C ABI and stages
# as web/ss, runnable on any backend for debugging the browser path outside a browser.
add_executable(pixal3d-ss-run src/pixal3d_wasm.cpp)
target_link_libraries(pixal3d-ss-run PRIVATE trellis_core)
set_target_properties(pixal3d-ss-run PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Native build of the browser texture entry point (src/pixal3d_texture_wasm.cpp) -- lets the
# same Texture Flow / Shape-512 stages run outside a browser, and dumps per-step latents
# (last CLI arg) so browser-vs-native parity can be measured with the same fixture.
add_executable(pixal3d-texture-run src/pixal3d_texture_wasm.cpp)
target_link_libraries(pixal3d-texture-run PRIVATE trellis_core)
# partial 経路（Texture Flow -> decode -> postprocess -> GLB）を native で回すドライバ。
# 重みの量子化や分割の A/B をブラウザ抜きで完結させるために使う。
add_executable(pixal3d-partial-run src/pixal3d_partial_e2e_wasm.cpp)
target_link_libraries(pixal3d-partial-run PRIVATE trellis_core)
set_target_properties(pixal3d-partial-run PROPERTIES BUILD_RPATH "${GGML_RPATH}")
set_target_properties(pixal3d-texture-run PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-ss-sample src/test_pixal3d_ss_sample.cpp)
target_link_libraries(trellis-test-pixal3d-ss-sample PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-ss-sample PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-slat-sample src/test_pixal3d_slat_sample.cpp)
target_link_libraries(trellis-test-pixal3d-slat-sample PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-slat-sample PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-shape-decode src/test_pixal3d_shape_decode.cpp)
target_link_libraries(trellis-test-pixal3d-shape-decode PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-shape-decode PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d-tex-decode src/test_pixal3d_tex_decode.cpp)
target_link_libraries(trellis-test-pixal3d-tex-decode PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d-tex-decode PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Native WebGPU (Dawn) backend infrastructure smoke test (docs/spec/31-webgpu-bringup.md
# §7). Self-contained (no Model/GGUF) -- only depends on ggml/ggml-backend/ggml-webgpu,
# so it links the ggml libs directly rather than the whole trellis_core.
if(GGML_WEBGPU)
  add_executable(trellis-webgpu-smoke src/test_webgpu_smoke.cpp)
  target_include_directories(trellis-webgpu-smoke PRIVATE ${GGML_DIR}/include)
  target_link_libraries(trellis-webgpu-smoke PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)
  set_target_properties(trellis-webgpu-smoke PROPERTIES BUILD_RPATH "${GGML_RPATH}")
  if(NOT EMSCRIPTEN)
    # test_webgpu_smoke.cpp's webgpu_smoke_limits() calls the wgpu C++ API directly (the
    # ggml-webgpu backend keeps its own wgpu::Adapter/Device private, no public accessor) --
    # re-resolve the same Dawn package ggml-webgpu/CMakeLists.txt already found via
    # CMAKE_PREFIX_PATH so this target gets <webgpu/webgpu_cpp.h> too.
    find_package(Dawn REQUIRED)
    target_link_libraries(trellis-webgpu-smoke PRIVATE dawn::webgpu_dawn)
  endif()
  # Op regression for patches/ggml-webgpu/0003 (2D dispatch of row-parallel ops) and the batched
  # mul_mat shapes of the NAF attention graph: WebGPU vs ggml-cpu on identical inputs.
  add_executable(trellis-webgpu-ops src/test_webgpu_ops.cpp)
  target_include_directories(trellis-webgpu-ops PRIVATE ${GGML_DIR}/include)
  target_link_libraries(trellis-webgpu-ops PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)
  set_target_properties(trellis-webgpu-ops PROPERTIES BUILD_RPATH "${GGML_RPATH}")
endif()
