# Standalone Emscripten project for the browser WebGPU smoke test
# (docs/spec/31-webgpu-bringup.md §8). Added as a subdirectory by the root CMakeLists.txt only
# when configured with -DTRELLIS_WASM_SMOKE=ON under emcmake; deliberately does not depend on
# trellis_core or any other repo target -- only ggml (+ its WebGPU backend) and
# src/test_webgpu_smoke.cpp, so it never needs the rest of the pipeline to build under
# Emscripten.
cmake_minimum_required(VERSION 3.18)
project(pixal3d_webgpu_smoke_wasm LANGUAGES CXX C)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "web/smoke/CMakeLists.txt is Emscripten-only; configure via scripts/build_wasm_smoke.sh")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/ggml)
# DiT/sparse tensor names in the rest of the repo exceed the default 64 chars; irrelevant to
# this smoke test's own tiny graph, but keeps ggml's compile definitions consistent with the
# native build in case anyone later reuses this same ggml build dir.
add_compile_definitions(GGML_MAX_NAME=128)
add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml)

add_executable(pixal3d-webgpu-smoke-wasm ${CMAKE_CURRENT_SOURCE_DIR}/../../src/test_webgpu_smoke.cpp)
target_include_directories(pixal3d-webgpu-smoke-wasm PRIVATE ${GGML_DIR}/include)
target_link_libraries(pixal3d-webgpu-smoke-wasm PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)

# ggml-webgpu/CMakeLists.txt only applies --use-port=emdawnwebgpu (which is what makes
# <webgpu/webgpu_cpp.h> resolvable) as a PRIVATE compile option on its own .cpp, plus an
# INTERFACE *link* option -- our own test_webgpu_smoke.cpp translation unit (which also
# includes that header directly, for webgpu_smoke_limits()) needs the same compile option
# itself, matching the same -fwasm-exceptions JSPI compile flag ggml-webgpu.cpp gets.
target_compile_options(pixal3d-webgpu-smoke-wasm PRIVATE "--use-port=emdawnwebgpu" "-fwasm-exceptions")

# JSPI needs the exported functions to suspend the wasm call stack across the WebGPU backend's
# async RequestAdapter/RequestDevice/buffer-map waits, so they must be callable as async from
# JS (ccall(..., {async: true})) -- that's -sJSPI_EXPORTS (emcc 6.0.9; ASYNCIFY_EXPORTS is the
# deprecated pre-JSPI name for the same setting). Neither function is `main`, and specifying
# EXPORTED_FUNCTIONS without `_main` already tells emcc this is a reactor (library) build with
# no entry point required, so no separate --no-entry flag is needed.
target_link_options(pixal3d-webgpu-smoke-wasm PRIVATE
  "--use-port=emdawnwebgpu"
  "-sJSPI"
  "-fwasm-exceptions"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sEXPORTED_FUNCTIONS=[_webgpu_smoke_run,_webgpu_smoke_limits,_malloc,_free]"
  "-sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString]"
  "-sMODULARIZE=1"
  "-sEXPORT_NAME=createPixal3dSmoke"
  "-sJSPI_EXPORTS=[webgpu_smoke_run,webgpu_smoke_limits]"
)

set_target_properties(pixal3d-webgpu-smoke-wasm PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_NAME "pixal3d_smoke"
  SUFFIX ".js"
)
