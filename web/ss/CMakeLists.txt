# Browser Pixal3D SS path (docs/spec/31-webgpu-bringup.md §9): the shared C++ pipeline
# (DINOv3 -> projection -> MV fusion -> ProjectAttention SS flow -> Euler sampler) compiled to
# WASM on top of ggml + its WebGPU backend, driven from a Worker (web/ss/worker.js) through
# the C ABI in src/pixal3d_wasm.cpp. Added by the root CMakeLists.txt only under
# -DTRELLIS_WASM_SS=ON with emcmake (scripts/build_wasm_ss.sh); like web/smoke it vendors
# thirdparty/ggml itself and compiles just the SS-path sources -- the rest of trellis_core
# (mesh postprocess, CUDA/Vulkan kernels, xatlas, ...) isn't needed here.
cmake_minimum_required(VERSION 3.18)
project(pixal3d_webgpu_ss_wasm LANGUAGES CXX C)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "web/ss/CMakeLists.txt is Emscripten-only; configure via scripts/build_wasm_ss.sh")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(GGML_DIR ${REPO}/thirdparty/ggml)
add_compile_definitions(GGML_MAX_NAME=128)
add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml)
# patches/ggml-webgpu/0002: one DiT forward is one graph submission and takes minutes in a
# browser; upstream's 30 s queue-wait ceiling aborts it. (0001 is moot under Emscripten.)
target_compile_definitions(ggml-webgpu PRIVATE WEBGPU_RUNTIME_WAIT_TIMEOUT_MS=600000u)

add_executable(pixal3d-webgpu-ss-wasm
  ${REPO}/src/pixal3d_wasm.cpp
  ${REPO}/src/trellis_model.cpp
  ${REPO}/src/dinov3.cpp
  ${REPO}/src/dit.cpp
  ${REPO}/src/flow_runner.cpp
  ${REPO}/src/proj_grid.cpp
  ${REPO}/src/pixal3d_cond.cpp
  ${REPO}/src/pixal3d_cond_gpu.cpp
  ${REPO}/src/naf.cpp
  ${REPO}/src/graph_dump.cpp
)
target_include_directories(pixal3d-webgpu-ss-wasm PRIVATE ${REPO}/include ${GGML_DIR}/include)
target_link_libraries(pixal3d-webgpu-ss-wasm PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)
target_compile_options(pixal3d-webgpu-ss-wasm PRIVATE "-fwasm-exceptions")

# WORKERFS mounts the GGUFs/fixtures the page hands over as File objects (read lazily through
# FileReaderSync, never copied into the wasm heap), so Model::load's fseeko/fread path works
# unchanged. JSPI lets pixal3d_ss_run suspend across the backend's async WebGPU waits.
target_link_options(pixal3d-webgpu-ss-wasm PRIVATE
  "--use-port=emdawnwebgpu"
  "-sJSPI"
  "-fwasm-exceptions"
  "-lworkerfs.js"
  "-sFORCE_FILESYSTEM=1"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sINITIAL_MEMORY=268435456"
  "-sMAXIMUM_MEMORY=4294967296"
  "-sEXPORTED_FUNCTIONS=[_pixal3d_ss_run,_pixal3d_ss_latent,_pixal3d_ss_latent_size,_pixal3d_ss_steps,_pixal3d_ss_n_steps,_malloc,_free]"
  "-sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString,FS,WORKERFS,HEAPF32]"
  "-sMODULARIZE=1"
  "-sEXPORT_NAME=createPixal3dSS"
  "-sJSPI_EXPORTS=[pixal3d_ss_run]"
  "-sENVIRONMENT=worker"
)

set_target_properties(pixal3d-webgpu-ss-wasm PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_NAME "pixal3d_ss"
  SUFFIX ".js"
)
