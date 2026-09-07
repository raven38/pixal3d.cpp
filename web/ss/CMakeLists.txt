# Browser Pixal3D stage tests + integration/full-model WASM targets.
# Every target executes shared C++ on ggml WebGPU; JavaScript only mounts files and invokes C ABI.
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
target_compile_definitions(ggml-webgpu PRIVATE WEBGPU_RUNTIME_WAIT_TIMEOUT_MS=7200000u)

set(PIXAL3D_WASM_COMMON
  ${REPO}/src/trellis_model.cpp
  ${REPO}/src/dinov3.cpp
  ${REPO}/src/dit.cpp
  ${REPO}/src/flow_runner.cpp
  ${REPO}/src/proj_grid.cpp
  ${REPO}/src/pixal3d_cond.cpp
  ${REPO}/src/pixal3d_cond_gpu.cpp
  ${REPO}/src/naf.cpp
  ${REPO}/src/naf_gpu.cpp
  ${REPO}/src/graph_dump.cpp
  ${REPO}/src/shape_decoder.cpp
  ${REPO}/src/sparse.cpp
  ${REPO}/src/dual_grid.cpp
)

function(pixal3d_wasm_target target source export_name output_dir output_name exported_functions jspi_exports)
  add_executable(${target} ${source} ${PIXAL3D_WASM_COMMON})
  target_include_directories(${target} PRIVATE ${REPO}/include ${GGML_DIR}/include)
  target_link_libraries(${target} PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)
  target_compile_options(${target} PRIVATE "-fwasm-exceptions")
  target_link_options(${target} PRIVATE
    "--use-port=emdawnwebgpu" "--profiling-funcs" "-sJSPI" "-fwasm-exceptions" "-lworkerfs.js"
    "-sFORCE_FILESYSTEM=1" "-sALLOW_MEMORY_GROWTH=1" "-sINITIAL_MEMORY=268435456" "-sMAXIMUM_MEMORY=4294967296"
    "-sEXPORTED_FUNCTIONS=${exported_functions}"
    "-sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString,FS,WORKERFS,HEAPF32,HEAPU8]"
    "-sMODULARIZE=1" "-sEXPORT_NAME=${export_name}" "-sJSPI_EXPORTS=${jspi_exports}" "-sENVIRONMENT=worker")
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${output_dir}" OUTPUT_NAME "${output_name}" SUFFIX ".js")
endfunction()

pixal3d_wasm_target(pixal3d-webgpu-ss-wasm ${REPO}/src/pixal3d_wasm.cpp createPixal3dSS ${CMAKE_CURRENT_SOURCE_DIR} pixal3d_ss
  "[_pixal3d_ss_run,_pixal3d_ss_latent,_pixal3d_ss_latent_size,_pixal3d_ss_steps,_pixal3d_ss_n_steps,_pixal3d_shape512_run,_pixal3d_shape512_latent,_pixal3d_shape512_latent_size,_pixal3d_shape512_steps,_pixal3d_shape512_n_steps,_pixal3d_shape_decode_run,_pixal3d_tex_decode_run,_malloc,_free]"
  "[pixal3d_ss_run,pixal3d_shape512_run,pixal3d_shape_decode_run,pixal3d_tex_decode_run]")

pixal3d_wasm_target(pixal3d-webgpu-texture-wasm ${REPO}/src/pixal3d_texture_wasm.cpp createPixal3dTexture ${REPO}/web/texture pixal3d_texture
  "[_pixal3d_texture_run,_pixal3d_texture_latent,_pixal3d_texture_latent_size,_pixal3d_texture_steps,_pixal3d_texture_n_steps,_malloc,_free]"
  "[pixal3d_texture_run]")

add_executable(pixal3d-webgpu-partial-e2e-wasm ${REPO}/src/pixal3d_partial_e2e_wasm.cpp ${PIXAL3D_WASM_COMMON} ${REPO}/src/mesh_glb.cpp)
target_include_directories(pixal3d-webgpu-partial-e2e-wasm PRIVATE ${REPO}/include ${GGML_DIR}/include ${REPO}/thirdparty/stb)
target_link_libraries(pixal3d-webgpu-partial-e2e-wasm PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)
target_compile_options(pixal3d-webgpu-partial-e2e-wasm PRIVATE "-fwasm-exceptions")
target_link_options(pixal3d-webgpu-partial-e2e-wasm PRIVATE "--use-port=emdawnwebgpu" "--profiling-funcs" "-sJSPI" "-fwasm-exceptions" "-lworkerfs.js" "-sFORCE_FILESYSTEM=1" "-sALLOW_MEMORY_GROWTH=1" "-sINITIAL_MEMORY=268435456" "-sMAXIMUM_MEMORY=4294967296" "-sEXPORTED_FUNCTIONS=[_pixal3d_partial_e2e_run,_malloc,_free]" "-sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString,FS,WORKERFS,HEAPU8]" "-sMODULARIZE=1" "-sEXPORT_NAME=createPixal3dPartialE2E" "-sJSPI_EXPORTS=[pixal3d_partial_e2e_run]" "-sENVIRONMENT=worker")
set_target_properties(pixal3d-webgpu-partial-e2e-wasm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${REPO}/web/partial_e2e" OUTPUT_NAME "pixal3d_partial_e2e" SUFFIX ".js")

# Phase-1 fixture-input full-model orchestration. Live through Shape-1024; Texture condition
# currently uses the explicitly reported fixture fallback until NAF@1024 is chunked for WebGPU.
add_executable(pixal3d-webgpu-full-e2e-wasm ${REPO}/src/pixal3d_full_e2e_wasm.cpp ${PIXAL3D_WASM_COMMON} ${REPO}/src/ss_decoder.cpp ${REPO}/src/mesh_glb.cpp)
target_include_directories(pixal3d-webgpu-full-e2e-wasm PRIVATE ${REPO}/include ${GGML_DIR}/include ${REPO}/thirdparty/stb)
target_link_libraries(pixal3d-webgpu-full-e2e-wasm PRIVATE ggml ggml-base ggml-cpu ggml-webgpu)
target_compile_options(pixal3d-webgpu-full-e2e-wasm PRIVATE "-fwasm-exceptions")
target_link_options(pixal3d-webgpu-full-e2e-wasm PRIVATE "--use-port=emdawnwebgpu" "--profiling-funcs" "-sJSPI" "-fwasm-exceptions" "-lworkerfs.js" "-sFORCE_FILESYSTEM=1" "-sALLOW_MEMORY_GROWTH=1" "-sINITIAL_MEMORY=268435456" "-sMAXIMUM_MEMORY=4294967296" "-sEXPORTED_FUNCTIONS=[_pixal3d_full_fixture_run,_malloc,_free]" "-sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString,FS,WORKERFS,HEAPU8]" "-sMODULARIZE=1" "-sEXPORT_NAME=createPixal3dFullE2E" "-sJSPI_EXPORTS=[pixal3d_full_fixture_run]" "-sENVIRONMENT=worker")
set_target_properties(pixal3d-webgpu-full-e2e-wasm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${REPO}/web/full_e2e" OUTPUT_NAME "pixal3d_full_e2e" SUFFIX ".js")
