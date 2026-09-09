// Infrastructure smoke test for the ggml WebGPU (Dawn / emdawnwebgpu) backend bring-up,
// native and browser/WASM. Self-contained: no Model/GGUF. Enumerates backend devices, picks
// the WebGPU device, builds one tiny ggml graph (mul_mat -> norm -> gelu/silu -> scale) on
// both WebGPU and CPU with identical deterministic inputs, and compares the results. Also
// exercises a large (64 MiB) buffer alloc/upload/readback and prints
// ggml_backend_supports_op() for every op used, per docs/spec/31-webgpu-bringup.md.
//
// Native:  ./build-webgpu/trellis-webgpu-smoke
// Browser: web/smoke/index.html calls the exported webgpu_smoke_limits()/webgpu_smoke_run()
//          (see scripts/build_wasm_smoke.sh, web/smoke/CMakeLists.txt).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

using T = ggml_tensor;

namespace {

// Accumulates the full report text (returned to JS via webgpu_smoke_run()/webgpu_smoke_limits(),
// and printed to stdout by main() natively) instead of printf'ing directly, so both entry points
// share one code path.
struct Report {
    std::string buf;
    void pf(const char* fmt, ...) {
        char tmp[8192];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(tmp, sizeof(tmp), fmt, ap);
        va_end(ap);
        buf += tmp;
    }
};

const char* dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "?";
}

ggml_context* mkctx(size_t n_tensors, size_t graph_nodes) {
    size_t meta = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(graph_nodes, false) + (1 << 16);
    return ggml_init({meta, nullptr, true});
}

// Deterministic pseudo-random f32 in [-1, 1], same generator used elsewhere in this repo's tests.
float detval(size_t i) { return ((i * 2654435761u) & 1023) / 512.0f - 1.0f; }

// One tiny graph: X[Cin,N] -> mul_mat(W[Cin,Cout]) -> [Cout,N] -> norm(eps) -> act -> scale.
struct Graph {
    ggml_context* ctx = nullptr;
    T* x = nullptr;
    T* w = nullptr;
    T* out = nullptr;
};

Graph build_graph(int Cin, int Cout, int N, ggml_type w_type, bool use_silu, float eps, float scale) {
    Graph g;
    g.ctx = mkctx(16, 16);
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, Cin, N);
    ggml_set_input(g.x);
    ggml_set_name(g.x, "x");
    g.w = ggml_new_tensor_2d(g.ctx, w_type, Cin, Cout);
    ggml_set_input(g.w);
    ggml_set_name(g.w, "w");
    T* mm = ggml_mul_mat(g.ctx, g.w, g.x);           // [Cout, N], always F32 output
    T* nrm = ggml_norm(g.ctx, mm, eps);              // [Cout, N]
    T* act = use_silu ? ggml_silu(g.ctx, nrm) : ggml_gelu(g.ctx, nrm);
    g.out = ggml_scale(g.ctx, act, scale);
    ggml_set_output(g.out);
    ggml_set_name(g.out, "out");
    return g;
}

// Run `g` on `backend`, uploading host x/w, filling `out_result` with the f32 readback of
// g.out. Returns false (and appends a [FAIL] line to R) on allocation/compute failure instead
// of aborting the process, so both the native and WASM entry points can report gracefully.
bool run_graph(ggml_backend_t backend, Graph& g,
                const std::vector<float>& host_x, const std::vector<uint8_t>& host_w_bytes,
                Report& R, std::vector<float>& out_result) {
    ggml_cgraph* cg = ggml_new_graph_custom(g.ctx, 16, false);
    ggml_build_forward_expand(cg, g.out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, cg)) {
        R.pf("  [FAIL] ggml_gallocr_alloc_graph failed on %s\n", ggml_backend_name(backend));
        return false;
    }
    ggml_backend_tensor_set(g.x, host_x.data(), 0, ggml_nbytes(g.x));
    ggml_backend_tensor_set(g.w, host_w_bytes.data(), 0, ggml_nbytes(g.w));
    if (ggml_backend_graph_compute(backend, cg) != GGML_STATUS_SUCCESS) {
        R.pf("  [FAIL] ggml_backend_graph_compute failed on %s\n", ggml_backend_name(backend));
        ggml_gallocr_free(alloc);
        return false;
    }
    out_result.resize((size_t)ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out_result.data(), 0, ggml_nbytes(g.out));
    ggml_gallocr_free(alloc);
    return true;
}

struct Diff { double max_abs, mean_abs, rel; };

Diff compare(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d{0, 0, 0};
    double sum_abs = 0, sum_ref = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double diff = std::fabs((double)a[i] - (double)b[i]);
        d.max_abs = std::max(d.max_abs, diff);
        sum_abs += diff;
        sum_ref += std::fabs((double)a[i]);
    }
    d.mean_abs = n ? sum_abs / n : 0.0;
    d.rel = sum_ref > 1e-12 ? sum_abs / sum_ref : d.mean_abs;
    return d;
}

// Build a minimal single-node op tensor, just to probe ggml_backend_supports_op().
void print_op_support(ggml_backend_t backend, const char* label, T* op, Report& R) {
    bool ok = ggml_backend_supports_op(backend, op);
    R.pf("  supports_op %-10s : %s\n", label, ok ? "yes" : "no");
}

void run_op_support_probe(ggml_backend_t backend, Report& R) {
    ggml_context* c = mkctx(16, 16);
    T* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 32);
    T* w = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 48);
    T* w16 = ggml_new_tensor_2d(c, GGML_TYPE_F16, 64, 48);
    T* mm = ggml_mul_mat(c, w, x);
    T* mm16 = ggml_mul_mat(c, w16, x);
    T* nrm = ggml_norm(c, mm, 1e-5f);
    T* gel = ggml_gelu(c, nrm);
    T* sil = ggml_silu(c, nrm);
    T* scl = ggml_scale(c, gel, 0.5f);
    R.pf("  device: %s\n", ggml_backend_name(backend));
    print_op_support(backend, "mul_mat(f32)", mm, R);
    print_op_support(backend, "mul_mat(f16w)", mm16, R);
    print_op_support(backend, "norm", nrm, R);
    print_op_support(backend, "gelu", gel, R);
    print_op_support(backend, "silu", sil, R);
    print_op_support(backend, "scale", scl, R);
    ggml_free(c);
}

// 64 MiB buffer alloc + upload + readback checksum, to exercise the backend's
// buffer-type get_max_size()/maxStorageBufferBindingSize handling (docs/spec/31 §5).
bool big_buffer_test(ggml_backend_t backend, Report& R) {
    const size_t MB = 1024 * 1024;
    const size_t N_ELEM = (64 * MB) / sizeof(float);   // exactly 64 MiB of f32
    ggml_context* c = mkctx(4, 4);
    T* big = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)N_ELEM);
    ggml_set_input(big);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (!buf) {
        R.pf("  [FAIL] 64 MiB buffer allocation failed on %s\n", ggml_backend_name(backend));
        ggml_free(c);
        return false;
    }
    std::vector<float> host(N_ELEM);
    for (size_t i = 0; i < N_ELEM; ++i) host[i] = detval(i * 7 + 3);
    ggml_backend_tensor_set(big, host.data(), 0, ggml_nbytes(big));
    std::vector<float> back(N_ELEM, 0.0f);
    ggml_backend_tensor_get(big, back.data(), 0, ggml_nbytes(big));
    double sum_host = 0, sum_back = 0, max_diff = 0;
    for (size_t i = 0; i < N_ELEM; ++i) {
        sum_host += host[i];
        sum_back += back[i];
        max_diff = std::max(max_diff, (double)std::fabs(host[i] - back[i]));
    }
    bool ok = max_diff == 0.0;
    R.pf("  64 MiB buffer (%zu f32 elems): alloc OK, upload+readback checksum host=%.6f back=%.6f max_diff=%.3g -> %s\n",
         N_ELEM, sum_host, sum_back, max_diff, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return ok;
}

// Full graph-comparison smoke test, shared by webgpu_smoke_run() (WASM export) and main()
// (native). Returns overall pass/fail; appends a trailing "RESULT: PASS"/"RESULT: FAIL" line.
bool run_smoke_impl(Report& R) {
    R.pf("=== pixal3d-webgpu-smoke ===\n");

    const size_t ndev = ggml_backend_dev_count();
    R.pf("ggml_backend_dev_count() = %zu\n", ndev);
    ggml_backend_dev_t webgpu_dev = nullptr;
    ggml_backend_dev_t cpu_dev = nullptr;
    for (size_t i = 0; i < ndev; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_dev_props pr{};
        ggml_backend_dev_get_props(d, &pr);
        const char* reg_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        R.pf("[%zu] name=%s desc=%s type=%s reg=%s mem_free=%.1fMB mem_total=%.1fMB\n",
             i, pr.name, pr.description, dev_type_str(pr.type), reg_name,
             pr.memory_free / 1e6, pr.memory_total / 1e6);
        if (std::string(reg_name) == "WebGPU" && !webgpu_dev) webgpu_dev = d;
        if (pr.type == GGML_BACKEND_DEVICE_TYPE_CPU && !cpu_dev) cpu_dev = d;
    }

    if (!webgpu_dev) {
        R.pf("[FAIL] no WebGPU device found (adapter/instance creation likely failed)\n");
        R.pf("\nRESULT: FAIL\n");
        return false;
    }
    if (!cpu_dev) {
        R.pf("[FAIL] no CPU device found\n");
        R.pf("\nRESULT: FAIL\n");
        return false;
    }

    R.pf("\n-- init backends --\n");
    ggml_backend_t webgpu = ggml_backend_dev_init(webgpu_dev, nullptr);
    if (!webgpu) {
        R.pf("[FAIL] ggml_backend_dev_init(WebGPU) returned null "
             "(adapter/device request failed, e.g. missing ShaderF16 feature)\n");
        R.pf("\nRESULT: FAIL\n");
        return false;
    }
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        R.pf("[FAIL] ggml_backend_dev_init(CPU) returned null\n");
        ggml_backend_free(webgpu);
        R.pf("\nRESULT: FAIL\n");
        return false;
    }
    R.pf("webgpu backend name: %s\n", ggml_backend_name(webgpu));
    R.pf("cpu    backend name: %s\n", ggml_backend_name(cpu));

    R.pf("\n-- ggml_backend_supports_op probe (WebGPU device) --\n");
    run_op_support_probe(webgpu, R);

    const int Cin = 64, Cout = 48, N = 32;
    const float eps = 1e-5f, scale = 0.5f;
    std::vector<float> host_x((size_t)Cin * N);
    for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = detval(i);
    std::vector<float> host_w_f32((size_t)Cin * Cout);
    for (size_t i = 0; i < host_w_f32.size(); ++i) host_w_f32[i] = detval(i * 31 + 11);
    std::vector<uint8_t> host_w_f32_bytes(host_w_f32.size() * sizeof(float));
    std::memcpy(host_w_f32_bytes.data(), host_w_f32.data(), host_w_f32_bytes.size());
    std::vector<ggml_fp16_t> host_w_f16(host_w_f32.size());
    ggml_fp32_to_fp16_row(host_w_f32.data(), host_w_f16.data(), (int64_t)host_w_f32.size());
    std::vector<uint8_t> host_w_f16_bytes(host_w_f16.size() * sizeof(ggml_fp16_t));
    std::memcpy(host_w_f16_bytes.data(), host_w_f16.data(), host_w_f16_bytes.size());

    bool all_pass = true;
    R.pf("\n-- graph comparison: X[%d,%d] @ W[%d,%d] -> norm(eps=%.0e) -> act -> scale(%.2f) --\n",
         Cin, N, Cin, Cout, eps, scale);

    struct Case { const char* label; ggml_type w_type; bool silu; double tol; const std::vector<uint8_t>* w_bytes; };
    std::vector<Case> cases = {
        // 5e-3 for all cases: the WebGPU backend's F32 mul_mat lands at rel ~1.3e-3 vs ggml-cpu
        // (f16-staged GEMM tiles, same as the CUDA/Metal backends — docs/spec/30 section 5), so
        // this is an infrastructure check, not a precision gate.
        {"f32-weight, gelu", GGML_TYPE_F32, false, 5e-3, &host_w_f32_bytes},
        {"f32-weight, silu", GGML_TYPE_F32, true,  5e-3, &host_w_f32_bytes},
        {"f16-weight, gelu", GGML_TYPE_F16, false, 5e-3, &host_w_f16_bytes},
    };

    for (auto& tc : cases) {
        Graph g_wgpu = build_graph(Cin, Cout, N, tc.w_type, tc.silu, eps, scale);
        std::vector<float> out_wgpu;
        bool ok_wgpu = run_graph(webgpu, g_wgpu, host_x, *tc.w_bytes, R, out_wgpu);
        ggml_free(g_wgpu.ctx);

        Graph g_cpu = build_graph(Cin, Cout, N, tc.w_type, tc.silu, eps, scale);
        std::vector<float> out_cpu;
        bool ok_cpu = run_graph(cpu, g_cpu, host_x, *tc.w_bytes, R, out_cpu);
        ggml_free(g_cpu.ctx);

        if (!ok_wgpu || !ok_cpu) {
            all_pass = false;
            R.pf("  %-16s [FAIL] graph execution error (see above)\n", tc.label);
            continue;
        }

        Diff d = compare(out_wgpu, out_cpu);
        bool pass = d.rel < tc.tol;
        all_pass &= pass;
        R.pf("  %-16s max|d|=%.3e mean|d|=%.3e rel=%.3e (tol=%.0e) -> %s\n",
             tc.label, d.max_abs, d.mean_abs, d.rel, tc.tol, pass ? "PASS" : "FAIL");
    }

    R.pf("\n-- 64 MiB buffer alloc/upload/readback --\n");
    R.pf(" WebGPU:\n");
    all_pass &= big_buffer_test(webgpu, R);
    R.pf(" CPU (control):\n");
    all_pass &= big_buffer_test(cpu, R);

    ggml_backend_free(webgpu);
    ggml_backend_free(cpu);

    R.pf("\nRESULT: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

// Independent adapter/device query for reporting WebGPU limits/features. The ggml WebGPU
// backend keeps its wgpu::Instance/Adapter/Device in a private static context with no public
// accessor (confirmed: ggml-webgpu.h exposes only ggml_backend_webgpu_init/_reg), so this asks
// the wgpu C++ API directly for its own adapter/device, mirroring
// thirdparty/ggml/src/ggml-webgpu/ggml-webgpu.cpp's create_webgpu_device() request pattern
// (docs/spec/31-webgpu-bringup.md §5/§7).
bool run_limits_impl(Report& R) {
    R.pf("=== webgpu_smoke_limits ===\n");
#ifdef __EMSCRIPTEN__
    R.pf("build: Emscripten/WASM (emdawnwebgpu)\n");
#else
    R.pf("build: native (Dawn)\n");
#endif

    // TimedWaitAny is required for instance.WaitAny(..., UINT64_MAX) below to actually block --
    // same instance-feature request ggml-webgpu.cpp's ggml_backend_webgpu_reg() makes (its
    // instance is private, so this queries its own separate one; see comment above).
    wgpu::InstanceDescriptor instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures     = instance_features.data();
    instance_descriptor.requiredFeatureCount = instance_features.size();
    wgpu::Instance instance = wgpu::CreateInstance(&instance_descriptor);
    if (!instance) {
        R.pf("[FAIL] wgpu::CreateInstance failed\n");
        return false;
    }

    wgpu::RequestAdapterOptions options = {};
    wgpu::Adapter adapter;
    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    std::fprintf(stderr, "webgpu_smoke_limits: RequestAdapter failed: %s\n",
                                 std::string(message).c_str());
                    return;
                }
                adapter = std::move(a);
            }),
        UINT64_MAX);
    if (!adapter) {
        R.pf("[FAIL] RequestAdapter failed (no WebGPU adapter available -- check navigator.gpu)\n");
        return false;
    }

    wgpu::Limits alimits{};
    adapter.GetLimits(&alimits);

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);

    bool adapter_shader_f16  = adapter.HasFeature(wgpu::FeatureName::ShaderF16);
    bool adapter_subgroups   = adapter.HasFeature(wgpu::FeatureName::Subgroups);

    R.pf("adapter: vendor=%s architecture=%s device=%s description=%s\n",
         std::string(info.vendor).c_str(), std::string(info.architecture).c_str(),
         std::string(info.device).c_str(), std::string(info.description).c_str());
    R.pf("adapter feature shader-f16: %s\n", adapter_shader_f16 ? "supported" : "NOT supported");
    R.pf("adapter feature subgroups : %s\n", adapter_subgroups ? "supported" : "NOT supported");

    std::vector<wgpu::FeatureName> required_features;
    if (adapter_shader_f16) required_features.push_back(wgpu::FeatureName::ShaderF16);
    if (adapter_subgroups)  required_features.push_back(wgpu::FeatureName::Subgroups);

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &alimits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            std::fprintf(stderr, "webgpu_smoke_limits: device error: %s\n", std::string(message).c_str());
        });

    wgpu::Device device;
    instance.WaitAny(
        adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [&device](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    std::fprintf(stderr, "webgpu_smoke_limits: RequestDevice failed: %s\n",
                                 std::string(message).c_str());
                    return;
                }
                device = std::move(d);
            }),
        UINT64_MAX);
    if (!device) {
        R.pf("[FAIL] RequestDevice failed (adapter may lack the required features)\n");
        return false;
    }

    wgpu::Limits dlimits{};
    device.GetLimits(&dlimits);
    bool device_shader_f16 = device.HasFeature(wgpu::FeatureName::ShaderF16);
    bool device_subgroups  = device.HasFeature(wgpu::FeatureName::Subgroups);

    R.pf("device feature shader-f16 enabled: %s\n", device_shader_f16 ? "yes" : "no");
    R.pf("device feature subgroups  enabled: %s\n", device_subgroups ? "yes" : "no");

    R.pf("maxBufferSize: %llu\n", (unsigned long long)dlimits.maxBufferSize);
    R.pf("maxStorageBufferBindingSize: %llu\n", (unsigned long long)dlimits.maxStorageBufferBindingSize);
    R.pf("maxComputeWorkgroupStorageSize: %u\n", (unsigned)dlimits.maxComputeWorkgroupStorageSize);
    R.pf("maxComputeInvocationsPerWorkgroup: %u\n", (unsigned)dlimits.maxComputeInvocationsPerWorkgroup);
    R.pf("maxComputeWorkgroupSizeX: %u\n", (unsigned)dlimits.maxComputeWorkgroupSizeX);
    R.pf("maxComputeWorkgroupSizeY: %u\n", (unsigned)dlimits.maxComputeWorkgroupSizeY);
    R.pf("maxComputeWorkgroupSizeZ: %u\n", (unsigned)dlimits.maxComputeWorkgroupSizeZ);
    R.pf("maxComputeWorkgroupsPerDimension: %u\n", (unsigned)dlimits.maxComputeWorkgroupsPerDimension);
    R.pf("maxBindGroups: %u\n", (unsigned)dlimits.maxBindGroups);
    R.pf("maxStorageBuffersPerShaderStage: %u\n", (unsigned)dlimits.maxStorageBuffersPerShaderStage);
    R.pf("maxUniformBufferBindingSize: %llu\n", (unsigned long long)dlimits.maxUniformBufferBindingSize);
    return true;
}

std::string g_run_report;
std::string g_limits_report;

}  // namespace

extern "C" {

const char* webgpu_smoke_run(void) {
    Report R;
    run_smoke_impl(R);
    g_run_report = R.buf;
    return g_run_report.c_str();
}

const char* webgpu_smoke_limits(void) {
    Report R;
    if (!run_limits_impl(R)) {
        R.pf("\n[webgpu_smoke_limits FAILED -- see messages above]\n");
    }
    g_limits_report = R.buf;
    return g_limits_report.c_str();
}

}  // extern "C"

// Native entry point only -- the browser build (web/smoke/) drives webgpu_smoke_limits()/
// webgpu_smoke_run() from JS instead, since main() would run automatically at module load
// before the page can set up async/JSPI-aware calls.
#ifndef __EMSCRIPTEN__
int main() {
    std::printf("%s\n", webgpu_smoke_limits());
    const char* report = webgpu_smoke_run();
    std::printf("%s\n", report);
    std::string s(report);
    bool pass = s.find("RESULT: PASS") != std::string::npos;
    return pass ? 0 : 1;
}
#endif
