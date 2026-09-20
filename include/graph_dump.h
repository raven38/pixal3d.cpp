// Diagnostic-only ggml graph dumper for the WebGPU op-gap survey (docs/PIXAL3D_WEBGPU_OP_GAP.md).
// Additive, off by default: a no-op unless env TRELLIS_DUMP_OPS is set. No effect on compute.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

namespace trellis {

// Appends one line per node of `g` (op, sub-op, src/dst shapes+dtypes+layout, op params) plus a
// per-graph summary (op histogram, largest tensor bytes) to the file named by env TRELLIS_DUMP_OPS
// (or stderr if the value is empty), tagged with `tag`. No-op if TRELLIS_DUMP_OPS is unset.
void trellis_graph_dump(const char* tag, ggml_cgraph* g);

// Allocation trace (docs/PIXAL3D_WEBGPU_MEMORY.md): for the graph-allocated tensors of `g` --
// inputs and non-view nodes, i.e. what ggml_gallocr places in its buffer; weights and views are
// excluded -- prints to stderr the op, shape, dtype and bytes of every tensor at or above
// TRELLIS_DBG_ALLOC_TRACE_MIN_MB (default 64; TRELLIS_DBG_ALLOC_TRACE=all prints every one), the
// ten largest, and a summary: largest single allocation, sum of all allocations, and a
// simultaneously-live estimate (interval sweep: a tensor is live from its producing node to its
// last consuming node, inputs and the output for the whole graph; gallocr's in-place reuse is
// not modelled, so this is an upper bound on gallocr's buffer, which the caller passes in as
// `gallocr_bytes` for calibration). No-op unless env TRELLIS_DBG_ALLOC_TRACE is set.
void trellis_graph_alloc_trace(const char* tag, ggml_cgraph* g, size_t gallocr_bytes);

// ---------------------------------------------------------------------------
// op-level profiling primitive (--profile / diagnostic binaries; docs/design/2026-09-20-conditioning-profiler.md §4)
//
// ggml exposes no per-op GPU timestamps, so the portable way to attribute a graph's time to its
// ops is to re-submit the already-allocated, already-executed graph one node at a time and wall-
// clock each submission (ggml_backend_graph_compute is synchronous). The numbers are ISOLATED:
// every submission carries a fixed submit/wait cost and no cross-node fusion / concurrency, so
// light ops are overstated relative to GEMMs. Report them next to the whole-graph time.
//
// Contract (the caller must guarantee all of it -- the function cannot check any):
//   * g has been executed and its outputs have already been read to the host; after this call
//     the tensors in g's gallocr buffer no longer hold the production values.
//   * `reupload` re-sets EVERY ggml_set_input tensor of g. It is called at the start of each
//     pass: gallocr reuses input buffers inside the graph, so after the whole run the inputs are
//     generally overwritten and a node-by-node replay from node 0 would read garbage otherwise.
//   * g must not write into persistent tensors outside its gallocr buffer (e.g. ggml_cpy into an
//     accumulator that later graphs read): the replay would apply that side effect again. Never
//     call this on such a graph -- the production paths in pixal3d_cond_gpu.cpp are excluded for
//     exactly this reason; only diagnostic invocations of self-contained graphs qualify.
//   * passes >= 1. The first pass compiles the non-fused pipeline variants (Metal) and is
//     discarded unless passes == 1; the last pass is returned. Each pass replays nodes in graph
//     order so every node sees the same buffer state as in the whole run.
// Returns node_s[j] = seconds of node j in the last pass (size ggml_graph_n_nodes(g)).
std::vector<double> trellis_graph_node_times(ggml_backend_t backend, ggml_cgraph* g, int passes,
                                             const std::function<void()>& reupload);

// op kind label used by the op tables: ggml_op_name, with MUL_MAT split by its src0 type
// ("MUL_MAT(q8_0)" weight GEMMs vs "MUL_MAT(f32)" activation GEMMs) and FLASH_ATTN_EXT by its
// K/V type.
std::string trellis_op_kind(const ggml_tensor* t);

// Prints the op-kind table (seconds, share of the node-pass total, count, ms/op; ops under 0.5 %
// omitted) and the `top_nodes` slowest nodes (op, shape, src0, name) of `node_s` to stdout,
// prefixed with `label`. `whole_s` is the whole-graph time of the same graph; the header prints
// node-sum / whole so the isolation overhead is visible next to every table.
void trellis_print_op_table(ggml_cgraph* g, const std::vector<double>& node_s, double whole_s,
                            const char* label, int top_nodes = 15);

} // namespace trellis
