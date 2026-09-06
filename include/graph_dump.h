// Diagnostic-only ggml graph dumper for the WebGPU op-gap survey (docs/PIXAL3D_WEBGPU_OP_GAP.md).
// Additive, off by default: a no-op unless env TRELLIS_DUMP_OPS is set. No effect on compute.
#pragma once

#include <cstddef>

struct ggml_cgraph;

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

} // namespace trellis
