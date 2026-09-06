// Diagnostic-only ggml graph dumper for the WebGPU op-gap survey (docs/PIXAL3D_WEBGPU_OP_GAP.md).
// Additive, off by default: a no-op unless env TRELLIS_DUMP_OPS is set. No effect on compute.
#pragma once

struct ggml_cgraph;

namespace trellis {

// Appends one line per node of `g` (op, sub-op, src/dst shapes+dtypes+layout, op params) plus a
// per-graph summary (op histogram, largest tensor bytes) to the file named by env TRELLIS_DUMP_OPS
// (or stderr if the value is empty), tagged with `tag`. No-op if TRELLIS_DUMP_OPS is unset.
void trellis_graph_dump(const char* tag, ggml_cgraph* g);

} // namespace trellis
