// Linux/GPU pod 上の headless Chrome が「実 GPU の WebGPU アダプタ」を返すかを判定する。
// ggml-webgpu と同じ要求（shader-f16 [+subgroups]、adapter の limits をそのまま requiredLimits）で
// requestDevice() まで通し、さらに設計上の単一バッファ（3,332 MB）が確保できるかを見る。
//
//   node probe_webgpu.mjs [--out report.json] [--headless new|xvfb]
import http from 'node:http';
import fs from 'node:fs';
import { chromium } from 'playwright';

const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf(n); return i >= 0 && argv[i + 1] ? argv[i + 1] : d; };
const outPath = flag('--out', 'webgpu_probe.json');
const mode = flag('--headless', 'new');
const angle = flag('--angle', 'vulkan');   // macOS の対照実験では 'metal' / 'default'

const PAGE = `<!doctype html><html><body><pre id="out">PENDING</pre><script>
window.__probe = (async () => {
  const r = { navigator_gpu: !!navigator.gpu };
  if (!navigator.gpu) return r;
  const a = await navigator.gpu.requestAdapter();
  if (!a) { r.adapter = null; return r; }
  const info = a.info || {};
  r.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
                isFallbackAdapter: a.isFallbackAdapter };
  r.features = [...a.features].sort();
  r.limits = {}; for (const k in Object.getPrototypeOf(a.limits)) { const v = a.limits[k]; if (typeof v === 'number') r.limits[k] = v; }
  r.shader_f16 = a.features.has('shader-f16');
  // ggml-webgpu と同じ要求で device を取る
  const req = ['shader-f16']; if (a.features.has('subgroups')) req.push('subgroups');
  const reqLimits = {}; for (const k of ['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup','maxStorageBuffersPerShaderStage','maxBindGroups']) reqLimits[k] = a.limits[k];
  r.device_request = { requiredFeatures: req, requiredLimits: reqLimits };
  try {
    const d = await a.requestDevice({ requiredFeatures: req, requiredLimits: reqLimits });
    r.device_ok = true;
    // 設計上の単一バッファ（docs/PIXAL3D_WEBGPU_MEMORY.md: 3,332 MB）を実確保する
    const bytes = 3332 * 1024 * 1024;
    d.pushErrorScope('out-of-memory'); d.pushErrorScope('validation');
    const buf = d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const v = await d.popErrorScope(); const oom = await d.popErrorScope();
    r.alloc_3332mb = { validation: v ? v.message : null, oom: oom ? oom.message : null, ok: !v && !oom };
    // 実際に GPU が計算していることの最小確認（1 dispatch）
    const mod = d.createShaderModule({ code: 'enable f16; @group(0) @binding(0) var<storage,read_write> o: array<f32>; @compute @workgroup_size(64) fn m(@builtin(global_invocation_id) g: vec3<u32>){ let h: f16 = f16(g.x) * 0.5h; o[g.x] = f32(h); }' });
    const pipe = await d.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'm' } });
    const out = d.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rb = d.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bg = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
    const enc = d.createCommandEncoder(); const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(4); p.end();
    enc.copyBufferToBuffer(out, 0, rb, 0, 256 * 4); d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const f = new Float32Array(rb.getMappedRange());
    r.compute_f16_check = { sample: [f[0], f[1], f[2], f[255]], ok: f[1] === 0.5 && f[255] === 127.5 };
    buf.destroy();
  } catch (e) { r.device_ok = false; r.device_error = String(e); }
  return r;
})();
</script></body></html>`;

const srv = http.createServer((_, res) => { res.writeHead(200, { 'content-type': 'text/html' }); res.end(PAGE); });
await new Promise((r) => srv.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${srv.address().port}/`;

const args = ['--enable-unsafe-webgpu', `--use-angle=${angle}`, ...(angle === 'vulkan' ? ['--enable-features=Vulkan,WebGPU'] : []), '--ignore-gpu-blocklist',
              '--no-sandbox', '--disable-dev-shm-usage', '--enable-logging=stderr', '--v=1',
              ...(process.env.EXTRA_ARGS ? process.env.EXTRA_ARGS.split(' ') : [])];   // 例: --enable-dawn-features=vulkan_enable_f16_on_nvidia
const stderrLines = [];
const browser = await chromium.launch({ headless: mode === 'new', args, chromiumSandbox: false,
  ...(mode === 'new' ? { ignoreDefaultArgs: ['--headless'] } : {}) });
const page = await browser.newPage();
await page.goto(url);
const result = await page.evaluate(() => window.__probe);
result.mode = mode; result.chromium = browser.version(); result.args = args; result.at = new Date().toISOString();
try { result.nvidia_smi = fs.readFileSync('/proc/driver/nvidia/version', 'utf8').trim().split('\n')[0]; } catch {}
await browser.close(); srv.close();
fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
const real = result.adapter && !/swiftshader|llvmpipe|software/i.test(JSON.stringify(result.adapter)) && !result.adapter.isFallbackAdapter;
// gate として使うので、f16 compute の結果まで含めて判定し、exit code にも伝える
// （SwiftShader・device 失敗・確保失敗・f16 の誤値はどれも FAIL / exit 1）。
const ok = Boolean(real && result.device_ok && result.alloc_3332mb?.ok && result.compute_f16_check?.ok);
console.log(`PROBE_VERDICT=${ok ? 'REAL_GPU_OK' : 'FAIL'}`);
process.exitCode = ok ? 0 : 1;
