// OPFS モデルキャッシュの失敗系ゲート。
//
// `test_headless.mjs` は正常系（remote lifecycle → OPFS → SHA → commit marker、delete、
// local fallback）を見ている。ここはその裏側だけを見る:
//
//   1. 同サイズ・別内容の配信 → SHA-256 で拒否し、ready にしない
//   2. 更新の中断（2本目で失敗）→ 混在セットが ready にならない
//   3. role の付け替えだけの manifest → 同一視されず再取得が要る
//   4. 必須ファイルの欠落 → ready にしない
//   5. delete は自 OPFS namespace だけ消す（無関係なディレクトリは残る）
//   6. **再読み込みで GGUF の再転送が 0**（#9 の中心的な受け入れ条件）
//   7. requestDevice() 拒否 → WebGPU の preflight が落ちる
//
// 実 7.54 GiB は `run_release_gate.mjs`（実 Chrome ゲート）の担当。ここは小さな模擬モデル。
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

const APP_URL = process.env.WEB_APP_URL || 'http://127.0.0.1:8199/web/app/';
const MANIFEST_URL = 'https://models.test/manifest.json';
const BASE_URL = 'https://models.test/files';

// release manifest は必須 9 ファイル・role 重複なしを要求する（release_store.js）。
const ROLES = ['image_encoder', 'naf', 'ss_flow', 'ss_decoder', 'shape_flow_512',
  'shape_decoder', 'shape_flow_1024', 'texture_flow_1024', 'texture_decoder'];
const bodies = Object.fromEntries(ROLES.map((r, i) => [`model-${i}.gguf`, `body-of-${r}-${'x'.repeat(8)}`]));

const manifestFor = (bodyMap, { version = 'v1', roles = ROLES } = {}) => ({
  schema_version: 1,
  model_set: 'pixal3d-test',
  version,
  files: Object.keys(bodyMap).map((name, i) => ({
    name,
    role: roles[i],
    required: true,
    size_bytes: Buffer.byteLength(bodyMap[name]),
    sha256: createHash('sha256').update(bodyMap[name]).digest('hex'),
  })),
});

const browser = await chromium.launch({ headless: true });
let failed = false;
const check = (ok, msg) => { console.log(`${ok ? 'ok' : 'FAIL'}: ${msg}`); if (!ok) failed = true; };

// 配信をルート単位で差し替えられるようにする。served には GGUF 本体のリクエストだけ数える。
async function newPage({ deviceFails = false, noF16 = false, manifest = manifestFor(bodies), bodyMap = bodies, failNames = [] } = {}) {
  const ctx = await browser.newContext();
  const served = { gguf: [], manifest: 0 };
  await ctx.route('https://models.test/**', async (route) => {
    const url = route.request().url();
    if (url === MANIFEST_URL) {
      served.manifest++;
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(manifest) });
    }
    const name = decodeURIComponent(url.split('/').pop());
    served.gguf.push(name);
    if (failNames.includes(name)) return route.fulfill({ status: 500, body: 'injected failure' });
    const body = bodyMap[name];
    if (body === undefined) return route.fulfill({ status: 404, body: 'no' });
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body });
  });
  const page = await ctx.newPage();
  await page.addInitScript(`Object.defineProperty(navigator, 'gpu', { configurable: true, value: {
    requestAdapter: async () => ({
      limits: { maxBufferSize: ${4 * 1024 ** 3}, maxStorageBufferBindingSize: ${2 * 1024 ** 3},
                maxComputeWorkgroupsPerDimension: 65535 },
      features: new Set(${noF16 ? '[]' : "['shader-f16']"}),
      info: { vendor: 'mock', architecture: 'mock', device: '', description: '' },
      requestDevice: async (desc) => {
        ${deviceFails ? "throw new DOMException('nope', 'OperationError');" : ''}
        // 実装と同じく、広告していない feature を要求されたら TypeError で拒む
        for (const f of (desc?.requiredFeatures || [])) {
          if (!new Set(${noF16 ? '[]' : "['shader-f16']"}).has(f)) throw new TypeError('Unsupported feature: ' + f);
        }
        // 契約: preflight は runtime と同じく shader-f16 を要求し、2 つの limits を指定すること。
        // これを落とす回帰（requiredFeatures/requiredLimits を空にする）をテストで検出する。
        if (!(desc?.requiredFeatures || []).includes('shader-f16')) throw new Error('CONTRACT: shader-f16 not requested');
        if (!desc?.requiredLimits?.maxBufferSize || !desc?.requiredLimits?.maxStorageBufferBindingSize) throw new Error('CONTRACT: requiredLimits missing');
        return { destroy() {} };
      },
    }),
  } });`);
  page.on('pageerror', (e) => { console.log('[pageerror]', e.message); failed = true; });
  // 直前の実行の bfcache/HTTP キャッシュで古い JS が来ると、無関係な null 参照エラーに化ける。
  await page.goto(APP_URL, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => /WebGPU/.test(document.querySelector('#preflight-status')?.textContent || ''), null, { timeout: 30_000 });
  return { ctx, page, served };
}

const install = (page) => page.evaluate(async ([manifestUrl, modelBaseUrl]) => {
  const { installReleaseModels } = await import('/web/app/release_store.js');
  try {
    const s = await installReleaseModels({ manifestUrl, modelBaseUrl });
    return { ok: true, ready: s.ready, version: s.version, files: s.files };
  } catch (e) {
    return { ok: false, error: String(e?.message || e) };
  }
}, [MANIFEST_URL, BASE_URL]);

const status = (page) => page.evaluate(async () => {
  const { cacheStatus } = await import('/web/app/model_store.js');
  const s = await cacheStatus();
  return { ready: s.ready, version: s.version, files: s.files, missing: s.missing?.length ?? 0 };
});

// ---- 1. 同サイズ・別内容 --------------------------------------------------
{
  const tampered = { ...bodies };
  const first = Object.keys(bodies)[3];
  tampered[first] = bodies[first].replace(/x{8}$/, 'y'.repeat(8)); // 長さは同じ
  if (Buffer.byteLength(tampered[first]) !== Buffer.byteLength(bodies[first])) throw new Error('setup: sizes must match');
  const { ctx, page } = await newPage({ bodyMap: tampered });
  const r = await install(page);
  check(!r.ok && /verification/i.test(r.error), `same-size different content rejected (${r.ok ? 'installed!' : r.error?.slice(0, 80)})`);
  check(!(await status(page)).ready, 'cache not ready after a rejected install');
  await ctx.close();
}

// ---- 2. 更新の中断 --------------------------------------------------------
{
  const { ctx, page, served } = await newPage();
  const ok = await install(page);
  check(ok.ok && ok.ready, 'baseline remote install ready');

  // v2 を配信し、3本目で落とす
  const v2 = manifestFor(bodies, { version: 'v2' });
  await ctx.unroute('https://models.test/**');
  const failName = Object.keys(bodies)[2];
  await ctx.route('https://models.test/**', async (route) => {
    const url = route.request().url();
    if (url === MANIFEST_URL) return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(v2) });
    const name = decodeURIComponent(url.split('/').pop());
    served.gguf.push(name);
    if (name === failName) return route.fulfill({ status: 500, body: 'injected failure' });
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: bodies[name] });
  });
  const r2 = await install(page);
  check(!r2.ok, `interrupted refresh fails loudly (${r2.ok ? 'succeeded!' : r2.error?.slice(0, 60)})`);
  const mid = await status(page);
  check(!mid.ready, `an interrupted refresh does not leave a mixed set ready (ready=${mid.ready}, missing=${mid.missing})`);

  // 3. role 付け替えのみ → 同一視されない（再取得が必要）
  const swapped = manifestFor(bodies, { version: 'v1', roles: [ROLES[1], ROLES[0], ...ROLES.slice(2)] });
  await ctx.unroute('https://models.test/**');
  await ctx.route('https://models.test/**', async (route) => {
    const url = route.request().url();
    if (url === MANIFEST_URL) return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(swapped) });
    const name = decodeURIComponent(url.split('/').pop());
    served.gguf.push(name);
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: bodies[name] });
  });
  const beforeSwap = served.gguf.length;
  const r3 = await install(page);
  check(r3.ok && r3.ready, 'role-reassigned manifest installs');
  check(served.gguf.length > beforeSwap,
        `role reassignment with identical hashes is not treated as the same set (${served.gguf.length - beforeSwap} files re-fetched)`);
  await ctx.close();
}

// ---- 4 + 5 + 6. 欠落・delete の範囲・再読み込みでの再転送 -------------------
{
  const { ctx, page, served } = await newPage();
  const ok = await install(page);
  check(ok.ok && ok.ready, 'install for the reload case ready');
  const afterInstall = served.gguf.length;

  // 6. 同じ context で新しいページ = 再読み込み。OPFS は残る。
  const page2 = await ctx.newPage();
  await page2.goto(APP_URL, { waitUntil: 'domcontentloaded' });
  await page2.waitForFunction(() => /Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: 30_000 });
  check(served.gguf.length === afterInstall,
        `reload re-transferred ${served.gguf.length - afterInstall} GGUF file(s) (expected 0)`);

  // 5. 無関係な OPFS ディレクトリは delete で消えない
  await page2.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const other = await root.getDirectoryHandle('someone-elses-data', { create: true });
    const w = await (await other.getFileHandle('keep.txt', { create: true })).createWritable();
    await w.write('do not delete me');
    await w.close();
  });
  await page2.evaluate(async () => {
    const { deleteCachedModels } = await import('/web/app/release_store.js');
    await deleteCachedModels();
  });
  const survivors = await page2.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const names = [];
    for await (const [name] of root.entries()) names.push(name);
    return names;
  });
  check(survivors.includes('someone-elses-data'), `delete kept unrelated OPFS data (${survivors.join(', ') || 'empty'})`);
  check(!(await status(page2)).ready, 'cache not ready after delete');

  // 4. 必須ファイルを 1 本消すと ready にならない
  const again = await install(page2);
  check(again.ok && again.ready, 're-install after delete ready');
  const removed = await page2.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const d = await root.getDirectoryHandle('pixal3d-models-v1');
    for await (const [name, h] of d.entries()) {
      if (h.kind === 'file' && name.endsWith('.gguf')) { await d.removeEntry(name); return name; }
    }
    return null;
  });
  const partial = await status(page2);
  check(!!removed && !partial.ready, `removing ${removed} makes the set not ready (ready=${partial.ready})`);
  await ctx.close();
}

// ---- 7. requestDevice 拒否 ------------------------------------------------
// #21: adapter と limits が使えても device が取れない GPU がある。preflight は
// ggml-webgpu と同じ要求（shader-f16 + adapter の limits）で使い捨ての device を
// 1 つ取り、失敗なら 7.54 GiB の取得前に落とす。通過してしまったら回帰。
{
  const { ctx, page } = await newPage({ deviceFails: true });
  const text = (await page.locator('#preflight-status').textContent()) || '';
  const gpuOk = /✓ WebGPU/.test(text);
  if (gpuOk) { console.log('FAIL: requestDevice() failure passed the preflight (#21)'); failed = true; }
  else console.log(`ok: requestDevice() failure is rejected by the preflight (${text.split('\n').find((l) => /WebGPU/.test(l))})`);
  await ctx.close();
}

// ---- 8. shader-f16 非対応 --------------------------------------------------
// ggml-webgpu は shader-f16 を必須にしている。adapter が広告していなければ device 要求は
// 失敗するので、preflight もそこで止まるべき。
{
  const { ctx, page } = await newPage({ noF16: true });
  const text = (await page.locator('#preflight-status').textContent()) || '';
  const gpuOk = /✓ WebGPU/.test(text);
  if (gpuOk) { console.log('FAIL: missing shader-f16 passed the preflight (#21)'); failed = true; }
  else console.log(`ok: missing shader-f16 is rejected by the preflight (${text.split('\n').find((l) => /WebGPU/.test(l))})`);
  await ctx.close();
}

// ---- 9. 正常なアダプタは通る（陽性ケース） -------------------------------
// mock の requestDevice は preflight の契約（shader-f16 の要求・limits の指定）を検査して
// 違反なら投げる。ここで ✓ を要求することで「契約を落とす回帰」と「本来動く GPU を弾く
// 誤検知」の両方を検出する。
{
  const { ctx, page } = await newPage();
  const text = (await page.locator('#preflight-status').textContent()) || '';
  const line = text.split('\n').find((l) => /WebGPU/.test(l)) || '';
  if (!/✓ WebGPU/.test(line)) { console.log(`FAIL: healthy adapter rejected by the preflight (${line})`); failed = true; }
  else console.log(`ok: healthy adapter passes the preflight (${line})`);
  await ctx.close();
}

console.log(failed ? 'WEB_CACHE_FAILURE_MODES_FAILED' : 'WEB_CACHE_FAILURE_MODES_OK');
await browser.close();
process.exit(failed ? 1 : 0);
