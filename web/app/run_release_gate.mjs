// Web alpha の実機リリースゲート（#37）。
//
// 実 Chrome + 実 WebGPU + 実モデルセット（7.54 GiB）で、初回取得 → OPFS 検証 →
// 1024 フル生成 → GLB 取得 → **ブラウザ再起動後にモデル本文の再転送ゼロ** →
// キャッシュ削除で使用量が減ることまでを 1 本で通し、記録用 JSON を書く。
//
//   node web/app/run_release_gate.mjs <models_dir> <views_dir> [--seed 1] [--out report.json]
//
// 前提: リポジトリを http://127.0.0.1:8199 で serve していること（app 本体の配信元）。
// モデル配信元はこのスクリプトが別ポートで立て、GGUF 本体へのリクエストを数える。
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';

const argv = process.argv.slice(2);
const modelsDir = argv[0];
const viewsDir = argv[1];
const seed = Number(flag('--seed', '1'));
const outPath = flag('--out', 'web_release_gate.json');
const appBase = process.env.WEB_APP_URL || 'http://127.0.0.1:8199/web/app/';
const genTimeoutMs = Number(flag('--gen-timeout-min', '120')) * 60 * 1000;

function flag(name, dflt) {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
}
if (!modelsDir || !viewsDir) throw new Error('usage: run_release_gate.mjs <models_dir> <views_dir> [--seed N] [--out FILE]');

const report = { started_at: new Date().toISOString(), steps: {}, ok: false };
const step = (name, value) => { report.steps[name] = value; console.log(`[gate] ${name}: ${JSON.stringify(value)}`); };

// ---- モデル配信元（GGUF 本体のリクエストを数える） -------------------------
const manifestPath = path.join(modelsDir, 'pixal3d-models.json');
if (!fs.existsSync(manifestPath)) throw new Error(`need ${manifestPath} (copy the release manifest next to the GGUFs)`);
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const hits = [];
// --models-url を渡すと、ローカルの配信元を立てずに実ホスト（例: Hugging Face）から取得する。
// その場合 GGUF のリクエスト数はブラウザの response イベントで数える。
const remoteModels = flag('--models-url', '');
const origin = http.createServer((req, res) => {
  const name = decodeURIComponent(req.url.replace(/^\//, '').split('?')[0]);
  hits.push({ name, at: Date.now(), range: req.headers.range || null });
  const file = path.join(modelsDir, path.basename(name));
  if (!fs.existsSync(file)) { res.writeHead(404).end('no'); return; }
  const size = fs.statSync(file).size;
  res.writeHead(200, {
    'content-type': name.endsWith('.json') ? 'application/json' : 'application/octet-stream',
    'content-length': String(size),
    'access-control-allow-origin': '*',
    'accept-ranges': 'bytes',
  });
  fs.createReadStream(file).pipe(res);
});
if (!remoteModels) await new Promise((r) => origin.listen(0, '127.0.0.1', r));
const modelsBase = remoteModels || `http://127.0.0.1:${origin.address().port}`;
const ggufHits = () => hits.filter((h) => h.name.endsWith('.gguf'));

// ---- 実 Chrome。プロファイルは使い回して「再起動」を作る -------------------
// プロファイルは固定できるようにする（--profile）。7.54 GiB の再取得を避けて
// 生成側だけを何度も見たいときに使う。
const fixedProfile = flag('--profile', '');
const userDataDir = fixedProfile || fs.mkdtempSync(path.join(os.tmpdir(), 'pixal3d-gate-'));
fs.mkdirSync(userDataDir, { recursive: true });
// 使い捨てプロファイルには OPFS の 7.5 GiB がそのまま入る。放置するとディスクを
// 詰まらせ、次の実行が「原因不明の停滞」に見える（実際に一度踏んだ）ので必ず消す。
const cleanupProfile = () => {
  if (fixedProfile) return;
  try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch {}
};
// Linux/NVIDIA では Dawn の shader-f16 が toggle 無しで出ないので、追加フラグを環境変数で受ける
// （docker/linux-webgpu-gate/README.md 条件 4）。macOS では未設定のままでよい。
const extraArgs = process.env.GATE_CHROMIUM_ARGS ? process.env.GATE_CHROMIUM_ARGS.split(' ').filter(Boolean) : [];
const launch = () => chromium.launchPersistentContext(userDataDir, {
  headless: false,
  args: ['--enable-unsafe-webgpu', ...extraArgs],
  acceptDownloads: true,
});

let ctx = null;
const shutdown = async () => {
  try { if (ctx) await ctx.close(); } catch {}
  ctx = null;
  try { if (!remoteModels) origin.close(); } catch {}
  cleanupProfile();
};
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, async () => { await shutdown(); process.exit(130); });

const viewFiles = fs.readdirSync(viewsDir, { withFileTypes: true })
  .flatMap((e) => (e.isDirectory()
    ? fs.readdirSync(path.join(viewsDir, e.name)).map((n) => path.join(viewsDir, e.name, n))
    : [path.join(viewsDir, e.name)]))
  .filter((f) => /\.(png|jpe?g|json)$/i.test(f));

async function openApp(context) {
  const page = await context.newPage();
  page.on('console', (m) => {
    const t = m.text();
    // 停滞の切り分けに必要なので、既定で全行出す（--quiet で絞る）。
    if (argv.includes('--quiet')
      ? /\[preflight\]|\[install\]|\[cache\]|\[storage\]|ERROR|Aborted/.test(t)
      : true) console.log('[browser]', t.slice(0, 500));
  });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  if (remoteModels) page.on('response', (r) => {
    const u = r.url();
    if (u.startsWith(remoteModels) || /\.gguf(\?|$)/.test(u)) {
      hits.push({ name: decodeURIComponent(u.split('?')[0].split('/').pop()), at: Date.now(), range: null, status: r.status() });
    }
  });
  // main の app は ?model_base_url= と #model-base-url を使う（release_store.js:73）。
  await page.goto(`${appBase}?model_base_url=${encodeURIComponent(modelsBase)}`, { waitUntil: 'domcontentloaded' });
  return page;
}

// main の app は状態を #model-status（セット同定と Ready/Not ready）と
// #preflight-status（storage/WebGPU の判定内容）に出す。
const stateText = (page) => page.locator('#model-status').textContent();
const readyNow = (page) => page.evaluate(() => /^Ready/.test(document.querySelector('#model-status')?.textContent || ''));

try {
  ctx = await launch();
  report.browser = ctx.browser()?.version() || 'persistent-context';
  let page = await openApp(ctx);

  // ---- 1. 初回: preflight → 取得 → OPFS 検証 -------------------------------
  await page.waitForFunction(() => {
    const t = document.querySelector('#model-status')?.textContent || '';
    return t && !/Checking/i.test(t);
  }, null, { timeout: 60_000 });
  step('initial_state', await stateText(page));
  step('preflight', (await page.locator('#preflight-status').textContent())?.split('\n').slice(0, 6));
  report.adapter = await page.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter?.();
    return a ? { info: a.info ? { ...a.info } : null, maxBufferSize: Number(a.limits.maxBufferSize), maxStorageBufferBindingSize: Number(a.limits.maxStorageBufferBindingSize) } : null;
  });
  step('adapter', report.adapter);

  const alreadyReady = await readyNow(page);
  const t0 = Date.now();
  if (alreadyReady) {
    // 固定プロファイルでの再実行。7.54 GiB の再取得はしない。
    step('install', { skipped: 'already cached in this profile', gguf_requests: ggufHits().length, status: await page.locator('#model-status').textContent() });
  } else {
    await page.fill('#model-base-url', modelsBase);
    await page.locator('#download-models').click();
    await page.waitForFunction(() => /^Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: genTimeoutMs });
    step('install', {
      seconds: Math.round((Date.now() - t0) / 1000),
      gguf_requests: ggufHits().length,
      status: await page.locator('#model-status').textContent(),
    });
  }
  report.storage_after_install = await page.evaluate(() => navigator.storage.estimate());
  step('storage_after_install', report.storage_after_install);

  // ---- 2. 1024 フル生成 → GLB ---------------------------------------------
  await page.locator('input[accept="image/*,.json,application/json"]').setInputFiles(viewFiles);
  await page.waitForFunction((n) => document.querySelectorAll('.cal-view').length === n, viewFiles.filter((f) => !f.endsWith('.json')).length, { timeout: 60_000 });
  await page.fill('#seed', String(seed));
  await page.waitForFunction(() => !document.querySelector('#run')?.disabled, null, { timeout: 60_000 });
  const g0 = Date.now();
  await page.locator('#run').click();
  let shownLog = 0;
  const tick = setInterval(async () => {
    try {
      const t = (await page.locator('#log').textContent()) || '';
      if (t.length > shownLog) {
        process.stdout.write(t.slice(shownLog).replace(/\n?$/, '\n'));
        shownLog = t.length;
      } else {
        console.log(`[gate] ${Math.round((Date.now() - g0) / 1000)}s elapsed, no new log line (a flow stage prints only when it ends)`);
      }
    } catch {}
  }, 60_000);
  await page.waitForFunction(() => ['complete', 'failed'].includes(document.querySelector('#progress-text')?.textContent?.split(' · ')[0]),
                             null, { timeout: genTimeoutMs });
  clearInterval(tick);
  const genState = (await page.locator('#progress-text').textContent())?.split(' · ')[0];
  step('generate', { state: genState, seconds: Math.round((Date.now() - g0) / 1000) });
  if (genState !== 'complete') throw new Error('generation did not complete');

  const dlPromise = page.waitForEvent('download', { timeout: 120_000 });
  await page.locator('#download').click();
  const dl = await dlPromise;
  const glbPath = path.resolve(flag('--glb', 'web_release_gate.glb'));
  await dl.saveAs(glbPath);
  step('glb', { path: glbPath, bytes: fs.statSync(glbPath).size, filename: dl.suggestedFilename() });

  // ---- 3. ブラウザ再起動 → 再転送ゼロ --------------------------------------
  const beforeRestart = ggufHits().length;
  await ctx.close();
  ctx = await launch();
  page = await openApp(ctx);
  await page.waitForFunction(() => /^Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: 120_000 });
  const afterRestart = ggufHits().length;
  step('restart', {
    ready: true,
    gguf_requests_before: beforeRestart,
    gguf_requests_after: afterRestart,
    gguf_retransfers: afterRestart - beforeRestart,
    requests_after_restart: hits.slice(beforeRestart).map((h) => h.name),
    status: await page.locator('#model-status').textContent(),
  });
  if (afterRestart !== beforeRestart) throw new Error('model bodies were re-transferred after restart');

  // 再起動後のキャッシュから実際に生成が始まることを確認する（完走まではやらない:
  // 1回の完走で 20〜40 分かかるため、ここでは「キャッシュからモデルを読んで
  // 最初のライブ段に入る」ところまでを証拠にする）。
  await page.locator('input[accept="image/*,.json,application/json"]').setInputFiles(viewFiles);
  await page.waitForFunction(() => !document.querySelector('#run')?.disabled, null, { timeout: 60_000 });
  await page.locator('#run').click();
  await page.waitForFunction(() => /ss|cond|flow|stage|proj/i.test(document.querySelector('#log')?.textContent || ''), null, { timeout: 15 * 60 * 1000 });
  const retransfersTotal = ggufHits().length - beforeRestart;
  step('restart_generation_started', {
    gguf_retransfers_total: retransfersTotal,
    log_head: (await page.locator('#log').textContent())?.split('\n').filter((l) => /stage|flow|WebGPU/i.test(l)).slice(0, 3),
  });
  // Ready 直後だけでなく、生成開始までの経路でも GGUF を取り直していないこと。
  // ここを記録だけにすると「Ready → Generate でモデルを再取得する回帰」が gate を通ってしまう。
  if (retransfersTotal !== 0) throw new Error(`GGUF re-transferred after restart: ${retransfersTotal} request(s)`);
  await page.locator('#cancel').click();

  // ---- 4. 削除で使用量が減る ------------------------------------------------
  const before = await page.evaluate(() => navigator.storage.estimate());
  page.on('dialog', (d) => d.accept());
  await page.locator('#delete-models').click();
  await page.waitForFunction(() => !/^Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: 300_000 });
  const after = await page.evaluate(() => navigator.storage.estimate());
  step('delete', { usage_before: before.usage, usage_after: after.usage, freed: before.usage - after.usage, state: await stateText(page) });
  if (!(after.usage < before.usage)) throw new Error('cache delete did not reduce OPFS usage');

  report.model_set = { model_set: manifest.model_set, version: manifest.version };
  // shasum/sha256sum の有無に依存しない（コンテナ内に shasum が無く ok=false になった実績）
  report.manifest_sha256 = createHash('sha256').update(fs.readFileSync(manifestPath)).digest('hex');
  // gate 側の commit と、実際にテストした app の配信元は別物なので分けて記録する
  // （PR branch の Docker で main 由来の公開 app を叩く構成があり得る）。コンテナ内には
  // git ツリーが無いので gate の SHA は GATE_COMMIT で明示的に渡す。無ければ手元の git から
  // 取るが、どちらも無い場合は report.ok を立てない。
  report.gate_commit = process.env.GATE_COMMIT || execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  report.app_url = appBase;
  report.ok = true;
} finally {
  report.finished_at = new Date().toISOString();
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + '\n');
  console.log(`[gate] wrote ${outPath} (ok=${report.ok})`);
  await shutdown();
}
process.exit(report.ok ? 0 : 1);
