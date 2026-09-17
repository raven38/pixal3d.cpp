// 本番 URL で SV セットの取得・検証（9 ファイルを本番 origin 経由で HF から取得）→ 再起動で再転送ゼロ → 削除、までを実 Chrome で通す。
// 生成はしない（生成込みは run_release_gate.mjs --mode sv）。docs/runbooks/web-deploy.md §4。
//   node web/app/smoke_sv_install.mjs <app url> <profile dir> <report.json>
import fs from 'node:fs'; import path from 'node:path'; import { chromium } from 'playwright';
const url = process.argv[2]; const profile = process.argv[3]; const out = process.argv[4];
const hits = []; const report = { url, steps: {}, ok: false };
const launch = () => chromium.launchPersistentContext(profile, { headless: false, args: ['--enable-unsafe-webgpu'] });
const open = async (ctx) => { const p = await ctx.newPage(); p.on('response', (r) => { if (/\.gguf(\?|$)/.test(r.url())) hits.push({ name: r.url().split('?')[0].split('/').pop(), status: r.status() }); }); await p.goto(`${url}?mode=sv`, { waitUntil: 'domcontentloaded' }); return p; };
const status = (p) => p.locator('#model-status').textContent();
let ctx = await launch();
try {
  let page = await open(ctx);
  await page.waitForFunction(() => { const t = document.querySelector('#model-status')?.textContent || ''; return t && !/Checking/i.test(t); }, null, { timeout: 60_000 });
  report.steps.initial = await status(page);
  await page.waitForFunction(() => !document.querySelector('#download-models')?.disabled, null, { timeout: 60_000 });
  const t0 = Date.now();
  await page.locator('#download-models').click();
  await page.waitForFunction(() => /^Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: 90 * 60 * 1000 });
  report.steps.install = { seconds: Math.round((Date.now() - t0) / 1000), gguf_requests: hits.length, statuses: [...new Set(hits.map((h) => h.status))], status: await status(page) };
  console.log('[prod-sv] install', JSON.stringify(report.steps.install));
  const before = hits.length; await ctx.close(); ctx = await launch(); page = await open(ctx);
  await page.waitForFunction(() => /^Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: 120_000 });
  report.steps.restart = { gguf_retransfers: hits.length - before, status: await status(page) };
  console.log('[prod-sv] restart', JSON.stringify(report.steps.restart));
  const u0 = await page.evaluate(() => navigator.storage.estimate()); page.on('dialog', (d) => d.accept());
  await page.locator('#delete-models').click();
  await page.waitForFunction(() => !/^Ready/.test(document.querySelector('#model-status')?.textContent || ''), null, { timeout: 300_000 });
  const u1 = await page.evaluate(() => navigator.storage.estimate());
  report.steps.delete = { usage_before: u0.usage, usage_after: u1.usage };
  console.log('[prod-sv] delete', JSON.stringify(report.steps.delete));
  report.ok = report.steps.restart.gguf_retransfers === 0 && u1.usage < u0.usage && /pixal3d-sv-q8_0 v1 · SV/.test(report.steps.install.status);
} finally { fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n'); console.log('[prod-sv] ok=' + report.ok); try { await ctx.close(); } catch {} }
process.exit(report.ok ? 0 : 1);
