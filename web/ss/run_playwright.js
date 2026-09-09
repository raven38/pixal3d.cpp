// Drives web/ss/index.html in a real Chrome (WebGPU + JSPI) and saves the report and the
// latents the module returned. Usage (from web/ss/, with `python3 -m http.server 8199` serving it):
//   node run_playwright.js <dinov3.gguf> <pixal3d_ss_flow_mv.gguf> <cond_ss_dir> <ss_sample_dir> <out_dir> [views]
// Files are handed to the page's <input type=file> elements as real File objects, so the
// browser reads them straight from disk (no fetch/copy). Needs `playwright` resolvable by node
// (npx playwright install chrome is not required: channel 'chrome' uses the installed Chrome).
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [dinov3, ssFlow, condDir, sampleDir, outDir, viewsArg] = process.argv.slice(2);
  const views = viewsArg || '0';
  fs.mkdirSync(outDir, { recursive: true });
  const listNpy = (d) => fs.readdirSync(d).filter(f => f.endsWith('.npy')).map(f => path.join(d, f));
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const consoleLog = [];
  page.on('console', m => consoleLog.push('[' + m.type() + '] ' + m.text()));
  await page.goto('http://localhost:8199/index.html', { waitUntil: 'load' });
  await page.setInputFiles('#models', [dinov3, ssFlow]);
  await page.setInputFiles('#cond', listNpy(condDir));
  await page.setInputFiles('#sample', listNpy(sampleDir));
  await page.fill('#views', views);
  const t0 = Date.now();
  await page.click('#run');
  try {
    await page.waitForFunction(() => document.title.startsWith('RESULT:'), null, { timeout: 3600000 });
  } catch (e) {
    console.error('timed out waiting for RESULT; partial page output follows');
    console.error(await page.$eval('#out', el => el.textContent));
    fs.writeFileSync(path.join(outDir, 'console.txt'), consoleLog.join('\n'));
    await browser.close();
    process.exit(2);
  }
  const wall = (Date.now() - t0) / 1000;
  const title = await page.title();
  const text = await page.$eval('#out', el => el.textContent);
  const res = await page.evaluate(() => {
    const r = window.pixal3dResult;
    return r ? { latent: Array.from(r.latent), steps: Array.from(r.steps), nSteps: r.nSteps } : null;
  });
  fs.writeFileSync(path.join(outDir, 'report.txt'), text + '\nplaywright wall ' + wall.toFixed(1) + ' s\n' + title + '\n');
  fs.writeFileSync(path.join(outDir, 'console.txt'), consoleLog.join('\n'));
  if (res) {
    // Minimal .npy writer (float32, C order) so the native decoder/IoU tool can read them.
    const npy = (data, shape) => {
      const hdr0 = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shape.join(', ') + (shape.length === 1 ? ',' : '') + "), }";
      const pad = (64 - ((10 + hdr0.length + 1) % 64)) % 64;
      const hdr = hdr0 + ' '.repeat(pad) + '\n';
      const buf = Buffer.alloc(10 + hdr.length + data.length * 4);
      buf.write('\x93NUMPY', 0, 'latin1'); buf[6] = 1; buf[7] = 0; buf.writeUInt16LE(hdr.length, 8);
      buf.write(hdr, 10, 'latin1');
      Buffer.from(new Float32Array(data).buffer).copy(buf, 10 + hdr.length);
      return buf;
    };
    fs.writeFileSync(path.join(outDir, 'browser_x_final.npy'), npy(res.latent, [1, 8, 16, 16, 16]));
    const n = res.latent.length;
    for (let k = 0; k < res.nSteps; ++k)
      fs.writeFileSync(path.join(outDir, 'browser_x_step' + (k + 1) + '.npy'), npy(res.steps.slice(k * n, (k + 1) * n), [1, 8, 16, 16, 16]));
  }
  console.log(title + '  (wall ' + wall.toFixed(1) + ' s)');
  await browser.close();
  process.exit(title === 'RESULT: OK' ? 0 : 1);
})();
