// Drives web/shape_decode/index.html in a real Chrome (WebGPU + JSPI), saves the report and the
// mesh files the module returned (browser_verts/faces/coords.npy) so the native scorer can
// compare them with the reference:
//   trellis-test-pixal3d-shape-decode <shape_dec.gguf> <fixture_dir> --ext-mesh <out_dir>/browser_
// Usage (from web/, with `python3 -m http.server 8199` serving it; PIXAL3D_WEB_PORT overrides the port):
//   node shape_decode/run_playwright.js <shape_dec.gguf> <fixture_dir> <out_dir> [res]
// Only the fixture files the module reads are handed over (coords / hr_coords, f32_slat /
// f32_shape_slat, f32_tex_coords). Needs `playwright` resolvable by node (channel 'chrome').
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [gguf, fixtureDir, outDir, resArg] = process.argv.slice(2);
  const res = resArg || '0';
  fs.mkdirSync(outDir, { recursive: true });
  const keep = /^(coords|hr_coords|f32_slat|f32_shape_slat|f32_tex_coords)\.npy$/;
  const fixtureFiles = fs.readdirSync(fixtureDir).filter(f => keep.test(f)).map(f => path.join(fixtureDir, f));
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal'],
  });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const consoleLog = [];
  page.on('console', m => consoleLog.push('[' + m.type() + '] ' + m.text()));
  const downloads = [];
  page.on('download', d => downloads.push(d));
  await page.goto('http://localhost:' + (process.env.PIXAL3D_WEB_PORT || '8199') + '/shape_decode/index.html', { waitUntil: 'load' });
  await page.setInputFiles('#model', [gguf]);
  await page.setInputFiles('#fixture', fixtureFiles);
  await page.fill('#res', res);
  const t0 = Date.now();
  await page.click('#run');
  try {
    await page.waitForFunction(() => document.title.startsWith('RESULT:'), null, { timeout: 7200000 });
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
  fs.writeFileSync(path.join(outDir, 'report.txt'), text + '\nplaywright wall ' + wall.toFixed(1) + ' s\n' + title + '\n');
  fs.writeFileSync(path.join(outDir, 'console.txt'), consoleLog.join('\n'));
  // the page starts one download per mesh file right before it sets the title
  const deadline = Date.now() + 120000;
  while (downloads.length < 3 && Date.now() < deadline) await new Promise(r => setTimeout(r, 500));
  for (const d of downloads) {
    const dest = path.join(outDir, d.suggestedFilename());
    await d.saveAs(dest);
    console.log('saved ' + dest + ' (' + fs.statSync(dest).size + ' B)');
  }
  console.log(title + '  (wall ' + wall.toFixed(1) + ' s)');
  await browser.close();
  process.exit(title === 'RESULT: OK' ? 0 : 1);
})();
