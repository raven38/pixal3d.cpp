// Drives web/ops/index.html in a real Chrome (WebGPU + JSPI) and saves the report.
// Usage (with `python3 -m http.server <port>` serving web/ops/):
//   node run_playwright.js <out_dir> [args] [port]
// e.g. node run_playwright.js /tmp/ops "--budget-mb 1400 --only set_rows" 8213
// Needs `playwright` resolvable by node (channel 'chrome' uses the installed Chrome).
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [outDir, argsArg, portArg] = process.argv.slice(2);
  const args = argsArg || '--budget-mb 1400';
  const port = portArg || '8213';
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const consoleLog = [];
  page.on('console', m => consoleLog.push('[' + m.type() + '] ' + m.text()));
  const url = 'http://localhost:' + port + '/index.html?args=' + encodeURIComponent(args);
  const t0 = Date.now();
  await page.goto(url, { waitUntil: 'load' });
  let title;
  try {
    await page.waitForFunction(() => document.title.startsWith('RESULT:'), null, { timeout: 3600000 });
    title = await page.title();
  } catch (e) {
    title = 'RESULT: TIMEOUT';
  }
  const wall = (Date.now() - t0) / 1000;
  const text = await page.$eval('#out', el => el.textContent);
  fs.writeFileSync(path.join(outDir, 'report.txt'), text + '\nplaywright wall ' + wall.toFixed(1) + ' s\n' + title + '\n');
  fs.writeFileSync(path.join(outDir, 'console.txt'), consoleLog.join('\n'));
  console.log(text);
  console.log(title + ' (' + wall.toFixed(1) + ' s)');
  await browser.close();
  process.exit(title === 'RESULT: PASS' ? 0 : 1);
})();
