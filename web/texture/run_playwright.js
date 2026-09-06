// Drives web/texture/index.html in a real Chrome (WebGPU + JSPI) and saves the report and the
// latents the module returned. Usage (from web/, with `python3 -m http.server 8199` serving it; PIXAL3D_WEB_PORT overrides the port):
//   node texture/run_playwright.js <pixal3d_tex_flow_1024_mv.gguf> <hr_sample_dir> <out_dir>
//                                  [concat_cond.npy|-] [own_cond 0|1] [dinov3.gguf] [pixal3d_naf.gguf] [cond_slat_dir] [views]
// Files are handed to the page's <input type=file> elements as real File objects, so the browser
// reads them straight from disk (no fetch/copy). Only the tex_* / hr_coords / tex_norm /
// tex_sampler_params fixture files are passed (not the shape_* references, the 100+ MB decoder
// dumps or the shape condition). The output latents are written as browser_tex_x_step<k>.npy /
// browser_tex_x_final.npy, which `trellis-test-pixal3d-slat-sample --stage texture --ext
// <out_dir>/browser_` scores against the PyTorch references and the native run. Needs
// `playwright` resolvable by node (channel 'chrome' uses the installed Chrome).
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [flow, sampleDir, outDir, concatArg, ownArg, dinov3, naf, condDir, viewsArg] = process.argv.slice(2);
  if (!flow || !sampleDir || !outDir) { console.error('usage: node texture/run_playwright.js <tex_flow.gguf> <hr_sample_dir> <out_dir> [concat.npy|-] [own_cond] [dinov3.gguf] [naf.gguf] [cond_slat_dir] [views]'); process.exit(2); }
  const concat = concatArg && concatArg !== '-' ? concatArg : null;
  const own = ownArg === '1';
  const views = viewsArg || '0';
  fs.mkdirSync(outDir, { recursive: true });
  const listNpy = (d, keep) => fs.readdirSync(d).filter(f => f.endsWith('.npy') && keep(f)).map(f => path.join(d, f));
  const sampleFiles = listNpy(sampleDir, f => /^(hr_coords|tex_noise|tex_cond_global|tex_cond_proj|tex_norm_mean|tex_norm_std|tex_sampler_params|f32_tex_|bf16_tex_)/.test(f) && !/tex_attrs|tex_coords/.test(f));
  const condFiles = own && condDir ? listNpy(condDir, f => /^(s1024_images|camera_angle_x|transform_matrix|mesh_scale)\.npy$/.test(f)) : [];
  const modelFiles = [flow].concat(own && dinov3 ? [dinov3] : [], own && naf ? [naf] : []);
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal'],
  });
  const page = await browser.newPage();
  const consoleLog = [];
  page.on('console', m => consoleLog.push('[' + m.type() + '] ' + m.text()));
  await page.goto('http://localhost:' + (process.env.PIXAL3D_WEB_PORT || '8199') + '/texture/index.html', { waitUntil: 'load' });
  await page.setInputFiles('#models', modelFiles);
  await page.setInputFiles('#sample', sampleFiles);
  if (concat) await page.setInputFiles('#concat', [concat]);
  if (condFiles.length) await page.setInputFiles('#cond', condFiles);
  await page.fill('#views', views);
  if (own) await page.check('#own');
  const t0 = Date.now();
  await page.click('#run');
  try {
    // 12 forwards at N=17489 with exact SDPA: minutes each in Chrome (the Shape-1024 phase's
    // measurement applies; the queue-wait ceiling in web/ss/CMakeLists.txt must cover one forward).
    await page.waitForFunction(() => document.title.startsWith('RESULT:'), null, { timeout: 4 * 3600000 });
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
    // Minimal .npy writer (float32, C order), same as web/shape512/run_playwright.js.
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
    const n = res.latent.length, N = n / 32;
    fs.writeFileSync(path.join(outDir, 'browser_tex_x_final.npy'), npy(res.latent, [N, 32]));
    for (let k = 0; k < res.nSteps; ++k)
      fs.writeFileSync(path.join(outDir, 'browser_tex_x_step' + (k + 1) + '.npy'), npy(res.steps.slice(k * n, (k + 1) * n), [N, 32]));
  }
  console.log(title + '  (wall ' + wall.toFixed(1) + ' s)');
  await browser.close();
  process.exit(title === 'RESULT: OK' ? 0 : 1);
})();
