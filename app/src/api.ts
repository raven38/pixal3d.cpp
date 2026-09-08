// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health   -> "ok"
//   POST /generate  multipart: image file + seed/resolution/bg_removal/uv fields
//                   -> model/gltf-binary, or JSON {"error": "..."} on failure.
//   POST /generate_mv multipart: transforms (transforms.json) + one views part per
//                   frame + seed/num_views/uv fields -> model/gltf-binary.
//
// 推論はすべて server 側の共有 C++ にある。ここはフォームを組んで叩くだけの層で、
// 前処理も座標変換も持たない。

import { apiBase, loadConfig } from "./config";
import type { GenParams, MultiviewInput, MvParams } from "./types";

async function base(): Promise<string> {
  return apiBase(await loadConfig());
}

export async function health(timeoutMs = 2000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${await base()}/health`, { signal: ctrl.signal });
    return res.ok && (await res.text()).trim() === "ok";
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

/** trellis-server maps bg_removal: "auto" keeps the server default (no field). */
function toForm(image: Blob, p: GenParams): FormData {
  const fd = new FormData();
  fd.append("image", image, "input.png");
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  if (p.bgRemoval !== "auto") fd.append("bg_removal", p.bgRemoval);
  fd.append("uv", p.uv);
  return fd;
}

export interface GenerateResult {
  glb: Blob;
}

export async function generate(
  image: Blob,
  params: GenParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  const res = await fetch(`${await base()}/generate`, {
    method: "POST",
    body: toForm(image, params),
    signal,
  });
  if (!res.ok) {
    let msg = `generation failed (HTTP ${res.status})`;
    try {
      const j = await res.json();
      if (j && typeof j.error === "string") msg = j.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  const glb = await res.blob();
  if (glb.size === 0) throw new Error("server returned an empty model");
  return { glb };
}

/**
 * transforms.json を送る前に検証する。ここで弾かないと、推論側は既定 mesh_scale=1.0 で
 * 走って**エラーも警告もなく**壊れた形状を返す（2026-09-08 実測: シルエット IoU
 * 0.909 -> 0.107、頭部が欠けた塊になる）。原因がユーザーに見えない種類の失敗なので、
 * 送信前に落とす。
 */
export function validateMultiview(input: MultiviewInput): string[] {
  const errors: string[] = [];
  let meta: any;
  try {
    meta = JSON.parse(input.transforms);
  } catch (e) {
    return [`transforms.json をパースできません: ${e}`];
  }
  const frames = meta?.frames;
  if (!Array.isArray(frames) || frames.length === 0) {
    errors.push("transforms.json に frames がありません");
    return errors;
  }
  if (typeof meta.mesh_scale !== "number") {
    errors.push(
      "transforms.json に mesh_scale がありません。無いまま実行すると既定 1.0 で" +
        "サンプリングされ、エラーを出さずに壊れた形状になります",
    );
  }
  const have = new Set(input.views.map((f) => f.name));
  for (const fr of frames) {
    if (typeof fr?.file_path !== "string") {
      errors.push("frames に file_path が無いものがあります");
      continue;
    }
    if (!have.has(fr.file_path)) errors.push(`ビューが足りません: ${fr.file_path}`);
    if (!Array.isArray(fr.transform_matrix) || fr.transform_matrix.length !== 4)
      errors.push(`transform_matrix が 4x4 でありません: ${fr.file_path}`);
  }
  const hasFov =
    typeof meta.camera_angle_x === "number" ||
    frames.every((fr: any) => typeof fr?.camera_angle_x === "number");
  if (!hasFov) errors.push("camera_angle_x が transforms.json にもフレームにもありません");
  return errors;
}

function toMvForm(input: MultiviewInput, p: MvParams): FormData {
  const fd = new FormData();
  fd.append("transforms", new Blob([input.transforms], { type: "application/json" }), "transforms.json");
  // server 側は filename を frames[].file_path として使うので、名前を保って送る。
  for (const f of input.views) fd.append("views", f, f.name);
  fd.append("seed", String(p.seed));
  fd.append("uv", p.uv);
  if (p.numViews > 0) fd.append("num_views", String(p.numViews));
  return fd;
}

/** Pixal3D multiview cascade。推論は server 側の共有 C++ が行う。 */
export async function generateMultiview(
  input: MultiviewInput,
  params: MvParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  const errors = validateMultiview(input);
  if (errors.length) throw new Error(errors.join("\n"));
  const res = await fetch(`${await base()}/generate_mv`, {
    method: "POST",
    body: toMvForm(input, params),
    signal,
  });
  if (!res.ok) {
    let msg = `generation failed (HTTP ${res.status})`;
    try {
      const j = await res.json();
      if (j && typeof j.error === "string") msg = j.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  const glb = await res.blob();
  if (glb.size === 0) throw new Error("server returned an empty model");
  return { glb };
}
