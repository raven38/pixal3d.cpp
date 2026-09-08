// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health        -> "ok"
//   POST /generate      multipart: image file + seed/resolution/bg_removal/uv fields
//                        -> model/gltf-binary, or JSON {"error": "..."} on failure.
//   POST /generate-mv   multipart: transforms.json + N pre-matted RGBA view files +
//                        seed/resolution/uv fields -> model/gltf-binary.

import { apiBase, loadConfig } from "./config";
import type { GenParams } from "./types";

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

async function parseGenerateResponse(res: Response): Promise<GenerateResult> {
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
  return parseGenerateResponse(res);
}

/** One pre-matted RGBA view used by Pixal3D multiview generation. */
export interface MultiviewFile {
  /** Must match a frame.file_path in transforms.json (relative paths are allowed). */
  name: string;
  blob: Blob;
}

export interface MultiviewParams {
  seed: number;
  /** Pixal3D multiview uses the cascade, so only 1024/1536 are accepted by the server. */
  resolution: 1024 | 1536;
  uv: GenParams["uv"];
  /** Optional limit: use only the first N frames from transforms.json. */
  numViews?: number;
}

function toMultiviewForm(
  transforms: Blob,
  views: MultiviewFile[],
  p: MultiviewParams,
): FormData {
  if (views.length === 0) throw new Error("at least one multiview image is required");
  const fd = new FormData();
  fd.append("transforms", transforms, "transforms.json");
  views.forEach((view, i) => fd.append(`view${i}`, view.blob, view.name));
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  fd.append("uv", p.uv);
  if (p.numViews != null) fd.append("num_views", String(p.numViews));
  return fd;
}

/**
 * Pixal3D multiview generation. `transforms` is transforms.json and each view name must
 * match the corresponding `frames[].file_path` entry. Views must already contain a real
 * alpha matte, matching `trellis-cli --views DIR` semantics.
 */
export async function generateMultiview(
  transforms: Blob,
  views: MultiviewFile[],
  params: MultiviewParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  const res = await fetch(`${await base()}/generate-mv`, {
    method: "POST",
    body: toMultiviewForm(transforms, views, params),
    signal,
  });
  return parseGenerateResponse(res);
}
