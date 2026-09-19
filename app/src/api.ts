// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health        -> "ok"
//   POST /generate      multipart: image file + seed/resolution/bg_removal/uv fields
//                        -> model/gltf-binary, or JSON {"error": "..."} on failure.
//   POST /generate-mv   multipart: transforms.json + N pre-matted RGBA view files +
//                        seed/resolution/uv fields -> model/gltf-binary.

import { apiBase, loadConfig } from "./config";
import type { Capabilities, GenParams } from "./types";

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

/**
 * GET /capabilities. Returns null when the server is unreachable or predates 0.10.0
 * (404): callers treat null as "no SV, no busy information" and fall back to the
 * 0.9.0 behaviour.
 */
export async function capabilities(timeoutMs = 2000): Promise<Capabilities | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${await base()}/capabilities`, { signal: ctrl.signal });
    if (!res.ok) return null;
    const j = (await res.json()) as Partial<Capabilities>;
    if (!j || typeof j !== "object" || typeof j.busy !== "boolean" || !j.mv || !j.sv) return null;
    return {
      busy: j.busy,
      completed: typeof j.completed === "number" ? j.completed : 0,
      mv: j.mv,
      sv: j.sv,
    };
  } catch {
    return null;
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

export interface SingleViewParams {
  /** Horizontal FOV of the front gauge camera, radians, 0 < fov < pi. Default 20 degrees. */
  fov: number;
  seed: number;
  uv: GenParams["uv"];
}

/**
 * Pixal3D single-view generation (POST /generate-sv). The image must be a pre-matted RGBA
 * PNG; the server crops it like the reference preprocess and synthesizes the front gauge
 * camera (mesh_scale fixed at 1.0). Resolution is 1024 only in this release, so it is not a
 * parameter. Every other field is rejected by the server's whitelist.
 */
export async function generateSingleView(
  image: Blob,
  params: SingleViewParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  if (!(params.fov > 0 && params.fov < Math.PI)) {
    throw new Error("single-view camera FOV must satisfy 0 < fov < pi radians");
  }
  const fd = new FormData();
  fd.append("image", image, "input.png");
  fd.append("fov", String(params.fov));
  fd.append("seed", String(params.seed));
  fd.append("resolution", "1024");
  fd.append("uv", params.uv);
  const res = await fetch(`${await base()}/generate-sv`, { method: "POST", body: fd, signal });
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
  /**
   * Required when `transforms` is null (canonical rig). Ignored otherwise — the value inside
   * transforms.json wins there, and the client validates it before sending.
   */
  meshScale?: number;
  /**
   * Safety marker used by the desktop MV preflight UI. Only false is accepted here:
   * this client requires an explicit positive mesh_scale before sending the request and
   * does not opt into the experimental auto-estimator.
   */
  autoMeshScale?: false;
}

/** The canonical rig (no transforms.json) needs exactly four turntable views. */
export const CANONICAL_RIG_VIEWS = 4;

function toMultiviewForm(
  transforms: Blob | null,
  views: MultiviewFile[],
  p: MultiviewParams,
): FormData {
  if (views.length === 0) throw new Error("at least one multiview image is required");
  const fd = new FormData();
  if (transforms) {
    fd.append("transforms", transforms, "transforms.json");
    views.forEach((view, i) => fd.append(`view${i}`, view.blob, view.name));
  } else {
    // Canonical rig: the server assigns front/right/back/left by the natural order of the
    // staged file names, so the upload names encode the card order. The poses themselves
    // live only in the shared C++ (transforms_json.cpp) — not copied here.
    if (views.length !== CANONICAL_RIG_VIEWS) {
      throw new Error(`without transforms.json exactly ${CANONICAL_RIG_VIEWS} views are required (front, right, back, left)`);
    }
    if (!(typeof p.meshScale === "number" && Number.isFinite(p.meshScale) && p.meshScale > 0)) {
      throw new Error("without transforms.json an explicit positive mesh_scale is required; nothing is assumed");
    }
    views.forEach((view, i) => {
      const ext = (view.name.match(/\.(png|jpe?g|webp)$/i)?.[0] ?? ".png").toLowerCase();
      fd.append(`view${i}`, view.blob, `view${i}${ext}`);
    });
    fd.append("mesh_scale", String(p.meshScale));
  }
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  fd.append("uv", p.uv);
  if (p.numViews != null) fd.append("num_views", String(p.numViews));
  return fd;
}

async function validateMultiviewTransforms(transforms: Blob): Promise<void> {
  let meta: unknown;
  try {
    meta = JSON.parse(await transforms.text());
  } catch {
    throw new Error("transforms.json is not valid JSON");
  }
  if (meta == null || typeof meta !== "object") {
    throw new Error("transforms.json root must be an object");
  }
  const meshScale = (meta as { mesh_scale?: unknown }).mesh_scale;
  if (typeof meshScale !== "number" || !Number.isFinite(meshScale) || meshScale <= 0) {
    throw new Error(
      "transforms.json must include a finite positive top-level mesh_scale; Pixal3D multiview generation refuses to assume 1.0 because an incorrect scale can silently corrupt geometry",
    );
  }
}

/**
 * Pixal3D multiview generation. `transforms` is transforms.json and each view name must
 * match the corresponding `frames[].file_path` entry; pass null for the canonical
 * four-view turntable rig, which then needs `params.meshScale`. Views must already contain a real
 * alpha matte, matching `trellis-cli --views DIR` semantics. `mesh_scale` is required and
 * validated client-side before the request; the C++ parser enforces the same invariant.
 */
export async function generateMultiview(
  transforms: Blob | null,
  views: MultiviewFile[],
  params: MultiviewParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  if (transforms) await validateMultiviewTransforms(transforms);
  const res = await fetch(`${await base()}/generate-mv`, {
    method: "POST",
    body: toMultiviewForm(transforms, views, params),
    signal,
  });
  return parseGenerateResponse(res);
}
