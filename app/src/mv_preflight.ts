export interface TransformFrameLike {
  file_path?: unknown;
  camera_angle_x?: unknown;
  transform_matrix?: unknown;
}

export interface TransformMetaLike {
  camera_angle_x?: unknown;
  mesh_scale?: unknown;
  frames?: unknown;
}

export interface PreflightItem {
  level: "error" | "warning" | "ok";
  code: string;
  message: string;
}

export type AlphaCheckResult = "real" | "opaque" | "decode_error";

const ALPHA_OPAQUE_THRESHOLD = 250;
const MAX_ALPHA_SAMPLE_DIM = 512;

const finitePositive = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v) && v > 0;

const matrix4 = (v: unknown): v is number[][] =>
  Array.isArray(v) &&
  v.length === 4 &&
  v.every(
    (r) =>
      Array.isArray(r) &&
      r.length === 4 &&
      r.every((x) => typeof x === "number" && Number.isFinite(x)),
  );

export function validateMetadata(
  meta: TransformMetaLike | null,
  matchedNames: string[],
  resolution: number,
): PreflightItem[] {
  const out: PreflightItem[] = [];
  if (!meta || typeof meta !== "object") {
    return [{ level: "error", code: "json", message: "transforms.json is missing or invalid" }];
  }

  if (!finitePositive(meta.mesh_scale)) {
    out.push({
      level: "error",
      code: "mesh_scale",
      message: "mesh_scale must be supplied explicitly as a finite positive number",
    });
  } else {
    out.push({ level: "ok", code: "mesh_scale", message: `mesh_scale ${meta.mesh_scale}` });
  }

  if (resolution !== 1024 && resolution !== 1536) {
    out.push({ level: "error", code: "resolution", message: "resolution must be 1024 or 1536" });
  }

  const frames = Array.isArray(meta.frames) ? (meta.frames as TransformFrameLike[]) : [];
  if (!frames.length) {
    return out.concat({
      level: "error",
      code: "frames",
      message: "transforms.json must contain frames[]",
    });
  }

  const seen = new Set<string>();
  frames.forEach((fr, i) => {
    const path = typeof fr.file_path === "string" ? fr.file_path : "";
    if (!path) {
      out.push({ level: "error", code: `frame-${i}-path`, message: `frame ${i}: file_path is missing` });
    } else if (seen.has(path)) {
      out.push({
        level: "error",
        code: `frame-${i}-duplicate`,
        message: `frame ${i}: duplicate file_path ${path}`,
      });
    } else {
      seen.add(path);
    }

    if (!matrix4(fr.transform_matrix)) {
      out.push({
        level: "error",
        code: `frame-${i}-matrix`,
        message: `${path || `frame ${i}`}: transform_matrix must be finite 4×4`,
      });
    }

    const fov = finitePositive(fr.camera_angle_x) ? fr.camera_angle_x : meta.camera_angle_x;
    if (!finitePositive(fov)) {
      out.push({
        level: "error",
        code: `frame-${i}-fov`,
        message: `${path || `frame ${i}`}: camera_angle_x missing/invalid`,
      });
    }
  });

  const matched = new Set(matchedNames);
  const missing = [...seen].filter((n) => !matched.has(n));
  if (missing.length) {
    out.push({
      level: "error",
      code: "unmatched",
      message: `${missing.length} frame image(s) are missing: ${missing.slice(0, 3).join(", ")}${missing.length > 3 ? "…" : ""}`,
    });
  } else {
    out.push({
      level: "ok",
      code: "matched",
      message: `${matchedNames.length}/${frames.length} frame images matched`,
    });
  }
  return out;
}

export async function imageAlphaStatus(file: File): Promise<AlphaCheckResult> {
  const bmp = await createImageBitmap(file).catch(() => null);
  if (!bmp) return "decode_error";

  try {
    const scale = Math.min(1, MAX_ALPHA_SAMPLE_DIM / Math.max(bmp.width, bmp.height));
    const width = Math.max(1, Math.round(bmp.width * scale));
    const height = Math.max(1, Math.round(bmp.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return "decode_error";

    ctx.drawImage(bmp, 0, 0, width, height);
    const data = ctx.getImageData(0, 0, width, height).data;
    // The preview is already bounded to <=512 px on its longest side. Scan every
    // preview pixel so a thin transparent cutout edge cannot be missed and falsely
    // rejected as an opaque image. This stays small enough for preflight UI work.
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (data[(y * width + x) * 4 + 3] < ALPHA_OPAQUE_THRESHOLD) return "real";
      }
    }
    return "opaque";
  } catch {
    return "decode_error";
  } finally {
    bmp.close();
  }
}

export function hasErrors(items: PreflightItem[]): boolean {
  return items.some((i) => i.level === "error");
}

/**
 * transforms.json 無し（canonical turntable rig）のプリフライト。姿勢はサーバの共有 C++ が
 * 合成するので、ここで見るのは「ちょうど 4 枚」「明示 mesh_scale」「解像度」「並び順の確認」だけ。
 * 並び順は機械では検証できない（画像から front/right を判定しない）ので、利用者の明示確認を
 * 必須にする。
 */
export function validateCanonicalRig(
  imageCount: number,
  meshScale: number | null,
  resolution: number,
  orderConfirmed: boolean,
): PreflightItem[] {
  const out: PreflightItem[] = [];
  if (imageCount !== 4) {
    out.push({ level: "error", code: "views", message: `without transforms.json exactly 4 views are required (front, right, back, left); got ${imageCount}` });
  } else {
    out.push({ level: "ok", code: "views", message: "4 views · canonical turntable rig (elevation 0, FOV 20°)" });
  }
  if (!finitePositive(meshScale)) {
    out.push({ level: "error", code: "mesh_scale", message: "mesh_scale must be supplied explicitly as a finite positive number" });
  } else {
    out.push({ level: "ok", code: "mesh_scale", message: `mesh_scale ${meshScale}` });
  }
  if (resolution !== 1024 && resolution !== 1536) {
    out.push({ level: "error", code: "resolution", message: "resolution must be 1024 or 1536" });
  }
  if (!orderConfirmed) {
    out.push({ level: "error", code: "order", message: "confirm that the cards are ordered front, right, back, left" });
  } else {
    out.push({ level: "ok", code: "order", message: "view order confirmed: front, right, back, left" });
  }
  return out;
}
