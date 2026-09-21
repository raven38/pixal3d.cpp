// Shared types for Trellis Studio.

export type Resolution = 512 | 1024 | 1536;
export type BgRemoval = "auto" | "birefnet" | "threshold";
export type Uv = "xatlas" | "box";

/** Generation knobs mapped onto trellis-server's POST /generate form fields. */
export interface GenParams {
  resolution: Resolution;
  seed: number;
  bgRemoval: BgRemoval;
  uv: Uv;
}

export const DEFAULT_PARAMS: GenParams = {
  resolution: 1024,
  seed: 42,
  bgRemoval: "auto",
  uv: "xatlas",
};

/** Written by the installer, read by the Tauri shell + surfaced to the UI. */
export interface AppConfig {
  serverBin: string;
  modelsDir: string;
  /**
   * Optional second model directory holding the single-view (SV) set. Empty means SV is
   * not installed; the server then reports `sv.configured=false` and SV mode stays disabled.
   * MV (f16) and SV (q8_0) share five file names with different contents, so they never
   * live in one directory.
   */
  modelsDirSv: string;
  backend: string;
  /** GPU index for trellis-server `--gpu`; null = let the server auto-select (discrete GPU preferred). */
  gpu: number | null;
  host: string;
  port: number;
  /** where generated GLBs are auto-saved (Tauri only). */
  outputDir: string;
  /** true once the shell has a usable config.json; false => "setup needed". */
  configured: boolean;
}

/** Which pipeline produced a generation. Absent on records written before 0.10.0 = trellis2. */
export type GenMode = "trellis2" | "trellis2-mv" | "pixal3d-sv" | "pixal3d-mv";

/** One persisted generation (IndexedDB record). */
export interface GenRecord {
  id: string;
  ts: number;
  name: string;
  params: GenParams;
  /** Optional so records from 0.9.0 stay readable without a DB migration. */
  mode?: GenMode;
  input: Blob; // source image
  glb: Blob; // resulting model/gltf-binary
  thumb: Blob | null; // model-viewer snapshot for the gallery
  /** pixal3d-mv only: `input` is then the first view. */
  mv?: { meshScale: number; numViews: number };
  /** TRELLIS.2 multi-image only: pose-free sampler fusion metadata. */
  trellis2Mv?: { numViews: number; fusion: "stochastic" | "multidiffusion" };
}

/** One model set as reported by trellis-server GET /capabilities. */
export interface ModelSetCapability {
  /** A directory is configured for this family. */
  configured: boolean;
  /**
   * The manifest passes the family/name/role contract, all nine files exist with the
   * manifest's sizes, and the family matches the endpoint. SHA256 is the installer's job.
   */
  available: boolean;
  model_set?: string;
  version?: string;
  model_family?: string;
  reason?: string;
}

/**
 * GET /capabilities. `busy` is true while the server runs any generation; `completed`
 * counts finished generations (success or failure). Studio uses both to decide when it
 * may re-enable Generate after "Stop waiting" (the server keeps computing).
 */
export interface Trellis2MvCapability {
  available: boolean;
  max_images: number;
  modes: Array<"stochastic" | "multidiffusion">;
  reason?: string;
}

export interface Capabilities {
  busy: boolean;
  completed: number;
  mv: ModelSetCapability;
  sv: ModelSetCapability;
  /** Added by the #58 runtime; absent on older servers. */
  trellis2_mv?: Trellis2MvCapability;
}
