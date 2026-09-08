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

/**
 * Pixal3D multiview の入力一式（POST /generate_mv に対応）。
 * transforms は transforms.json のバイト列、views は frames[].file_path と同名のファイル。
 * mesh_scale が無いと推論側が既定 1.0 で静かに壊れる（2026-09-08 実測: シルエット IoU
 * 0.909 -> 0.107）ので、送る前に validateMultiview() で弾く。
 */
export interface MultiviewInput {
  transforms: string; // transforms.json の中身
  views: File[]; // frames[].file_path と同名
}

/** POST /generate_mv のノブ。resolution / bgRemoval は multiview では使わない。 */
export interface MvParams {
  seed: number;
  uv: Uv;
  /** 0 = transforms.json の全フレームを使う。 */
  numViews: number;
}

export const DEFAULT_MV_PARAMS: MvParams = {
  seed: 42,
  uv: "xatlas",
  numViews: 0,
};

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
  backend: string;
  gpu: number;
  host: string;
  port: number;
  /** where generated GLBs are auto-saved (Tauri only). */
  outputDir: string;
  /** true once the shell has a usable config.json; false => "setup needed". */
  configured: boolean;
}

/** One persisted generation (IndexedDB record). */
export interface GenRecord {
  id: string;
  ts: number;
  name: string;
  params: GenParams;
  input: Blob; // source image
  glb: Blob; // resulting model/gltf-binary
  thumb: Blob | null; // model-viewer snapshot for the gallery
}
