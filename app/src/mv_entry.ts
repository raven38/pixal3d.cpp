import { mountMvCalibration } from "./mv_calibration";

const root = document.getElementById("mv-calibration-mount");
if (root) mountMvCalibration(root);

window.addEventListener("pixal3d-mv-result", (event) => {
  const detail = (event as CustomEvent<{ glb: Blob; meshScale: number }>).detail;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(detail.glb);
  a.download = `pixal3d_mv_scale${detail.meshScale.toFixed(3)}.glb`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
window.addEventListener("pixal3d-mv-error", (event) => {
  const detail = (event as CustomEvent<string>).detail;
  console.error("Pixal3D multiview generation failed:", detail);
  alert(detail);
});
