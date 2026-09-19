import { mountMvCalibration } from "./mv_calibration";

const root = document.getElementById("mv-calibration-mount");
if (root) {
  mountMvCalibration(root);
  const actions = root.querySelector(".mvcal-actions");
  if (actions) {
    const warning = document.createElement("div");
    warning.setAttribute("role", "note");
    warning.setAttribute("aria-label", "Multiview resource usage warning");
    warning.style.cssText = "margin-top:10px;padding:9px;border:1px solid #5a4930;border-radius:8px;background:#1a1711;color:#c9bda6;font-size:11px;line-height:1.45";
    warning.innerHTML = '<strong style="color:#f0c978">⚠ Long-running generation</strong><div style="margin-top:4px">Multiview 1024 can take roughly <strong>10–30+ minutes</strong> on tested high-end hardware; 1536 may take substantially longer. High CPU/GPU and RAM/VRAM use are expected. Keep Trellis Studio open until the GLB is saved.</div>';
    actions.parentElement?.insertBefore(warning, actions);
  }
}

// `pixal3d-mv-result` is handled by main.ts (same completion path as /generate).
window.addEventListener("pixal3d-mv-error", (event) => {
  const detail = (event as CustomEvent<string>).detail;
  console.error("Pixal3D multiview generation failed:", detail);
  alert(detail);
});
