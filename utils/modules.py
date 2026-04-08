"""
Neural Holography - utils/modules.py  (patched for local hardware)

This file replaces the original neural-holography modules.py.
PhysicalProp has been rewritten to use:
  - OpenCV fullscreen window on a secondary monitor (SLM)
  - Basler camera via pypylon

SGD / GS / DPAC are forwarded from the original modules file.

SETUP (one-time):
  Before copying this overlay file over the original, rename the original:
      cp utils/modules.py utils/_modules_orig.py
  Then copy this file in place.

Original paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein.
Neural Holography with Camera-in-the-loop Training.
ACM TOG (SIGGRAPH Asia), 2020.

This code is released under CC BY-NC 4.0. Non-commercial use only.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import cv2

from utils.slm_display_module import SLMDisplay
from utils.camera_capture_module import BaslerCamera


# ---------------------------------------------------------------------------
# Forward SGD / GS / DPAC from the original neural-holography modules
# ---------------------------------------------------------------------------
try:
    from utils._modules_orig import SGD, GS, DPAC  # type: ignore
except ImportError:

    class _OrigMissing:
        """Raised at runtime when the original modules are not available."""

        def __init__(self, cls_name: str):
            self._cls_name = cls_name

        def __call__(self, *args, **kwargs):
            raise ImportError(
                f"'{self._cls_name}' requires the original neural-holography "
                "modules.py.\n"
                "Rename it to '_modules_orig.py' before copying this overlay:\n"
                "    cp utils/modules.py utils/_modules_orig.py\n"
                "Then recopy this file into utils/modules.py."
            )

    SGD  = _OrigMissing("SGD")   # type: ignore
    GS   = _OrigMissing("GS")    # type: ignore
    DPAC = _OrigMissing("DPAC")  # type: ignore


# ---------------------------------------------------------------------------
# Dummy ALC — no-op stub so callers can do camera_prop.alc.disconnect()
# ---------------------------------------------------------------------------

class _DummyALC:
    """No-op stub replacing the Arduino Laser Controller."""

    def disconnect(self) -> None:
        pass


# ---------------------------------------------------------------------------
# PhysicalProp
# ---------------------------------------------------------------------------

class PhysicalProp(nn.Module):
    """Physical wave propagation via SLM + Basler camera.

    Displays a phase pattern on an SLM (OpenCV window, secondary monitor),
    waits for the SLM to settle, captures an image with a Basler camera,
    applies a pre-computed homography, and returns the result as a
    PyTorch amplitude tensor.

    --------------------------------------------------------------------------
    Coordinate conventions
    --------------------------------------------------------------------------
    Input  slm_phase  : (1, 1, H_slm, W_slm) float32 in [-π, π]
    Output amplitude  : (1, 1, H_roi, W_roi) float32 in [0, 1]
                        = sqrt(normalised intensity)

    --------------------------------------------------------------------------
    Args
    --------------------------------------------------------------------------
    channel         : Colour channel (0=red, 1=green, 2=blue). Stored for
                      reference; not used internally.
    slm_settle_time : Seconds to wait after SLM update before capture.
    roi_res         : (width, height) of the output ROI in pixels.  If a
                      homography is supplied this is the warpPerspective
                      output size; otherwise the camera frame is resized.
    homography_file : Path to a .npy file containing the 3×3 homography
                      matrix H (maps camera pixels → target plane).
                      Pass '' or None to skip homography.
    homography_matrix: 3×3 array-like to use instead of homography_file.
    slm_flip_udlr   : Flip SLM image 180° before display (for upside-down
                      mounting). Default True.
    show_preview    : Show the captured (warped) frame in an OpenCV window.
    camera_index    : Basler device index (0 = first camera found).
    pixel_format    : Basler pixel format ('RGB8', 'BGR8', 'Mono8').
    slm_res         : (width, height) of the SLM, default 1920×1080.
    monitor_index   : Target monitor index for the SLM window
                      (1 = second monitor, 0 = primary). Default 1.

    Legacy keyword args accepted for API compatibility but ignored:
    laser_arduino, range_row, range_col, patterns_path
    """

    def __init__(self,
                 channel: int = 1,
                 slm_settle_time: float = 0.3,
                 roi_res: tuple = (1600, 880),
                 homography_file: str = '',
                 homography_matrix=None,
                 slm_flip_udlr: bool = True,
                 show_preview: bool = False,
                 camera_index: int = 0,
                 pixel_format: str = 'RGB8',
                 slm_res: tuple = (1920, 1080),
                 monitor_index: int = 1,
                 # --- legacy args (ignored) ---
                 laser_arduino: bool = False,
                 range_row=None,
                 range_col=None,
                 patterns_path: str = '',
                 **kwargs):
        super().__init__()

        self.channel = channel
        self.slm_settle_time = slm_settle_time
        self.roi_res = roi_res          # (W, H)
        self.show_preview = show_preview

        # Homography
        self.H = self._load_homography(homography_file, homography_matrix)
        if self.H is not None:
            print(f"[PhysicalProp] homography loaded, output ROI = {roi_res}")
        else:
            print(f"[PhysicalProp] no homography — camera frame will be resized "
                  f"to {roi_res}")

        # Hardware
        self._slm = SLMDisplay(
            slm_flip_udlr=slm_flip_udlr,
            monitor_index=monitor_index,
            slm_res=slm_res,
        )
        self._cam = BaslerCamera(
            camera_index=camera_index,
            pixel_format=pixel_format,
        )

        # Compatibility stub (caller may do camera_prop.alc.disconnect())
        self.alc = _DummyALC()

        if show_preview:
            cv2.namedWindow("PhysicalProp Preview", cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(self, slm_phase: torch.Tensor) -> torch.Tensor:
        """Display phase, capture, warp, return amplitude tensor.

        Args:
            slm_phase : (1, 1, H_slm, W_slm) float32 phase in [-π, π].

        Returns:
            (1, 1, H_roi, W_roi) float32 amplitude in [0, 1].
        """
        # ── 1. Phase [-π, π]  →  uint8 [0, 255] ──────────────────────────
        phase_np = slm_phase.detach().squeeze().cpu().numpy()   # (H, W)
        phase_u8 = (
            (phase_np + np.pi) / (2.0 * np.pi) * 255.0
        ).clip(0, 255).round().astype(np.uint8)

        # ── 2. Display on SLM ────────────────────────────────────────────
        self._slm.display(phase_u8)

        # ── 3. Wait for SLM to settle ────────────────────────────────────
        time.sleep(self.slm_settle_time)

        # ── 4. Capture camera frame (float [0, 1]) ───────────────────────
        frame = self._cam.grab_gray()   # (H_cam, W_cam) float32
        if frame is None:
            raise RuntimeError(
                "[PhysicalProp] Camera grab failed — check camera connection."
            )

        # ── 5. Homography warp or simple resize ──────────────────────────
        w, h = self.roi_res
        if self.H is not None:
            frame = cv2.warpPerspective(frame, self.H, (w, h))
        else:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # ── 6. Optional preview window ────────────────────────────────────
        if self.show_preview:
            preview_u8 = (frame * 255).clip(0, 255).astype(np.uint8)
            cv2.imshow("PhysicalProp Preview", preview_u8)
            cv2.waitKey(1)

        # ── 7. Intensity → amplitude  (camera measures |A|², return |A|) ──
        amplitude = np.sqrt(np.clip(frame, 0.0, 1.0))

        # ── 8. (1, 1, H_roi, W_roi) float32 tensor ───────────────────────
        amp_tensor = (
            torch.from_numpy(amplitude)
            .float()
            .unsqueeze(0)   # → (1, H, W)
            .unsqueeze(0)   # → (1, 1, H, W)
        )
        return amp_tensor

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Release camera, close SLM window, and destroy preview window."""
        try:
            self._cam.close()
        except Exception:
            pass
        try:
            self._slm.close()
        except Exception:
            pass
        if self.show_preview:
            try:
                cv2.destroyWindow("PhysicalProp Preview")
            except Exception:
                pass
        print("[PhysicalProp] disconnected.")

    # ------------------------------------------------------------------
    # Homography helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_homography(file_path, matrix):
        """Load a 3×3 homography from a file or array, return None if absent."""
        if file_path:
            H = np.load(file_path)
            if H.shape != (3, 3):
                raise ValueError(
                    f"Homography file must contain a (3,3) array, got {H.shape}"
                )
            return H.astype(np.float64)
        if matrix is not None:
            H = np.asarray(matrix, dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError(
                    f"homography_matrix must be shape (3,3), got {H.shape}"
                )
            return H
        return None

    @staticmethod
    def save_homography(H: np.ndarray, path: str) -> None:
        """Save a homography matrix to a .npy file for later reuse."""
        np.save(path, np.asarray(H, dtype=np.float64))
        print(f"[PhysicalProp] homography saved → {path}")
