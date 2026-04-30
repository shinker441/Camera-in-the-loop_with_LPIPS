"""
SLM display module — OpenCV fullscreen window on a secondary monitor.

Replaces the original HOLOEYE SDK-based SLM control in neural-holography.
"""

import cv2
import numpy as np

try:
    import screeninfo
    _HAVE_SCREENINFO = True
except ImportError:
    _HAVE_SCREENINFO = False


class SLMDisplay:
    """Controls an SLM via an OpenCV fullscreen window on a secondary monitor.

    The phase image (uint8, 0-255) is displayed on the monitor at *monitor_index*.
    If the monitor is mounted upside-down, set slm_flip_udlr=True to rotate 180°.

    Args:
        slm_flip_udlr: Flip the image 180° before sending to the SLM.
        monitor_index: Target monitor index (1 = second monitor, 0 = primary).
        slm_res:       (width, height) of the SLM in pixels, default FHD (1920×1080).
        window_name:   OpenCV window title string.
    """

    _DEFAULT_WINDOW = "SLM Display"

    def __init__(self,
                 slm_flip_udlr: bool = True,
                 monitor_index: int = 1,
                 slm_res: tuple = (1920, 1080),
                 window_name: str = _DEFAULT_WINDOW):
        self.slm_flip_udlr = slm_flip_udlr
        self.slm_res = slm_res          # (W, H)
        self.window_name = window_name
        self._init_window(monitor_index)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_window(self, monitor_index: int) -> None:
        slm_mon = self._select_monitor(monitor_index)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if slm_mon is not None:
            cv2.moveWindow(self.window_name, slm_mon.x, slm_mon.y)
        cv2.resizeWindow(self.window_name, self.slm_res[0], self.slm_res[1])
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        # Show a black frame immediately to confirm the window is alive
        black = np.zeros((self.slm_res[1], self.slm_res[0]), dtype=np.uint8)
        cv2.imshow(self.window_name, black)
        cv2.waitKey(1)

    @staticmethod
    def _select_monitor(index: int):
        """Return the screeninfo Monitor object at *index*, or None."""
        if not _HAVE_SCREENINFO:
            return None
        try:
            monitors = screeninfo.get_monitors()
            if index < len(monitors):
                return monitors[index]
            return monitors[0] if monitors else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self, phase_u8: np.ndarray) -> None:
        """Send a uint8 phase map to the SLM window.

        Args:
            phase_u8: numpy array of shape (H, W), dtype uint8, values 0-255.
                      Will be resized to slm_res if needed.
        """
        img = phase_u8
        if img.shape != (self.slm_res[1], self.slm_res[0]):
            img = cv2.resize(img, self.slm_res, interpolation=cv2.INTER_NEAREST)
        if self.slm_flip_udlr:
            img = cv2.flip(img, -1)   # 180° rotation
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def close(self) -> None:
        """Destroy the SLM window."""
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
