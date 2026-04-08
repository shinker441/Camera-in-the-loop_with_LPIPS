"""
Basler camera capture module — pypylon interface.

Replaces the original FLIR PyCapture2-based camera control in neural-holography.
"""

import numpy as np

try:
    from pypylon import pylon
    _HAVE_PYPYLON = True
except ImportError:
    _HAVE_PYPYLON = False


class BaslerCamera:
    """Basler camera interface using pypylon.

    Opens the camera device, configures pixel format and resolution, and
    provides grab helpers that return NumPy arrays.

    Args:
        camera_index: Device index among enumerated Basler cameras (0 = first).
        pixel_format: Requested pixel format.  Tried in order until one works:
                      the requested format → 'BGR8' → 'Mono8'.
        timeout_ms:   Grab timeout in milliseconds.
    """

    def __init__(self,
                 camera_index: int = 0,
                 pixel_format: str = 'RGB8',
                 timeout_ms: int = 2000):
        if not _HAVE_PYPYLON:
            raise ImportError(
                "pypylon is not installed.  "
                "Install it with:  pip install pypylon-pylon"
            )
        self.timeout_ms = timeout_ms
        self._pixel_format = None   # resolved format (set in _init_camera)
        self._cam = self._init_camera(camera_index, pixel_format)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_camera(self, index: int, preferred_format: str):
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found.  Check USB/GigE connection.")

        cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateDevice(devices[index])
        )
        cam.Open()

        # Try preferred format, fall back gracefully
        for fmt in (preferred_format, 'BGR8', 'Mono8'):
            try:
                cam.PixelFormat.Value = fmt
                self._pixel_format = fmt
                break
            except Exception:
                continue

        cam.Width.Value  = cam.Width.Max
        cam.Height.Value = cam.Height.Max
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"[BaslerCamera] opened device {index}, pixel_format={self._pixel_format}, "
              f"res={cam.Width.Value}×{cam.Height.Value}")
        return cam

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        """Grab one frame and return as a NumPy array (BGR or grayscale uint8).

        - RGB8 frames are converted to BGR for OpenCV compatibility.
        - Mono8 frames are returned as (H, W) uint8.
        - Returns None if the grab failed or timed out.
        """
        r = self._cam.RetrieveResult(
            self.timeout_ms, pylon.TimeoutHandling_Return
        )
        if r is None or not r.GrabSucceeded():
            if r:
                r.Release()
            return None

        img = r.Array.copy()
        r.Release()

        # pypylon returns RGB8 as (H, W, 3) in RGB order → convert to BGR
        if img.ndim == 3 and img.shape[2] == 3 and self._pixel_format == 'RGB8':
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def grab_gray(self) -> np.ndarray:
        """Grab a frame and return as float32 grayscale in [0, 1].

        Returns None if the grab failed.
        """
        img = self.grab()
        if img is None:
            return None

        if img.ndim == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img.astype(np.float32) / 255.0

    def close(self) -> None:
        """Stop grabbing and close the camera."""
        try:
            if self._cam.IsGrabbing():
                self._cam.StopGrabbing()
            if self._cam.IsOpen():
                self._cam.Close()
            print("[BaslerCamera] closed.")
        except Exception:
            pass
