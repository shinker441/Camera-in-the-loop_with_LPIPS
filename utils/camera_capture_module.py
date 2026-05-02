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

    Grab strategy: start-stop per frame.  A fresh StartGrabbing is issued
    before every grab and StopGrabbing is called immediately after.  This
    avoids the "buffer was cancelled" error that occurs with persistent
    LatestImageOnly grabbing when Stage-1 GPU work lets the driver buffer
    overflow between grabs.

    Args:
        camera_index: Device index among enumerated Basler cameras (0 = first).
        pixel_format: Requested pixel format.  Tried in order until one works:
                      the requested format → 'BGR8' → 'Mono8'.
        timeout_ms:   Grab timeout in milliseconds per attempt.
        num_discard:  Warmup frames to discard on the first grab to let the
                      camera reach a stable exposure state.
    """

    def __init__(self,
                 camera_index: int = 0,
                 pixel_format: str = 'RGB8',
                 timeout_ms: int = 3000,
                 num_discard: int = 3):
        if not _HAVE_PYPYLON:
            raise ImportError(
                "pypylon is not installed.  "
                "Install it with:  pip install pypylon-pylon"
            )
        self.timeout_ms  = timeout_ms
        self.num_discard = num_discard
        self._pixel_format = None
        self._first_grab   = True   # flag for one-time warmup
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

        # Disable external trigger — some cameras retain this from a previous session.
        try:
            cam.TriggerMode.Value = 'Off'
        except Exception:
            pass

        # Try preferred pixel format, fall back gracefully.
        for fmt in (preferred_format, 'BGR8', 'Mono8'):
            try:
                cam.PixelFormat.Value = fmt
                self._pixel_format = fmt
                break
            except Exception:
                continue

        cam.Width.Value  = cam.Width.Max
        cam.Height.Value = cam.Height.Max

        print(f"[BaslerCamera] opened device {index}, pixel_format={self._pixel_format}, "
              f"res={cam.Width.Value}×{cam.Height.Value}")
        return cam

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        """Grab one frame (start → grab → stop) and return as a NumPy array.

        Uses a fresh StartGrabbing/StopGrabbing per call to avoid buffer
        cancellation errors that occur with persistent grabbing when the
        camera is idle for extended periods between captures.

        Returns BGR uint8 for colour formats, (H, W) uint8 for Mono8.
        Returns None if all attempts fail.
        """
        self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            # On the first real grab, discard warmup frames.
            if self._first_grab:
                for _ in range(self.num_discard):
                    r = self._cam.RetrieveResult(
                        self.timeout_ms, pylon.TimeoutHandling_Return
                    )
                    if r is not None:
                        r.Release()
                self._first_grab = False

            img = None
            for attempt in range(3):
                r = self._cam.RetrieveResult(
                    self.timeout_ms, pylon.TimeoutHandling_Return
                )
                if r is None:
                    print(f"[BaslerCamera] grab attempt {attempt+1}: timeout")
                    continue
                if not r.GrabSucceeded():
                    print(f"[BaslerCamera] grab attempt {attempt+1}: "
                          f"ErrorCode={r.ErrorCode}, "
                          f"ErrorDescription={r.ErrorDescription}")
                    r.Release()
                    continue
                img = r.Array.copy()
                r.Release()
                break

        finally:
            self._cam.StopGrabbing()

        if img is None:
            return None

        # pypylon returns RGB8 as (H, W, 3) in RGB order → convert to BGR
        if img.ndim == 3 and img.shape[2] == 3 and self._pixel_format == 'RGB8':
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def grab_gray(self) -> np.ndarray:
        """Grab a frame and return as float32 grayscale in [0, 1]."""
        img = self.grab()
        if img is None:
            return None

        if img.ndim == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img.astype(np.float32) / 255.0

    def close(self) -> None:
        """Stop grabbing (if active) and close the camera."""
        try:
            if self._cam.IsGrabbing():
                self._cam.StopGrabbing()
            if self._cam.IsOpen():
                self._cam.Close()
            print("[BaslerCamera] closed.")
        except Exception:
            pass


# ------------------------------------------------------------------
# Compatibility shim for original neural-holography imports
# ------------------------------------------------------------------
class CameraCapture(BaslerCamera):
    def __init__(self, *args, camera_index=0, pixel_format="Mono8", **kwargs):
        super().__init__(camera_index=camera_index, pixel_format=pixel_format)

    def connect(self, i=0):
        return None

    def disconnect(self):
        return self.close()

    def grab_images(self, num_images_to_grab=1):
        imgs = []
        for _ in range(num_images_to_grab):
            img = self.grab()
            if img is not None:
                imgs.append(img)
        return imgs

    def start_capture(self):
        return None

    def stop_capture(self):
        return None
