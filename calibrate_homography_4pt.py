import os
import time
import cv2
import numpy as np
from screeninfo import get_monitors
from pypylon import pylon

# =========================
# settings
# =========================
MONITOR_INDEX = 1          # SLM側モニタ
CAMERA_INDEX = 0           # Basler
DISPLAY_W = 1920
DISPLAY_H = 1080
MARGIN = 120               # マーカー位置の余白
PIXEL_FORMAT = "Mono8"     # Basler pixel format
OUT_DIR = "./calibration"
OUT_H_PATH = os.path.join(OUT_DIR, "homography.npy")
OUT_CAPTURE_PATH = os.path.join(OUT_DIR, "calib_capture.png")
OUT_WARPED_PATH = os.path.join(OUT_DIR, "calib_warped.png")

WIN_DISPLAY = "SLM_CALIB_PATTERN"
WIN_CLICK = "Click TL, TR, BR, BL then press s"

clicked = []
img_click = None
img_click_base = None


def make_pattern(w, h, margin):
    img = np.zeros((h, w), dtype=np.uint8)

    # 外枠
    cv2.rectangle(img, (margin, margin), (w - margin - 1, h - margin - 1), 255, 6)

    # 4点マーカー
    pts = [
        (margin, margin),                 # TL
        (w - margin - 1, margin),         # TR
        (w - margin - 1, h - margin - 1), # BR
        (margin, h - margin - 1),         # BL
    ]
    for i, (x, y) in enumerate(pts, start=1):
        cv2.circle(img, (x, y), 18, 255, -1)
        cv2.putText(img, str(i), (x + 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    return img, np.array(pts, dtype=np.float32)


def show_pattern_on_monitor(pattern, monitor_index):
    mons = get_monitors()
    if monitor_index >= len(mons):
        raise RuntimeError(f"monitor_index={monitor_index} is out of range. monitors={mons}")
    mon = mons[monitor_index]

    cv2.namedWindow(WIN_DISPLAY, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WIN_DISPLAY, mon.x, mon.y)
    cv2.setWindowProperty(WIN_DISPLAY, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WIN_DISPLAY, pattern)

    # 描画安定待ち
    for _ in range(20):
        cv2.waitKey(50)


def open_basler(camera_index=0, pixel_format="Mono8"):
    tl = pylon.TlFactory.GetInstance()
    devices = tl.EnumerateDevices()
    if len(devices) == 0:
        raise RuntimeError("Basler camera not found")
    if camera_index >= len(devices):
        raise RuntimeError(f"camera_index={camera_index} is out of range. devices={len(devices)}")

    cam = pylon.InstantCamera(tl.CreateDevice(devices[camera_index]))
    cam.Open()

    try:
        cam.PixelFormat.SetValue(pixel_format)
    except Exception:
        try:
            cam.PixelFormat.Value = pixel_format
        except Exception:
            print(f"[warn] could not set PixelFormat={pixel_format}, using current camera setting")

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return cam, converter


def grab_frame(cam, converter, num_discard=5):
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # 最初の数枚を捨てる
    for _ in range(num_discard):
        res = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            _ = converter.Convert(res).GetArray()
        res.Release()

    res = cam.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        res.Release()
        raise RuntimeError("camera capture failed")
    arr = converter.Convert(res).GetArray()
    res.Release()
    cam.StopGrabbing()
    return arr


def refresh_click_image(gray):
    global img_click, img_click_base, clicked
    if len(gray.shape) == 2:
        img_click_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        img_click_base = gray.copy()
    img_click = img_click_base.copy()

    # 既存クリック点を描き直す
    for i, (x, y) in enumerate(clicked, start=1):
        cv2.circle(img_click, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_click, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def on_mouse(event, x, y, flags, param):
    global clicked, img_click
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x, y))
        cv2.circle(img_click, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_click, str(len(clicked)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    global clicked

    os.makedirs(OUT_DIR, exist_ok=True)

    pattern, dst_pts = make_pattern(DISPLAY_W, DISPLAY_H, MARGIN)
    show_pattern_on_monitor(pattern, MONITOR_INDEX)
    time.sleep(0.5)

    cam, converter = open_basler(CAMERA_INDEX, PIXEL_FORMAT)

    try:
        frame = grab_frame(cam, converter)
    finally:
        cam.Close()

    cv2.imwrite(OUT_CAPTURE_PATH, frame)
    print(f"[saved] {OUT_CAPTURE_PATH}")

    clicked = []
    refresh_click_image(frame)

    cv2.namedWindow(WIN_CLICK, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_CLICK, on_mouse)

    print("Click 4 points in this order: TL -> TR -> BR -> BL")
    print("Keys: s=save  r=reset points  ESC=quit")

    while True:
        cv2.imshow(WIN_CLICK, img_click)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("cancelled")
            break
        elif key == ord("r"):
            clicked = []
            refresh_click_image(frame)
            print("points reset")
        elif key == ord("s"):
            if len(clicked) != 4:
                print("need exactly 4 points")
                continue

            src_pts = np.array(clicked, dtype=np.float32)

            # 4点なので perspective transform で十分
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            np.save(OUT_H_PATH, H)
            print(f"[saved] {OUT_H_PATH}")
            print(H)

            warped = cv2.warpPerspective(frame, H, (DISPLAY_W, DISPLAY_H))
            cv2.imwrite(OUT_WARPED_PATH, warped)
            print(f"[saved] {OUT_WARPED_PATH}")

            cv2.imshow("warped_preview", warped)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()