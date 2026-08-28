import subprocess
import threading
import time
import cv2
import numpy as np

from config import (CAP_W, CAP_H, CAP_FPS, CAP_DEV_FPS, PROC_W, PROC_H,
                    CAM_CTRLS)


class CameraCSI:
    def __init__(self, index=0):
        from picamera2 import Picamera2
        import libcamera
        infos = Picamera2.global_camera_info()
        csi = [c["Num"] for c in infos if "usb" not in c.get("Id", "")]
        if not csi:
            raise RuntimeError("no CSI camera")
        self.cam = Picamera2(csi[min(index, len(csi) - 1)])
        ctrls = {"FrameRate": float(CAP_FPS)}
        if "AfMode" in self.cam.camera_controls:
            ctrls["AfMode"] = libcamera.controls.AfModeEnum.Manual
            ctrls["LensPosition"] = 0.0
        cfg = self.cam.create_video_configuration(
            main={"size": (CAP_W, CAP_H), "format": "RGB888"},
            lores={"size": (PROC_W, PROC_H), "format": "YUV420"},
            controls=ctrls,
            buffer_count=4)
        self.cam.configure(cfg)
        self.cam.start()

    def read(self):
        req = self.cam.capture_request()
        try:
            main = req.make_array("main")
            lores = req.make_array("lores")
            md = req.get_metadata()
        finally:
            req.release()
        t = md.get("SensorTimestamp", 0) * 1e-9 or time.monotonic()
        proc_bgr = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
        gray = np.ascontiguousarray(lores[:PROC_H, :PROC_W])
        return main, proc_bgr, gray, t

    def release(self):
        self.cam.stop()
        self.cam.close()


class CameraV4L2:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAP_DEV_FPS)
        if not self.cap.isOpened():
            raise RuntimeError("V4L2 open failed")
        self._skip = max(1, round(CAP_DEV_FPS / CAP_FPS))
        for c in [c.strip() for c in CAM_CTRLS.split(",") if c.strip()]:
            subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}", "-c", c],
                           capture_output=True)
        got = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._cond = threading.Condition()
        self._frame, self._t, self._seq = None, 0.0, 0
        self._run = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        i = 0
        while self._run:
            if not self.cap.grab():
                continue
            i += 1
            if i % self._skip:
                continue
            ok, f = self.cap.retrieve()
            if ok:
                with self._cond:
                    self._frame, self._t = f, time.monotonic()
                    self._seq += 1
                    self._cond.notify_all()

    def read(self):
        with self._cond:
            seq0 = self._seq
            while self._seq == seq0:
                self._cond.wait(0.5)
            f, t = self._frame, self._t
        if f.shape[1] != CAP_W or f.shape[0] != CAP_H:
            f = cv2.resize(f, (CAP_W, CAP_H))
        proc_bgr = cv2.resize(f, (PROC_W, PROC_H))
        gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
        return f, proc_bgr, gray, t

    def release(self):
        self._run = False
        self.cap.release()


def Camera(index=0):
    try:
        return CameraCSI(index)
    except Exception as e:
        return CameraV4L2(index)
