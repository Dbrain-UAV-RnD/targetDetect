#!/usr/bin/env python3

import cv2
import numpy as np
import collections
import math
import os
import signal
import socket
import threading
import time
import json
import queue
import struct
import termios

CAMERA_INDEX = 0

CAP_W = int(os.environ.get("CAP_W", "1920"))
CAP_H = int(os.environ.get("CAP_H", "1080"))
OUT_W = int(os.environ.get("OUT_W", "1920"))
OUT_H = int(os.environ.get("OUT_H", "1080"))
FPS          = int(os.environ.get("FPS", "60"))
ROTATION     = 0

SENSOR_W = int(os.environ.get("SENSOR_W", "1920"))
SENSOR_H = int(os.environ.get("SENSOR_H", "1080"))

CAM_BACKEND  = os.environ.get("CAM_BACKEND", "auto")
V4L2_FOURCC  = os.environ.get("V4L2_FOURCC", "MJPG")
V4L2_BUFFERS = int(os.environ.get("V4L2_BUFFERS", "3"))
V4L2_TIMEOUT = float(os.environ.get("V4L2_TIMEOUT", "2.0"))

PROC_W, PROC_H = 480, 270
PROC_SCALE = PROC_W / float(CAP_W)

CV_THREADS = int(os.environ.get("CV_THREADS", "2"))
if CV_THREADS > 0:
    cv2.setNumThreads(CV_THREADS)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

STAB_ENABLE   = os.environ.get("STAB", "1") not in ("0", "false", "no")
STAB_W        = int(os.environ.get("STAB_W", "240"))
STAB_H        = int(os.environ.get("STAB_H", str(max(2, int(round(
                    STAB_W * CAP_H / float(CAP_W) / 2.0)) * 2))))
STAB_ZOOM     = float(os.environ.get("STAB_ZOOM", "1.15"))
STAB_FREE     = os.environ.get("STAB_FREE", "0") not in ("0", "false", "no")
STAB_FREE_PX  = float(os.environ.get("STAB_FREE_PX", "200"))
STAB_TAU      = float(os.environ.get("STAB_TAU", "0.60"))
STAB_TAU_MIN  = float(os.environ.get("STAB_TAU_MIN", "0.10"))
STAB_TAU_MAX  = float(os.environ.get("STAB_TAU_MAX", "2.00"))
STAB_CORNERS  = int(os.environ.get("STAB_CORNERS", "60"))
STAB_QUALITY  = float(os.environ.get("STAB_QUALITY", "0.01"))
STAB_MIN_DIST = 8
STAB_MIN_PTS  = 12
STAB_FB_ERR   = 1.0
STAB_HIST     = int(os.environ.get("STAB_HIST", "96"))
STAB_DEAD     = float(os.environ.get("STAB_DEAD", "1.0"))
STAB_DEAD_DEG = float(os.environ.get("STAB_DEAD_DEG", "0.03"))
STAB_WALL     = float(os.environ.get("STAB_WALL", "2.0"))
STAB_STEP_MAX = float(os.environ.get("STAB_STEP_MAX", "0"))
STAB_DC_TAU   = float(os.environ.get("STAB_DC_TAU", "0"))
STAB_RATE     = float(os.environ.get("STAB_RATE", str(FPS)))
STAB_RS_MS    = float(os.environ.get("STAB_RS_MS", "0.0"))
STAB_RS_TAU   = float(os.environ.get("STAB_RS_TAU", "0.08"))
STAB_KX       = CAP_W / float(STAB_W)
STAB_KY       = CAP_H / float(STAB_H)
STAB_LK = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

MIN_ZOOM     = 1.0
DEFAULT_ZOOM = 1.0
MAX_ZOOM     = float(os.environ.get("MAX_ZOOM", "5.0"))
REAL_ZOOM    = CAP_W / OUT_W
CAP_K        = CAP_W / 1920.0

CAM_HFOV_DEG = float(os.environ.get("CAM_HFOV_DEG", "35.5"))
CAM_VFOV_DEG = float(os.environ.get("CAM_VFOV_DEG", "20.41"))
CAM_HFOV_TAN = math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
CAM_VFOV_TAN = math.tan(math.radians(CAM_VFOV_DEG) / 2.0)
ROTATE_HFOV_DEG = 35.5
ROTATE_VFOV_DEG = 20.41
PAN_STEP     = 100 * CAP_K
ZOOM_STEP    = 0.1
TAU_PAN    = float(os.environ.get("TAU_PAN",    "0.205"))
TAU_ZOOM   = float(os.environ.get("TAU_ZOOM",   "0.205"))
TAU_FOLLOW = float(os.environ.get("TAU_FOLLOW", "0.065"))
TAU_VEL    = float(os.environ.get("TAU_VEL",    "0.093"))
LEAD_TIME  = float(os.environ.get("LEAD_TIME", "0.05"))
DT_MAX = 0.25
FOLLOW_LEAD_MAX  = 200 * CAP_K

SWITCH_SETTLE_FRAMES = 10

TRK_ROI_SIZE    = 40
TRK_MAX_CORNERS = 12
TRK_QUALITY     = 0.01
TRK_MIN_DIST    = 4
TRK_MIN_POINTS  = 3
TRK_FB_MAX_ERR  = 1.0

LFC_HEF   = "/home/gimbal/models/lightfc_backbone.hef"
LFC_HEAD  = "/home/gimbal/models/lightfc_head.onnx"
LFC_HEAD_FRONT = "/home/gimbal/models/split/A_front.onnx"
LFC_HEAD_HEF   = "/home/gimbal/models/lightfc_head_back.hef"
LFC_HEAD_HAILO = os.environ.get("LFC_HEAD_HAILO", "1") not in ("0", "false", "no")
LFC_TEMPLATE, LFC_TEMPLATE_FACTOR = 128, 2.0
LFC_SEARCH,   LFC_SEARCH_FACTOR   = 256, 4.0
LFC_STRIDE    = 16
LFC_HEAD_THREADS = int(os.environ.get("LFC_THREADS", "2"))
LFC_MAX_RATE = float(os.environ.get("LFC_RATE", str(FPS)))
LFC_SCORE_MIN = 0.60
LFC_SIZE_ALPHA = float(os.environ.get("LFC_SIZE_ALPHA", "0.0"))
LFC_TPL_ALPHA = float(os.environ.get("LFC_TPL_ALPHA", "0.0"))
LFC_TPL_MIN   = float(os.environ.get("LFC_TPL_MIN", "0.75"))
LFC_MAX_MISS_SEC = float(os.environ.get("LFC_MISS_SEC", "5.6"))
ORT_SPINNING = os.environ.get("ORT_SPINNING", "0") not in ("0", "false", "no")
PRED_ENABLE = os.environ.get("PRED_ENABLE", "1") not in ("0", "false", "no")
PRED_MAX_AGE = float(os.environ.get("PRED_MAX_AGE", "0.15"))
LFC_DUMP_DIR = os.environ.get("LFC_DUMP_DIR", "")
STATUS_FILE = os.environ.get("STATUS_FILE", "")

REC_ENABLE  = os.environ.get("REC_ENABLE", "1") not in ("0", "false", "no")
REC_MOUNT   = os.environ.get("REC_MOUNT", "/mnt/usb")
REC_DIR     = os.environ.get("REC_DIR", os.path.join(REC_MOUNT, "snap"))
REC_JPEG_Q  = int(os.environ.get("REC_JPEG_Q", "95"))
REC_MIN_MB  = int(os.environ.get("REC_MIN_MB", "128"))
REC_QUEUE   = int(os.environ.get("REC_QUEUE", "4"))
STATUS_EVERY = int(os.environ.get("STATUS_EVERY", str(FPS)))
LFC_DUMP_N   = int(os.environ.get("LFC_DUMP_N", "256"))
LFC_DUMP_EVERY = int(os.environ.get("LFC_DUMP_EVERY", "3"))
LFC_INIT_BOX  = 90 * CAP_K
LFC_STAB_DEAD = float(os.environ.get("LFC_STAB_DEAD", "4.0")) * CAP_K
LFC_STAB_SPAN = float(os.environ.get("LFC_STAB_SPAN", "14.0")) * CAP_K
LFC_STAB_MIN  = float(os.environ.get("LFC_STAB_MIN",  "0.12"))
LFC_SCALE_LR     = float(os.environ.get("LFC_SCALE_LR",     "0.40"))
LFC_SCALE_RATE   = float(os.environ.get("LFC_SCALE_RATE",   "1.05"))
LFC_SCALE_HOLD   = float(os.environ.get("LFC_SCALE_HOLD",   "1.005"))
LFC_SCALE_SCORE  = float(os.environ.get("LFC_SCALE_SCORE",  "0.70"))
LFC_SCALE_MIN_PX = float(os.environ.get("LFC_SCALE_MIN_PX", "16")) * CAP_K
LFC_SCALE_MAX_PX = float(os.environ.get("LFC_SCALE_MAX_PX", "0.75")) * CAP_H
BOX_DEAD     = float(os.environ.get("BOX_DEAD", "8.0")) * CAP_K
BOX_SPAN     = float(os.environ.get("BOX_SPAN", "12.0")) * CAP_K
BOX_MIN      = float(os.environ.get("BOX_MIN", "0.15"))
BOX_MAX_RATE = float(os.environ.get("BOX_MAX_RATE", "3000.0")) * CAP_K
BOX_HOLD_LOW = os.environ.get("BOX_HOLD_LOW", "1") not in ("0", "false", "no")

FLOW_ENABLE   = os.environ.get("FLOW", "0") not in ("0", "false", "no")
FLOW_MAX_PTS  = int(os.environ.get("FLOW_PTS", "24"))
FLOW_MIN_PTS  = int(os.environ.get("FLOW_MIN_PTS", "6"))
FLOW_QUALITY  = float(os.environ.get("FLOW_QUALITY", "0.01"))
FLOW_MIN_DIST = max(3.0, 6.0 * CAP_K)
FLOW_FB_ERR   = float(os.environ.get("FLOW_FB_ERR", "1.0")) * CAP_K
FLOW_PAD      = float(os.environ.get("FLOW_PAD", "2.0"))
FLOW_MASK     = 0.85
FLOW_CROP_MAX = int(os.environ.get("FLOW_CROP_MAX", "320"))
FLOW_MAX_PX_S = float(os.environ.get("FLOW_MAX_PX_S", "3000.0")) * CAP_K
FLOW_ANCHOR_W = float(os.environ.get("FLOW_ANCHOR_W", "0.50"))
FLOW_RESEED   = float(os.environ.get("FLOW_RESEED", "0.25"))
FLOW_HIST     = int(os.environ.get("FLOW_HIST", "96"))
LFC_BIG_FRAC    = float(os.environ.get("LFC_BIG_FRAC", "0.25"))
LFC_BIG_HOLD    = os.environ.get("LFC_BIG_HOLD", "1") not in ("0", "false", "no")
FLOW_SCALE_RATE = float(os.environ.get("FLOW_SCALE_RATE", "1.02"))

LOST_BOX      = os.environ.get("LOST_BOX", "1") not in ("0", "false", "no")
LOST_BOX_SEC  = float(os.environ.get("LOST_BOX_SEC", "0"))
FLOW_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

LFC_IN_Z  = "lightfc_backbone_scope1/input_layer1"
LFC_IN_X  = "lightfc_backbone_scope2/input_layer2"
LFC_OUT_Z = "lightfc_backbone_scope1/conv26"
LFC_OUT_X = "lightfc_backbone_scope2/conv52"


RTSP_PORT    = int(os.environ.get("RTSP_PORT", "554"))
RTSP_PATH    = "/video0"
RTSP_QUEUE   = 3
RTSP_BITRATE = int(os.environ.get("RTSP_BITRATE", "1200"))
RTSP_PRESET  = os.environ.get("RTSP_PRESET", "veryfast")
RTSP_CODEC   = os.environ.get("RTSP_CODEC", "h264").lower()
RTSP_X265_OPTS = os.environ.get(
    "RTSP_X265_OPTS",
    "no-rect=1:no-amp=1:wpp=1:pmode=1:pme=1:frame-threads=4:rd=1:me=0:subme=0")
RTSP_FPS     = int(os.environ.get("RTSP_FPS", "30"))
RTSP_GOP     = int(os.environ.get("RTSP_GOP", str(RTSP_FPS * 2)))
RTSP_VBV     = int(os.environ.get("RTSP_VBV", "300"))
RTSP_INTRA_REFRESH = os.environ.get("RTSP_INTRA_REFRESH", "1") not in ("0", "false", "no")

GCS_UDP_PORT = 37260
GCS_HEADER1  = 0x55
GCS_HEADER2  = 0x66
GCS_CMD_OFFSET     = 7
GCS_PAYLOAD_OFFSET = 8

CMD_CAM_HEARTBEAT   = 0
CMD_AI_MODE         = 4
CMD_GIMBAL_ZOOM     = 5
CMD_TRACK_ACTION    = 6
CMD_GIMBAL_ROTATE   = 7
CMD_GIMBAL_CENTER   = 8
CMD_SET_GAIN        = 10
CMD_SET_OSD_DISPLAY = 13
CMD_TEST_DIGITAL_ZOOM  = 20
CMD_TEST_OPTICAL_FOCUS = 21
CMD_TEST_ZOOM_RAW      = 22
CMD_SET_CSRT_PARAM  = 30
CMD_STABILIZER_MODE = 31
CMD_STABILIZER_ALPHA = 32
CMD_DIGITAL_TILT    = 33

GCS_REF_W, GCS_REF_H = 1920, 1080
GCS_ROTATE_FULL_SCALE = 100.0
GCS_ROTATE_STEP_PX    = 20.0
GCS_ZOOM_RAW_MAX      = 0x4000
GCS_ZOOM_RATE         = float(os.environ.get("GCS_ZOOM_RATE", "2.0"))
GCS_ZOOM_TIMEOUT      = float(os.environ.get("GCS_ZOOM_TIMEOUT", "0.5"))
GCS_STAB_RESET        = 0xFF

FCC_TX_HEADER1, FCC_TX_HEADER2 = 0xBB, 0x88
FCC_RX_HEADER1, FCC_RX_HEADER2 = 0xBB, 0x99
FCC_TX_FMT  = "<BBBffBffbBhhhhhh10bB"
FCC_TX_SIZE = struct.calcsize(FCC_TX_FMT)
FCC_RX_FMT  = "<BBfBffddffffffff32sB"
FCC_RX_SIZE = struct.calcsize(FCC_RX_FMT)

FCC_PORT = os.environ.get("FCC_PORT", "/dev/ttyAMA3")
FCC_BAUD = int(os.environ.get("FCC_BAUD", "115200"))
FCC_HZ   = float(os.environ.get("FCC_HZ", "50"))
FCC_RETRY = float(os.environ.get("FCC_RETRY", "2.0"))
assert FCC_TX_SIZE == 45, FCC_TX_SIZE
assert FCC_RX_SIZE == 96, FCC_RX_SIZE

OUT_ASPECT = OUT_W / OUT_H


def frame_jitter(samples):
    if len(samples) < 10:
        return None
    dts = sorted(x[0] for x in samples)
    reads = sorted(x[1] for x in samples)
    sens = sorted(x[2] for x in samples if x[2] > 0)
    q = lambda a, f: round(a[min(len(a) - 1, int(len(a) * f))], 1)
    out = {"n": len(samples),
           "dt_med": q(dts, .5), "dt_p95": q(dts, .95), "dt_max": round(dts[-1], 1),
           "read_med": q(reads, .5), "read_max": round(reads[-1], 1),
           "work_med": round(q(dts, .5) - q(reads, .5), 1),
           "loop_over_80ms": sum(1 for x in samples if x[0] > 80.0)}
    if sens:
        out.update({"sensor_med": q(sens, .5), "sensor_p95": q(sens, .95),
                    "sensor_max": round(sens[-1], 1),
                    "sensor_drops": sum(1 for v in sens if v > 100.0)})
    return out


def stab_margin():
    stab = STAB_REF.get("stab")
    if STAB_FREE or stab is None or not stab.enabled:
        return 1.0
    return STAB_ZOOM


def xform_box(m, box):
    x, y, w, h = box
    cx, cy = apply_pt(m, x + 0.5 * w, y + 0.5 * h)
    sc = math.hypot(m[0, 0], m[1, 0])
    w *= sc
    h *= sc
    return (cx - 0.5 * w, cy - 0.5 * h, w, h)


def box_smooth(prev, new, dt):
    if prev is None:
        return new
    px, py, pw, ph = prev
    nx, ny, nw, nh = new
    pcx, pcy = px + 0.5 * pw, py + 0.5 * ph
    ncx, ncy = nx + 0.5 * nw, ny + 0.5 * nh
    dx, dy = ncx - pcx, ncy - pcy
    d = math.hypot(dx, dy)
    lim = BOX_MAX_RATE * max(1e-3, min(DT_MAX, dt))
    if d > lim > 0.0:
        k = lim / d
        dx, dy, d = dx * k, dy * k, lim
    if d <= BOX_DEAD:
        a = 0.0
    else:
        a = min(1.0, max(BOX_MIN, (d - BOX_DEAD) / max(1e-6, BOX_SPAN)))
    w = pw + a * (nw - pw)
    h = ph + a * (nh - ph)
    return (pcx + a * dx - 0.5 * w, pcy + a * dy - 0.5 * h, w, h)


def _fit_free(m):
    return True


def crop_fits(cw3, m):
    inv = cv2.invertAffineTransform((cw3 @ np.vstack([m, (0.0, 0.0, 1.0)]))[:2].copy())
    for px, py in ((0.0, 0.0), (OUT_W, 0.0), (0.0, OUT_H), (OUT_W, OUT_H)):
        qx, qy = apply_pt(inv, px, py)
        if qx < 1.5 or qy < 1.5 or qx > CAP_W - 1.5 or qy > CAP_H - 1.5:
            return False
    return True


def base_window():
    w = min(CAP_W, CAP_H * OUT_ASPECT)
    return w, w / OUT_ASPECT


def apply_pt(m, x, y):
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2],
            m[1, 0] * x + m[1, 1] * y + m[1, 2])


class CameraCSI:

    def __init__(self, index, width, height, fps):
        from picamera2 import Picamera2
        import libcamera

        self.width = width
        self.height = height
        self.picam2 = Picamera2(index)

        tf = libcamera.Transform()
        if ROTATION == 180:
            tf = libcamera.Transform(hflip=1, vflip=1)

        ctrls = {"FrameRate": float(fps)}
        if "AfMode" in self.picam2.camera_controls:
            ctrls["AfMode"] = libcamera.controls.AfModeEnum.Manual
            ctrls["LensPosition"] = 0.0

        self.config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            lores={"size": (PROC_W, PROC_H), "format": "YUV420"},
            sensor={"output_size": (SENSOR_W, SENSOR_H), "bit_depth": 10},
            controls=ctrls,
            transform=tf,
            buffer_count=4,
        )
        self.picam2.configure(self.config)
        self.picam2.start()

        self.crop_max = tuple(self.picam2.camera_properties["ScalerCropMaximum"])
        self.crop_now = self.crop_max
        self.sensor_dt_ms = 0.0
        self._prev_ts = None

    def read(self):
        try:
            req = self.picam2.capture_request()
        except Exception:
            return None, None, None
        try:
            bgr = req.make_array("main")
            lo = req.make_array("lores")
            md = req.get_metadata()
        finally:
            req.release()
        proc = np.ascontiguousarray(lo[:PROC_H, :PROC_W])
        crop = md.get("ScalerCrop")
        if crop is not None:
            self.crop_now = tuple(crop)
        ts = md.get("SensorTimestamp")
        if ts is not None:
            if self._prev_ts is not None:
                self.sensor_dt_ms = (ts - self._prev_ts) / 1e6
            self._prev_ts = ts
        return bgr, proc, self.crop_now

    def release(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass


class CameraV4L2:

    def __init__(self, index, width, height, fps):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("v4l2 open failed: %s" % index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*V4L2_FOURCC))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, V4L2_BUFFERS)
        self.crop_now = (0, 0, width, height)
        self.sensor_dt_ms = 0.0
        self._prev_ts = None
        self._cv = threading.Condition()
        self._frame = None
        self._seq = 0
        self._taken = 0
        self._alive = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._alive:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                with self._cv:
                    self._alive = False
                    self._cv.notify_all()
                return
            if ROTATION == 180:
                bgr = cv2.flip(bgr, -1)
            now = time.monotonic()
            with self._cv:
                if self._prev_ts is not None:
                    self.sensor_dt_ms = (now - self._prev_ts) * 1e3
                self._prev_ts = now
                self._frame = bgr
                self._seq += 1
                self._cv.notify_all()

    def read(self):
        with self._cv:
            while self._alive and self._seq == self._taken:
                if not self._cv.wait(timeout=V4L2_TIMEOUT):
                    return None, None, None
            if not self._alive or self._frame is None:
                return None, None, None
            self._taken = self._seq
            bgr = self._frame
        proc = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                          (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        return bgr, proc, self.crop_now

    def release(self):
        self._alive = False
        try:
            self.cap.release()
        except Exception:
            pass


def Camera(index, width, height, fps):
    if CAM_BACKEND == "v4l2":
        return CameraV4L2(index, width, height, fps)
    if CAM_BACKEND == "csi":
        return CameraCSI(index, width, height, fps)
    try:
        return CameraCSI(index, width, height, fps)
    except Exception:
        return CameraV4L2(index, width, height, fps)


class LightFCTracker:

    def __init__(self):
        from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, FormatType,
                                    InferVStreams, HailoSchedulingAlgorithm)
        import onnxruntime as ort

        self.hailo_head = LFC_HEAD_HAILO and os.path.exists(LFC_HEAD_HEF)
        if self.hailo_head:
            vp = VDevice.create_params()
            vp.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self.vdev = VDevice(vp)
        else:
            self.vdev = VDevice()

        hef = HEF(LFC_HEF)
        self.ng = self.vdev.configure(
            hef, ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe))[0]
        self.pipe = InferVStreams(
            self.ng,
            InputVStreamParams.make(self.ng, format_type=FormatType.UINT8),
            OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32))
        self.pipe.__enter__()
        self.act = None
        if not self.hailo_head:
            self.act = self.ng.activate()
            self.act.__enter__()

        so = ort.SessionOptions()
        so.intra_op_num_threads = LFC_HEAD_THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not ORT_SPINNING:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        if self.hailo_head:
            self.head = ort.InferenceSession(LFC_HEAD_FRONT, so,
                                             providers=["CPUExecutionProvider"])
            self.hn = [i.name for i in self.head.get_inputs()]
            self.fo = [o.name for o in self.head.get_outputs()]
            hef2 = HEF(LFC_HEAD_HEF)
            self.ng2 = self.vdev.configure(
                hef2, ConfigureParams.create_from_hef(
                    hef2, interface=HailoStreamInterface.PCIe))[0]
            self.pipe2 = InferVStreams(
                self.ng2,
                InputVStreamParams.make(self.ng2, format_type=FormatType.FLOAT32),
                OutputVStreamParams.make(self.ng2, format_type=FormatType.FLOAT32))
            self.pipe2.__enter__()
            self.hb_in = {v.shape[-1]: v.name
                          for v in hef2.get_input_vstream_infos()}
            self.hb_out = [v.name for v in hef2.get_output_vstream_infos()]
            self.hb_map = None
        else:
            self.head = ort.InferenceSession(LFC_HEAD, so,
                                             providers=["CPUExecutionProvider"])
            self.hn = [i.name for i in self.head.get_inputs()]

        n = LFC_SEARCH // LFC_STRIDE
        w1 = 0.5 * (1 - np.cos(2 * np.pi / (n + 1) * np.arange(1, n + 1)))
        self.hann = (w1.reshape(-1, 1) * w1.reshape(1, -1)).astype(np.float32)
        self.feat_sz = n

        self.box = None
        self.sbox = None
        self.z = None
        self.score = 0.0
        self.miss_t0 = None
        self.t_result = 0.0
        self.t_frame = 0.0
        self.hold = False
        self.ms = 0.0
        self.stage = collections.deque(maxlen=200)
        self.dump_n = 0
        self.dump_seen = 0
        if LFC_DUMP_DIR:
            os.makedirs(LFC_DUMP_DIR, exist_ok=True)
        self.cyc = collections.deque(maxlen=200)
        self._prev_cyc = None

        self.lock = threading.Lock()
        self.pending = None
        self.req = None
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def active(self):
        with self.lock:
            return self.box is not None

    def stop(self):
        with self.lock:
            self.box = None
            self.sbox = None
            self.z = None
            self.score = 0.0
            self.miss_t0 = None
            self.t_result = 0.0
            self.t_frame = 0.0
            self.hold = False
            self.req = None
            self.pending = None

    def submit(self, bgr, t):
        with self.lock:
            self.pending = (bgr, t)

    def request_start(self, bgr, box):
        with self.lock:
            self.req = (bgr, box)

    def stage_stats(self):
        with self.lock:
            rows = list(self.stage)
        if len(rows) < 10:
            return None
        q = lambda a, f: round(sorted(a)[min(len(a) - 1, int(len(a) * f))], 2)
        out = {"n": len(rows)}
        tot = 0.0
        for i, k in enumerate(("crop", "backbone", "transpose", "head")):
            col = [r[i] for r in rows]
            out[k] = [q(col, .5), q(col, .95)]
            tot += out[k][0]
        out["measured_sum_med"] = round(tot, 2)
        with self.lock:
            cyc = list(self.cyc)
        if cyc:
            out["cycle_med"] = q(cyc, .5)
            out["cycle_p95"] = q(cyc, .95)
            out["rate_hz"] = round(1000.0 / q(cyc, .5), 2) if q(cyc, .5) > 0 else 0.0
        out["cv_threads"] = cv2.getNumThreads()
        out["ort_spinning"] = ORT_SPINNING
        return out

    def snapshot(self):
        with self.lock:
            if self.sbox is None:
                return None, None, 0.0, 0.0
            x, y, w, h = self.sbox
            return self.sbox, (x + 0.5 * w, y + 0.5 * h), self.score, self.t_result

    def set_hold(self, on):
        with self.lock:
            self.hold = bool(on)

    def retarget(self, box):
        with self.lock:
            if self.box is None:
                return
            self.box = tuple(float(v) for v in box)
            self.sbox = self.box

    def raw_snapshot(self):
        with self.lock:
            if self.box is None:
                return None, 0.0, 0.0
            return self.box, self.score, self.t_frame

    @staticmethod
    def _stab(prev, new):
        if prev is None:
            return new
        px, py, pw, ph = prev
        nx, ny, nw, nh = new
        pcx, pcy = px + 0.5 * pw, py + 0.5 * ph
        ncx, ncy = nx + 0.5 * nw, ny + 0.5 * nh
        dx, dy = ncx - pcx, ncy - pcy
        d = math.hypot(dx, dy)
        if d <= LFC_STAB_DEAD:
            a = 0.0
        else:
            a = min(1.0, max(LFC_STAB_MIN,
                             (d - LFC_STAB_DEAD) / max(1e-6, LFC_STAB_SPAN)))
        w = pw + a * (nw - pw)
        h = ph + a * (nh - ph)
        return (pcx + a * dx - 0.5 * w, pcy + a * dy - 0.5 * h, w, h)

    def _loop(self):
        while True:
            with self.lock:
                req, self.req = self.req, None
                pend, self.pending = self.pending, None
                busy = self.box is not None
            frame, t_cap = pend if pend is not None else (None, 0.0)
            try:
                if req is not None:
                    if not self._start(req[0], req[1]):
                        pass
                    continue
                if not busy or frame is None:
                    time.sleep(0.003)
                    continue
                t0 = time.time()
                self._update(frame, t_cap)
                el = time.time() - t0
                with self.lock:
                    self.ms = el * 1000.0
                    if self._prev_cyc is not None:
                        self.cyc.append((t0 - self._prev_cyc) * 1e3)
                    self._prev_cyc = t0
                time.sleep(max(0.0, 1.0 / LFC_MAX_RATE - el))
            except Exception as e:
                self.stop()
                time.sleep(0.05)

    def _map_outputs(self, r):
        score = size = off = None
        for n in self.hb_out:
            a = r[n]
            if a.shape[-1] == 1:
                score = n
            elif float(a.min()) >= -1e-6:
                size = n
            else:
                off = n
        if None in (score, size, off):
            raise RuntimeError(self.hb_out)
        return (score, size, off)

    @staticmethod
    def _crop(img, box, factor, out_sz):
        x, y, w, h = box
        crop_sz = int(np.ceil(np.sqrt(max(w * h, 1.0)) * factor))
        cx, cy = x + 0.5 * w, y + 0.5 * h
        x1 = int(round(cx - crop_sz * 0.5))
        y1 = int(round(cy - crop_sz * 0.5))
        x2, y2 = x1 + crop_sz, y1 + crop_sz
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, 1.0
        patch = img[cy1:cy2, cx1:cx2]
        patch = cv2.copyMakeBorder(patch, cy1 - y1, y2 - cy2, cx1 - x1, x2 - cx2,
                                   cv2.BORDER_CONSTANT)
        return cv2.resize(patch, (out_sz, out_sz)), out_sz / crop_sz

    def _start(self, bgr, box):
        z, _ = self._crop(bgr, box, LFC_TEMPLATE_FACTOR, LFC_TEMPLATE)
        if z is None:
            return False
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        with self.lock:
            self.z = z[None]
            self.box = tuple(float(v) for v in box)
            self.sbox = self.box
            self.score = 1.0
            self.miss_t0 = None
            self.t_result = time.time()
            self.t_frame = self.t_result
        return True

    def _infer(self, bgr, cur_box, cur_z):
        _t0 = time.perf_counter()
        x_patch, rf = self._crop(bgr, cur_box, LFC_SEARCH_FACTOR, LFC_SEARCH)
        if x_patch is None:
            return None
        x_patch = cv2.cvtColor(x_patch, cv2.COLOR_BGR2RGB)
        _t1 = time.perf_counter()

        r = self.pipe.infer({LFC_IN_Z: cur_z, LFC_IN_X: x_patch[None]})
        _t2 = time.perf_counter()
        fz = np.transpose(r[LFC_OUT_Z], (0, 3, 1, 2)).copy()
        fx = np.transpose(r[LFC_OUT_X], (0, 3, 1, 2)).copy()
        _t3 = time.perf_counter()
        if self.hailo_head:
            corr = self.head.run(None, {self.hn[0]: fz, self.hn[1]: fx})[0]
            r2 = self.pipe2.infer({
                self.hb_in[corr.shape[1]]:
                    np.ascontiguousarray(corr.transpose(0, 2, 3, 1)),
                self.hb_in[fx.shape[1]]:
                    np.ascontiguousarray(fx.transpose(0, 2, 3, 1))})
            if self.hb_map is None:
                self.hb_map = self._map_outputs(r2)
            score, size, offset = (
                np.ascontiguousarray(r2[n].transpose(0, 3, 1, 2))
                for n in self.hb_map)
        else:
            score, size, offset = self.head.run(None, {self.hn[0]: fz, self.hn[1]: fx})
        _t4 = time.perf_counter()
        if LFC_DUMP_DIR and self.dump_n < LFC_DUMP_N:
            self.dump_seen += 1
            if self.dump_seen % LFC_DUMP_EVERY == 0:
                np.savez_compressed(
                    os.path.join(LFC_DUMP_DIR, "s%04d.npz" % self.dump_n),
                    fz=fz, fx=fx, score=score, size=size, offset=offset,
                    box=np.asarray(cur_box, dtype=np.float32),
                    rf=np.float32(rf))
                self.dump_n += 1
        with self.lock:
            self.stage.append(((_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3,
                               (_t3 - _t2) * 1e3, (_t4 - _t3) * 1e3))

        resp = score[0, 0] * self.hann
        iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
        peak = float(score[0, 0, iy, ix])

        ox, oy = offset[0, 0, iy, ix], offset[0, 1, iy, ix]
        bw, bh = size[0, 0, iy, ix], size[0, 1, iy, ix]
        cx = (ix + ox) / self.feat_sz * LFC_SEARCH / rf
        cy = (iy + oy) / self.feat_sz * LFC_SEARCH / rf
        w = bw * LFC_SEARCH / rf
        h = bh * LFC_SEARCH / rf

        px, py, pw, ph = cur_box
        raw_wh = (float(w), float(h))
        w = pw + LFC_SIZE_ALPHA * (w - pw)
        h = ph + LFC_SIZE_ALPHA * (h - ph)
        half = 0.5 * LFC_SEARCH / rf
        rcx = cx + (px + 0.5 * pw - half)
        rcy = cy + (py + 0.5 * ph - half)
        nx = min(max(rcx - 0.5 * w, -w + 2), CAP_W - 2)
        ny = min(max(rcy - 0.5 * h, -h + 2), CAP_H - 2)
        return (peak, (float(nx), float(ny), float(max(4.0, w)),
                       float(max(4.0, h))), raw_wh)

    def _update(self, bgr, t_cap):
        with self.lock:
            if self.box is None:
                return None
            cur_box, cur_z = self.box, self.z

        r0 = self._infer(bgr, cur_box, cur_z)
        if r0 is None:
            self.stop()
            return None
        peak, new_box, raw_wh = r0

        moving = False
        if LFC_SCALE_LR > 0.0 and peak >= LFC_SCALE_SCORE:
            pw, ph = cur_box[2], cur_box[3]
            hint = math.sqrt(raw_wh[0] * raw_wh[1] / (pw * ph)) if pw * ph > 0 else 1.0
            f = min(LFC_SCALE_RATE, max(1.0 / LFC_SCALE_RATE,
                                        1.0 + LFC_SCALE_LR * (hint - 1.0)))
            k = math.sqrt(pw * ph) * f
            if LFC_SCALE_MIN_PX <= k <= LFC_SCALE_MAX_PX:
                bx, by, bw2, bh2 = new_box
                nw, nh = pw * f, ph * f
                new_box = (bx + 0.5 * bw2 - 0.5 * nw,
                           by + 0.5 * bh2 - 0.5 * nh, nw, nh)
                moving = f > LFC_SCALE_HOLD or f < 1.0 / LFC_SCALE_HOLD

        with self.lock:
            self.score = peak

        if peak < LFC_SCORE_MIN:
            with self.lock:
                if self.miss_t0 is None:
                    self.miss_t0 = time.time()
                over = (time.time() - self.miss_t0) > LFC_MAX_MISS_SEC
                if over and self.hold:
                    over = False
            if over:
                self.stop()
            return None
        with self.lock:
            self.miss_t0 = None
            if self.box is not None:
                self.box = new_box
                self.sbox = self._stab(self.sbox, new_box)
                self.t_result = time.time()
                self.t_frame = t_cap or self.t_result

        if LFC_TPL_ALPHA > 0.0 and peak >= LFC_TPL_MIN and not moving:
            nz, _ = self._crop(bgr, new_box, LFC_TEMPLATE_FACTOR, LFC_TEMPLATE)
            if nz is not None:
                nz = cv2.cvtColor(nz, cv2.COLOR_BGR2RGB).astype(np.float32)
                with self.lock:
                    if self.z is not None:
                        self.z = np.clip(
                            self.z.astype(np.float32) * (1.0 - LFC_TPL_ALPHA)
                            + nz[None] * LFC_TPL_ALPHA, 0.0, 255.0).astype(np.uint8)
        return None


class FlowTracker:

    def __init__(self):
        self.w = self.h = 0.0
        self.reseeds = 0
        self.anchors = 0
        self.stop()

    def stop(self):
        self.pts = None
        self.prev = None
        self.rect = None
        self.sx = self.sy = 1.0
        self.cx = self.cy = 0.0
        self.t = 0.0
        self.hist = collections.deque(maxlen=FLOW_HIST)
        self.last_m = None
        self.own_size = False
        self.pts_n = 0
        self.fail = 0
        self.corr = 0.0
        self.ms = 0.0

    @property
    def active(self):
        return self.pts is not None

    def set_size(self, w, h):
        self.w, self.h = float(w), float(h)

    def set_own_size(self, on):
        self.own_size = bool(on)

    def box(self):
        return (self.cx - 0.5 * self.w, self.cy - 0.5 * self.h, self.w, self.h)

    @staticmethod
    def _rect_for(box):
        x, y, w, h = box
        cx, cy = x + 0.5 * w, y + 0.5 * h
        r = 0.5 * FLOW_PAD * max(w, h)
        x0 = int(max(0, math.floor(cx - r)))
        y0 = int(max(0, math.floor(cy - r)))
        x1 = int(min(CAP_W, math.ceil(cx + r)))
        y1 = int(min(CAP_H, math.ceil(cy + r)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def _gray(bgr, rect):
        x0, y0, x1, y1 = rect
        g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        w, h = x1 - x0, y1 - y0
        s = min(1.0, FLOW_CROP_MAX / float(max(w, h)))
        if s < 1.0:
            ow, oh = max(8, int(round(w * s))), max(8, int(round(h * s)))
            g = cv2.resize(g, (ow, oh), interpolation=cv2.INTER_AREA)
            return g, ow / float(w), oh / float(h)
        return g, 1.0, 1.0

    def _to_crop(self, p):
        q = np.empty_like(p)
        q[:, 0, 0] = (p[:, 0, 0] - self.rect[0]) * self.sx
        q[:, 0, 1] = (p[:, 0, 1] - self.rect[1]) * self.sy
        return q

    def _to_cap(self, q):
        p = np.empty_like(q)
        p[:, 0, 0] = self.rect[0] + q[:, 0, 0] / self.sx
        p[:, 0, 1] = self.rect[1] + q[:, 0, 1] / self.sy
        return p

    def _seed(self, bgr, box):
        rect = self._rect_for(box)
        if rect is None:
            return False
        g, sx, sy = self._gray(bgr, rect)
        x, y, w, h = box
        cx, cy = x + 0.5 * w, y + 0.5 * h
        mw, mh = 0.5 * w * FLOW_MASK, 0.5 * h * FLOW_MASK
        mask = np.zeros(g.shape, np.uint8)
        mx0 = int(max(0, round((cx - mw - rect[0]) * sx)))
        my0 = int(max(0, round((cy - mh - rect[1]) * sy)))
        mx1 = int(min(g.shape[1], round((cx + mw - rect[0]) * sx)))
        my1 = int(min(g.shape[0], round((cy + mh - rect[1]) * sy)))
        if mx1 - mx0 < 8 or my1 - my0 < 8:
            return False
        mask[my0:my1, mx0:mx1] = 255
        p = cv2.goodFeaturesToTrack(
            g, maxCorners=FLOW_MAX_PTS, qualityLevel=FLOW_QUALITY,
            minDistance=max(2.0, FLOW_MIN_DIST * sx), blockSize=7, mask=mask)
        if p is None or len(p) < FLOW_MIN_PTS:
            return False
        self.rect, self.sx, self.sy = rect, sx, sy
        self.prev = g
        self.pts = p.astype(np.float32)
        self.pts_n = int(len(p))
        return True

    def _recenter(self, bgr, pts_cap):
        rect = self._rect_for(self.box())
        if rect is None:
            return False
        g, sx, sy = self._gray(bgr, rect)
        q = np.empty_like(pts_cap)
        q[:, 0, 0] = (pts_cap[:, 0, 0] - rect[0]) * sx
        q[:, 0, 1] = (pts_cap[:, 0, 1] - rect[1]) * sy
        inside = ((q[:, 0, 0] > 2) & (q[:, 0, 0] < g.shape[1] - 3) &
                  (q[:, 0, 1] > 2) & (q[:, 0, 1] < g.shape[0] - 3))
        if int(inside.sum()) < FLOW_MIN_PTS:
            return False
        self.rect, self.sx, self.sy = rect, sx, sy
        self.prev = g
        self.pts = np.ascontiguousarray(q[inside])
        self.pts_n = int(inside.sum())
        return True

    def _lost(self):
        self.fail += 1
        self.pts = None
        return None

    def start(self, bgr, box, t):
        self.stop()
        if bgr is None or not self._seed(bgr, box):
            self.pts = None
            return False
        self.set_size(box[2], box[3])
        self.cx = box[0] + 0.5 * box[2]
        self.cy = box[1] + 0.5 * box[3]
        self.t = t
        self.hist.append((t, self.cx, self.cy, np.eye(3)))
        return True

    def update(self, bgr, t):
        if self.pts is None or bgr is None:
            return None
        t0 = time.perf_counter()
        dt = max(1e-3, min(DT_MAX, t - self.t))
        g, _, _ = self._gray(bgr, self.rect)

        p0 = self.pts
        guess, flags = None, 0
        if self.last_m is not None:
            cap = self._to_cap(p0)
            pr = np.empty_like(cap)
            m = self.last_m
            pr[:, 0, 0] = m[0, 0] * cap[:, 0, 0] + m[0, 1] * cap[:, 0, 1] + m[0, 2]
            pr[:, 0, 1] = m[1, 0] * cap[:, 0, 0] + m[1, 1] * cap[:, 0, 1] + m[1, 2]
            guess = np.ascontiguousarray(self._to_crop(pr))
            flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, guess,
                                             flags=flags, **FLOW_LK)
        if p1 is None:
            return self._lost()
        ok = st.ravel() == 1
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, None, **FLOW_LK)
        if pb is not None:
            err = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) / max(1e-6, self.sx)
            ok &= (st2.ravel() == 1) & (err < FLOW_FB_ERR)
        a, b = p0[ok], p1[ok]
        self.pts_n = int(len(a))
        if self.pts_n < FLOW_MIN_PTS:
            return self._lost()

        m, _inl = cv2.estimateAffinePartial2D(
            self._to_cap(a), self._to_cap(b),
            method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if m is None:
            return self._lost()

        ncx, ncy = apply_pt(m, self.cx, self.cy)
        if math.hypot(ncx - self.cx, ncy - self.cy) > FLOW_MAX_PX_S * dt:
            return self._lost()

        if self.own_size and self.w > 0.0:
            sc = min(FLOW_SCALE_RATE,
                     max(1.0 / FLOW_SCALE_RATE, math.hypot(m[0, 0], m[1, 0])))
            k = math.sqrt(max(1e-6, self.w * self.h)) * sc
            if LFC_SCALE_MIN_PX <= k <= LFC_SCALE_MAX_PX:
                self.w *= sc
                self.h *= sc

        self.cx, self.cy = float(ncx), float(ncy)
        self.t = t
        self.last_m = m
        self.fail = 0
        self.hist.append((t, self.cx, self.cy, np.vstack([m, (0.0, 0.0, 1.0)])))

        b_cap = self._to_cap(b)
        self.prev, self.pts = g, b
        if not self._recenter(bgr, b_cap) or self.pts_n < 2 * FLOW_MIN_PTS:
            if self._seed(bgr, self.box()):
                self.reseeds += 1
            elif self.pts_n < FLOW_MIN_PTS:
                return self._lost()
        self.ms = (time.perf_counter() - t0) * 1e3
        return (self.cx, self.cy)

    def anchor(self, meas_cx, meas_cy, meas_t):
        if self.pts is None or not self.hist:
            return False
        base_ts = None
        for ts, cx, cy, _A in self.hist:
            if ts <= meas_t:
                base_ts, bx, by = ts, cx, cy
            else:
                break
        if base_ts is None:
            return False
        ex, ey = meas_cx - bx, meas_cy - by

        out = collections.deque(maxlen=FLOW_HIST)
        for ts, cx, cy, A in self.hist:
            if ts < base_ts:
                out.append((ts, cx, cy, A))
                continue
            if ts > base_ts:
                ex, ey = A[0, 0] * ex + A[0, 1] * ey, A[1, 0] * ex + A[1, 1] * ey
            out.append((ts, cx + FLOW_ANCHOR_W * ex, cy + FLOW_ANCHOR_W * ey, A))
        self.hist = out
        self.cx, self.cy = out[-1][1], out[-1][2]
        self.anchors += 1
        self.corr = math.hypot(ex, ey)
        return self.corr <= FLOW_RESEED * math.hypot(self.w, self.h)

    def status(self):
        return {"on": bool(self.pts is not None), "pts": self.pts_n,
                "fail": self.fail, "reseed": self.reseeds,
                "anchor": self.anchors, "corr": round(self.corr, 1),
                "own": bool(self.own_size),
                "wh": [round(self.w), round(self.h)],
                "ms": round(self.ms, 2)}


class TargetTracker:

    def __init__(self):
        self.pts = None
        self.center = None
        self.prev = None

    def start(self, proc_gray, x, y):
        h, w = proc_gray.shape
        half = TRK_ROI_SIZE // 2
        x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
        x1, y1 = min(w, int(x) + half), min(h, int(y) + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return False
        roi = proc_gray[y0:y1, x0:x1]
        p = cv2.goodFeaturesToTrack(
            roi, maxCorners=TRK_MAX_CORNERS, qualityLevel=TRK_QUALITY,
            minDistance=TRK_MIN_DIST, blockSize=5,
        )
        if p is None or len(p) < TRK_MIN_POINTS:
            return False
        p = p.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
        self.pts = p.reshape(-1, 1, 2).astype(np.float32)
        self.center = (float(x), float(y))
        self.prev = proc_gray
        return True

    def stop(self):
        self.pts = None
        self.center = None
        self.prev = None

    @property
    def active(self):
        return self.pts is not None

    def update(self, proc_gray):
        if self.pts is None:
            return None
        if self.prev is None:
            self.prev = proc_gray
            return self.center
        prev_img = self.prev
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, proc_gray, self.pts, None, **LK_PARAMS)
        self.prev = proc_gray
        if p1 is None:
            self.stop()
            return None
        st = st.reshape(-1).astype(bool)
        prev_good = self.pts.reshape(-1, 2)[st]
        cur_good = p1.reshape(-1, 2)[st]
        if len(cur_good) < TRK_MIN_POINTS:
            self.stop()
            return None

        pb, st2, _ = cv2.calcOpticalFlowPyrLK(
            proc_gray, prev_img,
            cur_good.reshape(-1, 1, 2).astype(np.float32), None, **LK_PARAMS)
        if pb is not None:
            err = np.linalg.norm(pb.reshape(-1, 2) - prev_good, axis=1)
            keep = err < TRK_FB_MAX_ERR
            if keep.sum() >= TRK_MIN_POINTS:
                prev_good, cur_good = prev_good[keep], cur_good[keep]
        d = np.median(cur_good - prev_good, axis=0)
        self.center = (self.center[0] + float(d[0]), self.center[1] + float(d[1]))
        self.pts = cur_good.reshape(-1, 1, 2).astype(np.float32)
        return self.center


class FrameStabilizer:

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enabled = STAB_ENABLE
        self.tau = STAB_TAU
        self.ms = 0.0
        self.hist = collections.deque(maxlen=STAB_HIST)
        self.reset()

    def request_reset(self):
        self._reset_req = True

    def reset(self):
        self._reset_req = False
        self.hist.clear()
        self.prev = None
        self.W = np.eye(3)
        self.warp = None
        self.pts = 0
        self.fail = 0
        self.sat = 0.0
        self.clock = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.dcx = 0.0
        self.dcy = 0.0

    def set_enabled(self, on):
        on = bool(on)
        if on and not self.enabled:
            self.reset()
        self.enabled = on

    def set_tau(self, tau):
        self.tau = max(STAB_TAU_MIN, min(STAB_TAU_MAX, float(tau)))

    def _gray(self, proc):
        g = proc if proc.ndim == 2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        if g.shape[1] != STAB_W or g.shape[0] != STAB_H:
            g = cv2.resize(g, (STAB_W, STAB_H), interpolation=cv2.INTER_AREA)
        return self.clahe.apply(g)

    @staticmethod
    def _track(prev, curr, p0):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **STAB_LK)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None, **STAB_LK)
        err = np.linalg.norm(p0 - pb, axis=2).ravel()
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (err < STAB_FB_ERR)
        return p0[ok], p1[ok]

    @staticmethod
    def _scale(W, s):
        B = np.eye(3)
        B[0, 0] = B[1, 1] = 1.0 + s * (W[0, 0] - 1.0)
        B[1, 0] = s * W[1, 0]
        B[0, 1] = -B[1, 0]
        B[0, 2] = s * W[0, 2]
        B[1, 2] = s * W[1, 2]
        return B

    @staticmethod
    def _to_capture(W):
        m = np.eye(3)
        m[0, 0] = W[0, 0]
        m[1, 1] = W[1, 1]
        m[0, 1] = W[0, 1] * STAB_KX / STAB_KY
        m[1, 0] = W[1, 0] * STAB_KY / STAB_KX
        m[0, 2] = W[0, 2] * STAB_KX
        m[1, 2] = W[1, 2] * STAB_KY
        return m

    def _rolling_shutter(self):
        if STAB_RS_MS <= 0.0:
            return None
        r = (STAB_RS_MS / 1000.0) / float(CAP_H)
        m = np.eye(3)
        m[0, 1] = -self.vx * r
        m[0, 2] = self.vx * r * CAP_H / 2.0
        m[1, 1] = 1.0 - self.vy * r
        m[1, 2] = self.vy * r * CAP_H / 2.0
        return m

    @staticmethod
    def _deadzone(W):
        tx, ty = W[0, 2] * STAB_KX, W[1, 2] * STAB_KY
        d = math.hypot(tx, ty)
        if d > 1e-9:
            f = max(0.0, d - STAB_DEAD) / d
            W[0, 2] *= f
            W[1, 2] *= f
        sc = math.hypot(W[0, 0], W[1, 0])
        ang = math.atan2(W[1, 0], W[0, 0])
        if abs(ang) > 1e-12:
            a2 = math.copysign(max(0.0, abs(ang) - math.radians(STAB_DEAD_DEG)), ang)
            W[0, 0] = W[1, 1] = sc * math.cos(a2)
            W[1, 0] = sc * math.sin(a2)
            W[0, 1] = -W[1, 0]
        return W

    def update(self, proc, fit, budget):
        if not self.enabled or proc is None:
            self.warp = None
            return None
        if self._reset_req:
            self.reset()
        t0 = time.perf_counter()
        now = time.monotonic()
        dt = min(DT_MAX, now - self.clock) if self.clock else 0.0
        self.clock = now

        g = self._gray(proc)
        if self.prev is None:
            self.prev = g
            self.warp = None
            return None


        m = None
        p0 = cv2.goodFeaturesToTrack(self.prev, maxCorners=STAB_CORNERS,
                                     qualityLevel=STAB_QUALITY,
                                     minDistance=STAB_MIN_DIST, blockSize=7)
        if p0 is not None and len(p0) >= STAB_MIN_PTS:
            a, b = self._track(self.prev, g, p0)
            self.pts = len(a)
            if len(a) >= STAB_MIN_PTS:
                m, _ = cv2.estimateAffinePartial2D(
                    a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        else:
            self.pts = 0
        self.prev = g

        if m is None:
            self.fail += 1
            step = np.eye(3)
        else:
            self.fail = 0
            step = np.vstack([m, (0.0, 0.0, 1.0)])

        if dt > 0.0:
            av = 1.0 - math.exp(-dt / STAB_RS_TAU)
            self.vx += av * (step[0, 2] * STAB_KX / dt - self.vx)
            self.vy += av * (step[1, 2] * STAB_KY / dt - self.vy)

        if STAB_DC_TAU > 0.0 and dt > 0.0:
            a_dc = 1.0 - math.exp(-dt / STAB_DC_TAU)
            self.dcx += a_dc * (step[0, 2] - self.dcx)
            self.dcy += a_dc * (step[1, 2] - self.dcy)
            step = step.copy()
            step[0, 2] -= self.dcx
            step[1, 2] -= self.dcy

        if STAB_STEP_MAX > 0.0:
            d = math.hypot(step[0, 2] * STAB_KX, step[1, 2] * STAB_KY)
            if d > STAB_STEP_MAX:
                f = STAB_STEP_MAX / d
                step = step.copy()
                step[0, 2] *= f
                step[1, 2] *= f

        det = step[0, 0] ** 2 + step[1, 0] ** 2
        if det < 1e-9:
            step_inv = np.eye(3)
        else:
            step_inv = np.eye(3)
            step_inv[0, 0] = step_inv[1, 1] = step[0, 0] / det
            step_inv[0, 1] = step[1, 0] / det
            step_inv[1, 0] = -step_inv[0, 1]
            step_inv[0, 2] = -(step_inv[0, 0] * step[0, 2] + step_inv[0, 1] * step[1, 2])
            step_inv[1, 2] = -(step_inv[1, 0] * step[0, 2] + step_inv[1, 1] * step[1, 2])
        W = self.W @ step_inv

        u = min(1.0, max(abs(W[0, 2]) * STAB_KX / budget[0],
                         abs(W[1, 2]) * STAB_KY / budget[1]))
        tau = self.tau / (1.0 + STAB_WALL * u * u)

        alpha = 1.0 - math.exp(-dt / tau) if dt > 0.0 else 0.0
        W = self._scale(W, 1.0 - alpha)
        self.W = W

        W = self._deadzone(W.copy())
        rs = self._rolling_shutter()

        def compose(f):
            Wc = self._to_capture(W if f >= 1.0 else self._scale(W, f))
            return Wc if rs is None else Wc @ rs

        s = 1.0
        total = compose(1.0)
        if not fit(total[:2]):
            lo, hi = 0.0, 1.0
            for _ in range(6):
                mid = (lo + hi) / 2.0
                if fit(compose(mid)[:2]):
                    lo = mid
                else:
                    hi = mid
            s = lo
            total = compose(s)
            if not fit(total[:2]):
                total = np.eye(3)
                s = 0.0
            self.W = self._scale(self.W, s)
        self.sat = 1.0 - s
        self.warp = np.ascontiguousarray(total[:2])
        self.hist.append((time.time(), step))
        self.ms = (time.perf_counter() - t0) * 1e3
        return self.warp

    def last_t(self):
        return self.hist[-1][0] if self.hist else 0.0

    def transport(self, t):
        if not self.hist:
            return None
        T = None
        for ts, step in self.hist:
            if ts <= t:
                continue
            T = step if T is None else step @ T
        if T is None:
            return None
        return self._to_capture(T)[:2]

    def status(self):
        return {"on": bool(self.enabled), "tau": round(self.tau, 2),
                "zoom": 1.0 if STAB_FREE else round(STAB_ZOOM, 3),
                "free": bool(STAB_FREE), "pts": int(self.pts),
                "fail": int(self.fail), "sat": round(self.sat, 2),
                "v": [round(self.vx), round(self.vy)],
                "rs_ms": round(STAB_RS_MS, 1), "ms": round(self.ms, 2)}


class VirtualPTZ:

    def __init__(self):
        self.cx = self.cx_t = CAP_W / 2.0
        self.cy = self.cy_t = CAP_H / 2.0
        self.zoom = self.zoom_t = DEFAULT_ZOOM
        self.zoom_rate = 0.0
        self.zoom_expire = 0.0
        self.zoom_clock = time.monotonic()
        self.lock = threading.Lock()

    def _win_for(self, zoom):
        w = base_window()[0] / zoom
        return w, w / OUT_ASPECT

    def _clamp_center(self, cx, cy, zoom):
        w, h = self._win_for(zoom)
        lo_x, hi_x = w / 2, CAP_W - w / 2
        lo_y, hi_y = h / 2, CAP_H - h / 2
        cx = (lo_x + hi_x) / 2 if lo_x > hi_x else max(lo_x, min(hi_x, cx))
        cy = (lo_y + hi_y) / 2 if lo_y > hi_y else max(lo_y, min(hi_y, cy))
        return cx, cy

    def set_zoom_rate(self, rate, ttl=GCS_ZOOM_TIMEOUT):
        with self.lock:
            self.zoom_rate = rate
            self.zoom_expire = (time.monotonic() + ttl) if rate else 0.0

    def zoom_rate_now(self):
        with self.lock:
            return self.zoom_rate

    def _integrate_zoom_locked(self, dt):
        now = time.monotonic()
        if self.zoom_rate and now > self.zoom_expire:
            self.zoom_rate = 0.0
        if self.zoom_rate:
            self.zoom_t = max(MIN_ZOOM, min(MAX_ZOOM,
                                            self.zoom_t + self.zoom_rate * dt))

    def step(self, tau_pan=TAU_PAN):
        with self.lock:
            now = time.monotonic()
            dt = min(DT_MAX, max(0.0, now - self.zoom_clock))
            self.zoom_clock = now
            self._integrate_zoom_locked(dt)
            a_zoom = 1.0 - math.exp(-dt / TAU_ZOOM) if dt > 0.0 else 0.0
            a_pan = 1.0 - math.exp(-dt / tau_pan) if dt > 0.0 else 0.0
            self.zoom += a_zoom * (self.zoom_t - self.zoom)
            self.cx += a_pan * (self.cx_t - self.cx)
            self.cy += a_pan * (self.cy_t - self.cy)
            self.cx_t, self.cy_t = self._clamp_center(self.cx_t, self.cy_t, self.zoom_t)
            self.cx, self.cy = self._clamp_center(self.cx, self.cy, self.zoom)

    def window(self):
        with self.lock:
            w, h = self._win_for(self.zoom)
            return self.cx, self.cy, w, h

    def pan(self, dx, dy):
        with self.lock:
            self.cx_t += dx / self.zoom_t
            self.cy_t += dy / self.zoom_t
            self.cx_t, self.cy_t = self._clamp_center(self.cx_t, self.cy_t, self.zoom_t)

    def set_target(self, tx, ty):
        with self.lock:
            self.cx_t, self.cy_t = self._clamp_center(tx, ty, self.zoom_t)


    def set_zoom(self, z):
        with self.lock:
            self.zoom_rate = 0.0
            self.zoom_t = max(MIN_ZOOM, min(MAX_ZOOM, z))
            self.cx_t, self.cy_t = self._clamp_center(self.cx_t, self.cy_t, self.zoom_t)

    def recenter(self, zoom=None):
        with self.lock:
            self.zoom_rate = 0.0
            self.cx_t, self.cy_t = CAP_W / 2.0, CAP_H / 2.0
            self.zoom_t = DEFAULT_ZOOM if zoom is None else zoom

    def state(self):
        with self.lock:
            return self.cx, self.cy, self.zoom

    def pan_range(self, margin=1.0):
        with self.lock:
            w, h = self._win_for(self.zoom * margin)
        return (max(0.0, (CAP_W - w) / 2),
                max(0.0, (CAP_H - h) / 2))


class RtspServer:

    def __init__(self, port=RTSP_PORT, path=RTSP_PATH):
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import Gst, GstRtspServer, GLib

        self.Gst = Gst
        self.GLib = GLib
        Gst.init(None)

        self.port = port
        self.path = path
        self.lock = threading.Lock()
        self.src = None
        self.clients = 0
        self.duration = Gst.SECOND // RTSP_FPS

        if RTSP_CODEC == "h265":
            enc = (f"x265enc tune=zerolatency speed-preset={RTSP_PRESET} "
                   f"bitrate={RTSP_BITRATE} key-int-max={RTSP_GOP} "
                   f"option-string={RTSP_X265_OPTS} "
                   "! rtph265pay name=pay0 pt=96 config-interval=1")
        else:
            enc = (f"x264enc tune=zerolatency speed-preset={RTSP_PRESET} "
                   f"bitrate={RTSP_BITRATE} key-int-max={RTSP_GOP} "
                   f"vbv-buf-capacity={RTSP_VBV} "
                   f"intra-refresh={'true' if RTSP_INTRA_REFRESH else 'false'} "
                   "! rtph264pay name=pay0 pt=96 config-interval=1")

        launch = (
            "appsrc name=src is-live=true format=time do-timestamp=true "
            f"block=false max-bytes=0 max-buffers={RTSP_QUEUE} leaky-type=downstream "
            f"caps=video/x-raw,format=BGR,width={OUT_W},height={OUT_H},framerate={RTSP_FPS}/1 "
            "! videoconvert ! video/x-raw,format=I420 "
            f"! {enc}"
        )

        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(f"( {launch} )")
        factory.set_shared(True)
        factory.connect("media-configure", self._on_configure)

        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))
        self.server.get_mount_points().add_factory(path, factory)

    def _on_configure(self, factory, media):
        src = media.get_element().get_by_name("src")
        media.connect("unprepared", self._on_unprepared)
        with self.lock:
            self.src = src
            self.clients += 1

    def _on_unprepared(self, media):
        with self.lock:
            self.src = None
            self.clients = 0

    def serve(self):
        if self.server.attach(None) == 0:
            return
        self.GLib.MainLoop().run()

    def push(self, bgr):
        with self.lock:
            src = self.src
        if src is None:
            return
        buf = self.Gst.Buffer.new_wrapped(bgr.tobytes())
        buf.duration = self.duration
        if src.emit("push-buffer", buf) != self.Gst.FlowReturn.OK:
            with self.lock:
                self.src = None

    def info(self):
        with self.lock:
            return {"on": True, "clients": self.clients,
                    "port": self.port, "path": self.path,
                    "codec": RTSP_CODEC, "fps": RTSP_FPS,
                    "bitrate": RTSP_BITRATE, "size": [OUT_W, OUT_H]}


class Renderer:

    def __init__(self):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.job = None
        self.alive = True
        self.dropped = 0
        self.ms = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, job):
        with self.cv:
            if self.job is not None:
                self.dropped += 1
            self.job = job
            self.cv.notify()

    def stop(self):
        with self.cv:
            self.alive = False
            self.cv.notify_all()

    def stats(self):
        with self.lock:
            return {"ms": round(self.ms, 2), "dropped": self.dropped}

    def _loop(self):
        while True:
            with self.cv:
                while self.alive and self.job is None:
                    self.cv.wait(0.5)
                if not self.alive:
                    return
                job = self.job
                self.job = None
            try:
                self._render(job)
            except Exception:
                pass

    def _render(self, job):
        t0 = time.perf_counter()
        bgr, M, rect, label, box, trk = job
        if rect is None:
            out = cv2.warpAffine(
                bgr, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                borderMode=(cv2.BORDER_CONSTANT if STAB_FREE
                            else cv2.BORDER_REPLICATE),
                borderValue=(0, 0, 0))
        else:
            x0, y0, x1, y1 = rect
            if (x1 - x0) == OUT_W and (y1 - y0) == OUT_H:
                out = bgr[y0:y1, x0:x1].copy()
            else:
                interp = cv2.INTER_AREA if (x1 - x0) > OUT_W else cv2.INTER_CUBIC
                out = cv2.resize(bgr[y0:y1, x0:x1], (OUT_W, OUT_H),
                                 interpolation=interp)
        if label:
            cv2.putText(out, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 128), 1, cv2.LINE_AA)
        if box is not None:
            a, b, col = box
            cv2.rectangle(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                          col, 2, cv2.LINE_AA)
        elif trk is not None:
            cv2.circle(out, (int(trk[0]), int(trk[1])), 18,
                       (0, 255, 255), 2, cv2.LINE_AA)
        if rtsp is not None:
            rtsp.push(out)
        t = (time.perf_counter() - t0) * 1e3
        with self.lock:
            self.ms = 0.8 * self.ms + 0.2 * t if self.ms else t


class SharedState:

    def __init__(self):
        self.lock = threading.Lock()
        self.follow = False
        self.inv_hist = collections.deque(maxlen=240)
        self.click = None
        self.clear_req = False
        self.click_count = 0
        self.last_click = None
        self.det_info = {"on": False, "n": 0, "ms": 0.0, "mode": "none"}
        self.pred = {"age_ms": 0.0, "dx": 0.0, "dy": 0.0, "vx": 0.0, "vy": 0.0,
                     "on": PRED_ENABLE}
        self.track_box = None
        self.track_score = 0.0
        self.lost_box = None
        self.show_det = True
        self.stats = {"fps": 0.0, "crop": None, "zoom": 1.0}
        self.osd_master = True
        self.osd_zoom = True
        self.gain = [0] * 10
        self.fcc_tgt = {"valid": False, "nx": 0.0, "ny": 0.0,
                        "yaw": 0.0, "pitch": 0.0}
        self.rotate = (0.0, 0.0)
        self.center_pending = False

    def set_rotate(self, yaw, pitch):
        with self.lock:
            self.rotate = (float(yaw), float(pitch))

    def rotate_cmd(self):
        with self.lock:
            return self.rotate

    def set_center(self, value):
        with self.lock:
            self.center_pending = True

    def center_cmd(self):
        with self.lock:
            if self.center_pending:
                self.center_pending = False
                return 1
            return 0

    def set_fcc_target(self, valid, nx, ny, yaw, pitch):
        with self.lock:
            self.fcc_tgt = {"valid": bool(valid), "nx": nx, "ny": ny,
                            "yaw": yaw, "pitch": pitch}

    def fcc_target(self):
        with self.lock:
            return dict(self.fcc_tgt)

    def set_osd_flags(self, master, zoom):
        with self.lock:
            self.osd_master = master
            self.osd_zoom = zoom

    def osd_flags(self):
        with self.lock:
            return self.osd_master, self.osd_zoom

    def set_gain(self, values):
        g = [max(-128, min(127, int(v))) for v in values[:10]]
        g += [0] * (10 - len(g))
        with self.lock:
            self.gain = g

    def gain_cmd(self):
        with self.lock:
            return list(self.gain)


    def set_show_det(self, on):
        with self.lock:
            self.show_det = on

    def show_dets(self):
        with self.lock:
            return self.show_det

    def request_clear(self):
        with self.lock:
            self.follow = False
            self.clear_req = True

    def push_click(self, x, y, frame_id, box=None):
        with self.lock:
            self.click = (x, y, frame_id, box)
            self.click_count += 1
            self.last_click = [round(x), round(y)]

    def snapshot(self):
        with self.lock:
            s = dict(self.stats)
            s["follow"] = self.follow
            s["clicks"] = self.click_count
            s["last_click"] = self.last_click
            s.update(self.det_info)
            s["show_det"] = self.show_det
            s["track_box"] = self.track_box
            s["track_score"] = self.track_score
            s["lost_box"] = self.lost_box
            s["pred"] = dict(self.pred)
            s["rotate"] = list(self.rotate)
            s["center"] = self.center_pending
            s["snap"] = snap_status()
            s["fcc"] = {"valid": self.fcc_tgt["valid"],
                        "nx": round(self.fcc_tgt["nx"], 4),
                        "ny": round(self.fcc_tgt["ny"], 4),
                        "yaw_deg": round(self.fcc_tgt["yaw"], 2),
                        "pitch_deg": round(self.fcc_tgt["pitch"], 2)}
            return s

    def begin_frame(self):
        with self.lock:
            click, self.click = self.click, None
            clear, self.clear_req = self.clear_req, False
            return self.follow, click, clear

    def inv_for(self, frame_id):
        with self.lock:
            if not self.inv_hist:
                return None
            for fid, inv in reversed(self.inv_hist):
                if fid == frame_id:
                    return inv
            return self.inv_hist[-1][1]

    def set_follow(self, value):
        with self.lock:
            self.follow = value

    def set_track_box(self, box, score=0.0):
        with self.lock:
            self.track_box = ([round(float(v), 1) for v in box] if box is not None else None)
            self.track_score = round(float(score), 3)

    def set_lost_box(self, box):
        with self.lock:
            self.lost_box = ([round(float(v), 1) for v in box]
                             if box is not None else None)

    def set_pred(self, age, dx, dy, vx, vy):
        with self.lock:
            self.pred = {"age_ms": round(age * 1e3, 1),
                         "dx": round(dx, 1), "dy": round(dy, 1),
                         "vx": round(vx, 1), "vy": round(vy, 1),
                         "on": PRED_ENABLE}

    def set_det_info(self, ms, mode):
        with self.lock:
            self.det_info = {"ms": round(ms, 1), "mode": mode}

    def publish_frame(self, frame_id, inv, fps, crop, zoom, tx=0.0, jitter=None):
        with self.lock:
            self.inv_hist.append((frame_id, inv))
            self.stats["fps"] = round(fps, 1)
            self.stats["tx_fps"] = round(tx, 1)
            self.stats["crop"] = list(crop) if crop else None
            self.stats["zoom"] = round(zoom, 3)
            if jitter:
                self.stats["jitter"] = jitter


CAMERA_REF = {}
LFC_REF = {}
STAB_REF = {}
RENDER_REF = {}
FLOW_REF = {}

ptz = VirtualPTZ()
state = SharedState()
rtsp = None


def _off_axis_rad(px, half, tan_half):
    return math.atan((px - half) / half * tan_half)


def target_angles(tx, ty):
    return (math.degrees(_off_axis_rad(tx, CAP_W / 2.0, CAM_HFOV_TAN)),
            math.degrees(-_off_axis_rad(ty, CAP_H / 2.0, CAM_VFOV_TAN)))


def _sum8(body):
    return sum(body) & 0xFF


def _xor8(body):
    x = 0
    for b in body:
        x ^= b
    return x


FCC_CHECKSUM = _sum8


def _deg_x10(v):
    return max(-32768, min(32767, int(round(v * 10.0))))


def fcc_tx_packet():
    t = state.fcc_target()
    on = t["valid"]
    xm, ym = state.rotate_cmd()
    fields = [FCC_TX_HEADER1, FCC_TX_HEADER2,
              1 if on else 0, t["nx"], t["ny"], 1 if on else 0,
              xm, ym, 0, state.center_cmd(),
              0, _deg_x10(t["pitch"]) if on else 0,
              _deg_x10(t["yaw"]) if on else 0, 0, 0, 0] + state.gain_cmd()
    raw = struct.pack(FCC_TX_FMT, *(fields + [0]))
    return raw[:-1] + bytes([FCC_CHECKSUM(raw[:-1])])


def fcc_rx_parse(raw):
    if len(raw) != FCC_RX_SIZE:
        return None
    if raw[0] != FCC_RX_HEADER1 or raw[1] != FCC_RX_HEADER2:
        return None
    x = 0
    for b in raw[:-1]:
        x ^= b
    if x != raw[-1]:
        return None
    return struct.unpack(FCC_RX_FMT, raw)


def _s8(v):
    return v - 256 if v > 127 else v


def gcs_track_action(msg):
    on = msg[GCS_PAYLOAD_OFFSET]
    if not on:
        state.request_clear()
        return
    sx, sy, ex, ey = struct.unpack_from("<HHHH", msg, GCS_PAYLOAD_OFFSET + 1)
    fx, fy = OUT_W / GCS_REF_W, OUT_H / GCS_REF_H
    x1, x2 = sorted((sx * fx, ex * fx))
    y1, y2 = sorted((sy * fy, ey * fy))
    if x2 - x1 < 2 or y2 - y1 < 2:
        state.push_click((x1 + x2) / 2, (y1 + y2) / 2, 0, None)
    else:
        state.push_click((x1 + x2) / 2, (y1 + y2) / 2, 0, (x1, y1, x2, y2))


def gcs_osd_display(msg):
    b1 = msg[GCS_PAYLOAD_OFFSET]
    state.set_osd_flags(bool(b1 & 0x01), bool(b1 & 0x20))


def handle_gcs_message(msg):
    if len(msg) <= GCS_CMD_OFFSET:
        return
    if msg[0] != GCS_HEADER1 or msg[1] != GCS_HEADER2:
        return
    cmd = msg[GCS_CMD_OFFSET]
    p = GCS_PAYLOAD_OFFSET

    if cmd == CMD_CAM_HEARTBEAT:
        return

    if cmd == CMD_AI_MODE:
        state.set_show_det(bool(msg[p]))

    elif cmd in (CMD_GIMBAL_ZOOM, CMD_TEST_DIGITAL_ZOOM):
        d = _s8(msg[p])
        ptz.set_zoom_rate(0.0 if d == 0 else
                          (GCS_ZOOM_RATE if d > 0 else -GCS_ZOOM_RATE))

    elif cmd == CMD_TRACK_ACTION:
        gcs_track_action(msg)

    elif cmd == CMD_GIMBAL_ROTATE:
        global CAM_HFOV_DEG, CAM_VFOV_DEG, CAM_HFOV_TAN, CAM_VFOV_TAN
        CAM_HFOV_DEG = ROTATE_HFOV_DEG
        CAM_VFOV_DEG = ROTATE_VFOV_DEG
        CAM_HFOV_TAN = math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
        CAM_VFOV_TAN = math.tan(math.radians(CAM_VFOV_DEG) / 2.0)
        state.set_rotate(_s8(msg[p]), _s8(msg[p + 1]))

    elif cmd == CMD_DIGITAL_TILT:
        yaw = _s8(msg[p]) / GCS_ROTATE_FULL_SCALE
        pitch = _s8(msg[p + 1]) / GCS_ROTATE_FULL_SCALE
        if yaw or pitch:
            state.request_clear()
            ptz.pan(yaw * GCS_ROTATE_STEP_PX, -pitch * GCS_ROTATE_STEP_PX)

    elif cmd == CMD_GIMBAL_CENTER:
        state.set_center(msg[p])

    elif cmd == CMD_SET_GAIN:
        state.set_gain([_s8(b) for b in msg[p:p + 10]])

    elif cmd == CMD_SET_OSD_DISPLAY:
        gcs_osd_display(msg)

    elif cmd == CMD_STABILIZER_MODE:
        stab = STAB_REF.get("stab")
        if stab is not None:
            if msg[p] == GCS_STAB_RESET:
                stab.request_reset()
            else:
                stab.set_enabled(msg[p])

    elif cmd == CMD_STABILIZER_ALPHA:
        stab = STAB_REF.get("stab")
        if stab is not None:
            a = max(0, min(100, msg[p])) / 100.0
            stab.set_tau(STAB_TAU_MIN + a * (STAB_TAU_MAX - STAB_TAU_MIN))

    elif cmd == CMD_TEST_ZOOM_RAW:
        raw = struct.unpack_from("<H", msg, p)[0]
        f = min(1.0, raw / GCS_ZOOM_RAW_MAX)
        ptz.set_zoom(MIN_ZOOM + f * (MAX_ZOOM - MIN_ZOOM))


_snap_q = queue.Queue(maxsize=REC_QUEUE)
_snap_stat = {"n": 0, "last": None, "err": None, "drop": 0}
_snap_lock = threading.Lock()


def snap_request(bgr, box, meta):
    if not REC_ENABLE or bgr is None:
        return
    try:
        _snap_q.put_nowait((bgr.copy(), box, meta))
    except queue.Full:
        with _snap_lock:
            _snap_stat["drop"] += 1


def _snap_write(bgr, box, meta):
    if not os.path.ismount(REC_MOUNT):
        raise OSError("%s not mounted" % REC_MOUNT)
    vfs = os.statvfs(REC_MOUNT)
    if vfs.f_bavail * vfs.f_frsize < REC_MIN_MB * 1024 * 1024:
        raise OSError("low space on %s" % REC_MOUNT)
    os.makedirs(REC_DIR, exist_ok=True)
    t = meta["ts"]
    name = "%s_%03d_f%06d" % (time.strftime("%Y%m%d_%H%M%S", time.localtime(t)),
                              int((t % 1.0) * 1000), meta["frame_id"])
    path = os.path.join(REC_DIR, name + ".jpg")
    ok, buf = cv2.imencode(".jpg", bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), REC_JPEG_Q])
    if not ok:
        raise OSError("jpeg encode failed")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    rec = {"file": os.path.basename(path), "ts": round(t, 3),
           "frame_id": meta["frame_id"],
           "size": [int(bgr.shape[1]), int(bgr.shape[0])],
           "box": [round(float(v), 1) for v in box] if box else None,
           "zoom": meta.get("zoom"), "crop": meta.get("crop")}
    with open(os.path.join(REC_DIR, "index.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def snap_loop():
    while True:
        bgr, box, meta = _snap_q.get()
        try:
            path = _snap_write(bgr, box, meta)
            with _snap_lock:
                _snap_stat["n"] += 1
                _snap_stat["last"] = os.path.basename(path)
                _snap_stat["err"] = None
        except Exception as e:
            with _snap_lock:
                _snap_stat["err"] = str(e)[:80]


def snap_status():
    with _snap_lock:
        return dict(_snap_stat)


def fcc_open(path, baud):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        speed = getattr(termios, "B%d" % baud)
        a = termios.tcgetattr(fd)
        a[0] = 0
        a[1] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[3] = 0
        a[4] = speed
        a[5] = speed
        cc = list(a[6])
        cc[termios.VMIN] = 0
        cc[termios.VTIME] = 0
        a[6] = cc
        termios.tcsetattr(fd, termios.TCSANOW, a)
        termios.tcflush(fd, termios.TCIOFLUSH)
    except Exception:
        os.close(fd)
        raise
    return fd


def fcc_loop():
    period = 1.0 / max(1.0, FCC_HZ)
    fd = None
    next_t = time.monotonic()
    while True:
        if fd is None:
            try:
                fd = fcc_open(FCC_PORT, FCC_BAUD)
                next_t = time.monotonic()
            except Exception:
                time.sleep(FCC_RETRY)
                continue
        try:
            os.write(fd, fcc_tx_packet())
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            fd = None
            time.sleep(FCC_RETRY)
            continue
        next_t += period
        d = next_t - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            next_t = time.monotonic()


def gcs_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", GCS_UDP_PORT))
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            handle_gcs_message(data)
        except Exception as e:
            pass


def write_status(frame_id):
    st = state.snapshot()
    cx, cy, z = ptz.state()
    rx, ry = ptz.pan_range(stab_margin())
    st["frame_id"] = frame_id
    st["pan"] = [round(cx), round(cy)]
    st["zoom"] = round(z, 2)
    st["zoom_max"] = round(MAX_ZOOM, 2)
    st["zoom_real"] = round(REAL_ZOOM / stab_margin(), 2)
    st["zoom_rate"] = round(ptz.zoom_rate_now(), 3)
    st["pan_range"] = [round(rx), round(ry)]
    st["rtsp"] = rtsp.info() if rtsp is not None else {"on": False}
    stab = STAB_REF.get("stab")
    if stab is not None:
        st["stab"] = stab.status()
    renderer = RENDER_REF.get("renderer")
    if renderer is not None:
        st["render"] = renderer.stats()
    flow = FLOW_REF.get("flow")
    if flow is not None:
        st["flow"] = flow.status()
        lfc0 = LFC_REF.get("lfc")
        if lfc0 is not None:
            st["flow"]["hold"] = bool(lfc0.hold)
    lfc = LFC_REF.get("lfc")
    if lfc is not None:
        sg = lfc.stage_stats()
        if sg is not None:
            st["lfc_stage"] = sg
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, STATUS_FILE)


def pipeline():
    cam = Camera(CAMERA_INDEX, CAP_W, CAP_H, FPS)
    CAMERA_REF["cam"] = cam
    tracker = TargetTracker()
    flow = FlowTracker() if FLOW_ENABLE else None
    if flow is not None:
        FLOW_REF["flow"] = flow
    stab = FrameStabilizer()
    STAB_REF["stab"] = stab
    renderer = Renderer()
    RENDER_REF["renderer"] = renderer
    try:
        lfc = LightFCTracker()
        LFC_REF["lfc"] = lfc
    except Exception as e:
        lfc = None

    t_last = time.time()
    fps_ema = 0.0
    jit = collections.deque(maxlen=150)
    t_loop_end = None
    settle = SWITCH_SETTLE_FRAMES
    prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0
    frame_id = 0
    lfc_box, lfc_center, lfc_score, lfc_t = None, None, 0.0, 0.0
    box_s, box_s_t, lfc_seen, meas_t = None, 0.0, 0.0, 0.0
    flow_seen = 0.0
    live_box, dead_box, dead_t = None, None, 0.0
    last_bgr = None
    rtsp_period = 1.0 / max(1, RTSP_FPS)
    rtsp_next = 0.0
    stab_period = 1.0 / max(1.0, STAB_RATE)
    stab_next = 0.0
    tx_hist = collections.deque(maxlen=max(8, RTSP_FPS * 2 + 1))
    tx_fps = 0.0

    try:
        while True:
            _t_r0 = time.perf_counter()
            bgr, proc, crop = cam.read()
            _t_r1 = time.perf_counter()
            _now = time.time()
            dt_f = _now - t_last
            t_last = _now
            if proc is None:
                cam.release()
                os._exit(1)
            if bgr is not None:
                last_bgr = bgr

            follow, click, clear = state.begin_frame()
            frame_id += 1

            lfc_active = False
            if lfc is not None:
                if bgr is not None:
                    lfc.submit(bgr, _now)
                lfc_box, lfc_center, lfc_score, lfc_t = lfc.snapshot()
                lfc_active = lfc_box is not None

            flow_on = False
            if flow is not None:
                if not lfc_active:
                    flow.stop()
                    flow_seen = 0.0
                else:
                    raw_box, raw_score, raw_t = lfc.raw_snapshot()
                    big = (LFC_BIG_HOLD and raw_box is not None and
                           max(raw_box[2] / CAP_W, raw_box[3] / CAP_H) >= LFC_BIG_FRAC)
                    lost = raw_score < LFC_SCORE_MIN
                    flow.set_own_size(big and lost)
                    if raw_box is not None and not flow.own_size:
                        flow.set_size(raw_box[2], raw_box[3])
                    fc = flow.update(bgr, _now) if flow.active else None
                    lfc.set_hold(big and fc is not None)
                    if big and lost and fc is not None:
                        lfc.retarget(flow.box())
                    if raw_box is not None and raw_t > flow_seen:
                        flow_seen = raw_t
                        if raw_score >= LFC_SCORE_MIN:
                            if fc is None or not flow.anchor(
                                    raw_box[0] + 0.5 * raw_box[2],
                                    raw_box[1] + 0.5 * raw_box[3], raw_t):
                                fc = ((flow.cx, flow.cy)
                                      if flow.start(bgr, raw_box, _now) else None)
                    if fc is not None and raw_box is not None:
                        bw, bh = ((flow.w, flow.h) if flow.own_size
                                  else (raw_box[2], raw_box[3]))
                        lfc_box = (fc[0] - 0.5 * bw, fc[1] - 0.5 * bh, bw, bh)
                        lfc_center = fc
                        lfc_t = _now
                        flow_on = True

            if not lfc_active:
                box_s, box_s_t, lfc_seen, meas_t = None, 0.0, 0.0, 0.0
            elif flow_on:
                box_s, box_s_t, lfc_seen, meas_t = None, 0.0, 0.0, 0.0
            elif stab.enabled:
                if box_s is not None:
                    Ts = stab.transport(box_s_t)
                    if Ts is not None:
                        box_s = xform_box(Ts, box_s)
                box_s_t = stab.last_t()
                if lfc_t > lfc_seen:
                    fresh = lfc_score >= LFC_SCORE_MIN or not BOX_HOLD_LOW
                    if fresh:
                        Tm = stab.transport(lfc_t)
                        meas = (xform_box(Tm, lfc_box) if Tm is not None
                                else tuple(lfc_box))
                        box_s = box_smooth(box_s, meas,
                                           lfc_t - meas_t if meas_t else 0.0)
                        meas_t = lfc_t
                    lfc_seen = lfc_t
                if box_s is not None:
                    lfc_box = box_s
                    lfc_center = (box_s[0] + 0.5 * box_s[2],
                                  box_s[1] + 0.5 * box_s[3])
                    lfc_t = box_s_t
            if lfc_active and lfc_box is not None:
                live_box, dead_box = lfc_box, None
            elif live_box is not None:
                dead_box, dead_t, live_box = live_box, _now, None
            if (dead_box is not None and LOST_BOX_SEC > 0.0
                    and _now - dead_t > LOST_BOX_SEC):
                dead_box = None

            pred_dx, pred_dy, pred_age = 0.0, 0.0, 0.0
            tgt = None

            ptz.step(TAU_FOLLOW if ((lfc_active or tracker.active) and follow)
                     else TAU_PAN)
            if settle > 0:
                settle -= 1

            if clear:
                tracker.stop()
                live_box, dead_box = None, None
                if flow is not None: flow.stop()
                if lfc is not None: lfc.stop()
                follow = False
                prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0

            inv = state.inv_for(click[2]) if click is not None else None
            if click is not None and inv is not None:
                capx, capy = apply_pt(inv, click[0], click[1])
                live_box, dead_box = None, None
                prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0
                tracker.stop()
                if lfc is not None:
                    lfc.stop()
                if click[3]:
                    ax, ay = apply_pt(inv, click[3][0], click[3][1])
                    bx2, by2 = apply_pt(inv, click[3][2], click[3][3])
                    box = (min(ax, bx2), min(ay, by2),
                           max(8.0, abs(bx2 - ax)), max(8.0, abs(by2 - ay)))
                else:
                    box = (capx - LFC_INIT_BOX / 2, capy - LFC_INIT_BOX / 2,
                           LFC_INIT_BOX, LFC_INIT_BOX)
                snap_request(last_bgr, box, {"ts": _now, "frame_id": frame_id,
                                        "zoom": round(ptz.state()[2], 3),
                                        "crop": list(crop) if crop else None})
                if lfc is not None and last_bgr is not None:
                    lfc.request_start(last_bgr, box)
                    if flow is not None:
                        flow.start(last_bgr, box, _now)
                        flow_seen = 0.0
                    follow = True
                    state.set_follow(True)
                elif tracker.start(proc, capx * PROC_SCALE, capy * PROC_SCALE):
                    follow = True
                    state.set_follow(True)
                else:
                    follow = False
                    state.set_follow(False)

            if lfc_active or tracker.active:
                if lfc_active:
                    c = (lfc_center[0] * PROC_SCALE,
                         lfc_center[1] * PROC_SCALE) if lfc_center else None
                else:
                    c = tracker.update(proc)
                if c is None:
                    follow = False
                    state.set_follow(False)
                    prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0
                elif follow:
                    tx, ty = c[0] / PROC_SCALE, c[1] / PROC_SCALE
                    tgt = (tx, ty)
                    m_t = lfc_t if (lfc_active and lfc_t > 0.0) else _now
                    sx, sy = tx, ty
                    if stab.enabled:
                        T = stab.transport(m_t)
                        if T is not None:
                            sx, sy = apply_pt(T, sx, sy)
                        if stab.warp is not None:
                            sx, sy = apply_pt(stab.warp, sx, sy)
                    if prev_tgt is None:
                        prev_tgt, prev_tgt_t = (sx, sy), m_t
                    elif (sx, sy) != prev_tgt:
                        dt_t = m_t - prev_tgt_t
                        if dt_t > 0.0:
                            a_vel = 1.0 - math.exp(-min(DT_MAX, dt_t) / TAU_VEL)
                            tvx += a_vel * ((sx - prev_tgt[0]) / dt_t - tvx)
                            tvy += a_vel * ((sy - prev_tgt[1]) / dt_t - tvy)
                        prev_tgt, prev_tgt_t = (sx, sy), m_t
                    if PRED_ENABLE:
                        pred_age = max(0.0, min(PRED_MAX_AGE, _now - m_t))
                        pred_dx = max(-FOLLOW_LEAD_MAX,
                                      min(FOLLOW_LEAD_MAX, tvx * pred_age))
                        pred_dy = max(-FOLLOW_LEAD_MAX,
                                      min(FOLLOW_LEAD_MAX, tvy * pred_age))
                    lx = max(-FOLLOW_LEAD_MAX, min(FOLLOW_LEAD_MAX, tvx * LEAD_TIME))
                    ly = max(-FOLLOW_LEAD_MAX, min(FOLLOW_LEAD_MAX, tvy * LEAD_TIME))
                    ptz.set_target(sx + pred_dx + lx, sy + pred_dy + ly)

            cx, cy, win_w, win_h = ptz.window()
            if stab.enabled and not STAB_FREE:
                win_w /= STAB_ZOOM
                win_h /= STAB_ZOOM
            x0f, y0f = cx - win_w / 2.0, cy - win_h / 2.0
            push_now = (rtsp is not None and bgr is not None
                        and _now >= rtsp_next)
            if push_now:
                rtsp_next = max(_now - rtsp_period, rtsp_next) + rtsp_period
                tx_hist.append(_now)
            if len(tx_hist) > 1 and tx_hist[-1] > tx_hist[0]:
                tx_fps = (len(tx_hist) - 1) / (tx_hist[-1] - tx_hist[0])

            stab_m = None
            cw3 = None
            if stab.enabled:
                cw3 = np.array([
                    [OUT_W / win_w, 0.0, -OUT_W / win_w * x0f],
                    [0.0, OUT_H / win_h, -OUT_H / win_h * y0f],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                if STAB_FREE:
                    fit = _fit_free
                    budget = (STAB_FREE_PX, STAB_FREE_PX)
                else:
                    fit = lambda m: crop_fits(cw3, m)
                    budget = (max(1.0, min(x0f, CAP_W - (x0f + win_w))),
                              max(1.0, min(y0f, CAP_H - (y0f + win_h))))
                if _now >= stab_next:
                    stab_next = max(_now - stab_period, stab_next) + stab_period
                    stab_m = stab.update(proc, fit, budget)
                else:
                    stab_m = stab.warp
                    if stab_m is not None and not fit(stab_m):
                        stab_m = None

            if stab_m is not None:
                rect = None
                M = (cw3 @ np.vstack([stab_m, (0.0, 0.0, 1.0)]))[:2].copy()
            else:
                x0 = max(0, int(math.floor(x0f)))
                y0 = max(0, int(math.floor(y0f)))
                x1 = min(CAP_W, int(math.ceil(x0f + win_w)))
                y1 = min(CAP_H, int(math.ceil(y0f + win_h)))
                rect = (x0, y0, x1, y1)
                sx = OUT_W / float(x1 - x0)
                sy = OUT_H / float(y1 - y0)
                M = np.array([
                    [sx, 0.0, -sx * x0],
                    [0.0, sy, -sy * y0],
                ], dtype=np.float64)

            Minv = cv2.invertAffineTransform(M)

            if tgt is not None:
                px, py = apply_pt(M, tgt[0], tgt[1])
                state.set_fcc_target(
                    True,
                    max(-1.0, min(1.0, (px - OUT_W / 2.0) / (OUT_W / 2.0))),
                    max(-1.0, min(1.0, (OUT_H / 2.0 - py) / (OUT_H / 2.0))),
                    *target_angles(tgt[0], tgt[1]))
            else:
                state.set_fcc_target(False, 0.0, 0.0, 0.0, 0.0)

            dt = dt_f
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst
            jit.append((dt * 1000.0,
                        (_t_r1 - _t_r0) * 1000.0,
                        cam.sensor_dt_ms))

            _, _, zoom = ptz.state()
            if push_now:
                osd_master, osd_zoom = state.osd_flags()
                label = None
                if osd_master:
                    label = f"{tx_fps:4.1f}fps"
                    if osd_zoom:
                        label += f"  ZOOM:{zoom:.2f}x"
                Tb = (stab.transport(lfc_t)
                      if (stab.enabled and not flow_on) else None)
                box = trk = None
                if lfc_box is not None:
                    bx, by, bw, bh = lfc_box
                    if Tb is not None:
                        bx, by = apply_pt(Tb, bx, by)
                        sc = math.hypot(Tb[0, 0], Tb[1, 0])
                        bw *= sc
                        bh *= sc
                    a = apply_pt(M, bx + pred_dx, by + pred_dy)
                    b = apply_pt(M, bx + bw + pred_dx, by + bh + pred_dy)
                    col = ((0, 255, 255) if lfc_score >= LFC_SCORE_MIN
                           else (0, 140, 255))
                    box = (a, b, col)
                elif dead_box is not None and LOST_BOX:
                    a = apply_pt(M, dead_box[0], dead_box[1])
                    b = apply_pt(M, dead_box[0] + dead_box[2],
                                 dead_box[1] + dead_box[3])
                    box = (a, b, (0, 0, 255))
                elif tracker.active and tracker.center is not None:
                    trk = apply_pt(M, tracker.center[0] / PROC_SCALE,
                                   tracker.center[1] / PROC_SCALE)
                renderer.submit((bgr, M, rect, label, box, trk))

            state.set_det_info(lfc.ms if lfc is not None else 0.0,
                               ("flow" if flow_on else
                                "lightfc" if lfc_active else
                                "lk" if tracker.active else "none"))
            state.set_track_box(lfc_box, lfc_score)
            state.set_lost_box(dead_box)
            state.set_pred(pred_age, pred_dx, pred_dy, tvx, tvy)
            state.publish_frame(frame_id, Minv,
                                fps_ema, crop, zoom, tx_fps,
                                frame_jitter(jit) if frame_id % 15 == 0 else None)

            if STATUS_FILE and frame_id % STATUS_EVERY == 0:
                try:
                    write_status(frame_id)
                except Exception:
                    pass
            t_loop_end = time.perf_counter()

    finally:
        renderer.stop()
        cam.release()


def main():
    def _bye(signum, frame):
        cam = CAMERA_REF.get("cam")
        if cam is not None:
            cam.release()
        os._exit(0)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    global rtsp
    try:
        rtsp = RtspServer()
        threading.Thread(target=rtsp.serve, daemon=True).start()
    except Exception as e:
        rtsp = None

    threading.Thread(target=gcs_loop, daemon=True).start()
    threading.Thread(target=fcc_loop, daemon=True).start()
    threading.Thread(target=snap_loop, daemon=True).start()
    pipeline()


if __name__ == "__main__":
    main()
