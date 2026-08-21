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
import struct
import termios

CAMERA_INDEX = 0

CAP_W = int(os.environ.get("CAP_W", "2304"))
CAP_H = int(os.environ.get("CAP_H", "1296"))
OUT_W = int(os.environ.get("OUT_W", "1280"))
OUT_H = int(os.environ.get("OUT_H", "720"))
FPS          = int(os.environ.get("FPS", "24"))
ROTATION     = 0

SENSOR_W = int(os.environ.get("SENSOR_W", "2304"))
SENSOR_H = int(os.environ.get("SENSOR_H", "1296"))

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

MIN_ZOOM     = 1.0
DEFAULT_ZOOM = 1.0
MAX_ZOOM     = float(os.environ.get("MAX_ZOOM", "5.0"))
REAL_ZOOM    = CAP_W / OUT_W
CAP_K        = CAP_W / 1920.0

CAM_HFOV_DEG = float(os.environ.get("CAM_HFOV_DEG", "66.0"))
CAM_VFOV_DEG = float(os.environ.get("CAM_VFOV_DEG", "41.0"))
CAM_HFOV_TAN = math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
CAM_VFOV_TAN = math.tan(math.radians(CAM_VFOV_DEG) / 2.0)
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

HEF_PATHS = ["/usr/share/hailo-models/yolov8s_h8.hef",
             "/usr/share/hailo-models/yolov8s_h8l.hef"]
DET_SIZE        = 640
DET_SCORE       = 0.40
DET_MATCH_IOU   = 0.20
DET_MAX_MISS    = 20
DET_CLICK_PAD   = 0.15
USE_DETECTOR    = False

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
LFC_TPL_ALPHA = float(os.environ.get("LFC_TPL_ALPHA", "0.15"))
LFC_TPL_MIN   = float(os.environ.get("LFC_TPL_MIN", "0.75"))
LFC_MAX_MISS_SEC = float(os.environ.get("LFC_MISS_SEC", "5.6"))
ORT_SPINNING = os.environ.get("ORT_SPINNING", "0") not in ("0", "false", "no")
PRED_ENABLE = os.environ.get("PRED_ENABLE", "1") not in ("0", "false", "no")
PRED_MAX_AGE = float(os.environ.get("PRED_MAX_AGE", "0.15"))
LFC_DUMP_DIR = os.environ.get("LFC_DUMP_DIR", "")
STATUS_FILE = os.environ.get("STATUS_FILE", "")
STATUS_EVERY = int(os.environ.get("STATUS_EVERY", str(FPS)))
LFC_DUMP_N   = int(os.environ.get("LFC_DUMP_N", "256"))
LFC_DUMP_EVERY = int(os.environ.get("LFC_DUMP_EVERY", "3"))
LFC_INIT_BOX  = 90 * CAP_K
LFC_IN_Z  = "lightfc_backbone_scope1/input_layer1"
LFC_IN_X  = "lightfc_backbone_scope2/input_layer2"
LFC_OUT_Z = "lightfc_backbone_scope1/conv26"
LFC_OUT_X = "lightfc_backbone_scope2/conv52"


RTSP_PORT    = int(os.environ.get("RTSP_PORT", "554"))
RTSP_PATH    = "/video0"
RTSP_QUEUE   = 3
RTSP_BITRATE = int(os.environ.get("RTSP_BITRATE", "2500"))
RTSP_PRESET  = os.environ.get("RTSP_PRESET", "veryfast")
RTSP_GOP     = int(os.environ.get("RTSP_GOP", str(FPS * 2)))
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


def base_window():
    w = min(CAP_W, CAP_H * OUT_ASPECT)
    return w, w / OUT_ASPECT


def apply_pt(m, x, y):
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2],
            m[1, 0] * x + m[1, 1] * y + m[1, 2])


class Camera:

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
        self.z = None
        self.score = 0.0
        self.miss_t0 = None
        self.t_result = 0.0
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
            self.z = None
            self.score = 0.0
            self.miss_t0 = None
            self.t_result = 0.0
            self.req = None
            self.pending = None

    def submit(self, bgr):
        with self.lock:
            self.pending = bgr

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
            if self.box is None:
                return None, None, 0.0, 0.0
            x, y, w, h = self.box
            return self.box, (x + 0.5 * w, y + 0.5 * h), self.score, self.t_result

    def _loop(self):
        while True:
            with self.lock:
                req, self.req = self.req, None
                frame, self.pending = self.pending, None
                busy = self.box is not None
            try:
                if req is not None:
                    if not self._start(req[0], req[1]):
                        pass
                    continue
                if not busy or frame is None:
                    time.sleep(0.003)
                    continue
                t0 = time.time()
                self._update(frame)
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
            self.score = 1.0
            self.miss_t0 = None
            self.t_result = time.time()
        return True

    def _update(self, bgr):
        with self.lock:
            if self.box is None:
                return None
            cur_box, cur_z = self.box, self.z
        _t0 = time.perf_counter()
        x_patch, rf = self._crop(bgr, cur_box, LFC_SEARCH_FACTOR, LFC_SEARCH)
        if x_patch is None:
            self.stop()
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
        with self.lock:
            self.score = peak

        if peak < LFC_SCORE_MIN:
            with self.lock:
                if self.miss_t0 is None:
                    self.miss_t0 = time.time()
                over = (time.time() - self.miss_t0) > LFC_MAX_MISS_SEC
            if over:
                self.stop()
            return None
        with self.lock:
            self.miss_t0 = None

        ox, oy = offset[0, 0, iy, ix], offset[0, 1, iy, ix]
        bw, bh = size[0, 0, iy, ix], size[0, 1, iy, ix]
        cx = (ix + ox) / self.feat_sz * LFC_SEARCH / rf
        cy = (iy + oy) / self.feat_sz * LFC_SEARCH / rf
        w = bw * LFC_SEARCH / rf
        h = bh * LFC_SEARCH / rf

        px, py, pw, ph = cur_box
        w = pw + LFC_SIZE_ALPHA * (w - pw)
        h = ph + LFC_SIZE_ALPHA * (h - ph)
        half = 0.5 * LFC_SEARCH / rf
        rcx = cx + (px + 0.5 * pw - half)
        rcy = cy + (py + 0.5 * ph - half)
        nx = min(max(rcx - 0.5 * w, -w + 2), CAP_W - 2)
        ny = min(max(rcy - 0.5 * h, -h + 2), CAP_H - 2)
        new_box = (nx, ny, max(4.0, w), max(4.0, h))
        with self.lock:
            if self.box is not None:
                self.box = new_box
                self.t_result = time.time()

        if LFC_TPL_ALPHA > 0.0 and peak >= LFC_TPL_MIN:
            nz, _ = self._crop(bgr, new_box, LFC_TEMPLATE_FACTOR, LFC_TEMPLATE)
            if nz is not None:
                nz = cv2.cvtColor(nz, cv2.COLOR_BGR2RGB).astype(np.float32)
                with self.lock:
                    if self.z is not None:
                        self.z = np.clip(
                            self.z.astype(np.float32) * (1.0 - LFC_TPL_ALPHA)
                            + nz[None] * LFC_TPL_ALPHA, 0.0, 255.0).astype(np.uint8)
        return None


class HailoDetector:

    def __init__(self):
        from hailo_platform import (HEF, VDevice, ConfigureParams,
                                    HailoStreamInterface, InputVStreamParams,
                                    OutputVStreamParams, FormatType, InferVStreams)
        last_err = None
        for path in HEF_PATHS:
            try:
                hef = HEF(path)
                self.vdev = VDevice()
                self.ng = self.vdev.configure(
                    hef, ConfigureParams.create_from_hef(
                        hef, interface=HailoStreamInterface.PCIe))[0]
                self.in_name = hef.get_input_vstream_infos()[0].name
                self.out_name = hef.get_output_vstream_infos()[0].name
                self.pipe = InferVStreams(
                    self.ng,
                    InputVStreamParams.make(self.ng, format_type=FormatType.UINT8),
                    OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32))
                self.pipe.__enter__()
                self.act = self.ng.activate()
                self.act.__enter__()
                break
            except Exception as e:
                last_err = e
                self.pipe = None
        else:
            raise RuntimeError(last_err)

        self.lock = threading.Lock()
        self.pending = None
        self.dets = []
        self.ms = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _letterbox(bgr):
        h, w = bgr.shape[:2]
        s = min(DET_SIZE / w, DET_SIZE / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        out = np.full((DET_SIZE, DET_SIZE, 3), 114, np.uint8)
        ox, oy = (DET_SIZE - nw) // 2, (DET_SIZE - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = cv2.resize(bgr, (nw, nh))
        return out, s, ox, oy

    def submit(self, bgr):
        with self.lock:
            self.pending = bgr

    def latest(self):
        with self.lock:
            return self.dets, self.ms

    def _loop(self):
        while True:
            with self.lock:
                frame, self.pending = self.pending, None
            if frame is None:
                time.sleep(0.003)
                continue
            try:
                t0 = time.time()
                lb, s, ox, oy = self._letterbox(frame)
                rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
                res = self.pipe.infer({self.in_name: np.expand_dims(rgb, 0)})
                raw = res[self.out_name][0]
                dets = []
                for cls, arr in enumerate(raw):
                    a = np.asarray(arr).reshape(-1, 5)
                    for ymin, xmin, ymax, xmax, score in a[a[:, 4] >= DET_SCORE]:
                        x1 = (xmin * DET_SIZE - ox) / s
                        y1 = (ymin * DET_SIZE - oy) / s
                        x2 = (xmax * DET_SIZE - ox) / s
                        y2 = (ymax * DET_SIZE - oy) / s
                        dets.append((cls, float(score), x1, y1, x2, y2))
                with self.lock:
                    self.dets = dets
                    self.ms = (time.time() - t0) * 1000.0
            except Exception as e:
                time.sleep(0.05)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class DetectionTracker:

    def __init__(self):
        self.box = None
        self.cls = None
        self.miss = 0

    @property
    def active(self):
        return self.box is not None

    def stop(self):
        self.box = None
        self.cls = None
        self.miss = 0

    def start(self, dets, x, y):
        best = None
        for cls, score, x1, y1, x2, y2 in dets:
            px, py = (x2 - x1) * DET_CLICK_PAD, (y2 - y1) * DET_CLICK_PAD
            if x1 - px <= x <= x2 + px and y1 - py <= y <= y2 + py:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best[0]:
                    best = (area, cls, (x1, y1, x2, y2))
        if best is None:
            return False
        _, self.cls, self.box = best
        self.miss = 0
        return True

    def update(self, dets):
        best, best_iou = None, DET_MATCH_IOU
        for cls, score, x1, y1, x2, y2 in dets:
            if cls != self.cls:
                continue
            v = iou(self.box, (x1, y1, x2, y2))
            if v >= best_iou:
                best, best_iou = (x1, y1, x2, y2), v
        if best is None:
            self.miss += 1
            if self.miss > DET_MAX_MISS:
                self.stop()
                return None
            return self.center()
        self.box = best
        self.miss = 0
        return self.center()

    def center(self):
        if self.box is None:
            return None
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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

    def pan_range(self):
        with self.lock:
            w, h = self._win_for(self.zoom)
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
        self.duration = Gst.SECOND // FPS

        launch = (
            "appsrc name=src is-live=true format=time do-timestamp=true "
            f"block=false max-bytes=0 max-buffers={RTSP_QUEUE} leaky-type=downstream "
            f"caps=video/x-raw,format=BGR,width={OUT_W},height={OUT_H},framerate={FPS}/1 "
            "! videoconvert ! video/x-raw,format=I420 "
            f"! x264enc tune=zerolatency speed-preset={RTSP_PRESET} "
            f"bitrate={RTSP_BITRATE} key-int-max={RTSP_GOP} "
            f"vbv-buf-capacity={RTSP_VBV} "
            f"intra-refresh={'true' if RTSP_INTRA_REFRESH else 'false'} "
            "! rtph264pay name=pay0 pt=96 config-interval=1"
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
                    "port": self.port, "path": self.path}


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
        with self.lock:
            self.gain = list(values)


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
            s["pred"] = dict(self.pred)
            s["rotate"] = list(self.rotate)
            s["center"] = self.center_pending
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

    def set_pred(self, age, dx, dy, vx, vy):
        with self.lock:
            self.pred = {"age_ms": round(age * 1e3, 1),
                         "dx": round(dx, 1), "dy": round(dy, 1),
                         "vx": round(vx, 1), "vy": round(vy, 1),
                         "on": PRED_ENABLE}

    def set_det_info(self, on, n, ms, mode, boxes=None):
        with self.lock:
            self.det_info = {"on": on, "n": n, "ms": round(ms, 1), "mode": mode,
                             "boxes": boxes or []}

    def publish_frame(self, frame_id, inv, fps, crop, zoom, jitter=None):
        with self.lock:
            self.inv_hist.append((frame_id, inv))
            self.stats["fps"] = round(fps, 1)
            self.stats["crop"] = list(crop) if crop else None
            self.stats["zoom"] = round(zoom, 3)
            if jitter:
                self.stats["jitter"] = jitter


CAMERA_REF = {}
LFC_REF = {}

ptz = VirtualPTZ()
state = SharedState()
rtsp = None


def _off_axis_rad(px, half, tan_half):
    return math.atan((px - half) / half * tan_half)


def target_angles(tx, ty, vcx, vcy):
    hx, hy = CAP_W / 2.0, CAP_H / 2.0
    yaw = (_off_axis_rad(tx, hx, CAM_HFOV_TAN)
           - _off_axis_rad(vcx, hx, CAM_HFOV_TAN))
    pitch = (_off_axis_rad(vcy, hy, CAM_VFOV_TAN)
             - _off_axis_rad(ty, hy, CAM_VFOV_TAN))
    return math.degrees(yaw), math.degrees(pitch)


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
              _deg_x10(t["yaw"]) if on else 0, 0, 0, 0] + [0] * 10
    raw = struct.pack(FCC_TX_FMT, *(fields + [0]))
    return raw[:-1] + bytes([sum(raw[:-1]) & 0xFF])


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

    elif cmd == CMD_TEST_ZOOM_RAW:
        raw = struct.unpack_from("<H", msg, p)[0]
        f = min(1.0, raw / GCS_ZOOM_RAW_MAX)
        ptz.set_zoom(MIN_ZOOM + f * (MAX_ZOOM - MIN_ZOOM))


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
    rx, ry = ptz.pan_range()
    st["frame_id"] = frame_id
    st["pan"] = [round(cx), round(cy)]
    st["zoom"] = round(z, 2)
    st["zoom_max"] = round(MAX_ZOOM, 2)
    st["zoom_real"] = round(REAL_ZOOM, 2)
    st["zoom_rate"] = round(ptz.zoom_rate_now(), 3)
    st["pan_range"] = [round(rx), round(ry)]
    st["rtsp"] = rtsp.info() if rtsp is not None else {"on": False}
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
    dtracker = DetectionTracker()
    dets, det_ms = [], 0.0
    detector = None
    if USE_DETECTOR:
        try:
            detector = HailoDetector()
        except Exception as e:
            pass
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

    try:
        while True:
            _t_r0 = time.perf_counter()
            bgr, proc, crop = cam.read()
            _t_r1 = time.perf_counter()
            _now = time.time()
            dt_f = _now - t_last
            t_last = _now
            if bgr is None:
                cam.release()
                os._exit(1)

            follow, click, clear = state.begin_frame()
            show_dets = state.show_dets()
            frame_id += 1

            if detector is not None:
                detector.submit(bgr)
                dets, det_ms = detector.latest()
            lfc_active = False
            if lfc is not None:
                lfc.submit(bgr)
                lfc_box, lfc_center, lfc_score, lfc_t = lfc.snapshot()
                lfc_active = lfc_box is not None
            pred_dx, pred_dy, pred_age = 0.0, 0.0, 0.0
            tgt = None

            ptz.step(TAU_FOLLOW if ((lfc_active or tracker.active
                                     or dtracker.active) and follow) else TAU_PAN)
            if settle > 0:
                settle -= 1

            if clear:
                tracker.stop()
                dtracker.stop()
                if lfc is not None: lfc.stop()
                follow = False
                prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0

            inv = state.inv_for(click[2]) if click is not None else None
            if click is not None and inv is not None:
                capx, capy = apply_pt(inv, click[0], click[1])
                prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0
                tracker.stop()
                dtracker.stop()
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
                if lfc is not None:
                    lfc.request_start(bgr, box)
                    follow = True
                    state.set_follow(True)
                elif tracker.start(proc, capx * PROC_SCALE, capy * PROC_SCALE):
                    follow = True
                    state.set_follow(True)
                else:
                    follow = False
                    state.set_follow(False)

            if lfc_active or dtracker.active or tracker.active:
                if lfc_active:
                    c = (lfc_center[0] * PROC_SCALE,
                         lfc_center[1] * PROC_SCALE) if lfc_center else None
                elif dtracker.active:
                    cc = dtracker.update(dets)
                    c = (cc[0] * PROC_SCALE, cc[1] * PROC_SCALE) if cc else None
                else:
                    c = tracker.update(proc)
                if c is None:
                    follow = False
                    state.set_follow(False)
                    prev_tgt, prev_tgt_t, tvx, tvy = None, 0.0, 0.0, 0.0
                elif follow:
                    tx, ty = c[0] / PROC_SCALE, c[1] / PROC_SCALE
                    sx, sy = tx, ty
                    tgt = (tx, ty)
                    m_t = lfc_t if (lfc_active and lfc_t > 0.0) else _now
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
            x0f, y0f = cx - win_w / 2.0, cy - win_h / 2.0
            x0 = max(0, int(math.floor(x0f)))
            y0 = max(0, int(math.floor(y0f)))
            x1 = min(CAP_W, int(math.ceil(x0f + win_w)))
            y1 = min(CAP_H, int(math.ceil(y0f + win_h)))
            interp = cv2.INTER_AREA if (x1 - x0) > OUT_W else cv2.INTER_CUBIC
            out = cv2.resize(bgr[y0:y1, x0:x1], (OUT_W, OUT_H), interpolation=interp)

            sx = OUT_W / float(x1 - x0)
            sy = OUT_H / float(y1 - y0)
            M = np.array([
                [sx, 0.0, -sx * x0],
                [0.0, sy, -sy * y0],
            ], dtype=np.float64)

            if tgt is not None:
                px, py = apply_pt(M, tgt[0], tgt[1])
                vcx, vcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                state.set_fcc_target(
                    True,
                    max(-1.0, min(1.0, (px - OUT_W / 2.0) / (OUT_W / 2.0))),
                    max(-1.0, min(1.0, (OUT_H / 2.0 - py) / (OUT_H / 2.0))),
                    *target_angles(tgt[0], tgt[1], vcx, vcy))
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
            osd_master, osd_zoom = state.osd_flags()
            if osd_master:
                label = f"{fps_ema:4.1f}fps"
                if osd_zoom:
                    label += f"  ZOOM:{zoom:.2f}x"
                cv2.putText(out, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 128), 1, cv2.LINE_AA)
            if show_dets:
                for _c, _s, bx1, by1, bx2, by2 in dets:
                    p1 = apply_pt(M, bx1, by1)
                    p2 = apply_pt(M, bx2, by2)
                    cv2.rectangle(out, (int(p1[0]), int(p1[1])),
                                  (int(p2[0]), int(p2[1])), (0, 190, 90), 1, cv2.LINE_AA)
            if lfc_box is not None:
                bx, by, bw, bh = lfc_box
                bx += pred_dx
                by += pred_dy
                a = apply_pt(M, bx, by)
                b = apply_pt(M, bx + bw, by + bh)
                col = (0, 255, 255) if lfc_score >= LFC_SCORE_MIN else (0, 140, 255)
                cv2.rectangle(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                              col, 2, cv2.LINE_AA)
            elif dtracker.active and dtracker.box is not None:
                x1, y1, x2, y2 = dtracker.box
                a = apply_pt(M, x1, y1)
                b = apply_pt(M, x2, y2)
                cv2.rectangle(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                              (0, 255, 255), 2, cv2.LINE_AA)
                cc = dtracker.center()
                oc = apply_pt(M, cc[0], cc[1])
                cv2.circle(out, (int(oc[0]), int(oc[1])), 4, (0, 255, 255), -1, cv2.LINE_AA)
            elif tracker.active and tracker.center is not None:
                tx, ty = tracker.center[0] / PROC_SCALE, tracker.center[1] / PROC_SCALE
                ox, oy = apply_pt(M, tx, ty)
                cv2.circle(out, (int(ox), int(oy)), 18, (0, 255, 255), 2, cv2.LINE_AA)

            state.set_det_info(detector is not None, len(dets),
                               (lfc.ms if lfc is not None else det_ms),
                               ("lightfc" if lfc_active else
                                "detect" if dtracker.active else
                                "lk" if tracker.active else "none"),
                               [[c, round(sc, 2), int(a), int(b), int(x), int(y)]
                                for c, sc, a, b, x, y in dets[:12]])
            state.set_track_box(lfc_box, lfc_score)
            state.set_pred(pred_age, pred_dx, pred_dy, tvx, tvy)
            state.publish_frame(frame_id, cv2.invertAffineTransform(M),
                                fps_ema, crop, zoom,
                                frame_jitter(jit) if frame_id % 15 == 0 else None)

            if rtsp is not None:
                rtsp.push(out)
            if STATUS_FILE and frame_id % STATUS_EVERY == 0:
                try:
                    write_status(frame_id)
                except Exception:
                    pass
            t_loop_end = time.perf_counter()

    finally:
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
    pipeline()


if __name__ == "__main__":
    main()
