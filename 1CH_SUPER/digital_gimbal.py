import collections
import csv
import faulthandler
import math
import os
import pickle
import signal
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
from enum import Enum
from multiprocessing import shared_memory

import cv2
import numpy as np


def _env_int(k, d):
    return int(os.environ.get(k, str(d)))


def _env_float(k, d):
    return float(os.environ.get(k, str(d)))


def _env_path(k, *parts):
    return os.environ.get(k, os.path.join(*parts))


BASE_DIR   = os.environ.get("GIMBAL_HOME", os.path.expanduser("~"))
APP_DIR    = _env_path("APP_DIR", BASE_DIR, "1CH_SUPER")
MODELS_DIR = _env_path("MODELS_DIR", BASE_DIR, "models")

NANOTRACK_DIR  = _env_path("NANOTRACK_DIR", MODELS_DIR, "nanotrack")
HEF_SUPERPOINT = _env_path("HEF_SUPERPOINT", MODELS_DIR, "superpoint.hef")
LOG_DIR        = _env_path("LOG_DIR", APP_DIR, "logs")


CAP_W   = _env_int("CAP_W", 1280)
CAP_H   = _env_int("CAP_H", 720)
CAP_FPS = _env_int("CAP_FPS", 30)
CAP_DEV_FPS = _env_int("CAP_DEV_FPS", CAP_FPS)
CAM_INDEX = _env_int("CAM_INDEX", 0)
CAM_CTRLS = os.environ.get(
    "CAM_CTRLS",
    "backlight_compensation=0,brightness=-15,saturation=56,auto_exposure=3")

PROC_W = _env_int("PROC_W", 640)
PROC_H = _env_int("PROC_H", 360)

SP_W, SP_H       = _env_int("SP_W", 640), _env_int("SP_H", 400)

CAM_HFOV_DEG = _env_float("CAM_HFOV_DEG", 66.0)
CAM_VFOV_DEG = _env_float("CAM_VFOV_DEG", 41.0)

TRACK_BUDGET_MS = _env_float("TRACK_BUDGET_MS", 20.0)
TRACK_CONF_THRESH = _env_float("TRACK_CONF_THRESH", 0.5)
TRACK_APCE_MIN = _env_float("TRACK_APCE_MIN", 0.0)
BOX_MAX_FRAC = _env_float("BOX_MAX_FRAC", 0.5)
BOX_SCALE_LR = _env_float("BOX_SCALE_LR", 0.5)
BOX_AUDIT_LR = _env_float("BOX_AUDIT_LR", 0.3)
BOX_SHRINK_MAX = _env_float("BOX_SHRINK_MAX", 0.02)
BOX_MIN_FRAC = _env_float("BOX_MIN_FRAC", 0.5)
ROI_REFINE = os.environ.get("ROI_REFINE", "1") not in ("0", "", "false", "no")
ROI_MIN_PX = _env_float("ROI_MIN_PX", 16.0)
ROI_PAD = _env_float("ROI_PAD", 0.2)
ROI_MIN_Z = _env_float("ROI_MIN_Z", 4.0)
ROI_MAX_FILL = _env_float("ROI_MAX_FILL", 0.6)
ROI_MAX_OTHER = _env_float("ROI_MAX_OTHER", 0.05)
TRK_KF = os.environ.get("TRK_KF", "1") not in ("0", "", "false", "no")
TRK_KF_ACC = _env_float("TRK_KF_ACC", 40.0)
TRK_KF_MEAS = _env_float("TRK_KF_MEAS", 0.1)
TRK_KF_ANISO_MAX = _env_float("TRK_KF_ANISO_MAX", 80.0)
TRK_KF_ANISO_HOLD = _env_float("TRK_KF_ANISO_HOLD", 4.0)
TRK_KF_GATE = _env_float("TRK_KF_GATE", 6.6)
TRK_KF_STRIKES = _env_int("TRK_KF_STRIKES", 30)
BOX_SCALE_MIN = _env_float("BOX_SCALE_MIN", 0.5)
BOX_SCALE_MAX = _env_float("BOX_SCALE_MAX", 2.0)
TRACK_GRACE_S = _env_float("TRACK_GRACE_S", 0.5)
COLOR_MAX_D = _env_float("COLOR_MAX_D", 0.45)
COLOR0_MAX_D = _env_float("COLOR0_MAX_D", 0.7)
COLOR_EMA_GATE = _env_float("COLOR_EMA_GATE", 0.1)
COLOR_GRID = _env_int("COLOR_GRID", 16)
COLOR_EMA = _env_float("COLOR_EMA", 0.002)
TRACK_LOST_FRAMES = _env_int("TRACK_LOST_FRAMES", 10)


TERM_HOLD_FRAC   = _env_float("TERM_HOLD_FRAC", 0.12)
TERM_CONF_THRESH = _env_float("TERM_CONF_THRESH", 0.25)

YAW_KP          = _env_float("YAW_KP", 0.8)
YAW_RATE_MAX    = _env_float("YAW_RATE_MAX", 45.0)
SPEED_MAX       = _env_float("SPEED_MAX", 1.0)
TOF_TERMINAL_M  = _env_float("TOF_TERMINAL_M", 1.0)
TOF_CONTACT_M   = _env_float("TOF_CONTACT_M", 0.10)
REACQ_TIMEOUT_S = _env_float("REACQ_TIMEOUT_S", 10.0)

HAILO_RESULT_MAX_AGE_S = _env_float("HAILO_RESULT_MAX_AGE_S", 0.5)

STAB_ENABLE  = os.environ.get("STAB", "1") not in ("0", "", "false", "no")
STAB_W       = _env_int("STAB_W", 240)
STAB_H       = _env_int("STAB_H", 135)
STAB_ZOOM    = _env_float("STAB_ZOOM", 1.15)
STAB_FREE    = os.environ.get("STAB_FREE", "1") not in ("0", "", "false", "no")
STAB_FREE_PX = _env_float("STAB_FREE_PX", 400.0)
STAB_TAU     = _env_float("STAB_TAU", 0.60)
STAB_TAU_MIN = _env_float("STAB_TAU_MIN", 0.10)
STAB_TAU_MAX = _env_float("STAB_TAU_MAX", 2.00)
STAB_CORNERS = _env_int("STAB_CORNERS", 150)
STAB_QUALITY = _env_float("STAB_QUALITY", 0.01)
STAB_MIN_DIST = _env_int("STAB_MIN_DIST", 8)
STAB_MIN_PTS = _env_int("STAB_MIN_PTS", 12)
STAB_FB_ERR  = _env_float("STAB_FB_ERR", 1.0)
STAB_HIST    = _env_int("STAB_HIST", 96)
STAB_DEAD    = _env_float("STAB_DEAD", 0.0)
STAB_DEAD_DEG = _env_float("STAB_DEAD_DEG", 0.0)
STAB_WALL    = _env_float("STAB_WALL", 2.0)
STAB_STEP_MAX = _env_float("STAB_STEP_MAX", 0.0)
STAB_DC_TAU  = _env_float("STAB_DC_TAU", 0.0)
STAB_RS_MS   = _env_float("STAB_RS_MS", 0.0)
STAB_RS_TAU  = _env_float("STAB_RS_TAU", 0.08)
STAB_LP_TAU  = _env_float("STAB_LP_TAU", 0.3)
STAB_MODEL   = os.environ.get("STAB_MODEL", "rs")
STAB_RS_LEAK_TAU = _env_float("STAB_RS_LEAK_TAU", 0.3)
STAB_QUAD    = os.environ.get("STAB_QUAD", "1") not in ("0", "", "false", "no")
RTSP_BANDS   = _env_int("RTSP_BANDS", 8)
STAB_CLAHE   = _env_float("STAB_CLAHE", 0.0)
STAB_KX      = CAP_W / float(STAB_W)
STAB_KY      = CAP_H / float(STAB_H)
STAB_LK = dict(winSize=(15, 15), maxLevel=2,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         20, 0.03))
DT_MAX = 0.25

TERM_LOCK_FRAC      = _env_float("TERM_LOCK_FRAC", 0.35)
TERM_LOCK_TIMEOUT_S = _env_float("TERM_LOCK_TIMEOUT_S", 3.0)
COAST_S             = _env_float("COAST_S", 0.4)
SLEW_ANG_DEG_S      = _env_float("SLEW_ANG_DEG_S", 90.0)
SLEW_N_S            = _env_float("SLEW_N_S", 3.0)

WDG_FRAME_TIMEOUT_S = _env_float("WDG_FRAME_TIMEOUT_S", 0.1)
WDG_WARMUP_FRAMES   = _env_int("WDG_WARMUP_FRAMES", 15)
WDG_TEMP_LOG_S      = _env_float("WDG_TEMP_LOG_S", 5.0)

FAST_LOOP_CORES = {int(c) for c in os.environ.get("FAST_CORES", "0,1").split(",")}
HAILO_CORES     = {int(c) for c in os.environ.get("HAILO_CORES", "2").split(",")}

SHM_FRAME  = "1chs_frame"
SHM_RESULT = "1chs_result"
SHM_CTRL   = "1chs_ctrl"

ANCHOR_MAX_KP = _env_int("ANCHOR_MAX_KP", 200)
SP_CONF_THRESH = _env_float("SP_CONF_THRESH", 0.015)
SP_NMS_RADIUS  = _env_int("SP_NMS_RADIUS", 4)
REACQ_MIN_MATCHES = _env_int("REACQ_MIN_MATCHES", 12)
SP_ZOOM_TARGET = _env_float("SP_ZOOM_TARGET", 120.0)
SP_ZOOM_MAX    = _env_float("SP_ZOOM_MAX", 8.0)
SP_STRONG_MIN_PX = _env_float("SP_STRONG_MIN_PX", 72.0)
AUDIT_EVERY = _env_int("AUDIT_EVERY", 10)
AUDIT_FAILS = _env_int("AUDIT_FAILS", 5)
AUDIT_REFRESH = _env_int("AUDIT_REFRESH", 10)

RTSP_ENABLE  = os.environ.get("RTSP", "1") not in ("0", "", "false")
RTSP_PORT    = _env_int("RTSP_PORT", 554)
RTSP_PATH    = os.environ.get("RTSP_PATH", "/video0")
RTSP_W       = _env_int("RTSP_W", 1920)
RTSP_H       = _env_int("RTSP_H", 1080)
RTSP_FPS     = _env_int("RTSP_FPS", 12)
RTSP_BITRATE = _env_int("RTSP_BITRATE", 2500)
RTSP_PRESET  = os.environ.get("RTSP_PRESET", "veryfast")
RTSP_CODEC   = os.environ.get("RTSP_CODEC", "h264").lower()
RTSP_GOP     = _env_int("RTSP_GOP", RTSP_FPS * 2)
RTSP_VBV     = _env_int("RTSP_VBV", 300)
RTSP_QUEUE   = _env_int("RTSP_QUEUE", 3)
RTSP_INTRA_REFRESH = os.environ.get("RTSP_INTRA_REFRESH", "1") not in ("0", "false", "no")
OSD_BOX_STICKY_FRAC = _env_float("OSD_BOX_STICKY_FRAC", 0.06)
RTSP_X265_OPTS = os.environ.get(
    "RTSP_X265_OPTS",
    "no-rect=1:no-amp=1:wpp=1:pmode=1:pme=1:frame-threads=4:rd=1:me=0:subme=0")


GCS_UDP_PORT = _env_int("GCS_UDP_PORT", 37260)
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
CMD_TEST_DIGITAL_ZOOM = 20
CMD_TEST_ZOOM_RAW     = 22
CMD_STABILIZER_MODE   = 31
CMD_STABILIZER_ALPHA  = 32
GCS_STAB_RESET        = 0xFF

GCS_REF_W, GCS_REF_H = 1920, 1080
GCS_ZOOM_RAW_MAX = 0x4000
MAX_ZOOM     = _env_float("MAX_ZOOM", 5.0)
ZOOM_RATE    = _env_float("GCS_ZOOM_RATE", 2.0)
ZOOM_TIMEOUT = _env_float("GCS_ZOOM_TIMEOUT", 3.0)
DEADBAND_FRAC = _env_float("DEADBAND_FRAC", 0.03)
FOLLOW_TAU     = _env_float("FOLLOW_TAU", 0.15)
FOLLOW_TAU_OFF = _env_float("FOLLOW_TAU_OFF", 0.205)
FOLLOW_DEAD_FRAC = _env_float("FOLLOW_DEAD_FRAC", 0.35)
FOLLOW_DEAD_MIN  = _env_float("FOLLOW_DEAD_MIN", 4.0)

FCC_TX_HEADER1, FCC_TX_HEADER2 = 0xBB, 0x88
FCC_RX_HEADER1, FCC_RX_HEADER2 = 0xBB, 0x99
FCC_TX_FMT  = "<BBBffBffbBhhhhhh10bB"
FCC_TX_SIZE = struct.calcsize(FCC_TX_FMT)
FCC_RX_FMT  = "<BBfBffddffffffff32sB"
FCC_RX_SIZE = struct.calcsize(FCC_RX_FMT)
assert FCC_TX_SIZE == 45, FCC_TX_SIZE
assert FCC_RX_SIZE == 96, FCC_RX_SIZE

FCC_PORT  = os.environ.get("FCC_PORT", "/dev/ttyAMA3")
FCC_BAUD  = _env_int("FCC_BAUD", 115200)
FCC_RETRY = _env_float("FCC_RETRY", 2.0)

TOF_I2C_BUS  = _env_int("TOF_I2C_BUS", 1)
TOF_I2C_ADDR = _env_int("TOF_I2C_ADDR", 0x29)
BUMPER_GPIO  = _env_int("BUMPER_GPIO", 17)


if sys.version_info >= (3, 13):
    def _shm_attach(name):
        return shared_memory.SharedMemory(name=name, track=False)
else:
    from multiprocessing import resource_tracker as _resource_tracker

    def _shm_attach(name):
        _real = _resource_tracker.register
        _resource_tracker.register = lambda *a, **k: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            _resource_tracker.register = _real


_HDR = struct.Struct("<IdQI")


class _Slot:
    def __init__(self, name, size, create):
        self.name = name
        if create:
            try:
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True,
                                                  size=_HDR.size + size)
            self.shm.buf[:_HDR.size] = _HDR.pack(0, 0.0, 0, 0)
        else:
            self.shm = _shm_attach(name)
        self._owner = create

    def _write(self, payload, t, frame_id):
        seq, _, _, _ = _HDR.unpack_from(self.shm.buf, 0)
        seq += 1
        _HDR.pack_into(self.shm.buf, 0, seq, t, frame_id, len(payload))
        self.shm.buf[_HDR.size:_HDR.size + len(payload)] = payload
        _HDR.pack_into(self.shm.buf, 0, seq + 1, t, frame_id, len(payload))

    def _read(self, retries=4):
        for _ in range(retries):
            seq0, t, fid, n = _HDR.unpack_from(self.shm.buf, 0)
            if seq0 == 0 or seq0 & 1 or n == 0:
                continue
            payload = bytes(self.shm.buf[_HDR.size:_HDR.size + n])
            seq1, _, _, _ = _HDR.unpack_from(self.shm.buf, 0)
            if seq0 == seq1:
                return payload, t, fid, seq0
        return None, 0.0, 0, 0

    def close(self):
        self.shm.close()


class FrameSlot(_Slot):

    def __init__(self, name, shape, dtype=np.uint8, create=False):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        super().__init__(name, self.nbytes, create)

    def write(self, img, t, frame_id):
        assert img.nbytes == self.nbytes, (img.shape, self.shape)
        self._write(img.tobytes(), t, frame_id)

    def read(self):
        payload, t, fid, seq = self._read()
        if payload is None:
            return None, 0.0, 0, 0
        img = np.frombuffer(payload, self.dtype).reshape(self.shape)
        return img, t, fid, seq


class BlobSlot(_Slot):
    def __init__(self, name, size=1 << 20, create=False):
        super().__init__(name, size, create)

    def write(self, obj, t, frame_id=0):
        self._write(pickle.dumps(obj, protocol=4), t, frame_id)

    def read(self):
        payload, t, fid, seq = self._read()
        if payload is None:
            return None, 0.0, 0, 0
        return pickle.loads(payload), t, fid, seq


class State(Enum):
    IDLE = 0
    TRACK = 1
    REACQUIRE = 2
    TERMINAL = 3
    CONTACT = 4
    ESTOP = 5


class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self._entered = time.monotonic()
        self.reason = ""

    def _to(self, s, reason=""):
        if s is not self.state:
            self.state = s
            self._entered = time.monotonic()
            self.reason = reason

    @property
    def age(self):
        return time.monotonic() - self._entered

    def on_target_selected(self):
        if self.state in (State.IDLE, State.TRACK, State.REACQUIRE):
            self._to(State.TRACK, "target selected")

    def on_target_cleared(self):
        if self.state is not State.ESTOP:
            self._to(State.IDLE, "target cleared")

    def on_watchdog_trip(self, why):
        self._to(State.ESTOP, why)

    def reset(self):
        self._to(State.IDLE, "reset")

    def step(self, track_ok, tof_m, bumper):
        s = self.state
        if s in (State.IDLE, State.ESTOP, State.CONTACT):
            return s

        if bumper or (tof_m is not None and tof_m <= TOF_CONTACT_M):
            self._to(State.CONTACT, f"bumper={bumper} tof={tof_m}")
            return self.state

        if s is State.TRACK:
            if not track_ok:
                self._to(State.REACQUIRE, "tracker lost")
            elif tof_m is not None and tof_m <= TOF_TERMINAL_M:
                self._to(State.TERMINAL, f"tof={tof_m}")

        elif s is State.REACQUIRE:
            if track_ok:
                self._to(State.TRACK, "reacquired")
            elif self.age > REACQ_TIMEOUT_S:
                self._to(State.IDLE, "reacquire timeout")

        elif s is State.TERMINAL:
            if not track_ok and tof_m is None:
                self._to(State.REACQUIRE, "terminal lost")

        return self.state


_HTAN = math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
_VTAN = math.tan(math.radians(CAM_VFOV_DEG) / 2.0)


def target_angles(cx, cy):
    yaw = math.degrees(math.atan((cx - PROC_W / 2.0) / (PROC_W / 2.0) * _HTAN))
    pitch = -math.degrees(math.atan((cy - PROC_H / 2.0) / (PROC_H / 2.0) * _VTAN))
    return yaw, pitch


def _soft_deadband(e, band):
    if e > band:
        return e - band
    if e < -band:
        return e + band
    return 0.0


def speed_profile(state):
    if state in (State.IDLE, State.REACQUIRE, State.CONTACT, State.ESTOP):
        return 0.0
    return SPEED_MAX


def view_norm(view, cx, cy):
    if view is None:
        return (max(-1.0, min(1.0, (cx - PROC_W / 2.0) / (PROC_W / 2.0))),
                max(-1.0, min(1.0, (PROC_H / 2.0 - cy) / (PROC_H / 2.0))))
    z, fx, fy, warp = view[:4]
    cw3 = crop_matrix(CAP_W, CAP_H, z, fx, fy)
    M = (cw3[:2] if warp is None
         else (cw3 @ np.vstack([warp, (0.0, 0.0, 1.0)]))[:2])
    px, py = apply_pt(M, cx * CAP_W / PROC_W, cy * CAP_H / PROC_H)
    return (max(-1.0, min(1.0, (px - RTSP_W / 2.0) / (RTSP_W / 2.0))),
            max(-1.0, min(1.0, (RTSP_H / 2.0 - py) / (RTSP_H / 2.0))))


def control_step(state, box, view=None):
    if state in (State.TRACK, State.TERMINAL) and box is not None:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        ex = _soft_deadband(cx - PROC_W / 2.0, DEADBAND_FRAC * PROC_W / 2.0)
        ey = _soft_deadband(cy - PROC_H / 2.0, DEADBAND_FRAC * PROC_H / 2.0)
        yaw, pitch = target_angles(PROC_W / 2.0 + ex, PROC_H / 2.0 + ey)
        yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, YAW_KP * yaw))

        nx, ny = view_norm(view, cx, cy)
        return {"valid": True,
                "nx": nx,
                "ny": ny,
                "yaw": yaw, "pitch": pitch,
                "yaw_rate": yaw_rate,
                "speed": speed_profile(state)}
    return {"valid": False, "nx": 0.0, "ny": 0.0,
            "yaw": 0.0, "pitch": 0.0, "yaw_rate": 0.0, "speed": 0.0}


def _box_frac(box):
    if box is None:
        return 0.0
    return (box[2] * box[3]) / (PROC_W * PROC_H)


class CmdFilter:
    def __init__(self):
        self.last = None
        self.last_t = 0.0
        self.lost_t = None
        self.lock_t = None

    def reset(self):
        self.last = None
        self.lost_t = None
        self.lock_t = None

    def _slew(self, cmd, now):
        if self.last is None or not self.last["valid"]:
            return cmd
        dt = max(1e-3, min(0.2, now - self.last_t))
        out = dict(cmd)
        for k, lim in (("yaw", SLEW_ANG_DEG_S), ("pitch", SLEW_ANG_DEG_S),
                       ("yaw_rate", SLEW_ANG_DEG_S),
                       ("nx", SLEW_N_S), ("ny", SLEW_N_S)):
            d = cmd[k] - self.last[k]
            m = lim * dt
            if d > m:
                out[k] = self.last[k] + m
            elif d < -m:
                out[k] = self.last[k] - m
        return out

    def step(self, state, cmd, box, now=None):
        now = time.monotonic() if now is None else now

        if state in (State.IDLE, State.CONTACT, State.ESTOP):
            self.reset()
            return cmd

        if self.lock_t is not None:
            if now - self.lock_t <= TERM_LOCK_TIMEOUT_S and self.last is not None:
                return dict(self.last)
            self.reset()
            return cmd

        if cmd["valid"]:
            if _box_frac(box) >= TERM_LOCK_FRAC:
                self.lock_t = now
                out = self._slew(cmd, now)
                self.last, self.last_t = dict(out), now
                return dict(out)
            self.lost_t = None
            out = self._slew(cmd, now)
            self.last, self.last_t = dict(out), now
            return out

        if self.last is not None and self.last["valid"]:
            if self.lost_t is None:
                self.lost_t = now
            if now - self.lost_t <= COAST_S:
                return dict(self.last)
        self.last = None
        return cmd


EXEMPLAR = 127
INSTANCE = 255
SCORE_SZ = 16
STRIDE = 16
CONTEXT = 0.5
PENALTY_K = 0.148
WIN_INFLUENCE = 0.462
LR = 0.390


class NanoTracker:
    name = "nanotrack"

    def __init__(self):
        import ncnn
        self.ncnn = ncnn
        bb = os.path.join(NANOTRACK_DIR, "nanotrack_backbone_sim-opt")
        hd = os.path.join(NANOTRACK_DIR, "nanotrack_head_sim-opt")
        for p in (bb + ".param", bb + ".bin", hd + ".param", hd + ".bin"):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        self.backbone = ncnn.Net()
        self.backbone.opt.num_threads = 2
        self.backbone.load_param(bb + ".param")
        self.backbone.load_model(bb + ".bin")
        self.head = ncnn.Net()
        self.head.opt.num_threads = 2
        self.head.load_param(hd + ".param")
        self.head.load_model(hd + ".bin")

        g = np.arange(SCORE_SZ, dtype=np.float32) * STRIDE
        self.grid_x = np.tile(g, (SCORE_SZ, 1))
        self.grid_y = self.grid_x.T
        han = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SCORE_SZ) / (SCORE_SZ - 1))
        self.window = np.outer(han, han).astype(np.float32)

        self.zf = None
        self.pos = None
        self.sz = None
        self.avg = None
        self.box = None
        self.score = 0.0
        self.apce = 0.0
        self.color_d = 0.0
        self.color_d0 = 0.0
        self.ref_chroma = None
        self.ref0 = None
        self.last_ms = 0.0
        self._miss = 0
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((240, 320, 3), np.uint8)
        self.start(dummy, (120, 90, 60, 60))
        self.update(dummy)
        self.stop()

    def _subwindow(self, img, pos, model_sz, original_sz):
        c = (original_sz + 1) / 2.0
        x0 = int(round(pos[0] - c))
        y0 = int(round(pos[1] - c))
        x1 = x0 + original_sz - 1
        y1 = y0 + original_sz - 1
        lp = max(0, -x0)
        tp = max(0, -y0)
        rp = max(0, x1 - img.shape[1] + 1)
        bp = max(0, y1 - img.shape[0] + 1)
        x0 += lp
        x1 += lp
        y0 += tp
        y1 += tp
        if lp or tp or rp or bp:
            img = cv2.copyMakeBorder(img, tp, bp, lp, rp,
                                     cv2.BORDER_CONSTANT, value=self.avg)
        patch = img[y0:y1 + 1, x0:x1 + 1]
        return cv2.resize(patch, (model_sz, model_sz))

    @staticmethod
    def _grid_at(img, cx, cy, w, h):
        x0 = int(np.clip(cx - w * 0.45, 0, img.shape[1] - 2))
        y0 = int(np.clip(cy - h * 0.45, 0, img.shape[0] - 2))
        x1 = int(np.clip(cx + w * 0.45, x0 + 1, img.shape[1]))
        y1 = int(np.clip(cy + h * 0.45, y0 + 1, img.shape[0]))
        g = COLOR_GRID
        cells = cv2.resize(img[y0:y1, x0:x1], (g, g),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        s = cells.sum(axis=2, keepdims=True) + 1e-6
        return cells / s

    @staticmethod
    def _grid_dist(a, b):
        return float(np.abs(a - b).sum(axis=2).mean())

    def _box_grid(self, img):
        return self._grid_at(img, float(self.pos[0]), float(self.pos[1]),
                             float(self.sz[0]), float(self.sz[1]))

    def matches_ref(self, img, box):
        if self.ref_chroma is None:
            return True
        x, y, w, h = box
        g = self._grid_at(img, x + w / 2.0, y + h / 2.0, w, h)
        if self._grid_dist(self.ref_chroma, g) > COLOR_MAX_D * 0.7:
            return False
        if self.ref0 is None:
            return True
        return self._grid_dist(self.ref0, g) <= COLOR0_MAX_D * 0.7

    def _extract(self, patch):
        m = self.ncnn.Mat.from_pixels(np.ascontiguousarray(patch),
                                      self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                                      patch.shape[1], patch.shape[0])
        ex = self.backbone.create_extractor()
        ex.set_light_mode(True)
        ex.input("input", m)
        _, out = ex.extract("output")
        return out

    def start(self, img, box, keep_ref=False):
        x, y, w, h = box
        self.pos = np.array([x + w / 2.0, y + h / 2.0], np.float32)
        self.sz = np.array([w, h], np.float32)
        self.avg = tuple(cv2.mean(img)[:3])
        wc = w + CONTEXT * (w + h)
        hc = h + CONTEXT * (w + h)
        s_z = round(np.sqrt(wc * hc))
        z = self._subwindow(img, self.pos, EXEMPLAR, int(s_z))
        self.zf = self._extract(z)
        self.box = tuple(box)
        self.score = 1.0
        self._miss = 0
        self._start_t = time.monotonic()
        if not (keep_ref and self.ref_chroma is not None):
            self.ref_chroma = self._box_grid(img)
            self.ref0 = self.ref_chroma.copy()
        self.aspect = float(w) / float(h)
        self.sz0 = (float(w), float(h))
        self.color_d = 0.0
        self.color_d0 = 0.0

    def rescale(self, h, img):
        h = float(np.clip(h, 10, img.shape[0]))
        self.sz[1] = h
        self.sz[0] = float(np.clip(h * self.aspect, 10, img.shape[1]))

    def feedforward(self, dx, dy):

        if self.zf is not None:
            self.pos[0] += dx
            self.pos[1] += dy

    def update(self, img):
        if self.zf is None:
            return False, None
        t0 = time.perf_counter()

        w, h = self.sz
        wc = w + CONTEXT * (w + h)
        hc = h + CONTEXT * (w + h)
        s_z = np.sqrt(wc * hc)
        scale_z = EXEMPLAR / s_z
        pad = (INSTANCE - EXEMPLAR) / 2.0 / scale_z
        s_x = s_z + 2 * pad

        x_crop = self._subwindow(img, self.pos, INSTANCE, int(s_x))
        xf = self._extract(x_crop)

        ex = self.head.create_extractor()
        ex.set_light_mode(True)
        ex.input("input1", self.zf)
        ex.input("input2", xf)
        _, cls = ex.extract("output1")
        _, reg = ex.extract("output2")
        cls = np.array(cls)
        reg = np.array(reg)

        score = 1.0 / (1.0 + np.exp(-cls[1]))
        x1 = self.grid_x - reg[0]
        y1 = self.grid_y - reg[1]
        x2 = self.grid_x + reg[2]
        y2 = self.grid_y + reg[3]
        pw = x2 - x1
        ph = y2 - y1

        tz = self.sz * scale_z
        pad_wh = (tz[0] + tz[1]) * 0.5
        sz_wh = np.sqrt((tz[0] + pad_wh) * (tz[1] + pad_wh))
        pad_p = (pw + ph) * 0.5
        s_c = np.sqrt((pw + pad_p) * (ph + pad_p)) / sz_wh
        s_c = np.maximum(s_c, 1.0 / s_c)
        ratio = tz[0] / tz[1]
        r_c = ratio / (pw / ph)
        r_c = np.maximum(r_c, 1.0 / r_c)
        penalty = np.exp(-(s_c * r_c - 1.0) * PENALTY_K)

        pscore = (penalty * score * (1 - WIN_INFLUENCE) +
                  self.window * WIN_INFLUENCE)
        r, c = np.unravel_index(np.argmax(pscore), pscore.shape)

        smin, smax = float(score.min()), float(score.max())
        self.apce = (smax - smin) ** 2 / (float(np.mean((score - smin) ** 2)) + 1e-12)

        px = (x1[r, c] + x2[r, c]) / 2.0
        py = (y1[r, c] + y2[r, c]) / 2.0
        bw = (x2[r, c] - x1[r, c]) / scale_z
        bh = (y2[r, c] - y1[r, c]) / scale_z
        dx = (px - INSTANCE / 2.0) / scale_z
        dy = (py - INSTANCE / 2.0) / scale_z

        lr = penalty[r, c] * score[r, c] * LR
        self.pos[0] = np.clip(self.pos[0] + dx, 0, img.shape[1])
        self.pos[1] = np.clip(self.pos[1] + dy, 0, img.shape[0])

        self.score = float(score[r, c])
        self.last_ms = (time.perf_counter() - t0) * 1e3

        self.frac = float((self.sz[0] * self.sz[1]) /
                          (img.shape[1] * img.shape[0]))
        cur = self._box_grid(img)
        self.color_d = self._grid_dist(self.ref_chroma, cur)
        self.color_d0 = (self._grid_dist(self.ref0, cur)
                         if self.ref0 is not None else 0.0)
        color_bad = (self.color_d > COLOR_MAX_D
                     or self.color_d0 > COLOR0_MAX_D)
        conf = (TERM_CONF_THRESH if self.frac >= TERM_HOLD_FRAC
                else TRACK_CONF_THRESH)
        if time.monotonic() - self._start_t < TRACK_GRACE_S:
            self._miss = self._miss + 1 if color_bad else 0
        elif (self.score < conf or self.apce < TRACK_APCE_MIN
                or color_bad):
            self._miss += 1
        else:
            self._miss = 0
            if self.color_d < COLOR_EMA_GATE:
                self.ref_chroma = ((1 - COLOR_EMA) * self.ref_chroma
                                   + COLOR_EMA * cur)
        if self._miss == 0:
            s_pred = np.sqrt(max(1e-6, bw * bh) /
                             max(1e-6, self.sz[0] * self.sz[1]))
            lr_s = lr * BOX_SCALE_LR
            s_new = (1 - lr_s) + s_pred * lr_s
            s_new = max(s_new, 1.0 - BOX_SHRINK_MAX)
            h_floor = max(10.0, BOX_MIN_FRAC * self.sz0[1])
            h_new = np.clip(self.sz[1] * s_new, h_floor, img.shape[0])
            self.sz[1] = h_new
            self.sz[0] = np.clip(h_new * self.aspect, 10, img.shape[1])
        if self._miss >= TRACK_LOST_FRAMES:
            self.box = None
            return False, None

        self.box = (float(self.pos[0] - self.sz[0] / 2),
                    float(self.pos[1] - self.sz[1] / 2),
                    float(self.sz[0]), float(self.sz[1]))
        return True, self.box

    def stop(self):
        self.zf = None
        self.box = None
        self._miss = 0

    @property
    def active(self):
        return self.zf is not None


class DisabledTracker:
    name = "disabled"
    active = False

    def __init__(self):
        self.last_ms = 0.0
        self.score = 0.0
        self.apce = 0.0
        self.frac = 0.0

    def start(self, img, box, keep_ref=False):
        pass

    def update(self, img):
        return False, None

    def feedforward(self, dx, dy):
        pass

    def stop(self):
        pass


def make_tracker():
    return NanoTracker()


_LK = dict(winSize=(15, 15), maxLevel=2,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def apply_pt(m, x, y):
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2],
            m[1, 0] * x + m[1, 1] * y + m[1, 2])


def xform_box(m, box):
    x, y, w, h = box
    cx, cy = apply_pt(m, x + 0.5 * w, y + 0.5 * h)
    sc = math.hypot(m[0, 0], m[1, 0])
    w *= sc
    h *= sc
    return (cx - 0.5 * w, cy - 0.5 * h, w, h)


def crop_matrix(W, H, z, fx, fy):
    cw = max(2.0, W / z)
    ch = max(2.0, H / z)
    x0 = (W - cw) / 2.0 + fx * W / PROC_W
    y0 = (H - ch) / 2.0 + fy * H / PROC_H
    x0 = max(0.0, min(W - cw, x0))
    y0 = max(0.0, min(H - ch, y0))
    return np.array([[RTSP_W / cw, 0.0, -RTSP_W / cw * x0],
                     [0.0, RTSP_H / ch, -RTSP_H / ch * y0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _gain_map(v, lo, mid, hi):
    v = max(-100.0, min(100.0, float(v)))
    if v >= 0.0:
        return mid + (hi - mid) * v / 100.0
    return mid + (mid - lo) * v / 100.0


def _gain_up(v, base, ratio):
    g = max(0.0, min(100.0, float(v))) / 100.0
    return base * (ratio ** g)


class Stabilizer:

    def __init__(self):
        self.lock = threading.Lock()
        self.clahe = (cv2.createCLAHE(clipLimit=STAB_CLAHE, tileGridSize=(8, 8))
                      if STAB_CLAHE > 0 else None)
        self.enabled = STAB_ENABLE
        self.tau = STAB_TAU
        self.corners = STAB_CORNERS
        self.min_pts = STAB_MIN_PTS
        self.free_px = STAB_FREE_PX
        self.lp_tau = STAB_LP_TAU
        self.rs_leak_tau = STAB_RS_LEAK_TAU
        self.ms = 0.0
        self.hist = collections.deque(maxlen=STAB_HIST)
        self.reset()

    def reset(self):
        with self.lock:
            self._reset_req = False
            self.hist.clear()
            self.prev = None
            self.W = np.eye(3)
            self.warp = None
            self.step_d = None
            self.response = 0
            self.fail = 0
            self.sat = 0.0
            self.clock = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.dcx = 0.0
            self.dcy = 0.0
            self.wsx = 0.0
            self.wsy = 0.0
            self.qx = 0.0
            self.qy = 0.0
            self.quad = None

    def request_reset(self):
        self._reset_req = True

    def set_mode(self, v):
        if v == GCS_STAB_RESET:
            self.reset()
            return
        on = bool(v)
        if on and not self.enabled:
            self.reset()
        self.enabled = on

    def set_alpha(self, a):
        a = max(0, min(100, int(a))) / 100.0
        with self.lock:
            self.tau = STAB_TAU_MIN + a * (STAB_TAU_MAX - STAB_TAU_MIN)

    def set_gain(self, v):
        if len(v) < 10:
            return
        with self.lock:
            self.free_px = _gain_up(v[7], STAB_FREE_PX, 3.0)
            self.lp_tau = _gain_up(v[8], STAB_LP_TAU, 5.0)
            self.rs_leak_tau = _gain_up(v[9], STAB_RS_LEAK_TAU, 5.0)

    def budget(self):
        with self.lock:
            return (self.free_px, self.free_px)

    def _gray(self, proc):
        g = proc if proc.ndim == 2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        if g.shape[1] != STAB_W or g.shape[0] != STAB_H:
            g = cv2.resize(g, (STAB_W, STAB_H), interpolation=cv2.INTER_AREA)
        return self.clahe.apply(g) if self.clahe is not None else g

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
        B[0, 0] = 1.0 + s * (W[0, 0] - 1.0)
        B[1, 1] = 1.0 + s * (W[1, 1] - 1.0)
        B[1, 0] = s * W[1, 0]
        B[0, 1] = s * W[0, 1]
        B[0, 2] = s * W[0, 2]
        B[1, 2] = s * W[1, 2]
        return B

    @staticmethod
    def _fit_rs(a, b):
        P = a.reshape(-1, 2)
        D = (b - a).reshape(-1, 2)
        x = P[:, 0] - STAB_W / 2.0
        y = P[:, 1] - STAB_H / 2.0
        hh = (STAB_H / 2.0) ** 2
        if STAB_QUAD:
            A = np.column_stack((np.ones_like(x), y, x, y * y / hh))
        else:
            A = np.column_stack((np.ones_like(x), y, x))

        def rfit(v):
            w = np.ones(len(v))
            for _ in range(3):
                sol = np.linalg.lstsq(A * w[:, None], v * w, rcond=None)[0]
                r = v - A @ sol
                sc = 1.4826 * np.median(np.abs(r)) + 1e-3
                w = (np.abs(r) < 2.5 * sc).astype(np.float64)
            return sol

        sx = rfit(D[:, 0])
        sy = rfit(D[:, 1])
        m = np.eye(3)[:2].copy()
        m[0, 0] = 1.0 + sx[2]
        m[0, 1] = sx[1]
        m[1, 0] = sy[2]
        m[1, 1] = 1.0 + sy[1]
        m[0, 2] = sx[0] - sx[1] * STAB_H / 2.0 - sx[2] * STAB_W / 2.0
        m[1, 2] = sy[0] - sy[1] * STAB_H / 2.0 - sy[2] * STAB_W / 2.0
        q = (float(sx[3]) / hh, float(sy[3]) / hh) if STAB_QUAD else (0.0, 0.0)
        return m, (float(sx[0]), float(sy[0])), q

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
        if STAB_MODEL == "rs":
            return W
        sc = math.hypot(W[0, 0], W[1, 0])
        ang = math.atan2(W[1, 0], W[0, 0])
        if abs(ang) > 1e-12:
            a2 = math.copysign(max(0.0, abs(ang) - math.radians(STAB_DEAD_DEG)), ang)
            W[0, 0] = W[1, 1] = sc * math.cos(a2)
            W[1, 0] = sc * math.sin(a2)
            W[0, 1] = -W[1, 0]
        return W

    def update(self, proc, fit, budget, mask_box=None):
        if not self.enabled or proc is None:
            self.warp = None
            self.quad = None
            self.step_d = None
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
            self.step_d = None
            return None

        mask = None
        if mask_box is not None:
            x, y, w, h = mask_box
            px, py = 0.1 * w, 0.1 * h
            mask = np.full((STAB_H, STAB_W), 255, np.uint8)
            mask[max(0, int((y - py) * STAB_H / PROC_H)):
                 int((y + h + py) * STAB_H / PROC_H) + 1,
                 max(0, int((x - px) * STAB_W / PROC_W)):
                 int((x + w + px) * STAB_W / PROC_W) + 1] = 0

        with self.lock:
            corners, min_pts = self.corners, self.min_pts
        m = None
        q_step = (0.0, 0.0)
        p0 = cv2.goodFeaturesToTrack(self.prev, maxCorners=corners,
                                     qualityLevel=STAB_QUALITY,
                                     minDistance=STAB_MIN_DIST, blockSize=7,
                                     mask=mask)
        if p0 is not None and len(p0) >= min_pts:
            a, b = self._track(self.prev, g, p0)
            self.response = len(a)
            if len(a) >= min_pts:
                if STAB_MODEL == "rs":
                    m, ctr_d, q_step = self._fit_rs(a, b)
                elif STAB_MODEL == "trans":
                    d = np.median((b - a).reshape(-1, 2), axis=0)
                    m = np.array([[1.0, 0.0, d[0]], [0.0, 1.0, d[1]]])
                else:
                    m, _ = cv2.estimateAffinePartial2D(
                        a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        else:
            self.response = 0
        self.prev = g

        if m is None:
            self.fail += 1
            step = np.eye(3)
            self.step_d = None
        else:
            self.fail = 0
            step = np.vstack([m, (0.0, 0.0, 1.0)])
            if STAB_MODEL == "rs":
                self.step_d = (ctr_d[0] * PROC_W / STAB_W,
                               ctr_d[1] * PROC_H / STAB_H)
            else:
                self.step_d = (step[0, 2] * PROC_W / STAB_W,
                               step[1, 2] * PROC_H / STAB_H)

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

        det = step[0, 0] * step[1, 1] - step[0, 1] * step[1, 0]
        step_inv = np.eye(3)
        if abs(det) >= 1e-9:
            step_inv[0, 0] = step[1, 1] / det
            step_inv[1, 1] = step[0, 0] / det
            step_inv[0, 1] = -step[0, 1] / det
            step_inv[1, 0] = -step[1, 0] / det
            step_inv[0, 2] = -(step_inv[0, 0] * step[0, 2]
                               + step_inv[0, 1] * step[1, 2])
            step_inv[1, 2] = -(step_inv[1, 0] * step[0, 2]
                               + step_inv[1, 1] * step[1, 2])
        W = self.W @ step_inv

        if dt > 0.0:
            with self.lock:
                lp_tau, rs_leak_tau = self.lp_tau, self.rs_leak_tau
            a_lp = 1.0 - math.exp(-dt / lp_tau)
            self.wsx += a_lp * (W[0, 2] * STAB_KX - self.wsx)
            self.wsy += a_lp * (W[1, 2] * STAB_KY - self.wsy)
        u = min(1.0, max(abs(self.wsx) / budget[0],
                         abs(self.wsy) / budget[1]))
        with self.lock:
            tau = self.tau / (1.0 + STAB_WALL * u * u)
        alpha = 1.0 - math.exp(-dt / tau) if dt > 0.0 else 0.0
        tx = W[0, 2] - alpha * self.wsx / STAB_KX
        ty = W[1, 2] - alpha * self.wsy / STAB_KY
        self.wsx -= alpha * self.wsx
        self.wsy -= alpha * self.wsy
        W[0, 2], W[1, 2] = tx, ty
        if dt <= 0.0:
            with self.lock:
                rs_leak_tau = self.rs_leak_tau
        a_rs = 1.0 - math.exp(-dt / rs_leak_tau) if dt > 0.0 else 0.0
        self.qx = (self.qx - q_step[0]) * (1.0 - a_rs)
        self.qy = (self.qy - q_step[1]) * (1.0 - a_rs)
        if a_rs > 0.0:
            ctr = np.array([STAB_W / 2.0, STAB_H / 2.0])
            t_c = W[:2, :2] @ ctr + W[:2, 2]
            W[:2, :2] = np.eye(2) + (1.0 - a_rs) * (W[:2, :2] - np.eye(2))
            W[:2, 2] = t_c - W[:2, :2] @ ctr
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
            self.wsx *= s
            self.wsy *= s
            self.qx *= s
            self.qy *= s
        self.sat = 1.0 - s
        self.warp = np.ascontiguousarray(total[:2])
        self.quad = ((self.qx * STAB_KX / (STAB_KY * STAB_KY),
                      self.qy / STAB_KY)
                     if STAB_QUAD and STAB_MODEL == "rs" else None)
        self.hist.append((time.time(), step))
        self.ms = (time.perf_counter() - t0) * 1e3
        return self.warp


class DigitalZoom:
    def __init__(self):
        self._z = 1.0
        self._rate = 0
        self._t = time.monotonic()
        self._rate_t = 0.0
        self._lock = threading.Lock()

    def _step(self, now):
        dt = max(0.0, now - self._t)
        self._t = now
        if self._rate and now - self._rate_t > ZOOM_TIMEOUT:
            self._rate = 0
        if self._rate and dt > 0:
            self._z = min(MAX_ZOOM,
                          max(1.0, self._z * (ZOOM_RATE ** (self._rate * dt))))

    def set_rate(self, d):
        with self._lock:
            now = time.monotonic()
            self._step(now)
            self._rate = max(-1, min(1, int(d)))
            self._rate_t = now

    def set_zoom(self, z):
        with self._lock:
            self._step(time.monotonic())
            self._rate = 0
            self._z = min(MAX_ZOOM, max(1.0, float(z)))

    def value(self):
        with self._lock:
            self._step(time.monotonic())
            return self._z


class FollowPan:

    def __init__(self):
        self._lock = threading.Lock()
        self._x = self._y = 0.0
        self._rx = self._ry = 0.0
        self._t = time.monotonic()

    def update(self, box):
        now = time.monotonic()
        with self._lock:
            dt = max(0.0, min(0.2, now - self._t))
            self._t = now
            if box is None:
                self._rx = self._ry = 0.0
                tx = ty = 0.0
                tau = FOLLOW_TAU_OFF
            else:
                bx = box[0] + box[2] / 2.0 - PROC_W / 2.0
                by = box[1] + box[3] / 2.0 - PROC_H / 2.0
                dead = max(FOLLOW_DEAD_MIN,
                           FOLLOW_DEAD_FRAC * max(box[2], box[3]))
                ex, ey = bx - self._rx, by - self._ry
                d = math.hypot(ex, ey)
                if d > dead:
                    f = (d - dead) / d
                    self._rx += ex * f
                    self._ry += ey * f
                tx, ty = self._rx, self._ry
                tau = FOLLOW_TAU
            a = 1.0 - math.exp(-dt / max(1e-3, tau))
            self._x += a * (tx - self._x)
            self._y += a * (ty - self._y)

    def offset(self):
        with self._lock:
            return self._x, self._y


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
        self.dev, self.cap = None, None
        nums = sorted(int(d[5:]) for d in os.listdir("/dev")
                      if d.startswith("video") and d[5:].isdigit()
                      and int(d[5:]) < 16)
        for n in [index] + [n for n in nums if n != index]:
            cap = self._try_open(f"/dev/video{n}")
            if cap is not None:
                self.dev, self.cap = f"/dev/video{n}", cap
                break
        if self.cap is None:
            raise RuntimeError("V4L2 open failed")
        self._skip = max(1, round(CAP_DEV_FPS / CAP_FPS))
        for c in [c.strip() for c in CAM_CTRLS.split(",") if c.strip()]:
            subprocess.run(["v4l2-ctl", "-d", self.dev, "-c", c],
                           capture_output=True)
        self._cond = threading.Condition()
        self._frame, self._t, self._seq = None, 0.0, 0
        self._run = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    @staticmethod
    def _try_open(path):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap.set(cv2.CAP_PROP_FPS, CAP_DEV_FPS)
        if not cap.grab():
            cap.release()
            return None
        return cap

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
        self._thread.join(timeout=2.0)
        self.cap.release()


def Camera(index=0):
    try:
        return CameraCSI(index)
    except Exception as e:
        return CameraV4L2(index)


UNDERVOLT_MASK = 0x1 | 0x10000


def _vcgencmd(arg):
    try:
        return subprocess.run(["vcgencmd", arg], capture_output=True,
                              text=True, timeout=1.0).stdout.strip()
    except Exception:
        return ""


def read_temp_c():
    out = _vcgencmd("measure_temp")
    try:
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def read_throttled():
    out = _vcgencmd("get_throttled")
    try:
        return int(out.split("=")[1], 16)
    except Exception:
        return 0


class Watchdog:
    def __init__(self, on_trip, logger=None):
        self.on_trip = on_trip
        self.logger = logger
        self._last_frame = time.monotonic()
        self._armed = False
        self._warm = 0
        self._run = True
        self.temp_c = None
        self.throttled = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def feed(self):
        self._last_frame = time.monotonic()
        if self._warm >= WDG_WARMUP_FRAMES:
            self._armed = True
        else:
            self._warm += 1

    def stop(self):
        self._run = False

    def _loop(self):
        next_temp = 0.0
        while self._run:
            now = time.monotonic()
            if self._armed and now - self._last_frame > WDG_FRAME_TIMEOUT_S:
                self.on_trip(f"frame stall {now - self._last_frame:.3f}s")
                self._last_frame = now
            if now >= next_temp:
                self.temp_c = read_temp_c()
                self.throttled = read_throttled()
                if self.throttled & 0x1:
                    self.on_trip(f"under-voltage 0x{self.throttled:x}")
                if self.logger:
                    self.logger.temp(self.temp_c, self.throttled)
                next_temp = now + WDG_TEMP_LOG_S
            time.sleep(0.02)


class LatencyLog:
    def __init__(self, every=1):
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, time.strftime("lat_%Y%m%d_%H%M%S.csv"))
        self._f = open(path, "w", newline="", buffering=1)
        self._w = csv.writer(self._f)
        self._w.writerow(["t", "kind", "frame_id", "state",
                          "cap_ms", "track_ms", "ctrl_ms", "total_ms",
                          "fps", "extra"])
        self._every = every
        self._n = 0
        self._periods = collections.deque(maxlen=120)
        self._last = None
        self.path = path

    def frame(self, frame_id, state, t_cap, t_track, t_ctrl, t_done, extra=""):
        now = time.monotonic()
        if self._last is not None:
            self._periods.append(now - self._last)
        self._last = now
        self._n += 1
        if self._n % self._every:
            return
        fps = (len(self._periods) / sum(self._periods)) if self._periods else 0.0
        self._w.writerow([f"{now:.4f}", "frame", frame_id, state,
                          f"{(t_track - t_cap) * 1e3:.2f}",
                          f"{(t_ctrl - t_track) * 1e3:.2f}",
                          f"{(t_done - t_ctrl) * 1e3:.2f}",
                          f"{(t_done - t_cap) * 1e3:.2f}",
                          f"{fps:.1f}", extra])

    def temp(self, temp_c, throttled):
        self._w.writerow([f"{time.monotonic():.4f}", "temp", "", "",
                          "", "", "", "", "",
                          f"temp={temp_c} throttled=0x{throttled:x}"])

    def event(self, msg):
        self._w.writerow([f"{time.monotonic():.4f}", "event", "", "",
                          "", "", "", "", "", msg])

    @property
    def fps(self):
        return (len(self._periods) / sum(self._periods)) if self._periods else 0.0

    def close(self):
        self._f.close()


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
            f"caps=video/x-raw,format=I420,width={RTSP_W},height={RTSP_H},framerate={RTSP_FPS}/1 "
            f"! {enc}"
        )

        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(f"( {launch} )")
        factory.set_shared(True)
        factory.connect("media-configure", self._on_configure)

        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))
        self.server.get_mount_points().add_factory(path, factory)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

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

    def _serve(self):
        if self.server.attach(None) == 0:
            return
        self.GLib.MainLoop().run()

    @property
    def active(self):
        with self.lock:
            return self.src is not None

    def push(self, bgr):
        with self.lock:
            src = self.src
        if src is None:
            return
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        buf = self.Gst.Buffer.new_wrapped(i420.tobytes())
        buf.duration = self.duration
        if src.emit("push-buffer", buf) != self.Gst.FlowReturn.OK:
            with self.lock:
                self.src = None

    def stop(self):
        pass


class RtspRenderer:
    def __init__(self, rtsp, stab=None):
        self.rtsp = rtsp
        self.stab = stab
        self.cv = threading.Condition()
        self.job = None
        self.dropped = 0
        self._run = True
        self._push_t = []
        self._sbox = None
        self._z = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, bgr, box, zoom, view=None):
        with self.cv:
            if self.job is not None:
                self.dropped += 1
            self.job = (bgr, box, zoom, view)
            self.cv.notify()

    def stop(self):
        with self.cv:
            self._run = False
            self.cv.notify_all()

    def _loop(self):
        while True:
            with self.cv:
                while self._run and self.job is None:
                    self.cv.wait(0.5)
                if not self._run:
                    return
                bgr, box, zoom, view = self.job
                self.job = None
            try:
                uz = z = max(1.0, zoom)
                stab_on = self.stab is not None and self.stab.enabled
                fx = fy = 0.0
                warp = None
                quad = None
                if view is not None:
                    z, fx, fy, warp = view[:4]
                    quad = view[4] if len(view) > 4 else None
                H, W = bgr.shape[:2]
                cw3 = crop_matrix(W, H, z, fx, fy)
                bmode = (cv2.BORDER_CONSTANT if STAB_FREE
                         else cv2.BORDER_REPLICATE)
                if warp is None:
                    M = cw3[:2].copy()
                    warp3 = None
                else:
                    warp3 = np.vstack([warp, (0.0, 0.0, 1.0)])
                    M = (cw3 @ warp3)[:2].copy()
                if (warp3 is None or quad is None or RTSP_BANDS <= 1
                        or (abs(quad[0]) + abs(quad[1])) * (H / 2.0) ** 2
                        < 0.25):
                    out = cv2.warpAffine(
                        bgr, M, (RTSP_W, RTSP_H), flags=cv2.INTER_LINEAR,
                        borderMode=bmode, borderValue=(0, 0, 0))
                else:
                    out = np.empty((RTSP_H, RTSP_W, 3), np.uint8)
                    cyc = H / 2.0
                    for k in range(RTSP_BANDS):
                        r0 = k * RTSP_H // RTSP_BANDS
                        r1 = (k + 1) * RTSP_H // RTSP_BANDS
                        ycap = ((r0 + r1) / 2.0 - cw3[1, 2]) / cw3[1, 1]
                        d2 = (ycap - cyc) ** 2
                        w3 = warp3.copy()
                        w3[0, 2] += quad[0] * d2
                        w3[1, 2] += quad[1] * d2
                        Mb = (cw3 @ w3)[:2].copy()
                        Mb[1, 2] -= r0
                        cv2.warpAffine(
                            bgr, Mb, (RTSP_W, r1 - r0), dst=out[r0:r1],
                            flags=cv2.INTER_LINEAR, borderMode=bmode,
                            borderValue=(0, 0, 0))
                if box is None:
                    vbox = None
                else:
                    kx, ky = W / PROC_W, H / PROC_H
                    vbox = xform_box(M, (box[0] * kx, box[1] * ky,
                                         box[2] * kx, box[3] * ky))
                if vbox is None:
                    self._sbox = None
                else:
                    if (self._sbox is None or z != self._z
                            or abs(vbox[0] - self._sbox[0]) > vbox[2]
                            or abs(vbox[1] - self._sbox[1]) > vbox[3]):
                        self._sbox = vbox
                    else:

                        tx = max(2.0, OSD_BOX_STICKY_FRAC * vbox[2])
                        ty = max(2.0, OSD_BOX_STICKY_FRAC * vbox[3])
                        if (abs(vbox[0] - self._sbox[0]) > tx
                                or abs(vbox[1] - self._sbox[1]) > ty
                                or abs(vbox[2] - self._sbox[2]) > tx
                                or abs(vbox[3] - self._sbox[3]) > ty):
                            a = 0.35
                            self._sbox = tuple((1 - a) * s + a * b
                                               for s, b in
                                               zip(self._sbox, vbox))
                self._z = z
                if self._sbox is not None:
                    x, y, w, h = self._sbox
                    cv2.rectangle(out, (int(x), int(y)),
                                  (int(x + w), int(y + h)),
                                  (0, 255, 0), 3)
                now = time.monotonic()
                self._push_t = [t for t in self._push_t if now - t < 2.0]
                self._push_t.append(now)
                out_fps = len(self._push_t) / 2.0
                label = f"FPS:{out_fps:.0f} ZOOM:x{uz:.1f}"
                if stab_on:
                    label += " STAB"
                cv2.putText(out, label, (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(out, label, (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                            cv2.LINE_AA)
                self.rtsp.push(out)
            except Exception:
                pass


def _s8(v):
    return v - 256 if v > 127 else v


class GcsLink:
    def __init__(self, on_track=None, on_clear=None, on_center=None,
                 on_ai_mode=None, on_zoom_rate=None, on_zoom_abs=None,
                 on_stab_mode=None, on_stab_alpha=None, on_gain=None,
                 on_rotate=None, zoom=None, view=None, port=GCS_UDP_PORT):
        self.on_track = on_track
        self.on_clear = on_clear
        self.on_center = on_center
        self.on_ai_mode = on_ai_mode
        self.on_zoom_rate = on_zoom_rate
        self.on_zoom_abs = on_zoom_abs
        self.on_stab_mode = on_stab_mode
        self.on_stab_alpha = on_stab_alpha
        self.on_gain = on_gain
        self.on_rotate = on_rotate
        self.zoom = zoom or (lambda: 1.0)
        self.view = view
        self.port = port
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._run = False

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        sock.settimeout(0.5)
        while self._run:
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(data)
            except Exception as e:
                print(f"GCS cmd error: {e!r}", file=sys.stderr)
        sock.close()

    def _handle(self, msg):
        if len(msg) <= GCS_CMD_OFFSET:
            return
        if msg[0] != GCS_HEADER1 or msg[1] != GCS_HEADER2:
            return
        cmd = msg[GCS_CMD_OFFSET]
        p = GCS_PAYLOAD_OFFSET

        if cmd == CMD_CAM_HEARTBEAT:
            return

        if cmd == CMD_TRACK_ACTION:
            self._track_action(msg)

        elif cmd == CMD_GIMBAL_CENTER:
            if self.on_center:
                self.on_center(msg[p])

        elif cmd in (CMD_GIMBAL_ZOOM, CMD_TEST_DIGITAL_ZOOM):
            if self.on_zoom_rate:
                d = _s8(msg[p])
                self.on_zoom_rate(0 if d == 0 else (1 if d > 0 else -1))

        elif cmd == CMD_TEST_ZOOM_RAW:
            if self.on_zoom_abs:
                raw = struct.unpack_from("<H", msg, p)[0]
                f = min(1.0, raw / GCS_ZOOM_RAW_MAX)
                self.on_zoom_abs(1.0 + f * (MAX_ZOOM - 1.0))

        elif cmd == CMD_STABILIZER_MODE:
            if self.on_stab_mode:
                self.on_stab_mode(msg[p])

        elif cmd == CMD_STABILIZER_ALPHA:
            if self.on_stab_alpha:
                self.on_stab_alpha(msg[p])

        elif cmd == CMD_GIMBAL_ROTATE:
            if self.on_rotate:
                self.on_rotate(_s8(msg[p]), _s8(msg[p + 1]))

        elif cmd == CMD_SET_GAIN:
            if self.on_gain:
                self.on_gain([_s8(b) for b in msg[p:p + 10]])

        elif cmd == CMD_AI_MODE:
            if self.on_ai_mode:
                self.on_ai_mode(msg[p])

    def _track_action(self, msg):
        p = GCS_PAYLOAD_OFFSET
        on = msg[p]
        if not on:
            if self.on_clear:
                self.on_clear()
            return
        sx, sy, ex, ey = struct.unpack_from("<HHHH", msg, p + 1)
        fx, fy = RTSP_W / GCS_REF_W, RTSP_H / GCS_REF_H
        x1, x2 = sorted((sx * fx, ex * fx))
        y1, y2 = sorted((sy * fy, ey * fy))
        if self.view is not None:
            z, ox, oy, warp = self.view()[:4]
        else:
            z, ox, oy, warp = max(1.0, self.zoom()), 0.0, 0.0, None
        cw3 = crop_matrix(CAP_W, CAP_H, z, ox, oy)
        M = (cw3 if warp is None
             else cw3 @ np.vstack([warp, (0.0, 0.0, 1.0)]))
        inv = cv2.invertAffineTransform(M[:2].copy())
        x1, y1 = apply_pt(inv, x1, y1)
        x2, y2 = apply_pt(inv, x2, y2)
        x1, x2 = sorted((x1 * PROC_W / CAP_W, x2 * PROC_W / CAP_W))
        y1, y2 = sorted((y1 * PROC_H / CAP_H, y2 * PROC_H / CAP_H))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box = None if (x2 - x1 < 2 or y2 - y1 < 2) else (x1, y1, x2 - x1, y2 - y1)
        if self.on_track:
            self.on_track(cx, cy, box)


def _sum8(body):
    return sum(body) & 0xFF


def _xor8(body):
    x = 0
    for b in body:
        x ^= b
    return x


def _deg_x10(v):
    return max(-32768, min(32767, int(round(v * 10.0))))


def tx_packet(cmd):
    on = 1 if cmd.get("valid") and not cmd.get("estop") else 0
    xm, ym = cmd.get("rotate", (0.0, 0.0))
    fields = [FCC_TX_HEADER1, FCC_TX_HEADER2,
              on, cmd.get("nx", 0.0), cmd.get("ny", 0.0), on,
              xm, ym, 0, cmd.get("center", 0),
              0,
              _deg_x10(cmd.get("pitch", 0.0)) if on else 0,
              _deg_x10(cmd.get("yaw", 0.0)) if on else 0,
              _deg_x10(cmd.get("yaw_rate", 0.0)) if on else 0,
              int(round(cmd.get("speed", 0.0) * 1000)) if on else 0,
              0] + cmd.get("gain", [0] * 10)
    raw = struct.pack(FCC_TX_FMT, *(fields + [0]))
    return raw[:-1] + bytes([_sum8(raw[:-1])])


def rx_parse(raw):
    if len(raw) != FCC_RX_SIZE:
        return None
    if raw[0] != FCC_RX_HEADER1 or raw[1] != FCC_RX_HEADER2:
        return None
    if _xor8(raw[:-1]) != raw[-1]:
        return None
    return struct.unpack(FCC_RX_FMT, raw)


def _open(path, baud):
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


class FccLink:
    def __init__(self, command_fn, sync, port=FCC_PORT, baud=FCC_BAUD):
        self.command_fn = command_fn
        self.sync = sync
        self.sync_timeout = 2.0 / max(1.0, CAP_FPS)
        self.port, self.baud = port, baud
        self.last_rx = None
        self.last_rx_t = 0.0
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._run = False

    def _loop(self):
        fd = None
        buf = b""
        while self._run:
            if fd is None:
                try:
                    fd = _open(self.port, self.baud)
                    buf = b""
                except Exception:
                    time.sleep(FCC_RETRY)
                    continue
            try:
                os.write(fd, tx_packet(self.command_fn()))
                try:
                    buf += os.read(fd, 4096)
                except BlockingIOError:
                    pass
                buf = self._drain_rx(buf)
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                fd = None
                time.sleep(FCC_RETRY)
                continue
            self.sync.wait(self.sync_timeout)
            self.sync.clear()
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass

    def _drain_rx(self, buf):
        while True:
            i = buf.find(bytes([FCC_RX_HEADER1, FCC_RX_HEADER2]))
            if i < 0:
                return buf[-1:]
            buf = buf[i:]
            if len(buf) < FCC_RX_SIZE:
                return buf
            pkt = rx_parse(buf[:FCC_RX_SIZE])
            if pkt is not None:
                self.last_rx = pkt
                self.last_rx_t = time.monotonic()
                buf = buf[FCC_RX_SIZE:]
            else:
                buf = buf[2:]


class Tof:
    def __init__(self):
        self._m = None
        self._t = 0.0
        self._run = True
        self._sensor = None
        try:
            import VL53L1X
            self._sensor = VL53L1X.VL53L1X(i2c_bus=TOF_I2C_BUS,
                                           i2c_address=TOF_I2C_ADDR)
            self._sensor.open()
            self._sensor.start_ranging(2)
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as e:
            pass

    def _loop(self):
        while self._run:
            try:
                mm = self._sensor.get_distance()
                if mm > 0:
                    self._m = mm / 1000.0
                    self._t = time.monotonic()
            except Exception:
                pass
            time.sleep(0.03)

    @property
    def latest_m(self):
        if self._m is None or time.monotonic() - self._t > 0.5:
            return None
        return self._m

    def stop(self):
        self._run = False
        if self._sensor:
            try:
                self._sensor.stop_ranging()
                self._sensor.close()
            except Exception:
                pass


class Bumper:
    def __init__(self):
        self._btn = None
        self.hit = False
        try:
            from gpiozero import Button
            self._btn = Button(BUMPER_GPIO, pull_up=True, bounce_time=0.01)
            self._btn.when_pressed = self._on_hit
        except Exception as e:
            pass

    def _on_hit(self):
        self.hit = True

    @property
    def pressed(self):
        return self.hit or (self._btn is not None and self._btn.is_pressed)

    def reset(self):
        self.hit = False

    def stop(self):
        if self._btn:
            self._btn.close()


CELL = 8


def postprocess(semi, desc, max_kp=ANCHOR_MAX_KP):
    hc, wc, _ = semi.shape
    e = np.exp(semi - semi.max(axis=2, keepdims=True))
    prob = e / e.sum(axis=2, keepdims=True)
    heat = prob[:, :, :64].reshape(hc, wc, CELL, CELL)
    heat = heat.transpose(0, 2, 1, 3).reshape(hc * CELL, wc * CELL)

    ys, xs = np.where(heat > SP_CONF_THRESH)
    if len(xs) == 0:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 256), np.float32))
    conf = heat[ys, xs]

    order = np.argsort(-conf)
    keep = []
    occupied = np.zeros_like(heat, dtype=bool)
    r = SP_NMS_RADIUS
    for i in order:
        y, x = ys[i], xs[i]
        if occupied[y, x]:
            continue
        keep.append(i)
        occupied[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        if len(keep) >= max_kp:
            break
    keep = np.array(keep)
    pts = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)

    fx = (pts[:, 0] / CELL).clip(0, wc - 1.001)
    fy = (pts[:, 1] / CELL).clip(0, hc - 1.001)
    x0, y0 = fx.astype(int), fy.astype(int)
    dx, dy = (fx - x0)[:, None], (fy - y0)[:, None]
    d = (desc[y0, x0] * (1 - dx) * (1 - dy) + desc[y0, x0 + 1] * dx * (1 - dy) +
         desc[y0 + 1, x0] * (1 - dx) * dy + desc[y0 + 1, x0 + 1] * dx * dy)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    return pts, d.astype(np.float32)


class SuperPointHef:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def preprocess(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (SP_W, SP_H))[None, :, :, None]

    def infer(self, bgr):
        outs = self.model.run(self.preprocess(bgr))
        semi = next(v for v in outs.values() if v.shape[-1] == 65)
        desc = next(v for v in outs.values() if v.shape[-1] == 256)
        return postprocess(semi, desc)


class HefModel:

    def __init__(self, vdevice, hef_path):
        from hailo_platform import FormatType
        self.im = vdevice.create_infer_model(hef_path)
        for o in self.im.outputs:
            o.set_format_type(FormatType.FLOAT32)
        self.cm = self.im.configure()
        self._bindings = self.cm.create_bindings()
        self._in_buf = np.empty(tuple(self.im.input().shape), dtype=np.uint8)
        self._bindings.input().set_buffer(self._in_buf)
        self._out_bufs = {}
        for o in self.im.outputs:
            buf = np.empty(tuple(o.shape), dtype=np.float32)
            self._out_bufs[o.name] = buf
            self._bindings.output(o.name).set_buffer(buf)

    def run(self, arr):
        np.copyto(self._in_buf, np.asarray(arr).reshape(self._in_buf.shape))
        self.cm.run([self._bindings], timeout=1000)
        return self._out_bufs

    def close(self):
        try:
            self.cm.shutdown()
        except Exception:
            pass
        self.cm = None
        self.im = None


def _pair_scale(a, b):
    if len(a) < 3:
        return None
    i, j = np.triu_indices(len(a), 1)
    da = np.linalg.norm(a[i] - a[j], axis=1)
    db = np.linalg.norm(b[i] - b[j], axis=1)
    ok = da > 1.0
    if not ok.any():
        return None
    return float(np.median(db[ok] / da[ok]))


class SpReacquirer:
    name = "superpoint"

    def __init__(self, sp):
        self.sp = sp
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.anchor_desc = None
        self.anchor_size = None
        self.anchor_pts = None
        self.anchor_ctr = (0.0, 0.0)
        self.strong = False
        self.zoom = 1.0
        self.last = None
        self._tiles = []
        self._tile_i = 0

    @staticmethod
    def _window(cx, cy, zoom):
        kx, ky = CAP_W / float(PROC_W), CAP_H / float(PROC_H)
        cw, ch = CAP_W / zoom, CAP_H / zoom
        x0 = min(max(0.0, cx * kx - cw / 2.0), CAP_W - cw)
        y0 = min(max(0.0, cy * ky - ch / 2.0), CAP_H - ch)
        return int(round(x0)), int(round(y0)), int(round(cw)), int(round(ch))

    def _infer(self, bgr, cx, cy, zoom):
        x0, y0, cw, ch = self._window(cx, cy, zoom)
        pts, desc = self.sp.infer(bgr[y0:y0 + ch, x0:x0 + cw])
        ox = x0 * PROC_W / float(CAP_W)
        oy = y0 * PROC_H / float(CAP_H)
        return pts, desc, (ox, oy)

    def _to_sp(self, v, axis):
        return v * self.zoom * (SP_W / float(PROC_W) if axis == 0
                                else SP_H / float(PROC_H))

    def _to_proc(self, v, axis):
        return v * (PROC_W / float(SP_W) if axis == 0
                    else PROC_H / float(SP_H)) / self.zoom

    def _make_tiles(self):
        n = max(1, int(math.ceil(self.zoom * 1.5)))
        xs = [(i + 0.5) * PROC_W / n for i in range(n)]
        ys = [(j + 0.5) * PROC_H / n for j in range(n)]
        self._tiles = [(x, y) for y in ys for x in xs]
        self._tile_i = 0

    def set_anchor(self, bgr, box):
        x, y, w, h = box
        bw1 = w * SP_W / float(PROC_W)
        bh1 = h * SP_H / float(PROC_H)
        zoom = SP_ZOOM_TARGET / max(1.0, max(bw1, bh1))
        self.zoom = float(np.clip(zoom, 1.0, SP_ZOOM_MAX))
        cx, cy = x + w / 2.0, y + h / 2.0
        pts, desc, (ox, oy) = self._infer(bgr, cx, cy, self.zoom)
        if len(pts) == 0:
            return False
        scx, scy = self._to_sp(cx - ox, 0), self._to_sp(cy - oy, 1)
        bw, bh = self._to_sp(w, 0), self._to_sp(h, 1)
        for scale in (1.0, 1.5):
            hw, hh = bw * scale / 2, bh * scale / 2
            sel = ((pts[:, 0] >= scx - hw) & (pts[:, 0] <= scx + hw) &
                   (pts[:, 1] >= scy - hh) & (pts[:, 1] <= scy + hh))
            if sel.sum() >= REACQ_MIN_MATCHES:
                self.anchor_desc = desc[sel][:ANCHOR_MAX_KP]
                self.anchor_size = (bw, bh)
                self.anchor_pts = pts[sel][:ANCHOR_MAX_KP]
                self.anchor_ctr = (float(scx), float(scy))
                self.strong = (scale == 1.0 and
                               max(w * CAP_W / PROC_W, h * CAP_H / PROC_H)
                               >= SP_STRONG_MIN_PX)
                self.last = (cx, cy)
                self._make_tiles()
                return True
        return False

    @property
    def ready(self):
        return self.anchor_desc is not None

    def clear(self):
        self.anchor_desc = self.anchor_size = self.anchor_pts = None
        self.strong = False
        self.last = None
        self._tiles = []

    def search(self, bgr, center=None):
        if not self.ready:
            return None, 0
        if center is None:
            if self.last is not None and self._tile_i == 0:
                cx, cy = self.last
            elif self._tiles:
                cx, cy = self._tiles[self._tile_i % len(self._tiles)]
            else:
                cx, cy = PROC_W / 2.0, PROC_H / 2.0
            self._tile_i = (self._tile_i + 1) % (len(self._tiles) + 1)
        else:
            cx, cy = center
        pts, desc, (ox, oy) = self._infer(bgr, cx, cy, self.zoom)
        if len(pts) < REACQ_MIN_MATCHES:
            return None, 0
        matches = self.bf.knnMatch(self.anchor_desc, desc, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.8 * n.distance]
        if len(good) < REACQ_MIN_MATCHES:
            return None, len(good)
        mpts = np.array([pts[m.trainIdx] for m in good], np.float32)
        mcx, mcy = np.median(mpts[:, 0]), np.median(mpts[:, 1])
        w, h = self.anchor_size
        inb = ((np.abs(mpts[:, 0] - mcx) <= w * 0.75) &
               (np.abs(mpts[:, 1] - mcy) <= h * 0.75))
        if inb.sum() < REACQ_MIN_MATCHES:
            return None, int(inb.sum())
        apts = self.anchor_pts[[m.queryIdx for m in good]]
        sc = _pair_scale(apts[inb], mpts[inb])
        if sc is None:
            sc = 1.0
        sc = float(np.clip(sc, BOX_SCALE_MIN, BOX_SCALE_MAX))
        acx, acy = self.anchor_ctr
        mcx = acx + np.median(mpts[inb, 0] - sc * (apts[inb, 0] - acx) - acx)
        mcy = acy + np.median(mpts[inb, 1] - sc * (apts[inb, 1] - acy) - acy)
        w, h = w * sc, h * sc
        px = ox + self._to_proc(mcx - w / 2, 0)
        py = oy + self._to_proc(mcy - h / 2, 1)
        pw, ph = self._to_proc(w, 0), self._to_proc(h, 1)
        self.last = (px + pw / 2, py + ph / 2)
        self._tile_i = 0
        return (px, py, pw, ph), int(inb.sum())


def _load_models():
    from hailo_platform import VDevice, HailoSchedulingAlgorithm
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdev = VDevice(params)
    return SuperPointHef(HefModel(vdev, HEF_SUPERPOINT)), vdev


_running = True


def _stop(signum, frame):
    global _running
    _running = False


def service_main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        os.sched_setaffinity(0, HAILO_CORES)
    except OSError:
        pass

    frame_slot = FrameSlot(SHM_FRAME, (CAP_H, CAP_W, 3))
    result_slot = BlobSlot(SHM_RESULT)
    ctrl_slot = BlobSlot(SHM_CTRL)

    sp_model, vdev = _load_models()
    reacq = SpReacquirer(sp_model)

    last_fid = 0
    last_anchor_fid = -1
    last_anchor_try = -100
    audit_cnt = 0
    ok_streak = 0
    try:
        while _running:
            ctrl, _, _, _ = ctrl_slot.read()
            if ctrl is None:
                time.sleep(0.02)
                continue
            if ctrl.get("quit"):
                break
            state = ctrl.get("state", "IDLE")

            img, t_cap, fid, seq = frame_slot.read()
            if img is None or fid == last_fid:
                time.sleep(0.005)
                continue
            last_fid = fid

            try:
                ab = ctrl.get("anchor_box")
                afid = ctrl.get("anchor_fid", -1)
                if (ab is not None and afid != last_anchor_fid
                        and fid - last_anchor_try >= 30):
                    last_anchor_try = fid
                    src = img
                    ok_anchor = reacq.set_anchor(src, ab)
                    if ok_anchor:
                        last_anchor_fid = afid
                    result_slot.write({"kind": "anchor", "ok": ok_anchor,
                                       "t_frame": t_cap}, t_cap, fid)
                elif ab is None and afid == -1 and reacq.ready:
                    reacq.clear()
                    last_anchor_fid = -1

                if state == "REACQUIRE":
                    src = img
                    t0 = time.perf_counter()
                    box, n = reacq.search(src)
                    result_slot.write({"kind": "reacq", "t_frame": t_cap,
                                       "box": box, "matches": n,
                                       "ms": (time.perf_counter() - t0) * 1e3,
                                       "backend": reacq.name}, t_cap, fid)
                elif state in ("TRACK", "TERMINAL"):
                    busy = False
                    audit_cnt += 1
                    if reacq.ready and audit_cnt >= AUDIT_EVERY:
                        audit_cnt = 0
                        src = img
                        tb0 = ctrl.get("track_box")
                        ctr = (None if tb0 is None else
                               (tb0[0] + tb0[2] / 2.0, tb0[1] + tb0[3] / 2.0))
                        sp_box, n = reacq.search(src, ctr)
                        result_slot.write({"kind": "audit", "t_frame": t_cap,
                                           "box": sp_box, "matches": n,
                                           "strong": reacq.strong},
                                          t_cap, fid)
                        tb = ctrl.get("track_box")
                        if sp_box is not None and tb is not None:
                            dx = (sp_box[0] + sp_box[2] / 2) - (tb[0] + tb[2] / 2)
                            dy = (sp_box[1] + sp_box[3] / 2) - (tb[1] + tb[3] / 2)
                            lim = max(tb[2], tb[3])
                            if dx * dx + dy * dy <= lim * lim:
                                ok_streak += 1
                                if ok_streak % AUDIT_REFRESH == 0:
                                    reacq.set_anchor(src, tb)
                            else:
                                ok_streak = 0
                        else:
                            ok_streak = 0
                        busy = True
                    if not busy:
                        time.sleep(0.005)
                else:
                    time.sleep(0.02)
            except Exception as e:
                time.sleep(0.1)
    finally:
        if sp_model is not None:
            sp_model.model.close()
        if vdev is not None:
            vdev.release()
        frame_slot.close()
        result_slot.close()
        ctrl_slot.close()


NO_HAILO = os.environ.get("NO_HAILO", "0") not in ("0", "", "false")
DEFAULT_BOX = 48


class TrackFilter:

    def __init__(self):
        self.x = None
        self.P = None
        self.strikes = 0
        self.aniso = 1.0
        self.gated = False

    def reset(self, box):
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        s = max(1.0, TRK_KF_MEAS * max(box[2], box[3]))
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([s * s, s * s, 2500.0, 2500.0])
        self.strikes = 0
        self.aniso = 1.0
        self.gated = False

    @staticmethod
    def _tensor(gray, box):
        x, y, w, h = box
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(gray.shape[1], x + w)), int(min(gray.shape[0], y + h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return np.eye(2), 1.0
        g = gray[y0:y1, x0:x1].astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        J = np.array([[(gx * gx).mean(), (gx * gy).mean()],
                      [(gx * gy).mean(), (gy * gy).mean()]])
        lam, V = np.linalg.eigh(J)
        ratio = math.sqrt(max(lam[1], 1e-6) / max(lam[0], 1e-6))
        return V, min(TRK_KF_ANISO_MAX, ratio)

    def step(self, box, ego, dt, gray):
        if self.x is None:
            self.reset(box)
            return box, True
        dt = max(1e-3, min(0.2, dt))
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=np.float64)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * (TRK_KF_ACC ** 2)
        V, aniso = self._tensor(gray, box)
        self.aniso = aniso
        ew = V[:, 0]
        hold = min(1.0, max(0.0, (aniso - 1.0) / (TRK_KF_ANISO_HOLD - 1.0)))
        vw = self.x[2] * ew[0] + self.x[3] * ew[1]
        self.x[2] -= ew[0] * vw * hold
        self.x[3] -= ew[1] * vw * hold
        self.x = F @ self.x
        if ego is not None:
            self.x[0] += ego[0]
            self.x[1] += ego[1]
        self.P = F @ self.P @ F.T + Q
        z = np.array([box[0] + box[2] / 2.0, box[1] + box[3] / 2.0])
        s = max(1.0, TRK_KF_MEAS * max(box[2], box[3]))
        sw = s * (1.0 + hold * (TRK_KF_ANISO_MAX - 1.0))
        R = V @ np.diag([sw * sw, s * s]) @ V.T
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        er = V[:, 1]
        Sr = float(er @ S @ er)
        yr = float(er @ y)
        d2 = yr * yr / max(Sr, 1e-6)
        if d2 > TRK_KF_GATE:
            self.strikes += 1
            self.gated = True
        else:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ H) @ self.P
            vw = self.x[2] * ew[0] + self.x[3] * ew[1]
            self.x[2] -= ew[0] * vw * hold
            self.x[3] -= ew[1] * vw * hold
            self.strikes = 0
            self.gated = False
        cx, cy = float(self.x[0]), float(self.x[1])
        out = (cx - box[2] / 2.0, cy - box[3] / 2.0, box[2], box[3])
        return out, self.strikes < TRK_KF_STRIKES


def refine_box(bgr, box):
    kx, ky = bgr.shape[1] / float(PROC_W), bgr.shape[0] / float(PROC_H)
    x0, y0 = max(0, int(box[0] * kx)), max(0, int(box[1] * ky))
    x1, y1 = int((box[0] + box[2]) * kx), int((box[1] + box[3]) * ky)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return box
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    H, W = lab.shape[:2]
    m = max(2, min(H, W) // 8)
    ring = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                           lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    med = np.median(ring, axis=0)
    mad = np.median(np.abs(ring - med), axis=0) * 1.4826 + 1.0
    best, best_score = None, 0.0
    for ch, min_abs in ((0, 25.0), (1, 8.0), (2, 8.0)):
        diff = np.abs(lab[:, :, ch] - med[ch])
        z = diff / mad[ch]
        mask = ((z > ROI_MIN_Z) & (diff > min_abs)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, cc, stats, cent = cv2.connectedComponentsWithStats(mask)
        on = int((mask > 0).sum())
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < 9 or bx == 0 or by == 0 or bx + bw >= W or by + bh >= H:
                continue
            if bw > 0.7 * W or bh > 0.7 * H or bw * bh > ROI_MAX_FILL * W * H:
                continue
            if (on - area) > ROI_MAX_OTHER * W * H:
                continue
            dc = math.hypot(cent[i][0] - W / 2.0, cent[i][1] - H / 2.0)
            score = float(z[cc == i].mean()) * math.sqrt(area) \
                * math.exp(-dc / (0.5 * max(W, H)))
            if score > best_score:
                best, best_score = (bx, by, bw, bh), score
    if best is None:
        return box
    bx, by, bw, bh = best
    px, py = bw * ROI_PAD, bh * ROI_PAD
    nx = (x0 + bx - px) / kx
    ny = (y0 + by - py) / ky
    nw = (bw + 2 * px) / kx
    nh = (bh + 2 * py) / ky
    if nw < ROI_MIN_PX:
        nx -= (ROI_MIN_PX - nw) / 2.0
        nw = ROI_MIN_PX
    if nh < ROI_MIN_PX:
        ny -= (ROI_MIN_PX - nh) / 2.0
        nh = ROI_MIN_PX
    return (float(nx), float(ny), float(nw), float(nh))


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.click = None
        self.clear_req = False
        self.cmd = {"valid": False, "estop": False}
        self.rotate = (0.0, 0.0)
        self.gain = [0] * 10
        self.center_pending = False
        self.sm = StateMachine()
        self.cmd_event = threading.Event()
        self.log = LatencyLog(every=int(os.environ.get("LAT_EVERY", "30")))
        try:
            self.tracker = make_tracker()
        except Exception as e:
            self.tracker = DisabledTracker()
            self.log.event(f"TRACKER_DISABLED {type(e).__name__}: {e}")

    def on_track(self, cx, cy, box):
        with self.lock:
            self.click = (cx, cy, box)

    def on_clear(self):
        with self.lock:
            self.clear_req = True

    def on_center(self, v):
        self.sm.reset()
        self.on_clear()
        with self.lock:
            self.center_pending = True
        self.cmd_event.set()

    def on_rotate(self, yaw, pitch):
        with self.lock:
            self.rotate = (float(yaw), float(pitch))

    def on_gain(self, values):
        g = [max(-128, min(127, int(v))) for v in values[:10]]
        g += [0] * (10 - len(g))
        with self.lock:
            self.gain = g

    def on_trip(self, why):
        self.log.event(f"WATCHDOG {why}")
        self.sm.on_watchdog_trip(why)

    def fcc_command(self):
        with self.lock:
            c = dict(self.cmd)
            c["rotate"] = self.rotate
            c["gain"] = list(self.gain)
            c["center"] = 1 if self.center_pending else 0
            self.center_pending = False
        c["estop"] = self.sm.state is State.ESTOP
        return c


def fast_loop(app):
    stab = Stabilizer()
    rtsp = None
    renderer = None
    if RTSP_ENABLE:
        try:
            rtsp = RtspServer().start()
            renderer = RtspRenderer(rtsp, stab)
        except Exception as e:
            pass

    try:
        os.sched_setaffinity(0, FAST_LOOP_CORES)
    except OSError:
        pass
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
    except (OSError, PermissionError):
        pass

    frame_slot = FrameSlot(SHM_FRAME, (CAP_H, CAP_W, 3), create=True)
    result_slot = BlobSlot(SHM_RESULT, create=True)
    ctrl_slot = BlobSlot(SHM_CTRL, create=True)

    svc = None
    if not NO_HAILO:
        svc = subprocess.Popen([sys.executable, "-u",
                                os.path.abspath(__file__),
                                "--hailo-service"])

    cam = Camera(CAM_INDEX)
    tof = Tof()
    bumper = Bumper()
    wdg = Watchdog(app.on_trip, logger=app.log)
    zoom = DigitalZoom()
    follow = FollowPan()
    if ROI_REFINE:
        _wf = np.full((CAP_H, CAP_W, 3), 120, np.uint8)
        _wf[CAP_H // 2 - 8:CAP_H // 2 + 8, CAP_W // 2 - 8:CAP_W // 2 + 8] = (0, 0, 200)
        refine_box(_wf, (PROC_W / 2 - 24, PROC_H / 2 - 24, 48, 48))

    def view_now():
        z = max(1.0, zoom.value())
        if stab.enabled and not STAB_FREE:
            z *= STAB_ZOOM
        fx, fy = follow.offset()
        return (z, fx, fy, stab.warp, stab.quad)

    def stab_fit(m):
        if STAB_FREE:
            return True
        z, fx, fy, _ = view_now()
        cw3 = crop_matrix(CAP_W, CAP_H, z, fx, fy)
        inv = cv2.invertAffineTransform(
            (cw3 @ np.vstack([m, (0.0, 0.0, 1.0)]))[:2].copy())
        for px, py in ((0.0, 0.0), (RTSP_W, 0.0),
                       (0.0, RTSP_H), (RTSP_W, RTSP_H)):
            qx, qy = apply_pt(inv, px, py)
            if qx < 1.5 or qy < 1.5 or qx > CAP_W - 1.5 or qy > CAP_H - 1.5:
                return False
        return True

    def on_gain(values):
        app.on_gain(values)
        stab.set_gain(values)

    gcs = GcsLink(on_track=app.on_track, on_clear=app.on_clear,
                  on_center=app.on_center, on_zoom_rate=zoom.set_rate,
                  on_zoom_abs=zoom.set_zoom, zoom=zoom.value,
                  on_stab_mode=stab.set_mode, on_stab_alpha=stab.set_alpha,
                  on_gain=on_gain, on_rotate=app.on_rotate,
                  on_ai_mode=stab.set_mode,
                  view=view_now).start()
    fcc = FccLink(app.fcc_command, app.cmd_event).start()

    cmdf = CmdFilter()
    tkf = TrackFilter()
    frame_id = 0
    box = None
    audit_fails = 0
    snap_pending = 0
    last_result_seq = 0
    anchor_box, anchor_fid = None, -1
    rtsp_next = 0.0
    rtsp_period = 1.0 / RTSP_FPS
    prev_t_cap = time.monotonic()

    try:
        while True:
            bgr, proc, gray, t_cap = cam.read()
            frame_id += 1
            wdg.feed()
            stab.update(gray, stab_fit, stab.budget(), box)
            ego_d = stab.step_d
            frame_slot.write(bgr, t_cap, frame_id)
            t0 = time.perf_counter()

            with app.lock:
                click, app.click = app.click, None
                clear, app.clear_req = app.clear_req, False
            if clear:
                app.tracker.stop()
                app.sm.on_target_cleared()
                cmdf.reset()
            if click is not None:
                cx, cy, box = click
                clicked = box is None
                if box is None:
                    s = DEFAULT_BOX
                    box = (cx - s / 2, cy - s / 2, s, s)
                box = (max(0, box[0]), max(0, box[1]),
                       min(box[2], PROC_W - 1), min(box[3], PROC_H - 1))
                if ROI_REFINE and clicked:
                    rbox = refine_box(bgr, box)
                    if rbox != box:
                        app.log.event(f"ROI_REFINE {tuple(round(v) for v in box)}"
                                      f" -> {tuple(round(v, 1) for v in rbox)}")
                        box = rbox
                app.tracker.start(proc, box)
                tkf.reset(box)
                app.sm.on_target_selected()
                cmdf.reset()
                audit_fails = 0
                anchor_box, anchor_fid = box, frame_id

            if (ego_d is not None and app.tracker.active
                    and getattr(app.tracker, "frac", 1.0) < TERM_HOLD_FRAC):
                app.tracker.feedforward(float(ego_d[0]), float(ego_d[1]))
            track_ok, box = (app.tracker.update(proc)
                             if app.tracker.active else (False, None))
            if TRK_KF and track_ok and box is not None:
                fbox, alive = tkf.step(box, ego_d, t_cap - prev_t_cap, gray)
                if alive:
                    box = fbox
                    app.tracker.pos[0] = box[0] + box[2] / 2.0
                    app.tracker.pos[1] = box[1] + box[3] / 2.0
                    app.tracker.box = box
                else:
                    app.tracker.stop()
                    track_ok, box = False, None
                    app.log.event(f"KF_LOST strikes={tkf.strikes}")
            prev_t_cap = t_cap
            t1 = time.perf_counter()

            res, res_t, res_fid, seq = result_slot.read()
            if res is not None and seq != last_result_seq:
                last_result_seq = seq
                age = time.monotonic() - res.get("t_frame", 0)
                if age <= HAILO_RESULT_MAX_AGE_S:
                    if (res["kind"] == "reacq" and res["box"] is not None
                            and app.sm.state is State.REACQUIRE):
                        ok_ref = getattr(app.tracker, "matches_ref",
                                         lambda i, b: True)(proc, res["box"])
                        if ok_ref:
                            app.tracker.start(proc, res["box"], keep_ref=True)
                            tkf.reset(res["box"])
                            audit_fails = 0
                        else:
                            app.log.event(f"REACQ_REJECT color {res['box']}")
                    elif res["kind"] == "anchor" and not res["ok"]:
                        app.log.event("ANCHOR_FAIL")
                    elif (res["kind"] == "audit" and app.tracker.active
                          and app.sm.state is State.TRACK
                          and res.get("strong", True)):
                        sp_box = res["box"]
                        if (sp_box is not None
                                and res["matches"] >= REACQ_MIN_MATCHES):
                            audit_fails = 0
                            if track_ok and box is not None:
                                dx = ((sp_box[0] + sp_box[2] / 2)
                                      - (box[0] + box[2] / 2))
                                dy = ((sp_box[1] + sp_box[3] / 2)
                                      - (box[1] + box[3] / 2))
                                lim = max(box[2], box[3])
                                if dx * dx + dy * dy > lim * lim:
                                    snap_pending += 1
                                    if snap_pending >= 2:
                                        snap_pending = 0
                                        app.tracker.start(proc, sp_box,
                                                          keep_ref=True)
                                        tkf.reset(sp_box)
                                        track_ok, box = True, tuple(sp_box)
                                        app.log.event(f"SNAPBACK {sp_box}")
                                else:
                                    snap_pending = 0
                                    rescale = getattr(app.tracker,
                                                      "rescale", None)
                                    if rescale is not None:
                                        hc = float(app.tracker.sz[1])
                                        rescale(hc + BOX_AUDIT_LR
                                                * (sp_box[3] - hc), proc)
                        elif (res["matches"] < max(4, REACQ_MIN_MATCHES // 2)
                              and getattr(app.tracker, "frac", 0.0)
                                  < TERM_HOLD_FRAC):
                            audit_fails += 1
                            if audit_fails >= AUDIT_FAILS:
                                audit_fails = 0
                                app.tracker.stop()
                                track_ok, box = False, None
                                app.log.event("AUDIT_LOST")
            state = app.sm.step(track_ok, tof.latest_m, bumper.pressed)
            fbox = box
            if box is not None and stab.warp is not None:
                kx, ky = CAP_W / PROC_W, CAP_H / PROC_H
                wcx = (box[0] + box[2] / 2.0) * kx
                wcy = (box[1] + box[3] / 2.0) * ky
                sw = stab.warp
                sx_ = sw[0, 0] * wcx + sw[0, 1] * wcy + sw[0, 2]
                sy_ = sw[1, 0] * wcx + sw[1, 1] * wcy + sw[1, 2]
                fbox = (sx_ / kx - box[2] / 2.0, sy_ / ky - box[3] / 2.0,
                        box[2], box[3])
            follow.update(fbox)
            cmd = cmdf.step(state, control_step(state, box, view_now()), box)
            with app.lock:
                app.cmd = cmd
            app.cmd_event.set()
            t2 = time.perf_counter()

            if clear:
                anchor_box, anchor_fid = None, -1
            ctrl_slot.write({"state": state.name, "track_box": box,
                             "anchor_box": anchor_box,
                             "anchor_fid": anchor_fid,
                             "kf": (tkf.strikes, round(tkf.aniso, 1),
                                    tkf.gated)}, t_cap, frame_id)

            t3 = time.perf_counter()
            app.log.frame(frame_id, state.name, t0, t1, t2, t3,
                          extra=f"trk={app.tracker.last_ms:.1f}ms "
                                f"s={app.tracker.score:.2f} "
                                f"a={getattr(app.tracker, 'apce', 0.0):.1f} "
                                f"c={getattr(app.tracker, 'color_d', 0.0):.3f} "
                                f"c0={getattr(app.tracker, 'color_d0', 0.0):.3f} "
                                f"f={getattr(app.tracker, 'frac', 0.0):.2f} "
                                f"sb={stab.ms:.1f}ms sr={stab.response} "
                                f"tof={tof.latest_m}")
            if rtsp is not None and rtsp.active:
                now = time.monotonic()
                if now >= rtsp_next:
                    rtsp_next = max(rtsp_next + rtsp_period, now)
                    renderer.submit(bgr, box, zoom.value(), view_now())
    finally:
        ctrl_slot.write({"quit": True}, time.monotonic())
        time.sleep(0.1)
        for x in (gcs, fcc, wdg, tof, bumper):
            x.stop()
        if renderer is not None:
            renderer.stop()
        cam.release()
        if svc is not None:
            svc.terminate()
            try:
                svc.wait(timeout=3)
            except Exception:
                svc.kill()
        for s in (frame_slot, result_slot, ctrl_slot):
            s.close()
        app.log.close()


def main():
    app = App()

    def _bye(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)
    fast_loop(app)


if __name__ == "__main__":
    if "--hailo-service" in sys.argv:
        service_main()
    else:
        main()
