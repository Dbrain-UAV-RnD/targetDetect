

import os
import struct


def _env_int(k, d):
    return int(os.environ.get(k, str(d)))


def _env_float(k, d):
    return float(os.environ.get(k, str(d)))


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
DEPTH_W, DEPTH_H = _env_int("DEPTH_W", 320), _env_int("DEPTH_H", 256)

CAM_HFOV_DEG = _env_float("CAM_HFOV_DEG", 66.0)
CAM_VFOV_DEG = _env_float("CAM_VFOV_DEG", 41.0)

TRACKER = os.environ.get("TRACKER", "nanotrack")
TRACK_BUDGET_MS = _env_float("TRACK_BUDGET_MS", 20.0)
TRACK_CONF_THRESH = _env_float("TRACK_CONF_THRESH", 0.5)
TRACK_APCE_MIN = _env_float("TRACK_APCE_MIN", 0.0)
BOX_MAX_FRAC = _env_float("BOX_MAX_FRAC", 0.5)
TRACK_GRACE_S = _env_float("TRACK_GRACE_S", 0.5)
COLOR_MAX_D = _env_float("COLOR_MAX_D", 0.45)
COLOR0_MAX_D = _env_float("COLOR0_MAX_D", 0.7)
COLOR_EMA_GATE = _env_float("COLOR_EMA_GATE", 0.1)
COLOR_GRID = _env_int("COLOR_GRID", 16)
COLOR_EMA = _env_float("COLOR_EMA", 0.002)
TRACK_LOST_FRAMES = _env_int("TRACK_LOST_FRAMES", 10)


TERM_HOLD_FRAC   = _env_float("TERM_HOLD_FRAC", 0.12)
TERM_CONF_THRESH = _env_float("TERM_CONF_THRESH", 0.25)
NANOTRACK_DIR = os.environ.get("NANOTRACK_DIR", "/home/gimbal/models/nanotrack")

YAW_KP          = _env_float("YAW_KP", 0.8)
YAW_RATE_MAX    = _env_float("YAW_RATE_MAX", 45.0)
SPEED_MAX       = _env_float("SPEED_MAX", 1.0)
SLOW_DEPTH_M    = _env_float("SLOW_DEPTH_M", 3.0)
TERMINAL_DEPTH_M = _env_float("TERMINAL_DEPTH_M", 1.0)
TOF_TERMINAL_M  = _env_float("TOF_TERMINAL_M", 1.0)
TOF_CONTACT_M   = _env_float("TOF_CONTACT_M", 0.10)
DEPTH_TOF_DIVERGE_M = _env_float("DEPTH_TOF_DIVERGE_M", 1.5)
REACQ_TIMEOUT_S = _env_float("REACQ_TIMEOUT_S", 10.0)

HAILO_RESULT_MAX_AGE_S = _env_float("HAILO_RESULT_MAX_AGE_S", 0.5)

STAB_ENABLE  = os.environ.get("STAB", "1") not in ("0", "", "false", "no")
STAB_W       = _env_int("STAB_W", 240)
STAB_H       = _env_int("STAB_H", 135)
STAB_MARGIN  = _env_float("STAB_MARGIN", 0.05)
STAB_TAU     = _env_float("STAB_TAU", 0.4)
STAB_TAU_MIN = _env_float("STAB_TAU_MIN", 0.1)
STAB_TAU_MAX = _env_float("STAB_TAU_MAX", 2.0)
STAB_CORNERS = _env_int("STAB_CORNERS", 40)
STAB_MIN_PTS = _env_int("STAB_MIN_PTS", 12)

TERM_LOCK_FRAC      = _env_float("TERM_LOCK_FRAC", 0.35)
TERM_LOCK_TIMEOUT_S = _env_float("TERM_LOCK_TIMEOUT_S", 3.0)
COAST_S             = _env_float("COAST_S", 0.4)
SLEW_ANG_DEG_S      = _env_float("SLEW_ANG_DEG_S", 90.0)
SLEW_N_S            = _env_float("SLEW_N_S", 3.0)

WDG_FRAME_TIMEOUT_S = _env_float("WDG_FRAME_TIMEOUT_S", 0.1)
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
AUDIT_EVERY = _env_int("AUDIT_EVERY", 10)
AUDIT_FAILS = _env_int("AUDIT_FAILS", 5)
AUDIT_REFRESH = _env_int("AUDIT_REFRESH", 10)

HEF_SUPERPOINT = os.environ.get("HEF_SUPERPOINT", "/home/gimbal/models/superpoint.hef")
HEF_DEPTH      = os.environ.get("HEF_DEPTH", "/home/gimbal/models/scdepthv3.hef")

LOG_DIR = os.environ.get("LOG_DIR", "/home/gimbal/1CH_SUPER/logs")

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
FOLLOW_TAU     = _env_float("FOLLOW_TAU", 0.065)
FOLLOW_TAU_OFF = _env_float("FOLLOW_TAU_OFF", 0.205)

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
FCC_HZ    = _env_float("FCC_HZ", 50)
FCC_RETRY = _env_float("FCC_RETRY", 2.0)

TOF_I2C_BUS  = _env_int("TOF_I2C_BUS", 1)
TOF_I2C_ADDR = _env_int("TOF_I2C_ADDR", 0x29)
BUMPER_GPIO  = _env_int("BUMPER_GPIO", 17)


import pickle
import struct
import numpy as np
from multiprocessing import shared_memory

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
            self.shm = shared_memory.SharedMemory(name=name, track=False)
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


import time
from enum import Enum


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

    def step(self, track_ok, tof_m, depth_m, bumper):
        s = self.state
        if s in (State.IDLE, State.ESTOP, State.CONTACT):
            return s

        if bumper or (tof_m is not None and tof_m <= TOF_CONTACT_M):
            self._to(State.CONTACT, f"bumper={bumper} tof={tof_m}")
            return self.state

        if (s is State.TERMINAL and tof_m is not None and depth_m is not None
                and abs(tof_m - depth_m) > DEPTH_TOF_DIVERGE_M):
            self._to(State.ESTOP, f"depth/tof diverge {depth_m:.2f}/{tof_m:.2f}")
            return self.state

        if s is State.TRACK:
            if not track_ok:
                self._to(State.REACQUIRE, "tracker lost")
            elif ((tof_m is not None and tof_m <= TOF_TERMINAL_M) or
                  (depth_m is not None and depth_m <= TERMINAL_DEPTH_M)):
                self._to(State.TERMINAL, f"tof={tof_m} depth={depth_m}")

        elif s is State.REACQUIRE:
            if track_ok:
                self._to(State.TRACK, "reacquired")
            elif self.age > REACQ_TIMEOUT_S:
                self._to(State.IDLE, "reacquire timeout")

        elif s is State.TERMINAL:
            if not track_ok and tof_m is None:
                self._to(State.REACQUIRE, "terminal lost")

        return self.state


import math
import time


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


def speed_profile(depth_m, state):
    if state in (State.IDLE, State.REACQUIRE, State.CONTACT, State.ESTOP):
        return 0.0
    if depth_m is None:
        return 0.3 * SPEED_MAX
    if depth_m <= TERMINAL_DEPTH_M:
        return 0.15 * SPEED_MAX
    if depth_m >= SLOW_DEPTH_M:
        return SPEED_MAX
    f = (depth_m - TERMINAL_DEPTH_M) / (SLOW_DEPTH_M - TERMINAL_DEPTH_M)
    return (0.15 + 0.85 * f) * SPEED_MAX


def control_step(state, box, depth_m, view=None):
    if state in (State.TRACK, State.TERMINAL) and box is not None:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        ex = _soft_deadband(cx - PROC_W / 2.0, DEADBAND_FRAC * PROC_W / 2.0)
        ey = _soft_deadband(cy - PROC_H / 2.0, DEADBAND_FRAC * PROC_H / 2.0)
        yaw, pitch = target_angles(PROC_W / 2.0 + ex, PROC_H / 2.0 + ey)
        yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, YAW_KP * yaw))

        z, ox, oy = view if view is not None else (1.0, 0.0, 0.0)
        hw, hh = PROC_W / 2.0, PROC_H / 2.0
        return {"valid": True,
                "nx": max(-1.0, min(1.0, z * (cx - hw - ox) / hw)),
                "ny": max(-1.0, min(1.0, z * (hh + oy - cy) / hh)),
                "yaw": yaw, "pitch": pitch,
                "yaw_rate": yaw_rate,
                "speed": speed_profile(depth_m, state)}
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


import os
import time
import cv2
import numpy as np


EXEMPLAR = 127
INSTANCE = 255
SCORE_SZ = 16
STRIDE = 16
CONTEXT = 0.5
PENALTY_K = 0.148
WIN_INFLUENCE = 0.462
LR = 0.390


class CsrtTracker:
    name = "csrt"

    def __init__(self):
        self._t = None
        self.box = None
        self.score = 0.0
        self.last_ms = 0.0

    def start(self, img, box, keep_ref=False):
        make = getattr(cv2, "TrackerCSRT_create", None) or cv2.legacy.TrackerCSRT_create
        self._t = make()
        self._t.init(img, tuple(int(v) for v in box))
        self.box = tuple(box)

    def update(self, img):
        if self._t is None:
            return False, None
        t0 = time.perf_counter()
        ok, b = self._t.update(img)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        self.score = 1.0 if ok else 0.0
        self.box = tuple(b) if ok else None
        return ok, self.box

    def feedforward(self, dx, dy):
        pass

    def stop(self):
        self._t = None
        self.box = None

    @property
    def active(self):
        return self._t is not None


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
        self.color_d = 0.0
        self.color_d0 = 0.0

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
            s_new = (1 - lr) + s_pred * lr
            h_new = np.clip(self.sz[1] * s_new, 10, img.shape[0])
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


def make_tracker():
    if TRACKER == "csrt":
        return CsrtTracker()
    try:
        return NanoTracker()
    except Exception as e:
        return CsrtTracker()


import math
import threading
import time

import cv2
import numpy as np


_LK = dict(winSize=(15, 15), maxLevel=2,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class Stabilizer:

    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = STAB_ENABLE
        self.tau = STAB_TAU
        self.margin = STAB_MARGIN
        self.ms = 0.0
        self.response = 0
        self.prev = None
        self.c = np.zeros(2)
        self.t = 0.0

    def reset(self):
        with self.lock:
            self.prev = None
            self.c[:] = 0.0
            self.t = 0.0

    def set_mode(self, v):
        if v == GCS_STAB_RESET:
            self.reset()
            return
        on = bool(v)
        with self.lock:
            if on and not self.enabled:
                self.prev = None
                self.c[:] = 0.0
                self.t = 0.0
            self.enabled = on

    def set_alpha(self, a):
        a = max(0, min(100, int(a))) / 100.0
        with self.lock:
            self.tau = STAB_TAU_MIN + a * (STAB_TAU_MAX - STAB_TAU_MIN)

    def _estimate(self, prev, small, mask=None):
        p0 = cv2.goodFeaturesToTrack(prev, STAB_CORNERS, 0.01, 8, mask=mask)
        if p0 is None or len(p0) < STAB_MIN_PTS:
            return None, 0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, small, p0, None, **_LK)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(small, prev, p1, None, **_LK)
        err = np.linalg.norm(p0 - pb, axis=2).ravel()
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (err < 1.0)
        n = int(ok.sum())
        if n < STAB_MIN_PTS:
            return None, n
        d = (p1 - p0)[ok].reshape(-1, 2)
        return np.array((np.median(d[:, 0]) * (PROC_W / STAB_W),
                         np.median(d[:, 1]) * (PROC_H / STAB_H))), n

    def update(self, gray, t, mask_box=None):
        with self.lock:
            enabled, tau = self.enabled, self.tau
            dt = t - self.t if self.t > 0.0 else 1.0 / 30.0
            self.t = t
        if not enabled:
            self.prev = None
            return None
        t0 = time.perf_counter()
        small = cv2.resize(gray, (STAB_W, STAB_H),
                           interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, small
        mask = None
        if mask_box is not None:
            x, y, w, h = mask_box
            px, py = 0.1 * w, 0.1 * h
            mask = np.full((STAB_H, STAB_W), 255, np.uint8)
            mask[max(0, int((y - py) * STAB_H / PROC_H)):
                 int((y + h + py) * STAB_H / PROC_H) + 1,
                 max(0, int((x - px) * STAB_W / PROC_W)):
                 int((x + w + px) * STAB_W / PROC_W) + 1] = 0
        d, n = (None, 0) if prev is None else self._estimate(prev, small, mask)
        raw_d = d
        if d is None:
            d = np.zeros(2)
        dt = max(1e-3, min(0.1, dt))
        keep = math.exp(-dt / max(1e-3, tau))
        with self.lock:
            self.c = (self.c + d) * keep
            cmax = (self.margin * PROC_W, self.margin * PROC_H)
            self.c[0] = max(-cmax[0], min(cmax[0], self.c[0]))
            self.c[1] = max(-cmax[1], min(cmax[1], self.c[1]))
            self.response = n
            self.ms = (time.perf_counter() - t0) * 1e3
        return raw_d

    def view(self, zoom):
        z = max(1.0, zoom)
        with self.lock:
            if not self.enabled:
                return z, 0.0, 0.0
            z = max(z, 1.0 / (1.0 - 2.0 * self.margin))
            sx = (PROC_W - PROC_W / z) / 2.0
            sy = (PROC_H - PROC_H / z) / 2.0
            return (z, max(-sx, min(sx, self.c[0])),
                    max(-sy, min(sy, self.c[1])))


import math
import threading
import time


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
        self._t = time.monotonic()

    def update(self, box):
        now = time.monotonic()
        with self._lock:
            dt = max(0.0, min(0.2, now - self._t))
            self._t = now
            if box is None:
                tx = ty = 0.0
                tau = FOLLOW_TAU_OFF
            else:
                tx = box[0] + box[2] / 2.0 - PROC_W / 2.0
                ty = box[1] + box[3] / 2.0 - PROC_H / 2.0
                tau = FOLLOW_TAU
            a = 1.0 - math.exp(-dt / max(1e-3, tau))
            self._x += a * (tx - self._x)
            self._y += a * (ty - self._y)

    def offset(self):
        with self._lock:
            return self._x, self._y


import subprocess
import threading
import time
import cv2
import numpy as np


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


import subprocess
import threading
import time


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
        self._run = True
        self.temp_c = None
        self.throttled = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def feed(self):
        self._last_frame = time.monotonic()
        self._armed = True

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


import collections
import csv
import os
import time


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


import threading
import time

import cv2


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
            f"caps=video/x-raw,format=BGR,width={RTSP_W},height={RTSP_H},framerate={RTSP_FPS}/1 "
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
        buf = self.Gst.Buffer.new_wrapped(bgr.tobytes())
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
                ox = oy = 0.0
                if view is not None:

                    z, ox, oy = view
                elif self.stab is not None:
                    z, ox, oy = self.stab.view(z)
                H, W = bgr.shape[:2]
                cw, ch = max(2, int(W / z)), max(2, int(H / z))
                x0 = (W - cw) // 2 + int(round(ox * W / PROC_W))
                y0 = (H - ch) // 2 + int(round(oy * H / PROC_H))
                x0 = max(0, min(W - cw, x0))
                y0 = max(0, min(H - ch, y0))
                out = cv2.resize(bgr[y0:y0 + ch, x0:x0 + cw],
                                 (RTSP_W, RTSP_H))
                if box is None:
                    vbox = None
                else:


                    kx, ky = W / PROC_W, H / PROC_H
                    vbox = ((box[0] * kx - x0) * RTSP_W / cw,
                            (box[1] * ky - y0) * RTSP_H / ch,
                            box[2] * kx * RTSP_W / cw,
                            box[3] * ky * RTSP_H / ch)
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


import socket
import struct
import threading


def _s8(v):
    return v - 256 if v > 127 else v


class GcsLink:
    def __init__(self, on_track=None, on_clear=None, on_center=None,
                 on_ai_mode=None, on_zoom_rate=None, on_zoom_abs=None,
                 on_stab_mode=None, on_stab_alpha=None,
                 zoom=None, view=None, port=GCS_UDP_PORT):
        self.on_track = on_track
        self.on_clear = on_clear
        self.on_center = on_center
        self.on_ai_mode = on_ai_mode
        self.on_zoom_rate = on_zoom_rate
        self.on_zoom_abs = on_zoom_abs
        self.on_stab_mode = on_stab_mode
        self.on_stab_alpha = on_stab_alpha
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
                pass
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
        fx, fy = PROC_W / GCS_REF_W, PROC_H / GCS_REF_H
        x1, x2 = sorted((sx * fx, ex * fx))
        y1, y2 = sorted((sy * fy, ey * fy))
        if self.view is not None:
            z, ox, oy = self.view()
        else:
            z, ox, oy = max(1.0, self.zoom()), 0.0, 0.0
        hw, hh = PROC_W / 2.0, PROC_H / 2.0
        x1 = hw + (x1 - hw) / z + ox
        x2 = hw + (x2 - hw) / z + ox
        y1 = hh + (y1 - hh) / z + oy
        y2 = hh + (y2 - hh) / z + oy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box = None if (x2 - x1 < 2 or y2 - y1 < 2) else (x1, y1, x2 - x1, y2 - y1)
        if self.on_track:
            self.on_track(cx, cy, box)


import os
import struct
import termios
import threading
import time


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
    fields = [FCC_TX_HEADER1, FCC_TX_HEADER2,
              on, cmd.get("nx", 0.0), cmd.get("ny", 0.0), on,
              0.0, 0.0, 0, cmd.get("center", 0),
              0,
              _deg_x10(cmd.get("pitch", 0.0)) if on else 0,
              _deg_x10(cmd.get("yaw", 0.0)) if on else 0,
              _deg_x10(cmd.get("yaw_rate", 0.0)) if on else 0,
              int(round(cmd.get("speed", 0.0) * 1000)) if on else 0,
              0] + [0] * 10
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
    def __init__(self, command_fn, port=FCC_PORT, baud=FCC_BAUD, hz=FCC_HZ):
        self.command_fn = command_fn
        self.port, self.baud = port, baud
        self.period = 1.0 / max(1.0, hz)
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
        next_t = time.monotonic()
        while self._run:
            if fd is None:
                try:
                    fd = _open(self.port, self.baud)
                    buf = b""
                    next_t = time.monotonic()
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
            next_t += self.period
            d = next_t - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.monotonic()
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


import threading
import time


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


import os
import numpy as np
import cv2


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


import os
import numpy as np
import cv2


DEPTH_SCALE = float(os.environ.get("DEPTH_SCALE", "1.0"))
DEPTH_SHIFT = float(os.environ.get("DEPTH_SHIFT", "0.0"))


def summarize(dmap, box=None):
    h, w = dmap.shape[:2]
    cy0, cy1 = int(h * 0.4), int(h * 0.6)
    cx0, cx1 = int(w * 0.4), int(w * 0.6)
    center = float(np.median(dmap[cy0:cy1, cx0:cx1]))
    target = None
    if box is not None:
        x, y, bw, bh = box
        x0 = max(0, int(x / PROC_W * w))
        y0 = max(0, int(y / PROC_H * h))
        x1 = min(w, int((x + bw) / PROC_W * w))
        y1 = min(h, int((y + bh) / PROC_H * h))
        if x1 > x0 and y1 > y0:
            target = float(np.median(dmap[y0:y1, x0:x1]))
    to_m = lambda v: None if v is None else v * DEPTH_SCALE + DEPTH_SHIFT
    return to_m(target), to_m(center)


class DepthHef:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def preprocess(bgr):
        return cv2.resize(bgr, (DEPTH_W, DEPTH_H))[None]

    def infer(self, bgr, box=None):
        outs = self.model.run(self.preprocess(bgr))
        dmap = np.squeeze(next(iter(outs.values())))
        return summarize(dmap, box)


import numpy as np
import cv2


class OrbReacquirer:
    name = "orb"

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.anchor_kp = None
        self.anchor_desc = None
        self.anchor_size = None

    def set_anchor(self, gray, box):
        x, y, w, h = (int(v) for v in box)
        roi = gray[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            return False
        kp, desc = self.orb.detectAndCompute(roi, None)
        if desc is None or len(kp) < REACQ_MIN_MATCHES:
            return False
        if len(kp) > ANCHOR_MAX_KP:
            idx = np.argsort([-k.response for k in kp])[:ANCHOR_MAX_KP]
            kp = [kp[i] for i in idx]
            desc = desc[idx]
        self.anchor_kp = np.array([(k.pt[0] - w / 2, k.pt[1] - h / 2) for k in kp],
                                  np.float32)
        self.anchor_desc = desc
        self.anchor_size = (w, h)
        return True

    @property
    def ready(self):
        return self.anchor_desc is not None

    def clear(self):
        self.anchor_kp = self.anchor_desc = self.anchor_size = None

    def search(self, gray):
        if not self.ready:
            return None, 0
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < REACQ_MIN_MATCHES:
            return None, 0
        matches = self.bf.knnMatch(self.anchor_desc, desc, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < REACQ_MIN_MATCHES:
            return None, len(good)
        pts = np.array([kp[m.trainIdx].pt for m in good], np.float32)
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        w, h = self.anchor_size
        return (float(cx - w / 2), float(cy - h / 2), float(w), float(h)), len(good)


import faulthandler
import os
import time
import numpy as np
import cv2

faulthandler.enable()


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


class SpReacquirer:
    name = "superpoint"

    def __init__(self, sp):
        self.sp = sp
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.anchor_desc = None
        self.anchor_size = None

    def set_anchor(self, bgr, box):
        pts, desc = self.sp.infer(bgr)
        if len(pts) == 0:
            return False
        x, y, w, h = box
        fx, fy = SP_W / PROC_W, SP_H / PROC_H
        cx, cy = (x + w / 2) * fx, (y + h / 2) * fy
        bw, bh = w * fx, h * fy
        for scale in (1.0, 1.5):
            hw, hh = bw * scale / 2, bh * scale / 2
            sel = ((pts[:, 0] >= cx - hw) & (pts[:, 0] <= cx + hw) &
                   (pts[:, 1] >= cy - hh) & (pts[:, 1] <= cy + hh))
            if sel.sum() >= REACQ_MIN_MATCHES:
                self.anchor_desc = desc[sel][:ANCHOR_MAX_KP]
                self.anchor_size = (bw, bh)
                return True
        return False

    @property
    def ready(self):
        return self.anchor_desc is not None

    def clear(self):
        self.anchor_desc = self.anchor_size = None

    def search(self, bgr):
        if not self.ready:
            return None, 0
        pts, desc = self.sp.infer(bgr)
        if len(pts) < REACQ_MIN_MATCHES:
            return None, 0
        matches = self.bf.knnMatch(self.anchor_desc, desc, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.8 * n.distance]
        if len(good) < REACQ_MIN_MATCHES:
            return None, len(good)
        mpts = np.array([pts[m.trainIdx] for m in good], np.float32)
        cx, cy = np.median(mpts[:, 0]), np.median(mpts[:, 1])
        w, h = self.anchor_size
        inb = ((np.abs(mpts[:, 0] - cx) <= w * 0.75) &
               (np.abs(mpts[:, 1] - cy) <= h * 0.75))
        if inb.sum() < REACQ_MIN_MATCHES:
            return None, int(inb.sum())
        cx = np.median(mpts[inb, 0])
        cy = np.median(mpts[inb, 1])
        gx, gy = PROC_W / SP_W, PROC_H / SP_H
        return ((cx - w / 2) * gx, (cy - h / 2) * gy, w * gx, h * gy), int(inb.sum())


def _load_models():
    sp_model, depth_model, vdev = None, None, None
    try:
        from hailo_platform import VDevice, HailoSchedulingAlgorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        vdev = VDevice(params)
        if os.path.exists(HEF_SUPERPOINT):
            sp_model = SuperPointHef(HefModel(vdev, HEF_SUPERPOINT))
        if os.path.exists(HEF_DEPTH):
            depth_model = DepthHef(HefModel(vdev, HEF_DEPTH))
    except Exception as e:
        pass
    return sp_model, depth_model, vdev


_running = True


def _stop(signum, frame):
    global _running
    _running = False


def service_main():
    import signal
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        os.sched_setaffinity(0, HAILO_CORES)
    except OSError:
        pass

    frame_slot = FrameSlot(SHM_FRAME, (CAP_H, CAP_W, 3))
    result_slot = BlobSlot(SHM_RESULT)
    ctrl_slot = BlobSlot(SHM_CTRL)

    sp_model, depth_model, vdev = _load_models()
    reacq = SpReacquirer(sp_model) if sp_model else OrbReacquirer()

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
                    src = img if reacq.name == "superpoint" else _to_proc_gray(img)
                    ok_anchor = reacq.set_anchor(src, ab)
                    if ok_anchor:
                        last_anchor_fid = afid
                    result_slot.write({"kind": "anchor", "ok": ok_anchor,
                                       "t_frame": t_cap}, t_cap, fid)
                elif ab is None and afid == -1 and reacq.ready:
                    reacq.clear()
                    last_anchor_fid = -1

                if state == "REACQUIRE":
                    src = img if reacq.name == "superpoint" else _to_proc_gray(img)
                    t0 = time.perf_counter()
                    box, n = reacq.search(src)
                    result_slot.write({"kind": "reacq", "t_frame": t_cap,
                                       "box": box, "matches": n,
                                       "ms": (time.perf_counter() - t0) * 1e3,
                                       "backend": reacq.name}, t_cap, fid)
                elif state in ("TRACK", "TERMINAL"):
                    busy = False
                    if depth_model is not None:
                        t0 = time.perf_counter()
                        target_m, center_m = depth_model.infer(
                            img, ctrl.get("track_box"))
                        result_slot.write({"kind": "depth", "t_frame": t_cap,
                                           "target_m": target_m,
                                           "center_m": center_m,
                                           "ms": (time.perf_counter() - t0) * 1e3},
                                          t_cap, fid)
                        busy = True
                    audit_cnt += 1
                    if reacq.ready and audit_cnt >= AUDIT_EVERY:
                        audit_cnt = 0
                        src = (img if reacq.name == "superpoint"
                               else _to_proc_gray(img))
                        sp_box, n = reacq.search(src)
                        result_slot.write({"kind": "audit", "t_frame": t_cap,
                                           "box": sp_box, "matches": n},
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
        for m in (sp_model, depth_model):
            if m is not None:
                m.model.close()
        if vdev is not None:
            vdev.release()
        frame_slot.close()
        result_slot.close()
        ctrl_slot.close()


def _to_proc_gray(bgr):
    return cv2.cvtColor(cv2.resize(bgr, (PROC_W, PROC_H)), cv2.COLOR_BGR2GRAY)


import os
import signal
import subprocess
import sys
import threading
import time


NO_HAILO = os.environ.get("NO_HAILO", "0") not in ("0", "", "false")
DEFAULT_BOX = 48


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.click = None
        self.clear_req = False
        self.cmd = {"valid": False, "estop": False}
        self.sm = StateMachine()
        self.tracker = make_tracker()
        self.depth_m = None
        self.depth_t = 0.0
        self.log = LatencyLog(every=int(os.environ.get("LAT_EVERY", "30")))

    def on_track(self, cx, cy, box):
        with self.lock:
            self.click = (cx, cy, box)

    def on_clear(self):
        with self.lock:
            self.clear_req = True

    def on_center(self, v):
        self.sm.reset()
        self.on_clear()

    def on_trip(self, why):
        self.log.event(f"WATCHDOG {why}")
        self.sm.on_watchdog_trip(why)

    def fcc_command(self):
        with self.lock:
            c = dict(self.cmd)
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

    def view_now():


        z, ox, oy = stab.view(zoom.value())
        fx, fy = follow.offset()
        sx = (PROC_W - PROC_W / z) / 2.0
        sy = (PROC_H - PROC_H / z) / 2.0
        return (z, max(-sx, min(sx, ox + fx)),
                max(-sy, min(sy, oy + fy)))

    gcs = GcsLink(on_track=app.on_track, on_clear=app.on_clear,
                  on_center=app.on_center, on_zoom_rate=zoom.set_rate,
                  on_zoom_abs=zoom.set_zoom, zoom=zoom.value,
                  on_stab_mode=stab.set_mode, on_stab_alpha=stab.set_alpha,
                  on_ai_mode=stab.set_mode,
                  view=view_now).start()
    fcc = FccLink(app.fcc_command).start()

    cmdf = CmdFilter()
    frame_id = 0
    box = None
    audit_fails = 0
    last_result_seq = 0
    anchor_box, anchor_fid = None, -1
    rtsp_next = 0.0
    rtsp_period = 1.0 / RTSP_FPS

    try:
        while True:
            bgr, proc, gray, t_cap = cam.read()
            frame_id += 1
            wdg.feed()
            ego_d = stab.update(gray, t_cap, box)
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
                if box is None:
                    s = DEFAULT_BOX
                    box = (cx - s / 2, cy - s / 2, s, s)
                box = (max(0, box[0]), max(0, box[1]),
                       min(box[2], PROC_W - 1), min(box[3], PROC_H - 1))
                app.tracker.start(proc, box)
                app.sm.on_target_selected()
                cmdf.reset()
                audit_fails = 0
                anchor_box, anchor_fid = box, frame_id

            if (ego_d is not None and app.tracker.active
                    and getattr(app.tracker, "frac", 1.0) < TERM_HOLD_FRAC):
                app.tracker.feedforward(float(ego_d[0]), float(ego_d[1]))
            track_ok, box = (app.tracker.update(proc)
                             if app.tracker.active else (False, None))
            t1 = time.perf_counter()

            res, res_t, res_fid, seq = result_slot.read()
            if res is not None and seq != last_result_seq:
                last_result_seq = seq
                age = time.monotonic() - res.get("t_frame", 0)
                if age <= HAILO_RESULT_MAX_AGE_S:
                    if res["kind"] == "depth":
                        app.depth_m = res["target_m"] or res["center_m"]
                        app.depth_t = time.monotonic()
                    elif (res["kind"] == "reacq" and res["box"] is not None
                          and app.sm.state is State.REACQUIRE):
                        ok_ref = getattr(app.tracker, "matches_ref",
                                         lambda i, b: True)(proc, res["box"])
                        if ok_ref:
                            app.tracker.start(proc, res["box"], keep_ref=True)
                            audit_fails = 0
                        else:
                            app.log.event(f"REACQ_REJECT color {res['box']}")
                    elif res["kind"] == "anchor" and not res["ok"]:
                        app.log.event("ANCHOR_FAIL")
                    elif (res["kind"] == "audit" and app.tracker.active
                          and app.sm.state is State.TRACK):
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
                                    app.tracker.start(proc, sp_box,
                                                      keep_ref=True)
                                    track_ok, box = True, tuple(sp_box)
                                    app.log.event(f"SNAPBACK {sp_box}")
                        elif (res["matches"] < max(4, REACQ_MIN_MATCHES // 2)
                              and getattr(app.tracker, "frac", 0.0)
                                  < TERM_HOLD_FRAC):
                            audit_fails += 1
                            if audit_fails >= AUDIT_FAILS:
                                audit_fails = 0
                                app.tracker.stop()
                                track_ok, box = False, None
                                app.log.event("AUDIT_LOST")
            if time.monotonic() - app.depth_t > 2.0:
                app.depth_m = None

            state = app.sm.step(track_ok, tof.latest_m, app.depth_m,
                                bumper.pressed)
            follow.update(box)
            cmd = cmdf.step(state, control_step(state, box, app.depth_m,
                                                view_now()), box)
            with app.lock:
                app.cmd = cmd
            t2 = time.perf_counter()

            if clear:
                anchor_box, anchor_fid = None, -1
            ctrl_slot.write({"state": state.name, "track_box": box,
                             "anchor_box": anchor_box,
                             "anchor_fid": anchor_fid}, t_cap, frame_id)

            t3 = time.perf_counter()
            app.log.frame(frame_id, state.name, t0, t1, t2, t3,
                          extra=f"trk={app.tracker.last_ms:.1f}ms "
                                f"s={app.tracker.score:.2f} "
                                f"a={getattr(app.tracker, 'apce', 0.0):.1f} "
                                f"c={getattr(app.tracker, 'color_d', 0.0):.3f} "
                                f"c0={getattr(app.tracker, 'color_d0', 0.0):.3f} "
                                f"f={getattr(app.tracker, 'frac', 0.0):.2f} "
                                f"sb={stab.ms:.1f}ms sr={stab.response} "
                                f"depth={app.depth_m} tof={tof.latest_m}")
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
