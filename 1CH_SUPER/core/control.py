import math
import time

from config import (PROC_W, PROC_H, CAM_HFOV_DEG, CAM_VFOV_DEG,
                    YAW_KP, YAW_RATE_MAX, SPEED_MAX, SLOW_DEPTH_M,
                    TERMINAL_DEPTH_M, TERM_LOCK_FRAC, TERM_LOCK_TIMEOUT_S,
                    COAST_S, SLEW_ANG_DEG_S, SLEW_N_S, DEADBAND_FRAC)
from core.state_machine import State

_HTAN = math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
_VTAN = math.tan(math.radians(CAM_VFOV_DEG) / 2.0)


def target_angles(cx, cy):
    yaw = math.degrees(math.atan((cx - PROC_W / 2.0) / (PROC_W / 2.0) * _HTAN))
    pitch = -math.degrees(math.atan((cy - PROC_H / 2.0) / (PROC_H / 2.0) * _VTAN))
    return yaw, pitch


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


def control_step(state, box, depth_m):
    if state in (State.TRACK, State.TERMINAL) and box is not None:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        if abs(cx - PROC_W / 2.0) < DEADBAND_FRAC * PROC_W / 2.0:
            cx = PROC_W / 2.0
        if abs(cy - PROC_H / 2.0) < DEADBAND_FRAC * PROC_H / 2.0:
            cy = PROC_H / 2.0
        yaw, pitch = target_angles(cx, cy)
        yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, YAW_KP * yaw))
        return {"valid": True,
                "nx": max(-1.0, min(1.0, (cx - PROC_W / 2.0) / (PROC_W / 2.0))),
                "ny": max(-1.0, min(1.0, (PROC_H / 2.0 - cy) / (PROC_H / 2.0))),
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
