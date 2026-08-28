import math
import threading
import time

import cv2
import numpy as np

from config import (PROC_W, PROC_H, STAB_ENABLE, STAB_W, STAB_H, STAB_MARGIN,
                    STAB_TAU, STAB_TAU_MIN, STAB_TAU_MAX, STAB_CORNERS,
                    STAB_MIN_PTS, GCS_STAB_RESET)

_LK = dict(winSize=(15, 15), maxLevel=2,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class Stabilizer:
    """표시 전용 안정화: LK 희소 플로우 미디언으로 전역 평행이동 추정,
    크롭 창 오프셋으로 보정.

    위상상관은 이 footage(저텍스처+모션블러)에서 5%가 오추정이라 기각,
    LK+fb체크+미디언이 상호정합 p95 0.9px. 트래커/제어/FCC는 원본 좌표
    그대로 두고, 렌더러의 줌 크롭 원점과 GCS 클릭 역변환에만 view()를
    반영한다.
    """

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

    def _estimate(self, prev, small):
        p0 = cv2.goodFeaturesToTrack(prev, STAB_CORNERS, 0.01, 8)
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

    def update(self, gray, t):
        with self.lock:
            enabled, tau = self.enabled, self.tau
            dt = t - self.t if self.t > 0.0 else 1.0 / 30.0
            self.t = t
        if not enabled:
            self.prev = None
            return
        t0 = time.perf_counter()
        small = cv2.resize(gray, (STAB_W, STAB_H),
                           interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, small
        d, n = (None, 0) if prev is None else self._estimate(prev, small)
        if d is None:
            d = np.zeros(2)  # 추정 불가: 보정만 감쇠(리센터)
        dt = max(1e-3, min(0.1, dt))
        keep = math.exp(-dt / max(1e-3, tau))
        with self.lock:
            self.c = (self.c + d) * keep
            cmax = (self.margin * PROC_W, self.margin * PROC_H)
            self.c[0] = max(-cmax[0], min(cmax[0], self.c[0]))
            self.c[1] = max(-cmax[1], min(cmax[1], self.c[1]))
            self.response = n
            self.ms = (time.perf_counter() - t0) * 1e3

    def view(self, zoom):
        """(z_eff, ox, oy) — proc 좌표계. 크롭 창 원점 = 중앙 크롭 원점 + (ox, oy)."""
        z = max(1.0, zoom)
        with self.lock:
            if not self.enabled:
                return z, 0.0, 0.0
            z = max(z, 1.0 / (1.0 - 2.0 * self.margin))
            sx = (PROC_W - PROC_W / z) / 2.0
            sy = (PROC_H - PROC_H / z) / 2.0
            return (z, max(-sx, min(sx, self.c[0])),
                    max(-sy, min(sy, self.c[1])))
