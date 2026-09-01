import math
import threading
import time

from config import (MAX_ZOOM, ZOOM_RATE, ZOOM_TIMEOUT,
                    FOLLOW_TAU, FOLLOW_TAU_OFF, PROC_W, PROC_H)


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
    """디지털 짐벌: 줌 크롭 중심이 트래킹 박스를 추종하는 팬 오프셋(proc 좌표).

    박스가 없으면 느린 시정수로 중앙 복귀. 오프셋 자체는 무제한으로 쌓고,
    프레임 경계 클램프는 뷰 합성(view_now) 쪽에서 줌 배율 기준으로 건다.
    """

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
