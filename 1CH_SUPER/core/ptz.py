import threading
import time

from config import MAX_ZOOM, ZOOM_RATE, ZOOM_TIMEOUT


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
