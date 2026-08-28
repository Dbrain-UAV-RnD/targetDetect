import collections
import csv
import os
import time

from config import LOG_DIR


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
