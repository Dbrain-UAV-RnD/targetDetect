import subprocess
import threading
import time

from config import WDG_FRAME_TIMEOUT_S, WDG_TEMP_LOG_S

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
