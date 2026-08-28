import threading
import time

from config import TOF_I2C_BUS, TOF_I2C_ADDR


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
