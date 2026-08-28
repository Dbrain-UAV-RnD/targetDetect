import os
import struct
import termios
import threading
import time

from config import (FCC_TX_HEADER1, FCC_TX_HEADER2, FCC_RX_HEADER1,
                    FCC_RX_HEADER2, FCC_TX_FMT, FCC_RX_FMT, FCC_RX_SIZE,
                    FCC_PORT, FCC_BAUD, FCC_HZ, FCC_RETRY)


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
