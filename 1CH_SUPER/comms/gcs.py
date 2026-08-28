import socket
import struct
import threading

from config import (GCS_UDP_PORT, GCS_HEADER1, GCS_HEADER2,
                    GCS_CMD_OFFSET, GCS_PAYLOAD_OFFSET,
                    CMD_CAM_HEARTBEAT, CMD_AI_MODE, CMD_TRACK_ACTION,
                    CMD_GIMBAL_CENTER, CMD_GIMBAL_ZOOM,
                    CMD_TEST_DIGITAL_ZOOM, CMD_TEST_ZOOM_RAW,
                    GCS_REF_W, GCS_REF_H, GCS_ZOOM_RAW_MAX, MAX_ZOOM,
                    PROC_W, PROC_H)


def _s8(v):
    return v - 256 if v > 127 else v


class GcsLink:
    def __init__(self, on_track=None, on_clear=None, on_center=None,
                 on_ai_mode=None, on_zoom_rate=None, on_zoom_abs=None,
                 zoom=None, port=GCS_UDP_PORT):
        self.on_track = on_track
        self.on_clear = on_clear
        self.on_center = on_center
        self.on_ai_mode = on_ai_mode
        self.on_zoom_rate = on_zoom_rate
        self.on_zoom_abs = on_zoom_abs
        self.zoom = zoom or (lambda: 1.0)
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
        z = max(1.0, self.zoom())
        hw, hh = PROC_W / 2.0, PROC_H / 2.0
        x1 = hw + (x1 - hw) / z
        x2 = hw + (x2 - hw) / z
        y1 = hh + (y1 - hh) / z
        y2 = hh + (y2 - hh) / z
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box = None if (x2 - x1 < 2 or y2 - y1 < 2) else (x1, y1, x2 - x1, y2 - y1)
        if self.on_track:
            self.on_track(cx, cy, box)
