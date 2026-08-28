import threading
import time

import cv2

from config import (RTSP_PORT, RTSP_PATH, RTSP_W, RTSP_H, RTSP_FPS,
                    RTSP_BITRATE, RTSP_PRESET, RTSP_CODEC, RTSP_GOP,
                    RTSP_VBV, RTSP_QUEUE, RTSP_INTRA_REFRESH, RTSP_X265_OPTS,
                    PROC_W, PROC_H)


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
    def __init__(self, rtsp):
        self.rtsp = rtsp
        self.cv = threading.Condition()
        self.job = None
        self.dropped = 0
        self._run = True
        self._push_t = []
        self._sbox = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, bgr, box, zoom):
        with self.cv:
            if self.job is not None:
                self.dropped += 1
            self.job = (bgr, box, zoom)
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
                bgr, box, zoom = self.job
                self.job = None
            try:
                z = max(1.0, zoom)
                H, W = bgr.shape[:2]
                cw, ch = max(2, int(W / z)), max(2, int(H / z))
                x0, y0 = (W - cw) // 2, (H - ch) // 2
                out = cv2.resize(bgr[y0:y0 + ch, x0:x0 + cw],
                                 (RTSP_W, RTSP_H))
                if box is None:
                    self._sbox = None
                else:
                    if self._sbox is None or (
                            abs(box[0] - self._sbox[0]) > box[2]
                            or abs(box[1] - self._sbox[1]) > box[3]):
                        self._sbox = tuple(box)
                    else:
                        a = 0.35
                        self._sbox = tuple((1 - a) * s + a * b
                                           for s, b in zip(self._sbox, box))
                if self._sbox is not None:
                    x, y, w, h = self._sbox
                    kx, ky = W / PROC_W, H / PROC_H
                    ox = (x * kx - x0) * RTSP_W / cw
                    oy = (y * ky - y0) * RTSP_H / ch
                    ow = w * kx * RTSP_W / cw
                    oh = h * ky * RTSP_H / ch
                    cv2.rectangle(out, (int(ox), int(oy)),
                                  (int(ox + ow), int(oy + oh)),
                                  (0, 255, 0), 3)
                now = time.monotonic()
                self._push_t = [t for t in self._push_t if now - t < 2.0]
                self._push_t.append(now)
                out_fps = len(self._push_t) / 2.0
                label = f"FPS:{out_fps:.0f} ZOOM:x{z:.1f}"
                cv2.putText(out, label, (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(out, label, (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                            cv2.LINE_AA)
                self.rtsp.push(out)
            except Exception:
                pass
