import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CAP_W, CAP_H, PROC_W, PROC_H, CAM_INDEX,
                    SHM_FRAME, SHM_RESULT, SHM_CTRL, FAST_LOOP_CORES,
                    HAILO_RESULT_MAX_AGE_S, REACQ_MIN_MATCHES, AUDIT_FAILS,
                    RTSP_ENABLE, RTSP_FPS)
from core.camera import Camera
from core.tracker import make_tracker
from core.state_machine import StateMachine, State
from core.control import control_step, CmdFilter
from core.watchdog import Watchdog
from core.ptz import DigitalZoom
from core.latency import LatencyLog
from core.shm import FrameSlot, BlobSlot
from comms.gcs import GcsLink
from comms.fcc import FccLink
from sensors.tof import Tof
from sensors.bumper import Bumper

NO_HAILO = os.environ.get("NO_HAILO", "0") not in ("0", "", "false")
DEFAULT_BOX = 48


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.click = None
        self.clear_req = False
        self.cmd = {"valid": False, "estop": False}
        self.sm = StateMachine()
        self.tracker = make_tracker()
        self.depth_m = None
        self.depth_t = 0.0
        self.log = LatencyLog(every=int(os.environ.get("LAT_EVERY", "30")))

    def on_track(self, cx, cy, box):
        with self.lock:
            self.click = (cx, cy, box)

    def on_clear(self):
        with self.lock:
            self.clear_req = True

    def on_center(self, v):
        self.sm.reset()
        self.on_clear()

    def on_trip(self, why):
        self.log.event(f"WATCHDOG {why}")
        self.sm.on_watchdog_trip(why)

    def fcc_command(self):
        with self.lock:
            c = dict(self.cmd)
        c["estop"] = self.sm.state is State.ESTOP
        return c


def fast_loop(app):
    rtsp = None
    renderer = None
    if RTSP_ENABLE:
        try:
            from core.rtsp import RtspServer, RtspRenderer
            rtsp = RtspServer().start()
            renderer = RtspRenderer(rtsp)
        except Exception as e:
            pass

    try:
        os.sched_setaffinity(0, FAST_LOOP_CORES)
    except OSError:
        pass
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
    except (OSError, PermissionError):
        pass

    frame_slot = FrameSlot(SHM_FRAME, (CAP_H, CAP_W, 3), create=True)
    result_slot = BlobSlot(SHM_RESULT, create=True)
    ctrl_slot = BlobSlot(SHM_CTRL, create=True)

    svc = None
    if not NO_HAILO:
        svc = subprocess.Popen([sys.executable, "-u", "-m", "hailo.service"],
                               cwd=os.path.dirname(os.path.abspath(__file__)))

    cam = Camera(CAM_INDEX)
    tof = Tof()
    bumper = Bumper()
    wdg = Watchdog(app.on_trip, logger=app.log)
    zoom = DigitalZoom()
    gcs = GcsLink(on_track=app.on_track, on_clear=app.on_clear,
                  on_center=app.on_center, on_zoom_rate=zoom.set_rate,
                  on_zoom_abs=zoom.set_zoom, zoom=zoom.value).start()
    fcc = FccLink(app.fcc_command).start()

    cmdf = CmdFilter()
    frame_id = 0
    audit_fails = 0
    last_result_seq = 0
    anchor_box, anchor_fid = None, -1
    rtsp_next = 0.0
    rtsp_period = 1.0 / RTSP_FPS

    try:
        while True:
            bgr, proc, gray, t_cap = cam.read()
            frame_id += 1
            wdg.feed()
            frame_slot.write(bgr, t_cap, frame_id)
            t0 = time.perf_counter()

            with app.lock:
                click, app.click = app.click, None
                clear, app.clear_req = app.clear_req, False
            if clear:
                app.tracker.stop()
                app.sm.on_target_cleared()
                cmdf.reset()
            if click is not None:
                cx, cy, box = click
                if box is None:
                    s = DEFAULT_BOX
                    box = (cx - s / 2, cy - s / 2, s, s)
                box = (max(0, box[0]), max(0, box[1]),
                       min(box[2], PROC_W - 1), min(box[3], PROC_H - 1))
                app.tracker.start(proc, box)
                app.sm.on_target_selected()
                cmdf.reset()
                audit_fails = 0
                anchor_box, anchor_fid = box, frame_id

            track_ok, box = (app.tracker.update(proc)
                             if app.tracker.active else (False, None))
            t1 = time.perf_counter()

            res, res_t, res_fid, seq = result_slot.read()
            if res is not None and seq != last_result_seq:
                last_result_seq = seq
                age = time.monotonic() - res.get("t_frame", 0)
                if age <= HAILO_RESULT_MAX_AGE_S:
                    if res["kind"] == "depth":
                        app.depth_m = res["target_m"] or res["center_m"]
                        app.depth_t = time.monotonic()
                    elif (res["kind"] == "reacq" and res["box"] is not None
                          and app.sm.state is State.REACQUIRE):
                        ok_ref = getattr(app.tracker, "matches_ref",
                                         lambda i, b: True)(proc, res["box"])
                        if ok_ref:
                            app.tracker.start(proc, res["box"], keep_ref=True)
                            audit_fails = 0
                        else:
                            app.log.event(f"REACQ_REJECT color {res['box']}")
                    elif (res["kind"] == "audit" and app.tracker.active
                          and app.sm.state is State.TRACK):
                        sp_box = res["box"]
                        if (sp_box is not None
                                and res["matches"] >= REACQ_MIN_MATCHES):
                            audit_fails = 0
                            if track_ok and box is not None:
                                dx = ((sp_box[0] + sp_box[2] / 2)
                                      - (box[0] + box[2] / 2))
                                dy = ((sp_box[1] + sp_box[3] / 2)
                                      - (box[1] + box[3] / 2))
                                lim = max(box[2], box[3])
                                if dx * dx + dy * dy > lim * lim:
                                    app.tracker.start(proc, sp_box,
                                                      keep_ref=True)
                                    track_ok, box = True, tuple(sp_box)
                                    app.log.event(f"SNAPBACK {sp_box}")
                        else:
                            audit_fails += 1
                            if audit_fails >= AUDIT_FAILS:
                                audit_fails = 0
                                app.tracker.stop()
                                track_ok, box = False, None
                                app.log.event("AUDIT_LOST")
            if time.monotonic() - app.depth_t > 2.0:
                app.depth_m = None

            state = app.sm.step(track_ok, tof.latest_m, app.depth_m,
                                bumper.pressed)
            cmd = cmdf.step(state, control_step(state, box, app.depth_m), box)
            with app.lock:
                app.cmd = cmd
            t2 = time.perf_counter()

            if clear:
                anchor_box, anchor_fid = None, -1
            ctrl_slot.write({"state": state.name, "track_box": box,
                             "anchor_box": anchor_box,
                             "anchor_fid": anchor_fid}, t_cap, frame_id)

            t3 = time.perf_counter()
            app.log.frame(frame_id, state.name, t0, t1, t2, t3,
                          extra=f"trk={app.tracker.last_ms:.1f}ms "
                                f"s={app.tracker.score:.2f} "
                                f"a={getattr(app.tracker, 'apce', 0.0):.1f} "
                                f"c={getattr(app.tracker, 'color_d', 0.0):.3f} "
                                f"c0={getattr(app.tracker, 'color_d0', 0.0):.3f} "
                                f"f={getattr(app.tracker, 'frac', 0.0):.2f} "
                                f"depth={app.depth_m} tof={tof.latest_m}")
            if rtsp is not None and rtsp.active:
                now = time.monotonic()
                if now >= rtsp_next:
                    rtsp_next = max(rtsp_next + rtsp_period, now)
                    renderer.submit(bgr, box, zoom.value())
    finally:
        ctrl_slot.write({"quit": True}, time.monotonic())
        time.sleep(0.1)
        for x in (gcs, fcc, wdg, tof, bumper):
            x.stop()
        if renderer is not None:
            renderer.stop()
        cam.release()
        if svc is not None:
            svc.terminate()
            try:
                svc.wait(timeout=3)
            except Exception:
                svc.kill()
        for s in (frame_slot, result_slot, ctrl_slot):
            s.close()
        app.log.close()


def main():
    app = App()

    def _bye(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)
    fast_loop(app)


if __name__ == "__main__":
    main()
