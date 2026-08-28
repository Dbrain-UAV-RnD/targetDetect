import faulthandler
import os
import time
import numpy as np
import cv2

faulthandler.enable()

from config import (SHM_FRAME, SHM_RESULT, SHM_CTRL, CAP_W, CAP_H,
                    HAILO_CORES, HEF_SUPERPOINT, HEF_DEPTH,
                    ANCHOR_MAX_KP, REACQ_MIN_MATCHES,
                    AUDIT_EVERY, AUDIT_REFRESH)
from core.shm import FrameSlot, BlobSlot
from hailo import superpoint, depth as depth_mod
from hailo.orb_fallback import OrbReacquirer


class HefModel:

    def __init__(self, vdevice, hef_path):
        from hailo_platform import FormatType
        self.im = vdevice.create_infer_model(hef_path)
        for o in self.im.outputs:
            o.set_format_type(FormatType.FLOAT32)
        self.cm = self.im.configure()
        self._bindings = self.cm.create_bindings()
        self._in_buf = np.empty(tuple(self.im.input().shape), dtype=np.uint8)
        self._bindings.input().set_buffer(self._in_buf)
        self._out_bufs = {}
        for o in self.im.outputs:
            buf = np.empty(tuple(o.shape), dtype=np.float32)
            self._out_bufs[o.name] = buf
            self._bindings.output(o.name).set_buffer(buf)

    def run(self, arr):
        np.copyto(self._in_buf, np.asarray(arr).reshape(self._in_buf.shape))
        self.cm.run([self._bindings], timeout=1000)
        return self._out_bufs

    def close(self):
        try:
            self.cm.shutdown()
        except Exception:
            pass
        self.cm = None
        self.im = None


class SpReacquirer:
    name = "superpoint"

    def __init__(self, sp):
        self.sp = sp
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.anchor_desc = None
        self.anchor_size = None

    def set_anchor(self, bgr, box):
        pts, desc = self.sp.infer(bgr)
        if len(pts) == 0:
            return False
        from config import PROC_W, PROC_H, SP_W, SP_H
        x, y, w, h = box
        fx, fy = SP_W / PROC_W, SP_H / PROC_H
        cx, cy = (x + w / 2) * fx, (y + h / 2) * fy
        bw, bh = w * fx, h * fy
        for scale in (1.0, 1.5):
            hw, hh = bw * scale / 2, bh * scale / 2
            sel = ((pts[:, 0] >= cx - hw) & (pts[:, 0] <= cx + hw) &
                   (pts[:, 1] >= cy - hh) & (pts[:, 1] <= cy + hh))
            if sel.sum() >= REACQ_MIN_MATCHES:
                self.anchor_desc = desc[sel][:ANCHOR_MAX_KP]
                self.anchor_size = (bw, bh)
                return True
        return False

    @property
    def ready(self):
        return self.anchor_desc is not None

    def clear(self):
        self.anchor_desc = self.anchor_size = None

    def search(self, bgr):
        if not self.ready:
            return None, 0
        pts, desc = self.sp.infer(bgr)
        if len(pts) < REACQ_MIN_MATCHES:
            return None, 0
        matches = self.bf.knnMatch(self.anchor_desc, desc, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.8 * n.distance]
        if len(good) < REACQ_MIN_MATCHES:
            return None, len(good)
        mpts = np.array([pts[m.trainIdx] for m in good], np.float32)
        cx, cy = np.median(mpts[:, 0]), np.median(mpts[:, 1])
        w, h = self.anchor_size
        inb = ((np.abs(mpts[:, 0] - cx) <= w * 0.75) &
               (np.abs(mpts[:, 1] - cy) <= h * 0.75))
        if inb.sum() < REACQ_MIN_MATCHES:
            return None, int(inb.sum())
        cx = np.median(mpts[inb, 0])
        cy = np.median(mpts[inb, 1])
        from config import PROC_W, PROC_H, SP_W, SP_H
        gx, gy = PROC_W / SP_W, PROC_H / SP_H
        return ((cx - w / 2) * gx, (cy - h / 2) * gy, w * gx, h * gy), int(inb.sum())


def _load_models():
    sp_model, depth_model, vdev = None, None, None
    try:
        from hailo_platform import VDevice, HailoSchedulingAlgorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        vdev = VDevice(params)
        if superpoint.available():
            sp_model = superpoint.SuperPointHef(HefModel(vdev, HEF_SUPERPOINT))
        if depth_mod.available():
            depth_model = depth_mod.DepthHef(HefModel(vdev, HEF_DEPTH))
    except Exception as e:
        pass
    return sp_model, depth_model, vdev


_running = True


def _stop(signum, frame):
    global _running
    _running = False


def service_main():
    import signal
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        os.sched_setaffinity(0, HAILO_CORES)
    except OSError:
        pass

    frame_slot = FrameSlot(SHM_FRAME, (CAP_H, CAP_W, 3))
    result_slot = BlobSlot(SHM_RESULT)
    ctrl_slot = BlobSlot(SHM_CTRL)

    sp_model, depth_model, vdev = _load_models()
    reacq = SpReacquirer(sp_model) if sp_model else OrbReacquirer()

    last_fid = 0
    last_anchor_fid = -1
    last_anchor_try = -100
    audit_cnt = 0
    ok_streak = 0
    try:
        while _running:
            ctrl, _, _, _ = ctrl_slot.read()
            if ctrl is None:
                time.sleep(0.02)
                continue
            if ctrl.get("quit"):
                break
            state = ctrl.get("state", "IDLE")

            img, t_cap, fid, seq = frame_slot.read()
            if img is None or fid == last_fid:
                time.sleep(0.005)
                continue
            last_fid = fid

            try:
                ab = ctrl.get("anchor_box")
                afid = ctrl.get("anchor_fid", -1)
                if (ab is not None and afid != last_anchor_fid
                        and fid - last_anchor_try >= 30):
                    last_anchor_try = fid
                    src = img if reacq.name == "superpoint" else _to_proc_gray(img)
                    ok_anchor = reacq.set_anchor(src, ab)
                    if ok_anchor:
                        last_anchor_fid = afid
                    result_slot.write({"kind": "anchor", "ok": ok_anchor,
                                       "t_frame": t_cap}, t_cap, fid)
                elif ab is None and afid == -1 and reacq.ready:
                    reacq.clear()
                    last_anchor_fid = -1

                if state == "REACQUIRE":
                    src = img if reacq.name == "superpoint" else _to_proc_gray(img)
                    t0 = time.perf_counter()
                    box, n = reacq.search(src)
                    result_slot.write({"kind": "reacq", "t_frame": t_cap,
                                       "box": box, "matches": n,
                                       "ms": (time.perf_counter() - t0) * 1e3,
                                       "backend": reacq.name}, t_cap, fid)
                elif state in ("TRACK", "TERMINAL"):
                    busy = False
                    if depth_model is not None:
                        t0 = time.perf_counter()
                        target_m, center_m = depth_model.infer(
                            img, ctrl.get("track_box"))
                        result_slot.write({"kind": "depth", "t_frame": t_cap,
                                           "target_m": target_m,
                                           "center_m": center_m,
                                           "ms": (time.perf_counter() - t0) * 1e3},
                                          t_cap, fid)
                        busy = True
                    audit_cnt += 1
                    if reacq.ready and audit_cnt >= AUDIT_EVERY:
                        audit_cnt = 0
                        src = (img if reacq.name == "superpoint"
                               else _to_proc_gray(img))
                        sp_box, n = reacq.search(src)
                        result_slot.write({"kind": "audit", "t_frame": t_cap,
                                           "box": sp_box, "matches": n},
                                          t_cap, fid)
                        tb = ctrl.get("track_box")
                        if sp_box is not None and tb is not None:
                            dx = (sp_box[0] + sp_box[2] / 2) - (tb[0] + tb[2] / 2)
                            dy = (sp_box[1] + sp_box[3] / 2) - (tb[1] + tb[3] / 2)
                            lim = max(tb[2], tb[3])
                            if dx * dx + dy * dy <= lim * lim:
                                ok_streak += 1
                                if ok_streak % AUDIT_REFRESH == 0:
                                    reacq.set_anchor(src, tb)
                            else:
                                ok_streak = 0
                        else:
                            ok_streak = 0
                        busy = True
                    if not busy:
                        time.sleep(0.005)
                else:
                    time.sleep(0.02)
            except Exception as e:
                time.sleep(0.1)
    finally:
        for m in (sp_model, depth_model):
            if m is not None:
                m.model.close()
        if vdev is not None:
            vdev.release()
        frame_slot.close()
        result_slot.close()
        ctrl_slot.close()


def _to_proc_gray(bgr):
    from config import PROC_W, PROC_H
    return cv2.cvtColor(cv2.resize(bgr, (PROC_W, PROC_H)), cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    service_main()
