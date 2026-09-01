import os
import time
import cv2
import numpy as np

from config import (TRACKER, NANOTRACK_DIR, TRACK_CONF_THRESH,
                    TRACK_APCE_MIN, TRACK_LOST_FRAMES,
                    TRACK_GRACE_S, COLOR_MAX_D, COLOR0_MAX_D,
                    COLOR_EMA, COLOR_EMA_GATE, COLOR_GRID,
                    TERM_HOLD_FRAC, TERM_CONF_THRESH)

EXEMPLAR = 127
INSTANCE = 255
SCORE_SZ = 16
STRIDE = 16
CONTEXT = 0.5
PENALTY_K = 0.148
WIN_INFLUENCE = 0.462
LR = 0.390


class CsrtTracker:
    name = "csrt"

    def __init__(self):
        self._t = None
        self.box = None
        self.score = 0.0
        self.last_ms = 0.0

    def start(self, img, box, keep_ref=False):
        make = getattr(cv2, "TrackerCSRT_create", None) or cv2.legacy.TrackerCSRT_create
        self._t = make()
        self._t.init(img, tuple(int(v) for v in box))
        self.box = tuple(box)

    def update(self, img):
        if self._t is None:
            return False, None
        t0 = time.perf_counter()
        ok, b = self._t.update(img)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        self.score = 1.0 if ok else 0.0
        self.box = tuple(b) if ok else None
        return ok, self.box

    def feedforward(self, dx, dy):
        pass

    def stop(self):
        self._t = None
        self.box = None

    @property
    def active(self):
        return self._t is not None


class NanoTracker:
    name = "nanotrack"

    def __init__(self):
        import ncnn
        self.ncnn = ncnn
        bb = os.path.join(NANOTRACK_DIR, "nanotrack_backbone_sim-opt")
        hd = os.path.join(NANOTRACK_DIR, "nanotrack_head_sim-opt")
        for p in (bb + ".param", bb + ".bin", hd + ".param", hd + ".bin"):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        self.backbone = ncnn.Net()
        self.backbone.opt.num_threads = 2
        self.backbone.load_param(bb + ".param")
        self.backbone.load_model(bb + ".bin")
        self.head = ncnn.Net()
        self.head.opt.num_threads = 2
        self.head.load_param(hd + ".param")
        self.head.load_model(hd + ".bin")

        g = np.arange(SCORE_SZ, dtype=np.float32) * STRIDE
        self.grid_x = np.tile(g, (SCORE_SZ, 1))
        self.grid_y = self.grid_x.T
        han = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SCORE_SZ) / (SCORE_SZ - 1))
        self.window = np.outer(han, han).astype(np.float32)

        self.zf = None
        self.pos = None
        self.sz = None
        self.avg = None
        self.box = None
        self.score = 0.0
        self.apce = 0.0
        self.color_d = 0.0
        self.color_d0 = 0.0
        self.ref_chroma = None
        self.ref0 = None
        self.last_ms = 0.0
        self._miss = 0
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((240, 320, 3), np.uint8)
        self.start(dummy, (120, 90, 60, 60))
        self.update(dummy)
        self.stop()

    def _subwindow(self, img, pos, model_sz, original_sz):
        c = (original_sz + 1) / 2.0
        x0 = int(round(pos[0] - c))
        y0 = int(round(pos[1] - c))
        x1 = x0 + original_sz - 1
        y1 = y0 + original_sz - 1
        lp = max(0, -x0)
        tp = max(0, -y0)
        rp = max(0, x1 - img.shape[1] + 1)
        bp = max(0, y1 - img.shape[0] + 1)
        x0 += lp
        x1 += lp
        y0 += tp
        y1 += tp
        if lp or tp or rp or bp:
            img = cv2.copyMakeBorder(img, tp, bp, lp, rp,
                                     cv2.BORDER_CONSTANT, value=self.avg)
        patch = img[y0:y1 + 1, x0:x1 + 1]
        return cv2.resize(patch, (model_sz, model_sz))

    @staticmethod
    def _grid_at(img, cx, cy, w, h):
        x0 = int(np.clip(cx - w * 0.45, 0, img.shape[1] - 2))
        y0 = int(np.clip(cy - h * 0.45, 0, img.shape[0] - 2))
        x1 = int(np.clip(cx + w * 0.45, x0 + 1, img.shape[1]))
        y1 = int(np.clip(cy + h * 0.45, y0 + 1, img.shape[0]))
        g = COLOR_GRID
        cells = cv2.resize(img[y0:y1, x0:x1], (g, g),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        s = cells.sum(axis=2, keepdims=True) + 1e-6
        return cells / s

    @staticmethod
    def _grid_dist(a, b):
        return float(np.abs(a - b).sum(axis=2).mean())

    def _box_grid(self, img):
        return self._grid_at(img, float(self.pos[0]), float(self.pos[1]),
                             float(self.sz[0]), float(self.sz[1]))

    def matches_ref(self, img, box):
        if self.ref_chroma is None:
            return True
        x, y, w, h = box
        g = self._grid_at(img, x + w / 2.0, y + h / 2.0, w, h)
        if self._grid_dist(self.ref_chroma, g) > COLOR_MAX_D * 0.7:
            return False
        if self.ref0 is None:
            return True
        return self._grid_dist(self.ref0, g) <= COLOR0_MAX_D * 0.7

    def _extract(self, patch):
        m = self.ncnn.Mat.from_pixels(np.ascontiguousarray(patch),
                                      self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                                      patch.shape[1], patch.shape[0])
        ex = self.backbone.create_extractor()
        ex.set_light_mode(True)
        ex.input("input", m)
        _, out = ex.extract("output")
        return out

    def start(self, img, box, keep_ref=False):
        x, y, w, h = box
        self.pos = np.array([x + w / 2.0, y + h / 2.0], np.float32)
        self.sz = np.array([w, h], np.float32)
        self.avg = tuple(cv2.mean(img)[:3])
        wc = w + CONTEXT * (w + h)
        hc = h + CONTEXT * (w + h)
        s_z = round(np.sqrt(wc * hc))
        z = self._subwindow(img, self.pos, EXEMPLAR, int(s_z))
        self.zf = self._extract(z)
        self.box = tuple(box)
        self.score = 1.0
        self._miss = 0
        self._start_t = time.monotonic()
        if not (keep_ref and self.ref_chroma is not None):
            self.ref_chroma = self._box_grid(img)
            self.ref0 = self.ref_chroma.copy()
        self.aspect = float(w) / float(h)
        self.color_d = 0.0
        self.color_d0 = 0.0

    def feedforward(self, dx, dy):
        # 안정화 전역 이동(자기운동) 프라이어: 탐색 중심만 미리 옮긴다
        if self.zf is not None:
            self.pos[0] += dx
            self.pos[1] += dy

    def update(self, img):
        if self.zf is None:
            return False, None
        t0 = time.perf_counter()

        w, h = self.sz
        wc = w + CONTEXT * (w + h)
        hc = h + CONTEXT * (w + h)
        s_z = np.sqrt(wc * hc)
        scale_z = EXEMPLAR / s_z
        pad = (INSTANCE - EXEMPLAR) / 2.0 / scale_z
        s_x = s_z + 2 * pad

        x_crop = self._subwindow(img, self.pos, INSTANCE, int(s_x))
        xf = self._extract(x_crop)

        ex = self.head.create_extractor()
        ex.set_light_mode(True)
        ex.input("input1", self.zf)
        ex.input("input2", xf)
        _, cls = ex.extract("output1")
        _, reg = ex.extract("output2")
        cls = np.array(cls)
        reg = np.array(reg)

        score = 1.0 / (1.0 + np.exp(-cls[1]))
        x1 = self.grid_x - reg[0]
        y1 = self.grid_y - reg[1]
        x2 = self.grid_x + reg[2]
        y2 = self.grid_y + reg[3]
        pw = x2 - x1
        ph = y2 - y1

        tz = self.sz * scale_z
        pad_wh = (tz[0] + tz[1]) * 0.5
        sz_wh = np.sqrt((tz[0] + pad_wh) * (tz[1] + pad_wh))
        pad_p = (pw + ph) * 0.5
        s_c = np.sqrt((pw + pad_p) * (ph + pad_p)) / sz_wh
        s_c = np.maximum(s_c, 1.0 / s_c)
        ratio = tz[0] / tz[1]
        r_c = ratio / (pw / ph)
        r_c = np.maximum(r_c, 1.0 / r_c)
        penalty = np.exp(-(s_c * r_c - 1.0) * PENALTY_K)

        pscore = (penalty * score * (1 - WIN_INFLUENCE) +
                  self.window * WIN_INFLUENCE)
        r, c = np.unravel_index(np.argmax(pscore), pscore.shape)

        smin, smax = float(score.min()), float(score.max())
        self.apce = (smax - smin) ** 2 / (float(np.mean((score - smin) ** 2)) + 1e-12)

        px = (x1[r, c] + x2[r, c]) / 2.0
        py = (y1[r, c] + y2[r, c]) / 2.0
        bw = (x2[r, c] - x1[r, c]) / scale_z
        bh = (y2[r, c] - y1[r, c]) / scale_z
        dx = (px - INSTANCE / 2.0) / scale_z
        dy = (py - INSTANCE / 2.0) / scale_z

        lr = penalty[r, c] * score[r, c] * LR
        self.pos[0] = np.clip(self.pos[0] + dx, 0, img.shape[1])
        self.pos[1] = np.clip(self.pos[1] + dy, 0, img.shape[0])

        self.score = float(score[r, c])
        self.last_ms = (time.perf_counter() - t0) * 1e3

        self.frac = float((self.sz[0] * self.sz[1]) /
                          (img.shape[1] * img.shape[0]))
        cur = self._box_grid(img)
        self.color_d = self._grid_dist(self.ref_chroma, cur)
        self.color_d0 = (self._grid_dist(self.ref0, cur)
                         if self.ref0 is not None else 0.0)
        color_bad = (self.color_d > COLOR_MAX_D
                     or self.color_d0 > COLOR0_MAX_D)
        conf = (TERM_CONF_THRESH if self.frac >= TERM_HOLD_FRAC
                else TRACK_CONF_THRESH)
        if time.monotonic() - self._start_t < TRACK_GRACE_S:
            self._miss = self._miss + 1 if color_bad else 0
        elif (self.score < conf or self.apce < TRACK_APCE_MIN
                or color_bad):
            self._miss += 1
        else:
            self._miss = 0
            if self.color_d < COLOR_EMA_GATE:
                self.ref_chroma = ((1 - COLOR_EMA) * self.ref_chroma
                                   + COLOR_EMA * cur)
        if self._miss == 0:
            s_pred = np.sqrt(max(1e-6, bw * bh) /
                             max(1e-6, self.sz[0] * self.sz[1]))
            s_new = (1 - lr) + s_pred * lr
            h_new = np.clip(self.sz[1] * s_new, 10, img.shape[0])
            self.sz[1] = h_new
            self.sz[0] = np.clip(h_new * self.aspect, 10, img.shape[1])
        if self._miss >= TRACK_LOST_FRAMES:
            self.box = None
            return False, None

        self.box = (float(self.pos[0] - self.sz[0] / 2),
                    float(self.pos[1] - self.sz[1] / 2),
                    float(self.sz[0]), float(self.sz[1]))
        return True, self.box

    def stop(self):
        self.zf = None
        self.box = None
        self._miss = 0

    @property
    def active(self):
        return self.zf is not None


def make_tracker():
    if TRACKER == "csrt":
        return CsrtTracker()
    try:
        return NanoTracker()
    except Exception as e:
        return CsrtTracker()
