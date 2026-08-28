import os
import numpy as np
import cv2

from config import HEF_DEPTH, DEPTH_W, DEPTH_H, PROC_W, PROC_H

DEPTH_SCALE = float(os.environ.get("DEPTH_SCALE", "1.0"))
DEPTH_SHIFT = float(os.environ.get("DEPTH_SHIFT", "0.0"))


def summarize(dmap, box=None):
    h, w = dmap.shape[:2]
    cy0, cy1 = int(h * 0.4), int(h * 0.6)
    cx0, cx1 = int(w * 0.4), int(w * 0.6)
    center = float(np.median(dmap[cy0:cy1, cx0:cx1]))
    target = None
    if box is not None:
        x, y, bw, bh = box
        x0 = max(0, int(x / PROC_W * w))
        y0 = max(0, int(y / PROC_H * h))
        x1 = min(w, int((x + bw) / PROC_W * w))
        y1 = min(h, int((y + bh) / PROC_H * h))
        if x1 > x0 and y1 > y0:
            target = float(np.median(dmap[y0:y1, x0:x1]))
    to_m = lambda v: None if v is None else v * DEPTH_SCALE + DEPTH_SHIFT
    return to_m(target), to_m(center)


class DepthHef:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def preprocess(bgr):
        return cv2.resize(bgr, (DEPTH_W, DEPTH_H))[None]

    def infer(self, bgr, box=None):
        outs = self.model.run(self.preprocess(bgr))
        dmap = np.squeeze(next(iter(outs.values())))
        return summarize(dmap, box)


def available():
    return os.path.exists(HEF_DEPTH)
