import numpy as np
import cv2

from config import ANCHOR_MAX_KP, REACQ_MIN_MATCHES


class OrbReacquirer:
    name = "orb"

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.anchor_kp = None
        self.anchor_desc = None
        self.anchor_size = None

    def set_anchor(self, gray, box):
        x, y, w, h = (int(v) for v in box)
        roi = gray[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            return False
        kp, desc = self.orb.detectAndCompute(roi, None)
        if desc is None or len(kp) < REACQ_MIN_MATCHES:
            return False
        if len(kp) > ANCHOR_MAX_KP:
            idx = np.argsort([-k.response for k in kp])[:ANCHOR_MAX_KP]
            kp = [kp[i] for i in idx]
            desc = desc[idx]
        self.anchor_kp = np.array([(k.pt[0] - w / 2, k.pt[1] - h / 2) for k in kp],
                                  np.float32)
        self.anchor_desc = desc
        self.anchor_size = (w, h)
        return True

    @property
    def ready(self):
        return self.anchor_desc is not None

    def clear(self):
        self.anchor_kp = self.anchor_desc = self.anchor_size = None

    def search(self, gray):
        if not self.ready:
            return None, 0
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < REACQ_MIN_MATCHES:
            return None, 0
        matches = self.bf.knnMatch(self.anchor_desc, desc, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < REACQ_MIN_MATCHES:
            return None, len(good)
        pts = np.array([kp[m.trainIdx].pt for m in good], np.float32)
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        w, h = self.anchor_size
        return (float(cx - w / 2), float(cy - h / 2), float(w), float(h)), len(good)
