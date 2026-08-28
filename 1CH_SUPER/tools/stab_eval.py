"""flight_crop.mp4 등에 안정화를 적용해 전/후 비교 영상과 잔여 진동을 측정한다.

OBY 화면 녹화에는 정적 OSD(좌상단 텍스트, 중앙 십자선)가 박혀 있어
추정·측정을 0으로 끌어당긴다. --mask-osd(기본 on)로 제거한 프레임을
추정/측정에 쓰고, 비교 영상은 원본으로 만든다. 실기에서는 카메라 원본에서
돌므로 마스킹된 쪽이 실운용에 가깝다.

usage: python3 -m tools.stab_eval <video> [--zoom 1.0] [--tau 0.4] [--out cmp.mp4]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROC_W, PROC_H
from core.stab import Stabilizer

_LK = dict(winSize=(15, 15), maxLevel=2,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def lk_shift(prev, cur):
    """전/후 크롭 프레임 간 전역 이동(px). 실패 시 None."""
    p0 = cv2.goodFeaturesToTrack(prev, 60, 0.01, 8)
    if p0 is None or len(p0) < 8:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **_LK)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **_LK)
    err = np.linalg.norm(p0 - pb, axis=2).ravel()
    ok = (st.ravel() == 1) & (stb.ravel() == 1) & (err < 1.0)
    if ok.sum() < 8:
        return None
    d = (p1 - p0)[ok].reshape(-1, 2)
    return np.median(d[:, 0]), np.median(d[:, 1])


def mask_osd(f):
    f = f[40:, :].copy()
    H, W = f.shape[:2]
    cy, cx, r = H // 2 + 20, W // 2, 45
    f[cy - r:cy + r, cx - r:cx + r] = \
        cv2.blur(f[cy - r:cy + r, cx - r:cx + r], (31, 31))
    return f


def hp(v, fps, win_s=0.4):
    """프레임간 이동 시계열에서 이동평균(의도된 기동·리센터)을 뺀 고주파 성분."""
    v = np.asarray(v)
    k = max(1, int(round(win_s * fps)))
    kern = np.ones(k) / k
    lp = np.vstack([np.convolve(v[:, i], kern, mode="same") for i in (0, 1)]).T
    return v - lp


def stats(name, s):
    mag = np.hypot(s[:, 0], s[:, 1])
    print(f"{name}: p50 {np.percentile(mag, 50):.2f}px"
          f"  p95 {np.percentile(mag, 95):.2f}px  max {mag.max():.1f}px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default="/home/gimbal/stab_eval_cmp.mp4")
    ap.add_argument("--mask-osd", type=int, default=1)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stab = Stabilizer()
    stab.set_mode(1)
    if args.tau is not None:
        stab.tau = args.tau

    wr = None
    prev_raw = prev_stab = None
    d_raw, d_stab, mss, fails = [], [], [], 0
    px_raw, px_stab = [], []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meas = mask_osd(frame) if args.mask_osd else frame
        H, W = meas.shape[:2]
        gray = cv2.cvtColor(cv2.resize(meas, (PROC_W, PROC_H)),
                            cv2.COLOR_BGR2GRAY)
        stab.update(gray, n / fps)
        mss.append(stab.ms)
        z, ox, oy = stab.view(args.zoom)

        cw, ch = max(2, int(W / z)), max(2, int(H / z))
        cx0, cy0 = (W - cw) // 2, (H - ch) // 2
        x0 = max(0, min(W - cw, cx0 + int(round(ox * W / PROC_W))))
        y0 = max(0, min(H - ch, cy0 + int(round(oy * H / PROC_H))))

        mg = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
        for (yy, xx), prev_attr, acc in (((cy0, cx0), "prev_raw", d_raw),
                                         ((y0, x0), "prev_stab", d_stab)):
            g = mg[yy:yy + ch, xx:xx + cw]
            prev = prev_raw if prev_attr == "prev_raw" else prev_stab
            if prev is not None:
                d = lk_shift(prev, g)
                if d is None:
                    fails += 1
                    d = (0.0, 0.0)
                acc.append(d)
                (px_raw if prev_attr == "prev_raw" else px_stab).append(
                    np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean())
            if prev_attr == "prev_raw":
                prev_raw = g
            else:
                prev_stab = g

        vh, vw = meas.shape[0], meas.shape[1]
        raw_v = frame[40:][cy0:cy0 + ch, cx0:cx0 + cw] if args.mask_osd \
            else frame[cy0:cy0 + ch, cx0:cx0 + cw]
        stb_v = frame[40:][y0:y0 + ch, x0:x0 + cw] if args.mask_osd \
            else frame[y0:y0 + ch, x0:x0 + cw]
        vis = np.hstack([cv2.resize(raw_v, (960, 540)),
                         cv2.resize(stb_v, (960, 540))])
        for txt, x in (("RAW", 16), ("STAB", 976)):
            cv2.putText(vis, txt, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2, cv2.LINE_AA)
        if wr is None:
            wr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (vis.shape[1], vis.shape[0]))
        wr.write(vis)
        n += 1

    cap.release()
    if wr is not None:
        wr.release()

    d_raw, d_stab = np.array(d_raw), np.array(d_stab)
    quiet = np.hypot(d_raw[:, 0], d_raw[:, 1]) < 20
    print(f"frames {n}  update {np.mean(mss):.2f}ms (p95 {np.percentile(mss, 95):.2f})"
          f"  measure-fail {fails}  quiet {quiet.mean() * 100:.0f}%")
    print("frame-to-frame shift, all frames:")
    stats("  raw ", d_raw)
    stats("  stab", d_stab)
    print("high-freq vibration only (0.4s moving-avg removed, quiet frames):")
    stats("  raw ", hp(d_raw, fps)[quiet])
    stats("  stab", hp(d_stab, fps)[quiet])
    print("inter-frame pixel diff (lower = steadier; most robust metric):")
    print(f"  raw : p50 {np.percentile(px_raw, 50):.2f}  p95 {np.percentile(px_raw, 95):.2f}")
    print(f"  stab: p50 {np.percentile(px_stab, 50):.2f}  p95 {np.percentile(px_stab, 95):.2f}")
    print(f"compare video: {args.out}")


if __name__ == "__main__":
    main()
