import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tracker import NanoTracker


def main():
    video = sys.argv[1]
    x, y, w, h = (float(v) for v in sys.argv[2:6])
    start = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    out_path = sys.argv[7] if len(sys.argv) > 7 else None

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, f = cap.read()
    if not ok:
        sys.exit(1)
    proc = cv2.resize(f, (640, 360))
    t = NanoTracker()
    t.start(proc, (x, y, w, h))
    t._start_t -= 1.0

    wr = None
    if out_path:
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (640, 360))

    i = start
    scores, c0s, events = [], [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        proc = cv2.resize(f, (640, 360))
        tok, box = t.update(proc)
        scores.append(t.score)
        c0s.append(t.color_d0)
        if not tok:
            events.append(("LOST", i))
            break
        if wr is not None:
            vis = proc.copy()
            bx, by, bw, bh = (int(v) for v in box)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(vis, f"{i} s={t.score:.2f} c0={t.color_d0:.2f}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            wr.write(vis)
    cap.release()
    if wr is not None:
        wr.release()

    scores = np.array(scores)
    c0s = np.array(c0s)
    print(f"frames={len(scores)} ({start}->{i})")
    print(f"events={events if events else 'none'}")
    print(f"score p50={np.median(scores):.2f} p05={np.percentile(scores, 5):.2f} min={scores.min():.2f}")
    print(f"c0    p50={np.median(c0s):.3f} p95={np.percentile(c0s, 95):.3f} max={c0s.max():.3f}")
    print(f"final_box={tuple(round(v, 1) for v in box) if box else None}")


if __name__ == "__main__":
    main()
