import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.camera import Camera


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/gimbal/calib"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    os.makedirs(out, exist_ok=True)
    cam = Camera()
    try:
        for i in range(n):
            bgr = cam.read()[0]
            path = os.path.join(out, f"calib_{i:05d}.png")
            cv2.imwrite(path, bgr)
            time.sleep(dt)
    finally:
        cam.release()


if __name__ == "__main__":
    main()
