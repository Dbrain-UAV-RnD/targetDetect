import sys

import cv2
import numpy as np
import onnxruntime as ort

from superpoint_postprocess import postprocess

W, H = 640, 400


def main():
    onnx_path = sys.argv[1]
    img_path = sys.argv[2]

    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (W, H))
    x = (g.astype(np.float32) / 255.0)[None, None, :, :]

    sess = ort.InferenceSession(onnx_path)
    inp = sess.get_inputs()[0]
    outs = sess.run(None, {inp.name: x})
    semi = next(v for v in outs if 65 in v.shape)
    desc = next(v for v in outs if 256 in v.shape)
    if semi.shape[1] == 65:
        semi = semi.transpose(0, 2, 3, 1)
    if desc.shape[1] == 256:
        desc = desc.transpose(0, 2, 3, 1)
    assert semi.shape == (1, H // 8, W // 8, 65), semi.shape
    assert desc.shape == (1, H // 8, W // 8, 256), desc.shape

    pts, d = postprocess(semi[0], desc[0])

    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for px, py in pts.astype(int):
        cv2.circle(vis, (px, py), 3, (0, 255, 0), 1)
    cv2.imwrite("out.png", vis)


if __name__ == "__main__":
    main()
