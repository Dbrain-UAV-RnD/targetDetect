import numpy as np

CELL = 8
CONF_THRESH = 0.015
NMS_RADIUS = 4
MAX_KP = 200


def postprocess(semi, desc, max_kp=MAX_KP):
    hc, wc, _ = semi.shape
    e = np.exp(semi - semi.max(axis=2, keepdims=True))
    prob = e / e.sum(axis=2, keepdims=True)
    heat = prob[:, :, :64].reshape(hc, wc, CELL, CELL)
    heat = heat.transpose(0, 2, 1, 3).reshape(hc * CELL, wc * CELL)

    ys, xs = np.where(heat > CONF_THRESH)
    if len(xs) == 0:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 256), np.float32))
    conf = heat[ys, xs]

    order = np.argsort(-conf)
    keep = []
    occupied = np.zeros_like(heat, dtype=bool)
    r = NMS_RADIUS
    for i in order:
        y, x = ys[i], xs[i]
        if occupied[y, x]:
            continue
        keep.append(i)
        occupied[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        if len(keep) >= max_kp:
            break
    keep = np.array(keep)
    pts = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)

    fx = (pts[:, 0] / CELL).clip(0, wc - 1.001)
    fy = (pts[:, 1] / CELL).clip(0, hc - 1.001)
    x0, y0 = fx.astype(int), fy.astype(int)
    dx, dy = (fx - x0)[:, None], (fy - y0)[:, None]
    d = (desc[y0, x0] * (1 - dx) * (1 - dy) + desc[y0, x0 + 1] * dx * (1 - dy) +
         desc[y0 + 1, x0] * (1 - dx) * dy + desc[y0 + 1, x0 + 1] * dx * dy)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    return pts, d.astype(np.float32)
