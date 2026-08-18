#!/usr/bin/env python3
import argparse, glob, time
import numpy as np

CALIB = "/home/gimbal/models/split/calib"
SPLIT = "/home/gimbal/models/split"
SCORE_MIN = 0.60
FEAT_SZ, SEARCH = 16, 256


def hann(n):
    w = 0.5 * (1 - np.cos(2 * np.pi / (n + 1) * np.arange(1, n + 1)))
    return (w.reshape(-1, 1) * w.reshape(1, -1)).astype(np.float32)


def decode(score, size, offset, win):
    resp = score[0, 0] * win
    iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
    ox, oy = offset[0, 0, iy, ix], offset[0, 1, iy, ix]
    cx = (ix + ox) / FEAT_SZ * SEARCH
    cy = (iy + oy) / FEAT_SZ * SEARCH
    w = size[0, 0, iy, ix] * SEARCH
    h = size[0, 1, iy, ix] * SEARCH
    return cx, cy, w, h, float(score[0, 0, iy, ix]), (iy, ix)


def nchw(a):
    return np.ascontiguousarray(a.transpose(0, 3, 1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"], default="A")
    ap.add_argument("--hef", required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                InputVStreamParams, OutputVStreamParams, FormatType,
                                InferVStreams)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 2
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    front = ort.InferenceSession(f"{SPLIT}/{args.variant}_front.onnx", so,
                                 providers=["CPUExecutionProvider"])
    fi = [i.name for i in front.get_inputs()]
    fo = [o.name for o in front.get_outputs()]

    gold = np.load(f"{CALIB}/golden.npz")
    fs = sorted(glob.glob("/home/gimbal/lfc_calib/*.npz"))
    assert len(fs) == gold["score"].shape[0]

    hef = HEF(args.hef)
    ivs = hef.get_input_vstream_infos()
    ovs = hef.get_output_vstream_infos()
    in_by_ch = {v.shape[-1]: v.name for v in ivs}

    dev = VDevice()
    ng = dev.configure(hef, ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe))[0]
    in_p = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    win = hann(FEAT_SZ)
    cerr, serr, ref_s, got_s, lat, cell_hit = [], [], [], [], [], []

    with InferVStreams(ng, in_p, out_p) as pipe, ng.activate():
        def run(d):
            mid = dict(zip(fo, front.run(None, {fi[0]: d["fz"], fi[1]: d["fx"]})))
            corr = mid[fo[0]]
            feed = {in_by_ch[64]: np.ascontiguousarray(corr.transpose(0, 2, 3, 1)),
                    in_by_ch[96]: np.ascontiguousarray(d["fx"].transpose(0, 2, 3, 1))}
            return pipe.infer(feed)

        d0 = np.load(fs[0])
        for _ in range(5):
            r0 = run(d0)

        name_score = name_size = name_off = None
        for v in ovs:
            a = r0[v.name]
            if a.shape[-1] == 1:
                name_score = v.name
            elif a.min() >= -1e-6:
                name_size = v.name
            else:
                name_off = v.name
        assert None not in (name_score, name_size, name_off)
        for v in ovs:
            a = r0[v.name]

        for i, f in enumerate(fs):
            d = np.load(f)
            t0 = time.perf_counter()
            r = run(d)
            lat.append((time.perf_counter() - t0) * 1e3)
            sc = nchw(r[name_score]); sz = nchw(r[name_size]); of = nchw(r[name_off])

            gx, gy, gw, gh, gs, gc = decode(gold["score"][i:i+1], gold["size"][i:i+1],
                                            gold["offset"][i:i+1], win)
            hx, hy, hw, hh, hs, hc = decode(sc, sz, of, win)
            diag = float(np.hypot(gw, gh)) or 1.0
            cerr.append(float(np.hypot(hx - gx, hy - gy)) / diag)
            serr.append(abs(hs - gs))
            ref_s.append(gs); got_s.append(hs)
            cell_hit.append(gc == hc)

    q = lambda a, p: float(np.percentile(a, p))
    ref_s, got_s = np.array(ref_s), np.array(got_s)
    hi = ref_s >= SCORE_MIN
    flip = int(((ref_s >= SCORE_MIN) != (got_s >= SCORE_MIN)).sum())
    rank = float(np.corrcoef(np.argsort(np.argsort(ref_s)),
                             np.argsort(np.argsort(got_s)))[0, 1])
    hit = np.array(cell_hit)




if __name__ == "__main__":
    main()
