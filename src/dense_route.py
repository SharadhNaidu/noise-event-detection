"""Decode a dense-head member (--box-head --box-kind dense) on DECIDE natural and compare with the hysteresis route.

Lanes: 0 = ch0 (any), 1 = ch7 (human_non_speech), 2 = max(ch1..6); distances from the cached "box" (T, 6) in seconds.
    python src/dense_route.py v12bnD [--thr 0.5,0.6] [--group 0.5] [--use 0 | 0,1,2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import densedec as DD                   # noqa: E402
import postproc as PP                   # noqa: E402
from solo_measure import Decide, fmt    # noqa: E402


def lanes_from_post8(p8):
    return np.stack([p8[:, 0], p8[:, 7], p8[:, 1:7].max(1)], 1)


def dense_preds(dec, tag, thr, group=0.5, use=(0, 1, 2), min_dur=0.06):
    C = dec.post(tag)
    assert "box" in C and C.get("box_kind") == "dense", f"{tag} has no dense box output in its cache"
    out = {}
    for c in dec.nids:
        p8 = C["post"][c]; bx = C["box"][c]; n = min(len(p8), len(bx))
        ev = DD.decode(lanes_from_post8(p8[:n]), bx[:n], dec.dur[c], thr=thr, group=group, min_dur=min_dur, use=use)
        out[c] = [(a, b) for a, b in ev if b > a]
    out = PP.clip_to_duration(PP.empty_clip_fallback(out, {c: C["post"][c][:, 0] for c in dec.nids}, q=0.7), dec.dur)
    return {c: [tuple(e[:2]) for e in out[c]] for c in dec.nids}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--thr", default="0.4,0.5,0.6,0.7")
    ap.add_argument("--group", default="0.5"); ap.add_argument("--use", default="0;0,1,2"); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dec = Decide()
    hyst = dec.natural_preds([a.tag]); f0, d0 = dec.point(hyst)
    print(f"{a.tag} hysteresis route (md02): natural F1 {f0:.4f} Dice {d0:.4f}")
    best = None
    for use in a.use.split(";"):
        u = tuple(int(x) for x in use.split(","))
        for thr in (float(x) for x in a.thr.split(",")):
            for g in (float(x) for x in a.group.split(",")):
                P = dense_preds(dec, a.tag, (thr, thr, thr), g, u)
                f, d = dec.point(P)
                print(f"  dense use {u} thr {thr:.2f} group {g:.2f}: F1 {f:.4f} ({f-f0:+.4f}) Dice {d:.4f} ({d-d0:+.4f}) preds {sum(len(v) for v in P.values())}", flush=True)
                if best is None or f + d > best[0]: best = (f + d, u, thr, g, P)
    print("best:", best[1:4])
    r = dec.compare(best[4], hyst, n=a.n)
    for pop in ("plain", "matched"):
        print(f"  best dense vs hysteresis [{pop}]: F1 {fmt(r[(pop,'event_f1')])} | Dice {fmt(r[(pop,'segment_dice')])}")


if __name__ == "__main__":
    main()
