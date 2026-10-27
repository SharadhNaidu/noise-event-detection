"""One entry point for every decoder, shared by validation and submission.

A clip's saved outputs are (frame (T,8), dist (T,>=2), box (T,6)).

  frame  : threshold readout of the any-noise channel + boundary refine
  box    : instance box head only (boxdec.decode)
  hybrid : box events, plus frame events that no box event already covers
           (IoU below `gap_iou`) - the frame track keeps coverage (Dice) on
           events the heatmaps missed, the boxes supply edges and nesting.
"""
from __future__ import annotations

import boxdec as BD
import train as T


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def frame(out, cfg):
    fr, dist = out[0], out[1]
    return T.decode_all({0: fr[:, 0]}, cfg, {0: dist[:, :2]}, [0])[0]


def apply(spec, out):
    k = spec["kind"]
    if k == "frame":
        return frame(out, spec["cfg"])
    boxes = BD.decode(out[2], *spec["prm"])
    if k == "box":
        return boxes
    if k == "hybrid":
        extra = [e for e in frame(out, spec["cfg"])
                 if all(_iou(e, b) < spec["gap_iou"] for b in boxes)]
        return sorted(boxes + extra)
    if k == "collar":
        return collar_snap(frame(out, spec["cfg"]), out[3], spec["R"], spec["h"])
    raise ValueError(k)


def collar_snap(spans, col, R, h, fs=0.02):
    """Move each span edge to the collar head's strongest peak within +-R s,
    if that peak's probability is >= h; onset stays before offset."""
    import numpy as np
    out = []
    n = len(col)
    for a, b in spans:
        ev = [a, b]
        for e, t in ((0, a), (1, b)):
            i0 = max(int((t - R) / fs), 0)
            i1 = min(int((t + R) / fs) + 1, n)
            if i1 <= i0:
                continue
            j = i0 + int(np.argmax(col[i0:i1, e]))
            if col[j, e] >= h:
                ev[e] = (j + 0.5) * fs
        if ev[1] - ev[0] >= 0.04:
            out.append((max(0.0, ev[0]), ev[1]))
        else:
            out.append((a, b))
    return out
