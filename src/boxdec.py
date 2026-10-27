"""Decode the instance box head (dataset.box_targets) into events.

Each onset peak emits a forward box (onset, onset + duration); each offset
peak emits a backward box (offset - duration, offset). Forward and backward
boxes describing the same event are paired by IoU (Hungarian), and the pair
takes its onset from the forward box and its offset from the backward one -
each edge comes from the head that was looking straight at it. Every event is
its own box, so a breath inside a traffic span survives as two events.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

FS = 0.02


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _match(fwd, bwd, iou_thr):
    """Pair forward and backward boxes maximising the NUMBER of admissible pairs (IoU >= iou_thr)
    first, then their IoU: admissibility is part of the assignment, not a filter applied after
    a max-sum matching that may spend a box on a sub-threshold pair (professor 2026-10-02)."""
    if not fwd or not bwd:
        return []
    return _assign(np.array([[_iou(f, b) for b in bwd] for f in fwd]), iou_thr)


def _assign(M, iou_thr):
    # bonus per admissible pair > any possible IoU-sum difference (each sum <= min(n_f, n_b)), so the
    # number of pairs strictly dominates; plain 1+IoU lets 4 perfect pairs beat 5 at IoU 0.5
    bonus = min(M.shape) + 1.0
    score = np.where(M >= iou_thr, bonus + M, 0.0)
    r, c = linear_sum_assignment(-score)
    return [(int(i), int(j)) for i, j in zip(r, c) if M[i, j] >= iou_thr]


def decode(box, h=0.3, iou_thr=0.5, h_single=0.6, nms_iou=0.7, min_dur=0.04,
           dur_max=None):
    """box: (T, 6) = [p_on, p_off, log_fwd, log_bwd, res_on, res_off], with
    the two heatmaps already as probabilities."""
    T = len(box)
    fwd, bwd = [], []
    # pad with a zero at both ends: find_peaks never reports index 0 or T-1, which dropped
    # events touching the clip edges (a whole-clip event entirely) - professor 2026-10-02
    pk = lambda x: find_peaks(np.r_[0.0, x, 0.0], height=h, distance=2)[0] - 1
    for i in pk(box[:, 0]):
        t = (i + float(np.clip(box[i, 4], -1.5, 1.5)) + 0.5) * FS
        fwd.append((t, t + float(np.exp(box[i, 2])), float(box[i, 0])))
    for j in pk(box[:, 1]):
        t = (j + float(np.clip(box[j, 5], -1.5, 1.5)) + 0.5) * FS
        bwd.append((t - float(np.exp(box[j, 3])), t, float(box[j, 1])))

    ev, used_f, used_b = [], set(), set()
    for i, j in _match(fwd, bwd, iou_thr):
        ev.append((fwd[i][0], bwd[j][1], 0.5 * (fwd[i][2] + bwd[j][2])))
        used_f.add(i)
        used_b.add(j)
    ev += [f for i, f in enumerate(fwd) if i not in used_f and f[2] >= h_single]
    ev += [b for j, b in enumerate(bwd) if j not in used_b and b[2] >= h_single]

    end = T * FS
    ev = [(max(0.0, a), min(end, b), s) for a, b, s in ev]
    ev = [e for e in ev if e[1] - e[0] >= min_dur]
    if dur_max:
        ev = [e for e in ev if e[1] - e[0] <= dur_max]
    return sorted((a, b) for a, b, _ in _dedupe(ev))


def _dedupe(ev, tol=0.04):
    """Drop a box only when BOTH its edges lie within `tol` (2 frames) of a
    higher-scoring kept box: the same event proposed twice (e.g. an unpaired
    forward and backward box). Neither IoU nor the metric tolerance is a duplicate
    test - two references [1.0,3.0] and [1.2,3.2] (IoU 0.83, inside each other's
    0.4 s tolerance) need two predictions to score two matches (external probe
    2026-10-01). Nested and partially overlapping events survive."""
    keep = []
    for e in sorted(ev, key=lambda e: -e[2]):
        dup = False
        for k in keep:
            if abs(e[0] - k[0]) <= tol and abs(e[1] - k[1]) <= tol:
                dup = True
                break
        if not dup:
            keep.append(e)
    return keep
