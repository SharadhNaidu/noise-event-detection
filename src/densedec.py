"""Decode the dense interval head (--box-kind dense) into events.

Each lane (0 = any event, 1 = human_non_speech, 2 = other categories) has a presence track and, at every
frame, regressed distances to the onset and offset of the event containing it. Every frame above the lane
threshold proposes an interval; proposals are grouped greedily by presence (highest first, IoU >= `group`)
and each group's edges are the presence-weighted MEDIAN of its members' proposals - many frames vote on
each edge, so an event no longer depends on one sparse peak (v7's peak decoder lost 2.7x more events).
Lanes may overlap; a lane-1/2 event that duplicates a kept event (both edges within the metric tolerance)
is dropped, so nested events survive and duplicates do not.
"""
from __future__ import annotations

import numpy as np

FS = 0.02


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def lane_events(prob, dist, duration, thr, group=0.5, min_dur=0.06, frame_sec=FS):
    """prob: (T,) presence; dist: (T, 2) seconds to onset / offset. Returns [(on, off, score)]."""
    t = (np.arange(len(prob)) + 0.5) * frame_sec
    keep = np.where((prob >= thr) & (t < duration))[0]
    if not len(keep):
        return []
    on = np.clip(t[keep] - dist[keep, 0], 0.0, duration)
    off = np.clip(t[keep] + dist[keep, 1], 0.0, duration)
    p = prob[keep].astype(np.float64)
    free = np.ones(len(keep), bool)
    out = []
    for i in np.argsort(-p):
        if not free[i]:
            continue
        ref = (on[i], off[i])
        g = np.array([i] + [j for j in np.where(free)[0] if j != i and _iou(ref, (on[j], off[j])) >= group], dtype=int)
        free[g] = False
        a, b = _wmedian(on[g], p[g]), _wmedian(off[g], p[g])
        if b - a >= min_dur:
            out.append((a, b, float(p[g].mean())))
    # overlapping survivors within one lane are the same event seen twice: keep the stronger
    out.sort(key=lambda e: -e[2])
    sel = []
    for e in out:
        if all(min(e[1], s[1]) <= max(e[0], s[0]) for s in sel):
            sel.append(e)
    return sorted(sel)


def _dup(e, kept):
    for k in kept:
        tol = max(0.2 * (k[1] - k[0]), 0.05)
        if abs(e[0] - k[0]) <= tol and abs(e[1] - k[1]) <= tol:
            return True
    return False


def decode(lanes, dist, duration, thr=(0.5, 0.5, 0.5), group=0.5, min_dur=0.06, use=(0, 1, 2)):
    """lanes: (T, 3) presence per lane; dist: (T, 6) distances; returns sorted [(on, off)]."""
    kept = []
    for k in use:
        for e in lane_events(lanes[:, k], dist[:, 2 * k:2 * k + 2], duration, thr[k], group, min_dur):
            if not _dup(e, kept):
                kept.append(e)
    return sorted((round(a, 4), round(b, 4)) for a, b, _ in kept)
