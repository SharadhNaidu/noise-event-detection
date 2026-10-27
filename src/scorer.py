"""Fast, exact reimplementation of the Datathon@IndoML 2026 Track 1 metric.

Combined = Event-based F1 (micro, tol = max(0.2*d, 0.05)) + Segment Dice (10 ms, macro)

Correctness contract
--------------------
Every function here must produce bit-identical results to src/reference_scorer.py,
which is a verbatim copy of the organizers' published code. tests/test_scorer.py
enforces that with randomised property tests plus hand-built edge cases.

Faithfully reproduced quirks (do NOT "fix" these -- the real scorer has them):
  * events_to_frames marks frames int(on/L) .. int(off/L) INCLUSIVE, so a
    [0.00, 0.05] event covers 6 frames (60 ms), not 5.
  * n_frames = int(max_time/L) + 1, and max_time is derived from the events
    themselves (last offset + 0.5 s), never from the clip duration.
  * Greedy matching sorts (distance, ref_idx, pred_idx), so ties break by
    reference index then prediction index.
  * A clip with no reference events and no predicted events scores Dice 1.0.
  * clip_ids present in predictions but absent from the reference contribute
    their whole event count as false positives.
"""
from __future__ import annotations

import numpy as np

FRAME_LEN = 0.01
TOL_FRAC = 0.20
TOL_FLOOR = 0.05

Events = list  # list[tuple[float, float]]


# --------------------------------------------------------------------------
# Event-based F1
# --------------------------------------------------------------------------
def match_events(ref_events, pred_events, tolerance_frac=TOL_FRAC):
    """Greedy closest-first one-to-one matching. Returns (tp, fp, fn)."""
    matched_ref, matched_pred = set(), set()
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), TOL_FLOOR)
        for pi, (p_on, p_off) in enumerate(pred_events):
            d_on = abs(p_on - r_on)
            d_off = abs(p_off - r_off)
            if d_on <= tol and d_off <= tol:
                candidates.append((d_on + d_off, ri, pi))
    candidates.sort()
    for _, ri, pi in candidates:
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp


def event_based_f1(ref_data, pred_data, return_counts=False):
    TP = FP = FN = 0
    for clip_id, ref_events in ref_data.items():
        tp, fp, fn = match_events(ref_events, pred_data.get(clip_id, []))
        TP += tp
        FP += fp
        FN += fn
    for clip_id in pred_data:
        if clip_id not in ref_data:
            FP += len(pred_data[clip_id])
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if return_counts:
        return f1, {"TP": TP, "FP": FP, "FN": FN,
                    "precision": precision, "recall": recall}
    return f1


# --------------------------------------------------------------------------
# Segment Dice
# --------------------------------------------------------------------------
def events_to_frames(events, max_time, frame_len=FRAME_LEN):
    """Rasterise events to a boolean frame mask, matching the reference exactly."""
    n_frames = int(max_time / frame_len) + 1
    mask = np.zeros(n_frames, dtype=bool)
    for on, off in events:
        a = int(on / frame_len)
        b = min(int(off / frame_len) + 1, n_frames)
        if a < 0:
            raise ValueError(f"negative onset {on!r}: reference scorer would wrap the index")
        if b > a:
            mask[a:b] = True
    return mask


def segment_dice(ref_data, pred_data, return_per_clip=False):
    scores = []
    per_clip = {}
    for clip_id, ref_events in ref_data.items():
        pred_events = pred_data.get(clip_id, [])
        all_events = list(ref_events) + list(pred_events)
        if not all_events:
            scores.append(1.0)
            per_clip[clip_id] = 1.0
            continue
        max_time = max(off for _, off in all_events) + 0.5
        ref_mask = events_to_frames(ref_events, max_time)
        pred_mask = events_to_frames(pred_events, max_time)
        inter = int(np.count_nonzero(ref_mask & pred_mask))
        total = int(ref_mask.sum() + pred_mask.sum())
        s = 1.0 if total == 0 else 2.0 * inter / total
        scores.append(s)
        per_clip[clip_id] = s
    mean = sum(scores) / len(scores) if scores else 0.0
    if return_per_clip:
        return mean, per_clip
    return mean


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------
def score(ref_data, pred_data, verbose=False):
    """Return the full metric dict: event_f1, segment_dice, combined."""
    f1, counts = event_based_f1(ref_data, pred_data, return_counts=True)
    dice = segment_dice(ref_data, pred_data)
    out = {"event_f1": f1, "segment_dice": dice, "combined": f1 + dice, **counts}
    if verbose:
        print(f"  Event F1 {f1:.4f} | Dice {dice:.4f} | Combined {f1 + dice:.4f}"
              f"   (TP={counts['TP']} FP={counts['FP']} FN={counts['FN']}"
              f"  P={counts['precision']:.4f} R={counts['recall']:.4f})")
    return out
