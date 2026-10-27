"""VERBATIM copy of the organizers' reference scoring code.

Source: Codabench competition 17825, "Evaluation" page, section "Reference Code".
https://www.codabench.org/competitions/17825/

DO NOT OPTIMISE OR "FIX" ANYTHING IN THIS FILE.
Its only job is to be byte-faithful to what the organizers published, so that
our fast implementation in scorer.py can be tested against it. If this file and
scorer.py ever disagree, scorer.py is wrong.
"""


def match_events(ref_events, pred_events, tolerance_frac=0.20):
    matched_ref, matched_pred = set(), set()
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)   # at least 50 ms
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):        # closest first
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri); matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp   # tp, fp, fn


def event_based_f1(ref_data, pred_data):
    TP = FP = FN = 0
    for clip_id, ref_events in ref_data.items():
        tp, fp, fn = match_events(ref_events, pred_data.get(clip_id, []))
        TP += tp; FP += fp; FN += fn
    for clip_id in pred_data:                   # extra clips -> false positives
        if clip_id not in ref_data:
            FP += len(pred_data[clip_id])
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0)


def events_to_frames(events, max_time, frame_len=0.01):
    n_frames = int(max_time / frame_len) + 1
    mask = [0] * n_frames
    for on, off in events:
        for i in range(int(on / frame_len), min(int(off / frame_len) + 1, n_frames)):
            mask[i] = 1
    return mask


def segment_dice(ref_data, pred_data):
    scores = []
    for clip_id, ref_events in ref_data.items():
        pred_events = pred_data.get(clip_id, [])
        all_events = ref_events + pred_events
        if not all_events:
            scores.append(1.0); continue
        max_time = max(off for _, off in all_events) + 0.5   # pad 0.5 s
        ref_mask  = events_to_frames(ref_events,  max_time)
        pred_mask = events_to_frames(pred_events, max_time)
        inter = sum(r & p for r, p in zip(ref_mask, pred_mask))
        total = sum(ref_mask) + sum(pred_mask)
        scores.append(1.0 if total == 0 else 2.0 * inter / total)
    return sum(scores) / len(scores) if scores else 0.0
