"""Post-processing measured to help on DECIDE.

empty_clip_fallback: a clip where the readout produced no event gets its most
confident region (frames >= q * clip max). [MEASURED 2026-09-30 on hon_A:
decide +0.0037 at q=0.7, tune +0.0035] — must be re-validated per shipped model.
"""
from __future__ import annotations

import numpy as np

FS = 0.02


def clip_to_duration(preds, dur):
    """Trim events to the real clip length; drop events that start past it.

    Short clips are zero-padded to a full crop at inference, so the decoder can
    emit events in the padded tail. [MEASURED 2026-10-01 DECIDE: bft_S
    1.4021 -> 1.4060, hon_A 1.3914 -> 1.3960.] A rule, not a tuned value.
    """
    out = {}
    for c, ev in preds.items():
        d = dur.get(c)
        out[c] = ev if d is None else [(a, min(b, d)) for a, b in ev if a < d]
    return out


def empty_clip_fallback(preds, post, q=0.7, floor=0.05):
    out = dict(preds)
    for c, ev in preds.items():
        if ev or c not in post:
            continue
        p = np.asarray(post[c], dtype=np.float32)
        if not len(p):
            continue
        m = p >= max(q * float(p.max()), floor)
        if m.any():
            idx = np.where(m)[0]
            out[c] = [(idx[0] * FS, (idx[-1] + 1) * FS)]
    return out


def overlay_nested(base, track, thr, ratio, fs=0.02, min_dur=0.06):
    """Add events of a second (category-head) track that sit INSIDE a clearly longer
    base event. A single decoded track can never emit overlapping events, yet 73% of
    merged natural refs are nested cross-category pairs (mostly human_non_speech in a
    background event) [MEASURED 2026-10-01]; the scorer is category-agnostic and accepts
    overlapping predictions. Measured on DECIDE (cross-fit): +0.004..0.005 combined.

    base: [(on, off)] seconds; track: (T,) posterior of the second channel at `fs`."""
    p = np.asarray(track) >= thr
    cand, i = [], 0
    while i < len(p):
        if p[i]:
            j = i
            while j < len(p) and p[j]:
                j += 1
            if (j - i) * fs >= min_dur - 1e-9:
                cand.append((round(i * fs, 6), round(j * fs, 6)))
            i = j
        else:
            i += 1
    out = list(base)
    for h in cand:
        for e in base:
            t = max(0.2 * (e[1] - e[0]), 0.05)
            inside = e[0] - fs <= h[0] and h[1] <= e[1] + fs
            same = abs(h[0] - e[0]) <= t and abs(h[1] - e[1]) <= t
            if inside and not same and (e[1] - e[0]) >= ratio * (h[1] - h[0]):
                out.append(h)
                break
    return sorted(out)


def _runs(track, thr, fs=0.02, min_dur=0.06):
    p = np.asarray(track) >= thr
    out, i = [], 0
    while i < len(p):
        if p[i]:
            j = i
            while j < len(p) and p[j]:
                j += 1
            if (j - i) * fs >= min_dur - 1e-9:
                out.append((round(i * fs, 6), round(j * fs, 6)))
            i = j
        else:
            i += 1
    return out


def category_union(base, post8, cfg, fs=0.02, min_dur=0.06):
    """Base (any-noise) events plus per-category track events that do not duplicate
    a base event. cfg = {channel: threshold or None (off)}. Keeps cross-category
    overlaps (perfect per-category tracks score natural F1 0.996 vs 0.894 for a
    perfect union track, rank1_gap_probe 2026-10-01); a duplicate is a category
    event whose both edges lie within the metric tolerance of a base event."""
    out = list(base)
    for ch, thr in sorted(cfg.items()):
        if thr is None:
            continue
        for h in _runs(post8[:, ch], thr, fs, min_dur):
            dup = any(abs(h[0] - e[0]) <= max(0.2 * (e[1] - e[0]), 0.05) and
                      abs(h[1] - e[1]) <= max(0.2 * (e[1] - e[0]), 0.05) for e in out)
            if not dup:
                out.append(h)
    return sorted(out)


def proposal_union(base, proposals, tol=0.04, require_overlap=False):
    """Coverage-preserving fusion of box-head proposals (boxdec.decode) with the base
    (union-presence) events: every base event is KEPT (Dice depends on its coverage), and a
    proposal is added unless both its edges lie within `tol` of an event already kept (a
    duplicate). Nested / overlapping proposals add event identity; replacing a base event by
    a proposal is never done here - it shrank coverage (Dice 1.0 -> 0.669 in the professor's
    toy case, 2026-10-02). require_overlap=True admits only proposals that overlap some base
    event (identity inside detected activity; no free-standing additions)."""
    out = list(base)
    for h in sorted(proposals):
        if require_overlap and not any(min(h[1], e[1]) > max(h[0], e[0]) for e in base):
            continue
        if any(abs(h[0] - e[0]) <= tol and abs(h[1] - e[1]) <= tol for e in out):
            continue
        out.append(h)
    return sorted(out)
