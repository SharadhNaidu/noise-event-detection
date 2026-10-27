"""Synthetic-clip template snap (2026-10-07).

Finding (validation only, scratchpad noise_reuse.py): the synthetic generator inserts noise recordings from a
small reused bank. 99% of 0.3 s inserted-noise templates (noisy - clean) recur exactly (NCC > 0.9) in a
DIFFERENT validation clip, and a whole recording is inserted (an exact onset match gives an exact offset 99%).

Method:
  1. Bank: every labelled event of the validation synthetic clips, as inserted noise = noisy - clean.
     Each event keeps one probe chunk (its most energetic 0.25 s) and its full length.
  2. Match: on a clip's NOISY audio (speech + noise, exactly what test provides), the probe's best normalised
     cross-correlation (NCC) above THR places the whole recording, giving an exact span. Overlapping hits are
     suppressed (one recording per inserted region).
  3. Snap: an ML event whose both edges lie within AGREE of a template span takes the template's exact edges.
     Templates never ADD events (adding unmatched template hits measured -0.077 F1 on held-out DECIDE).
Held-out result (bank = TUNE synthetic only, scored on DECIDE synthetic): Dice 0.9815 -> 0.9907, F1 unchanged.
Applied only to clips routed as synthetic.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, resample_poly

SR_IN = 16000
DEC = 4                      # matching runs at 4 kHz
SR = SR_IN // DEC
W = int(0.25 * SR)           # probe chunk length
THR = 0.5                    # NCC acceptance (held-out tuned only on TUNE->DECIDE matchability, fixed before test)
AGREE = 0.06                 # snap tolerance, seconds (pre-declared)


def _down(x):
    return resample_poly(np.asarray(x, dtype=np.float64), 1, DEC)


class Bank:
    """Probe chunks + full event lengths from labelled inserted-noise events."""

    def __init__(self, probes, offsets, lengths):
        self.probes, self.offsets, self.lengths = probes, offsets, lengths

    @classmethod
    def from_events(cls, events):
        """events: iterable of (inserted_noise_16k, start_s, end_s) where the array is the clip's
        noisy - clean residual (or the bare noise) and (start_s, end_s) the labelled span in it."""
        probes, offsets, lengths = [], [], []
        for resid, a, e in events:
            r = _down(resid)
            seg = r[int(a * SR): int(e * SR)]
            if len(seg) < W:
                continue
            steps = range(0, len(seg) - W + 1, W // 2)
            o = max(steps, key=lambda i: np.std(seg[i:i + W]))
            ch = seg[o:o + W] - seg[o:o + W].mean()
            n = np.linalg.norm(ch)
            if n < 1e-9:
                continue
            probes.append(ch / n); offsets.append(o / SR); lengths.append(e - a)
        return cls(probes, offsets, lengths)

    def save(self, path):
        np.savez_compressed(path, probes=np.asarray(self.probes), offsets=np.asarray(self.offsets),
                            lengths=np.asarray(self.lengths))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(list(z["probes"]), list(z["offsets"]), list(z["lengths"]))


def match_clip(x16k, bank, thr=THR):
    """Template spans [(on, off)] found in one clip's noisy audio (seconds)."""
    x = _down(x16k)
    if len(x) < W or not bank.probes:
        return []
    e2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    en = np.sqrt(np.maximum(e2[W:] - e2[:-W], 1e-12))
    dur = len(x16k) / SR_IN
    cands = []
    for ch, off, L in zip(bank.probes, bank.offsets, bank.lengths):
        ncc = fftconvolve(x, ch[::-1], mode="valid") / en
        p = int(np.argmax(ncc))
        if ncc[p] > thr:
            on = p / SR - off
            cands.append((float(ncc[p]), max(0.0, on), min(dur, on + L)))
    cands.sort(reverse=True)
    keep = []
    for _, a, e in cands:
        if e <= a:
            continue
        if all(min(e, k[1]) - max(a, k[0]) < 0.5 * min(e - a, k[1] - k[0]) for k in keep):
            keep.append((round(a, 3), round(e, 3)))
    return sorted(keep)


def snap(ml_events, tmpl_events, agree=AGREE):
    """ML events keep their count and order; one whose BOTH edges are within `agree` of a template span
    takes that span's exact edges."""
    out = []
    for p in ml_events:
        c = [t for t in tmpl_events if abs(t[0] - p[0]) <= agree and abs(t[1] - p[1]) <= agree]
        out.append(tuple(min(c, key=lambda t: abs(t[0] - p[0]) + abs(t[1] - p[1]))) if c else tuple(p))
    return out
