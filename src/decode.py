"""Frame posteriors -> events, and the post-processing search over that mapping.

This is the readout stage. EXP-01 showed it cannot raise the ceiling (a perfect
frame classifier at 20 ms tops out at F1 0.9370), but it decides how much of the
available ceiling a real model actually reaches - and the leaderboard's F1/Dice
asymmetry says most teams lose points here.

Kept deliberately separate from the model so that alternative readouts (SEBB,
boundary regression, HSMM) can be swapped in without touching training.
"""
from __future__ import annotations

import itertools

import numpy as np


def prob_to_events(p, thr=0.5, med=7, frame_sec=0.02, min_dur=0.05, merge_gap=0.10):
    """Binarise a frame-probability vector and read contiguous runs as events.

    Order of operations matters and mirrors the organizers' baseline:
      median filter -> threshold -> contiguous runs -> merge close events ->
      drop short events.

    Args:
        p:          (T,) float array of per-frame probabilities.
        thr:        decision threshold.
        med:        median filter width in frames (1 disables).
        frame_sec:  seconds per frame (0.02 for WavLM's 50 Hz).
        min_dur:    drop events shorter than this, in seconds.
        merge_gap:  merge events separated by less than this, in seconds.

    Returns:
        list[(onset, offset)] in seconds.
    """
    p = np.asarray(p, dtype=np.float32).ravel()
    if p.size == 0:
        return []
    if med and med > 1:
        p = _median_filter_1d(p, int(med))

    b = p >= thr
    if not b.any():
        return []

    d = np.diff(b.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    ev = [[float(s * frame_sec), float(e * frame_sec)] for s, e in zip(starts, ends)]

    if merge_gap and merge_gap > 0:
        merged = [ev[0]]
        for on, off in ev[1:]:
            if on - merged[-1][1] <= merge_gap:
                merged[-1][1] = off
            else:
                merged.append([on, off])
        ev = merged

    return [(on, off) for on, off in ev if off - on >= min_dur]


def _median_filter_1d(x, k):
    """Odd-width median filter with edge replication (scipy-free)."""
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(strided, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Post-processing search
# ---------------------------------------------------------------------------
DEFAULT_GRID = dict(
    thr=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    med=[1, 3, 5, 7, 9, 11, 15],
    min_dur=[0.00, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
    merge_gap=[0.00, 0.03, 0.06, 0.10, 0.16, 0.24],
    clip_gate=[0.0],
)
# The organizers' baseline sweeps min_dur over only [0.05, 0.12, 0.25] and does
# not sweep merge_gap at all. Given 14.1% of events are under 250 ms and 14.3%
# overlap, both parameters should matter more here than on DESED.


def sweep(posteriors, refs, scorer, grid=None, clip_probs=None, frame_sec=0.02,
          top_k=10, verbose=True):
    """Grid-search readout parameters against the real combined metric.

    Args:
        posteriors: dict clip_id -> (T,) frame probability array.
        refs:       dict clip_id -> list[(onset, offset)] ground truth.
        scorer:     module exposing score(refs, preds) -> dict.
        grid:       dict of parameter -> list of values (defaults to DEFAULT_GRID).
        clip_probs: optional dict clip_id -> clip-level probability, for the gate.
        top_k:      how many configurations to return.

    Returns:
        (best_cfg, results) where results is a list of (score_dict, cfg) sorted desc.
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if verbose:
        print(f"  sweeping {len(combos):,} configurations over "
              f"{len(posteriors):,} clips ...", flush=True)

    results = []
    for i, values in enumerate(combos):
        cfg = dict(zip(keys, values))
        gate = cfg.get("clip_gate", 0.0)
        preds = {}
        for cid, p in posteriors.items():
            if clip_probs is not None and gate > 0 and clip_probs.get(cid, 1.0) < gate:
                preds[cid] = []
                continue
            preds[cid] = prob_to_events(
                p, thr=cfg["thr"], med=cfg["med"], frame_sec=frame_sec,
                min_dur=cfg["min_dur"], merge_gap=cfg["merge_gap"])
        out = scorer.score(refs, preds)
        results.append((out, cfg))
        if verbose and (i + 1) % max(1, len(combos) // 10) == 0:
            best = max(results, key=lambda r: r[0]["combined"])
            print(f"    {i+1:>6}/{len(combos)}  best so far "
                  f"{best[0]['combined']:.4f}", flush=True)

    results.sort(key=lambda r: -r[0]["combined"])
    best_out, best_cfg = results[0]
    if verbose:
        print(f"  best: combined={best_out['combined']:.4f} "
              f"(F1 {best_out['event_f1']:.4f} + Dice {best_out['segment_dice']:.4f})")
        print(f"        {best_cfg}")
    return best_cfg, results[:top_k]
