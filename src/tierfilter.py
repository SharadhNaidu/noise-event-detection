"""Tier-aware label filtering, driven by measurement rather than by tier name.

EDA findings this implements (docs/DATA_UNDERSTANDING.md sections 2 and 4):

  * 26.7% of the entire Silver tier is ONE event spanning 0 -> clip end. That is
    a clip-level label wearing a timestamp: it carries no boundary information
    and actively teaches the model to emit one blob per clip, which is the exact
    failure mode that destroys Event-F1.
  * Event "kind" matters more than tier. A POINT event ([breathing], empty
    <horn></horn>) is a real acoustic event. A WRAPPER event (<bird sound>…text…
    </bird sound>) spans however much transcript the annotator enclosed - its
    duration correlates r=0.936 with character count, not with the sound.
  * human_non_speech is 100% point events and 35% of all events; animal,
    singing_music and appliance_machine are 78-96% wrappers.

So instead of a blanket silver_weight we classify each CLIP by how much usable
boundary information it carries, and weight accordingly.
"""
from __future__ import annotations

import numpy as np

# Fraction of the clip above which a single event is treated as clip-level.
WHOLE_CLIP_FRAC = 0.95
# Coverage above which a clip tells us almost nothing about where noise is NOT.
SATURATED_COV = 0.95


def clip_diagnosis(events, duration):
    """Describe how useful one clip's labels are for frame-level training.

    Returns dict with:
      coverage        fraction of the clip marked noise (union of events)
      n_events        number of events
      max_event_frac  longest single event as a fraction of the clip
      degenerate      True if a single event effectively covers the clip
      saturated       True if the union covers almost everything
    """
    if duration <= 0:
        return dict(coverage=0.0, n_events=len(events), max_event_frac=0.0,
                    degenerate=False, saturated=False)
    iv = sorted((max(0.0, a), min(b, duration)) for a, b in events)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append([a, b])
    cov = sum(b - a for a, b in merged) / duration
    longest = max(((b - a) / duration for a, b in iv), default=0.0)
    return dict(coverage=min(cov, 1.0), n_events=len(events),
                max_event_frac=longest,
                degenerate=bool(longest >= WHOLE_CLIP_FRAC),
                saturated=bool(cov >= SATURATED_COV))


def frame_weight(diag, tier, base_weights=None):
    """Per-clip weight for the FRAME-level loss.

    A degenerate or saturated clip gets weight 0 for the frame head: it cannot
    teach boundaries. It keeps full weight for the CLIP head, where "this clip
    contains noise" is exactly the right supervision - the same role Bronze
    plays.
    """
    base = (base_weights or {}).get(tier, 1.0)
    if diag["degenerate"] or diag["saturated"]:
        return 0.0
    return base


def clip_weight(diag, tier, base_weights=None):
    """Per-clip weight for the CLIP-level (MIL) loss. Always usable."""
    return (base_weights or {}).get(tier, 1.0)


def summarise(index, tier_of=None, label=""):
    """Print how much of a feature store survives frame-level filtering."""
    n = deg = sat = 0
    cov = []
    for cid, e in index.items():
        d = clip_diagnosis(e.get("events", []), e.get("dur_measured") or e.get("dur", 0))
        n += 1
        deg += d["degenerate"]
        sat += d["saturated"]
        cov.append(d["coverage"])
    usable = n - sum(1 for cid, e in index.items()
                     if clip_diagnosis(e.get("events", []),
                                       e.get("dur_measured") or e.get("dur", 0))
                     ["degenerate"] or
                     clip_diagnosis(e.get("events", []),
                                    e.get("dur_measured") or e.get("dur", 0))
                     ["saturated"])
    print(f"  {label:<10} {n:>7,} clips | mean coverage {100*np.mean(cov):>5.1f}% | "
          f"degenerate {deg:>6,} ({100*deg/max(n,1):>4.1f}%) | "
          f"saturated {sat:>6,} ({100*sat/max(n,1):>4.1f}%) | "
          f"usable for frame loss {usable:>7,} ({100*usable/max(n,1):>4.1f}%)")
    return usable
