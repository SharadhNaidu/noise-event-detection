"""Speaker-disjoint splitting.

The real test set is speaker-disjoint from everything else - confirmed twice
(the HF dataset card, and `dataset_info.txt` shipped inside input_data.zip:
"split by state/district/speaker"). EXP-01's companion measurement found zero
shared speaker keys between validation and test against 139 shared districts,
so the split is by speaker, not region.

Two consequences this module enforces:

  1. Any local split must group by speaker. A random per-clip split leaks the
     same speaker into both sides and inflates every number we measure.
  2. Validation must be cut in half: one half to TUNE readout parameters, the
     other held back to DECIDE whether a submission is worth spending. Tuning
     four sensitive parameters and then gating on the same clips is how you
     talk yourself into a bad submission.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


def speaker_key(filename: str) -> str:
    """Grouping key from the Vaani filename grammar.

    Every filename in validation and test splits into exactly 11 underscore
    fields (verified on all 2,501 + 5,517):

        IISc_VaaniProject_<batch>_<State>_<District>_<id1>_<id2>_<prompt>_<pid>_<startMs>_<endMs>

    Fields 2..5 (batch, state, district, id1) form a key that never crosses a
    district and averages ~1.3 clips. Three batch schemes exist (K/M/S) with
    different id layouts, so we split positionally rather than by regex -
    district names contain digits (e.g. North24Parganas) which breaks [A-Za-z]+.

    NOTE: that fields 5-6 encode a speaker is *inferred*, not documented. It is
    consistent with the evidence (never crosses a district; 15-digit ids in
    batch K) but treat it as a grouping key of unknown exact semantics.
    """
    stem = filename[:-4] if filename.lower().endswith(".wav") else filename
    f = stem.split("_")
    if len(f) < 6:
        return stem  # degrade to per-clip rather than silently mis-group
    return "_".join(f[2:6])


def _bucket(key: str, seed: int) -> float:
    """Deterministic uniform hash in [0,1) for a key. Stable across runs and
    machines, unlike Python's salted hash()."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_by_speaker(clip_ids, frac: float = 0.5, seed: int = 42):
    """Partition clip_ids into (a, b) with no speaker appearing in both.

    `frac` is the target share of SPEAKERS in `a`; the clip share will be close
    but not exact because speakers hold differing numbers of clips.
    """
    groups = defaultdict(list)
    for cid in clip_ids:
        groups[speaker_key(cid)].append(cid)
    a, b = [], []
    for key, cids in groups.items():
        (a if _bucket(key, seed) < frac else b).extend(cids)
    return sorted(a), sorted(b)


def split_report(a, b, label_a="A", label_b="B"):
    ka = {speaker_key(c) for c in a}
    kb = {speaker_key(c) for c in b}
    overlap = ka & kb
    n = len(a) + len(b)
    return {
        f"{label_a}_clips": len(a),
        f"{label_b}_clips": len(b),
        f"{label_a}_speakers": len(ka),
        f"{label_b}_speakers": len(kb),
        f"{label_a}_frac": len(a) / n if n else 0.0,
        "shared_speakers": len(overlap),
        "disjoint": len(overlap) == 0,
    }
