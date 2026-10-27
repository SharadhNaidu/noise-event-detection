"""Loaders for the Datathon@IndoML 2026 Track 1 data.

Validation ships metadata; the test set ships audio only (see COMPETITION_DETAILS.md
sections 12 and 13). Everything here is read-only and validates as it loads --
silent data corruption is the most expensive kind of bug in a competition.
"""
from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("INDOML_DATA", str(ROOT / "data")))   # folder holding the organisers' zips
VAL_ZIP = DATA / "validation.zip"
TEST_ZIP = DATA / "input_data.zip"


def clip_id_from_filename(name: str) -> str:
    """Our working convention: the WAV filename stem.

    WARNING: unverified against the live scorer. The Evaluation page uses
    'vaani_eval_001' placeholders that match nothing in the distributed data,
    and the only identifier participants receive is the filename. Confirm with
    a probe submission before trusting this for a real run.
    """
    return Path(name).stem


def load_validation(path: Path = VAL_ZIP, strict: bool = True):
    """Return (refs, meta).

    refs : dict clip_id -> list[(onset, offset)] sorted by onset
    meta : dict clip_id -> full metadata record
    """
    with zipfile.ZipFile(path) as z:
        records = json.loads(z.read("validation/validationMetadata.json").decode("utf-8"))

    refs, meta = {}, {}
    issues = Counter()
    for rec in records:
        cid = clip_id_from_filename(rec["segmentFileName"])
        if cid in refs:
            issues["duplicate_clip_id"] += 1
        events = []
        for e in rec["NoiseSubCategoryTimeStamp"]:
            # start/end are STRINGS in this dataset, despite the README's example
            on, off = float(e["start"]), float(e["end"])
            if on < 0 or off < 0:
                issues["negative_time"] += 1
                continue
            if off < on:
                issues["offset_before_onset"] += 1
                on, off = off, on
            if off > rec["duration"] + 1e-6:
                issues["event_past_clip_end"] += 1
            events.append((on, off))
        events.sort()
        refs[cid] = events
        meta[cid] = rec

    if issues:
        msg = f"data issues while loading validation: {dict(issues)}"
        if strict and (issues["negative_time"] or issues["duplicate_clip_id"]):
            raise ValueError(msg)
        print(f"  [data.py] note: {msg}")
    return refs, meta


def validation_stats(refs, meta):
    n_ev = sum(len(v) for v in refs.values())
    durs = [m["duration"] for m in meta.values()]
    lens = [off - on for v in refs.values() for on, off in v]
    syn = sum(1 for m in meta.values() if m["syntheticData"])
    return {
        "clips": len(refs),
        "synthetic": syn,
        "natural": len(refs) - syn,
        "total_hours": sum(durs) / 3600,
        "events": n_ev,
        "events_per_clip": n_ev / len(refs),
        "clips_with_zero_events": sum(1 for v in refs.values() if not v),
        "mean_event_len": sum(lens) / len(lens),
        "median_event_len": sorted(lens)[len(lens) // 2],
        "events_under_250ms": sum(1 for x in lens if x < 0.25),
        "frac_under_250ms": sum(1 for x in lens if x < 0.25) / len(lens),
    }


def count_overlapping(refs):
    """How many reference events overlap another event in the same clip?"""
    n_overlap_events, n_clips_with_overlap = 0, 0
    for ev in refs.values():
        if len(ev) < 2:
            continue
        flagged = set()
        for i in range(len(ev)):
            for j in range(i + 1, len(ev)):
                if ev[j][0] < ev[i][1] and ev[i][0] < ev[j][1]:
                    flagged.add(i)
                    flagged.add(j)
        if flagged:
            n_clips_with_overlap += 1
            n_overlap_events += len(flagged)
    return n_overlap_events, n_clips_with_overlap


def list_test_clip_ids(path: Path = TEST_ZIP):
    with zipfile.ZipFile(path) as z:
        wavs = [n for n in z.namelist() if n.endswith(".wav")]
    return sorted(clip_id_from_filename(n) for n in wavs)
