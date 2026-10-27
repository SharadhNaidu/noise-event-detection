"""postproc.overlay_nested: add short category-head events nested inside a clearly longer base event.

Why: on natural DECIDE 22.6% of refs are merged into one prediction, 73% of those pairs are OVERLAPPING
cross-category refs (mostly human_non_speech inside a background event) - a single track can never emit both.
Production changes that make these fail:
- adding an event that is NOT inside a base event (would create free false positives);
- adding an event that already equals its base event within the metric tolerance (duplicate);
- ignoring the length ratio (a breath as long as its host is not nested);
- emitting events shorter than min_dur.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import postproc as PP  # noqa: E402

FS = 0.02


def _track(n, spans):
    p = np.zeros(n, np.float32)
    for a, b in spans:
        p[int(a / FS):int(b / FS)] = 0.9
    return p


def test_adds_nested_short_event():
    base = [(1.0, 5.0)]
    out = PP.overlay_nested(base, _track(300, [(2.0, 2.4)]), thr=0.6, ratio=3.0)
    assert out == [(1.0, 5.0), (2.0, 2.4)]


def test_not_inside_base_is_not_added():
    out = PP.overlay_nested([(1.0, 3.0)], _track(300, [(4.0, 4.4)]), thr=0.6, ratio=3.0)
    assert out == [(1.0, 3.0)]


def test_same_as_base_is_not_duplicated():
    out = PP.overlay_nested([(2.0, 2.4)], _track(300, [(2.0, 2.42)]), thr=0.6, ratio=1.0)
    assert out == [(2.0, 2.4)]


def test_ratio_and_min_dur():
    assert PP.overlay_nested([(1.0, 2.0)], _track(300, [(1.2, 1.6)]), thr=0.6, ratio=3.0) == [(1.0, 2.0)]
    assert PP.overlay_nested([(1.0, 5.0)], _track(300, [(2.0, 2.04)]), thr=0.6, ratio=3.0) == [(1.0, 5.0)]


def test_below_threshold_ignored():
    p = _track(300, [(2.0, 2.4)]) * 0.5                     # 0.45 < 0.6
    assert PP.overlay_nested([(1.0, 5.0)], p, thr=0.6, ratio=3.0) == [(1.0, 5.0)]


def test_apply_overlay_natural_only_and_channel():
    import submit_ens as SE
    T = 300
    post = {"n": np.zeros((T, 8), np.float32), "s": np.zeros((T, 8), np.float32)}
    for c in post:
        post[c][100:120, 7] = 0.9                                 # human head fires 2.0-2.4 s
    preds = {"n": [(1.0, 5.0)], "s": [(1.0, 5.0)]}
    ov = {"ch": 7, "thr": 0.6, "ratio": 3.0, "natural_only": True}
    out = SE.apply_overlay(preds, post, ["n", "s"], ov, {"n": True, "s": False})
    assert out["n"] == [(1.0, 5.0), (2.0, 2.4)] and out["s"] == [(1.0, 5.0)]
    assert SE.apply_overlay(preds, post, ["n", "s"], None, {"n": True, "s": False}) == preds
    cat = SE.apply_categories(preds, post, ["n", "s"], {"7": 0.6}, {"n": True, "s": False})   # json keys are strings
    assert cat["n"] == [(1.0, 5.0), (2.0, 2.4)] and cat["s"] == [(1.0, 5.0)]
    assert SE.apply_categories(preds, post, ["n", "s"], None, {"n": True, "s": False}) == preds


def test_category_union_adds_non_duplicate_tracks():
    T = 300
    p8 = np.zeros((T, 8), np.float32)
    p8[50:150, 1] = 0.9          # animal 1.0-3.0 (== base event -> duplicate, not added)
    p8[100:120, 7] = 0.9         # human 2.0-2.4 nested
    p8[200:260, 2] = 0.9         # vehicle 4.0-5.2 partially overlapping base (3.0-4.6) -> added
    base = [(1.0, 3.0), (3.0, 4.6)]
    cfg = {1: 0.6, 2: 0.6, 7: 0.6}
    out = PP.category_union(base, p8, cfg)
    assert (2.0, 2.4) in out and (4.0, 5.2) in out
    assert out.count((1.0, 3.0)) == 1 and len(out) == 4, out
    assert PP.category_union(base, p8, {}) == sorted(base)              # no category enabled = base unchanged
    assert PP.category_union(base, p8, {1: None, 7: 0.95}) == sorted(base)  # off / above-threshold


def test_proposal_union_preserves_coverage_and_adds_identity():
    """Box proposals add event identity without ever shrinking base coverage (professor 2026-10-02: replacing
    activity [0.5,2.5] by proposal [1,2] cut Dice 1.0 -> 0.669)."""
    base = [(0.5, 2.5)]
    out = PP.proposal_union(base, [(1.0, 2.0), (0.52, 2.48), (3.0, 3.4)], tol=0.04)
    assert (0.5, 2.5) in out                 # coverage kept
    assert (1.0, 2.0) in out                 # nested proposal = new identity
    assert (0.52, 2.48) not in out           # duplicate of base within 2 frames
    assert (3.0, 3.4) in out                 # free-standing proposal (a missed event)
    assert PP.proposal_union(base, [], tol=0.04) == base
    assert PP.proposal_union(base, [(3.0, 3.4)], tol=0.04, require_overlap=True) == base   # only identity inside base


if __name__ == "__main__":
    bad = 0
    for f in (test_adds_nested_short_event, test_not_inside_base_is_not_added, test_same_as_base_is_not_duplicated,
              test_ratio_and_min_dur, test_below_threshold_ignored, test_apply_overlay_natural_only_and_channel,
              test_category_union_adds_non_duplicate_tracks, test_proposal_union_preserves_coverage_and_adds_identity):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
