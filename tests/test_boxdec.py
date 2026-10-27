"""boxdec.decode round trip from exact box_targets: distinct overlapping events must survive.

Production changes that make these fail:
- IoU-based suppression dropping a DISTINCT high-overlap event ([1.0,3.0] vs [1.2,3.2], IoU 0.83:
  the metric would score them as two separate references) - external probe 2026-10-01;
- duplicates (both edges within the metric tolerance) not being merged;
- nested events (breath inside traffic) lost.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boxdec as BD  # noqa: E402
import dataset as DS  # noqa: E402


def _rt(events, T=300):
    tgt, _ = DS.box_targets(events, T)
    return BD.decode(tgt)


def _close(got, want, tol=0.03):
    return len(got) == len(want) and all(abs(g[0] - w[0]) <= tol and abs(g[1] - w[1]) <= tol
                                         for g, w in zip(sorted(got), sorted(want)))


def test_distinct_high_iou_events_both_survive():
    ev = [(1.0, 3.0), (1.2, 3.2)]
    assert _close(_rt(ev), ev), _rt(ev)


def test_nested_events_survive():
    ev = [(1.0, 4.0), (2.0, 2.4)]
    assert _close(_rt(ev), ev), _rt(ev)


def test_same_onset_and_same_offset_collisions_survive():
    """Forward durations collide at a shared onset (and backward at a shared offset); the
    other direction still separates them."""
    for ev in ([(1.0, 3.0), (1.0, 1.4)], [(1.0, 3.0), (2.6, 3.0)]):
        assert _close(_rt(ev), ev), (ev, _rt(ev))


def test_events_touching_clip_edges_survive():
    """find_peaks never reports index 0 or T-1: a whole-clip event (both peaks at the ends) was
    dropped entirely (professor 2026-10-02)."""
    for ev in ([(0.0, 6.0)], [(0.0, 1.0)], [(4.0, 6.0)]):
        assert _close(_rt(ev), ev), (ev, _rt(ev))


def test_box_targets_order_independent():
    """Nearby onsets compete for the same regression frames; targets must not depend on annotation order
    (professor 2026-10-02: reversing order flipped log-dur -0.97 <-> 0.69)."""
    import numpy as np
    ev = [(1.00, 3.00), (1.02, 1.40), (2.0, 2.3)]
    t1, m1 = DS.box_targets(ev, 300); t2, m2 = DS.box_targets(ev[::-1], 300)
    assert np.allclose(t1, t2) and np.allclose(m1, m2)


def test_matcher_maximises_admissible_pairs():
    """Admissibility (IoU >= thr) is part of the assignment, not a filter after max-sum matching."""
    fwd = [(0.0, 1.0, .9), (0.0, 2.0, .9)]
    bwd = [(0.0, 1.9, .9), (0.05, 1.0, .9)]          # admissible pairing: f0-b1 and f1-b0
    pairs = BD._match(fwd, bwd, 0.5)
    assert sorted(pairs) == [(0, 1), (1, 0)], pairs


def test_assignment_is_cardinality_first_for_many_pairs():
    """Score 1+IoU is not cardinality-first once >=4 pairs compete (professor 2026-10-02): 4 perfect
    pairs (sum 8) beat 5 pairs at IoU 0.5 (sum 7.5). The bonus must exceed any IoU-sum difference."""
    n = 5
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = 0.5                     # the full 5-pair matching, every pair admissible but weak
    for i in range(n - 1):
        M[i, i + 1] = 1.0                 # a 4-pair matching of perfect pairs that blocks the 5th
    pairs = BD._assign(M, 0.5)
    assert len(pairs) == n, pairs


def test_near_identical_duplicates_merged():
    assert BD._dedupe([(1.0, 3.0, 0.9), (1.02, 3.03, 0.8), (5.0, 5.3, 0.7)]) == [(1.0, 3.0, 0.9), (5.0, 5.3, 0.7)]


if __name__ == "__main__":
    bad = 0
    for f in (test_distinct_high_iou_events_both_survive, test_nested_events_survive,
              test_same_onset_and_same_offset_collisions_survive, test_events_touching_clip_edges_survive,
              test_box_targets_order_independent, test_matcher_maximises_admissible_pairs,
              test_assignment_is_cardinality_first_for_many_pairs,
              test_near_identical_duplicates_merged):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
