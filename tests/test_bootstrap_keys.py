"""Bootstrap resampling must count a clip drawn twice as two clips.

Production change that makes this fail: scoring a resample through a dict keyed
by clip id, which silently collapses duplicate draws (audit 2026-10-01: CIs were
~24% too narrow because every replicate was a ~63% unique-clip subsample).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boot_ens as B  # noqa: E402
import scorer as S    # noqa: E402

REFS = {"a": [(0.0, 1.0)], "b": [(2.0, 3.0)]}
PRED = {"a": [(0.0, 1.0)], "b": []}          # a perfect, b fully missed


def test_duplicate_draw_counts_twice():
    once = B.resample_score(REFS, PRED, ["a", "b"])
    twice = B.resample_score(REFS, PRED, ["a", "a", "b"])
    assert abs(once["event_f1"] - 2 * 1 / 3) < 1e-9          # TP1 FN1 -> P1 R.5 -> F1 2/3
    assert abs(twice["event_f1"] - 0.8) < 1e-9               # TP2 FN1 -> P1 R2/3 -> F1 .8
    assert abs(twice["segment_dice"] - 2 / 3) < 1e-9          # dice a=1,a=1,b=0 -> 2/3


def test_matches_plain_score_without_duplicates():
    assert B.resample_score(REFS, PRED, ["a", "b"]) == S.score(REFS, PRED)


if __name__ == "__main__":
    bad = 0
    for t in (test_duplicate_draw_counts_twice, test_matches_plain_score_without_duplicates):
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
