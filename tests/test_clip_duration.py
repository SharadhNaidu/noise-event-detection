"""clip_to_duration: no event may extend past the real clip length.

Production change that makes this fail: padded-tail events shipped as-is
(audit 2026-10-01: 80 bft_S DECIDE events ended past the clip, -0.004 combined),
or trimming that drops/alters events fully inside the clip.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import postproc as PP  # noqa: E402


def test_trims_and_drops_past_end_keeps_inside():
    preds = {"a": [(0.1, 0.5), (2.0, 3.4), (3.5, 4.9)], "b": [(0.0, 1.0)], "c": [(1.0, 2.0)]}
    out = PP.clip_to_duration(preds, {"a": 3.2, "b": 4.0})
    assert out["a"] == [(0.1, 0.5), (2.0, 3.2)]      # trimmed; (3.5, 4.9) starts past end -> dropped
    assert out["b"] == [(0.0, 1.0)]                    # inside -> untouched
    assert out["c"] == [(1.0, 2.0)]                    # unknown duration -> untouched


def test_event_ending_exactly_at_end_is_kept():
    assert PP.clip_to_duration({"a": [(1.0, 2.0)]}, {"a": 2.0})["a"] == [(1.0, 2.0)]


if __name__ == "__main__":
    bad = 0
    for t in (test_trims_and_drops_past_end_keeps_inside, test_event_ending_exactly_at_end_is_kept):
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
