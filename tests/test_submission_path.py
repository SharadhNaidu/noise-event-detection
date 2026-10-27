"""Submission writing (shared by single-model and ensemble paths) and the empty-clip fallback.

Production changes that make these fail: an archive that does not round-trip,
a clip-id mismatch that is not refused, malformed events written, the
fallback touching clips that already have events.
"""
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import postproc as PP     # noqa: E402
import submit_ft as SF    # noqa: E402


def test_write_zip_round_trips():
    out = Path(tempfile.mkdtemp()) / "s.zip"
    preds = {"a": [(0.1, 0.5)], "b": []}
    SF.write_zip(preds, out, {"a", "b"})
    with zipfile.ZipFile(out) as z:
        rows = [json.loads(l) for l in z.read("predictions.jsonl").decode().splitlines() if l]
    assert {r["clip_id"] for r in rows} == {"a", "b"}
    assert [r for r in rows if r["clip_id"] == "a"][0]["events"] == [{"onset": 0.1, "offset": 0.5}]


def test_write_zip_refuses_id_mismatch():
    out = Path(tempfile.mkdtemp()) / "s.zip"
    try:
        SF.write_zip({"a": []}, out, {"a", "b"})
    except SystemExit:
        return
    raise AssertionError("a submission missing a test clip must be refused")


def test_fallback_only_fills_empty_clips():
    post = {"a": np.array([0.1, 0.2, 0.9, 0.95, 0.2]), "b": np.array([0.9, 0.9, 0.1])}
    preds = {"a": [], "b": [(0.0, 0.04)]}
    out = PP.empty_clip_fallback(preds, post, q=0.7)
    assert out["b"] == [(0.0, 0.04)], "clip with events must be untouched"
    assert len(out["a"]) == 1 and abs(out["a"][0][0] - 0.04) < 1e-9 and abs(out["a"][0][1] - 0.08) < 1e-9


if __name__ == "__main__":
    bad = 0
    for t in [test_write_zip_round_trips, test_write_zip_refuses_id_mismatch, test_fallback_only_fills_empty_clips]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
