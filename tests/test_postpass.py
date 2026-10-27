"""Unit tests for the post-passes shipped after the 1.5831 champion (src/postpass.py)."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import postpass as PP  # noqa: E402

CFG = PP.load_cfg()
FS = 0.02


def p8_with(n, ch0=0.0, cat=1):
    q = np.zeros((n, 8), np.float32)
    q[:, 0] = ch0
    q[:, cat] = 0.5
    return q


def test_bgext_extends_background_run():
    q = p8_with(200, ch0=0.35, cat=1)        # argmax over ch1.. = category index 0 (ch1) -> in {0,3,5}
    assert PP.bgext([(1.0, 2.0)], q, 4.0, CFG["bgext"]) == [(0.0, 4.0)]


def test_bgext_leaves_short_events():
    q = p8_with(200, ch0=0.35, cat=1)
    assert PP.bgext([(1.0, 1.5)], q, 4.0, CFG["bgext"]) == [(1.0, 1.5)]


def test_habit_span_adds_full_clip():
    h = CFG["habit"]
    assert PP.habit_span([(1.0, 2.0)], 0.6, 5.0, h) == [(0.0, 5.0), (1.0, 2.0)]
    assert PP.habit_span([(0.2, 4.9)], 0.6, 5.0, h) == [(0.2, 4.9)]          # an event already covers >= 90 %
    assert PP.habit_span([(1.0, 2.0)], 0.4, 5.0, h) == [(1.0, 2.0)]
    assert PP.habit_span([(1.0, 2.0)], None, 5.0, h) == [(1.0, 2.0)]


def test_typec_adds_nested_inner_event():
    n = 200
    q = p8_with(n, ch0=0.8, cat=1)            # host dominated by category 0 (ch1)
    q[50:80, 3] = 0.7                          # category 2 (ch3) active 1.0-1.6 s
    assert PP.typec_events([(0.0, 4.0)], q, 4.0, False, CFG["typec"]) == [(0.0, 4.0), (1.0, 1.6)]
    q2 = p8_with(n, ch0=0.8, cat=1)
    q2[50:80, 1] = 0.9                         # same category as the host -> not an inner event
    assert PP.typec_events([(0.0, 4.0)], q2, 4.0, False, CFG["typec"]) == [(0.0, 4.0)]


def test_typec_skips_habit_host():
    q = p8_with(200, ch0=0.8, cat=1)
    q[50:80, 3] = 0.7
    assert PP.typec_events([(0.0, 4.0)], q, 4.0, True, CFG["typec"]) == [(0.0, 4.0)]


def test_syn_shift_two_step_exact():
    assert PP.syn_shift([(1.0, 2.0)], 3.0, CFG["syn_shift"]) == [(1.008, 1.994)]
    assert PP.syn_shift([(0.0, 0.009)], 3.0, CFG["syn_shift"]) == []


def test_write_predictions_keeps_empty_clips(tmp_path):
    out = tmp_path / "p.zip"
    PP.write_predictions({"a": [(0.1, 0.2)], "b": []}, str(out), ["a", "b"])
    lines = zipfile.ZipFile(out).read("predictions.jsonl").decode().splitlines()
    rows = {json.loads(x)["clip_id"]: json.loads(x)["events"] for x in lines}
    assert rows == {"a": [{"onset": 0.1, "offset": 0.2}], "b": []}
    assert zipfile.ZipFile(out).namelist() == ["predictions.jsonl"]


def test_agreement_identical_and_shifted():
    a = {"x": [(0.0, 1.0), (2.0, 3.0)], "y": []}
    r = PP.agreement(a, a)
    assert r["clips_differing"] == 0 and r["event_f1"] == 1.0 and r["dice"] == 1.0
    b = {"x": [(0.02, 1.0), (2.0, 3.5)], "y": [(0.0, 0.5)]}
    r = PP.agreement(a, b)
    assert r["clips_differing"] == 2
    assert r["event_f1"] == 2 * 1 / (2 + 3)          # only (0,1)~(0.02,1) within 0.2 s collar
    assert abs(r["dice"] - 2 * 1.98 / (2.0 + 2.98)) < 1e-9   # exact interval union, no frame grid
