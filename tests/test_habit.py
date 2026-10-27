"""Unit tests for the labelling-habit model features (src/habit.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import habit as H  # noqa: E402


def test_state_of_parses_vaani_id():
    cid = "IISc_VaaniProject_K_Jharkhand_Deoghar_Deoghar844425030203839_009_GENERIC_0628_19520_23198"
    assert H.state_of(cid) == "jharkhand"
    assert H.state_of("IISc_VaaniProject_S_Uttar-Pradesh_X") == "uttarpradesh"
    assert H.state_of("clip01") == ""                    # non-Vaani names -> unknown state, no crash


def test_tagfeat_shape():
    X = np.random.default_rng(0).random((3, 447 * 3 + 768)).astype(np.float32)
    assert H.tagfeat(X).shape == (3, 27)


def test_features_onehot_unknown_state():
    STATES = ["?", "bihar", "odisha"]
    tf = np.zeros(27)
    r = H.features(tf, "Odisha", 4.5, STATES)
    assert r.shape == (27 + 1 + 3,) and r[27] == 4.5 and list(r[28:]) == [0.0, 0.0, 1.0]
    assert list(H.features(tf, "Narnia", 1.0, STATES)[28:]) == [0.0, 0.0, 0.0]
    assert list(H.features(tf, "?", 1.0, STATES)[28:]) == [0.0, 0.0, 0.0]   # NS("?") == "" (as shipped)
