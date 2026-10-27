"""Ensemble gate: averaging is exact on identical members and truncates, never misaligns.

Production change that makes these fail: averaging without truncation (shape
error or silent misalignment), or averaging that drops clips.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ens_gate as E  # noqa: E402


def test_average_of_identical_members_is_identity():
    p = {"a": np.linspace(0, 1, 50), "b": np.full(30, 0.3)}
    out = E.average([p, p])
    assert set(out) == set(p) and all(np.allclose(out[k], p[k]) for k in p)


def test_average_truncates_to_common_length():
    out = E.average([{"a": np.ones(50)}, {"a": np.zeros(48)}])
    assert len(out["a"]) == 48 and np.allclose(out["a"], 0.5)


def test_average_of_2d_boundary_distances():
    out = E.average([{"a": np.ones((50, 2))}, {"a": np.zeros((49, 2))}])
    assert out["a"].shape == (49, 2) and np.allclose(out["a"], 0.5)


def test_save_readout_round_trips():
    import json, tempfile
    path = Path(tempfile.mkdtemp()) / "r.json"
    E.save_readout(path, ["hon_A", "bts_C"], {"kind": "global", "thr": 0.7})
    d = json.load(open(path))
    assert d["members"] == ["hon_A", "bts_C"] and d["readout"]["thr"] == 0.7


if __name__ == "__main__":
    bad = 0
    for t in [test_average_of_identical_members_is_identity, test_average_truncates_to_common_length,
              test_average_of_2d_boundary_distances, test_save_readout_round_trips]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
