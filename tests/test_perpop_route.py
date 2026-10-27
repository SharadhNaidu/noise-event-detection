"""submit_ens per-population routing + peak-based natural detection.

Production changes that make these fail:
- a natural clip (peak at int16 full scale) decoded with the synthetic readout or vice versa
  (DATA_DEEP_DIVE: test splits exactly 3437/2080 by peak >= 0.9999 of full scale);
- a plain single readout no longer decoding every clip identically to before.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import submit_ens as SE  # noqa: E402


def test_natural_by_peak():
    arr = np.concatenate([np.array([0, 32767, -5], np.int16), np.array([0, 32440, -32440], np.int16)])
    idx = {"nat": {"off": 0, "n": 3}, "syn": {"off": 3, "n": 3}}
    assert SE.natural_by_peak(arr, idx, ["nat", "syn"]) == {"nat": True, "syn": False}
    arr3 = np.array([0, 32766, 5], np.int16)                    # the store rounds full scale to 32766 too:
    assert SE.natural_by_peak(arr3, {"c": {"off": 0, "n": 3}}, ["c"]) == {"c": True}   # 2734/3437 test naturals
    arr2 = np.array([-32768, 0], np.int16)                      # negative full scale also natural
    assert SE.natural_by_peak(arr2, {"c": {"off": 0, "n": 2}}, ["c"]) == {"c": True}


def test_route_uses_population_readout():
    T = 100
    post = {"nat": np.full(T, 0.5, np.float32), "syn": np.full(T, 0.5, np.float32)}
    dist = {c: np.zeros((T, 2), np.float32) for c in post}
    ro = {"kind": "perpop", "natural": {"kind": "global", "thr": 0.4, "med": 1, "min_dur": 0.0, "merge_gap": 0.0},
          "synthetic": {"kind": "global", "thr": 0.6, "med": 1, "min_dur": 0.0, "merge_gap": 0.0}}
    out = SE.decode_routed(post, dist, ["nat", "syn"], ro, {"nat": True, "syn": False})
    assert len(out["nat"]) == 1 and out["syn"] == []          # 0.5 > 0.4 natural; 0.5 < 0.6 synthetic


def test_plain_readout_unchanged():
    T = 100
    post = {"a": np.full(T, 0.5, np.float32)}
    dist = {"a": np.zeros((T, 2), np.float32)}
    ro = {"kind": "global", "thr": 0.4, "med": 1, "min_dur": 0.0, "merge_gap": 0.0}
    import train as TR
    assert SE.decode_routed(post, dist, ["a"], ro, {"a": True}) == TR.decode_all(post, ro, dist, ["a"])


if __name__ == "__main__":
    bad = 0
    for f in (test_natural_by_peak, test_route_uses_population_readout, test_plain_readout_unchanged):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
