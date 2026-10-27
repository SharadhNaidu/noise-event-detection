"""External-noise store (synth_ext): category mapping and event placement.

Production changes that make these fail:
- a DEMAND environment or DESED class mapped to the wrong competition category,
  or an unlabelled / speech-bearing source (cafeteria, DESED Speech) getting into
  the bank: the model would learn speech as noise or noise as background;
- placement that never produces a whole-clip event (the gap we measured:
  0% whole-clip events in every synthetic store), or produces spans outside the host.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import synth_ext as X  # noqa: E402


def test_category_map():
    assert X.category("demand", "DWASHING") == "appliance_machine"
    assert X.category("demand", "TCAR") == "vehicle_traffic"
    assert X.category("demand", "STRAFFIC") == "vehicle_traffic"
    assert X.category("demand", "PCAFETER") is None            # contains speech
    assert X.category("desed", "Dog") == "animal"
    assert X.category("desed", "Vacuum_cleaner") == "appliance_machine"
    assert X.category("desed", "Alarm_bell_ringing") == "phone_signal_alarm"
    assert X.category("desed", "Speech") is None


def test_placement_whole_clip_fraction_and_bounds():
    rng = np.random.default_rng(0)
    L = 16000 * 6
    spans = [X.place(L, rng, p_whole=0.3) for _ in range(4000)]
    whole = np.mean([(a == 0 and b == L) for a, b in spans])
    assert 0.26 < whole < 0.34, whole
    assert all(0 <= a < b <= L for a, b in spans)
    assert all((b - a) >= int(1.5 * 16000) for a, b in spans)  # long backgrounds only


if __name__ == "__main__":
    bad = 0
    for t in (test_category_map, test_placement_whole_clip_fraction_and_bounds):
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
