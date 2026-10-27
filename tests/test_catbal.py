"""--p-catbal: event-aware category-balanced crops (professor 2026-10-02). With p=1 every crop is centred
(with jitter) on an event of a UNIFORMLY chosen category, so a rare category is drawn ~1/K of the time,
and all co-occurring labels in the crop are kept (multi-label). Fails if sampling stays clip-uniform
(rare category ~its clip share) or if co-occurring labels are dropped."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as DS  # noqa: E402
import finetune as FT  # noqa: E402

SR = 16000


def _ds(p):
    n = 10 * SR
    idx = {}
    for i in range(18):                                    # 18 clips with only animal events
        idx[f"a{i}"] = {"off": 0, "n": n, "events": [(1.0, 2.0), (6.0, 7.0)], "cats": ["animal", "animal"], "tier": "gold"}
    idx["app"] = {"off": 0, "n": n, "events": [(8.0, 9.0), (8.2, 8.6)], "cats": ["appliance_machine", "human_non_speech"], "tier": "gold"}
    arr = np.zeros(n, np.int16)
    return FT.CropDataset(arr, idx, list(idx), crop_s=5.0, train=True, seed=0, n_frames=249, p_center=0.0, p_catbal=p)


def _share(ds, k=2000):
    ia = 1 + DS.CAT2IDX["appliance_machine"]; ih = 1 + DS.CAT2IDX["human_non_speech"]
    app = both = 0
    for t in range(k):
        item = ds[t % len(ds)]
        c = item[2].numpy()
        app += c[ia] > 0; both += (c[ia] > 0) and (c[ih] > 0)
    return app / k, both / max(app, 1)


def test_uniform_category_draws_and_cooccurrence_kept():
    share, co = _share(_ds(1.0))
    assert 0.55 < share < 0.78, share          # appliance or human (nested in it) chosen: ~2/3 (clip-uniform: ~1/19)
    assert co > 0.9, co                        # the nested human event stays labelled


def test_off_by_default():
    share, _ = _share(_ds(0.0))
    assert share < 0.12, share


if __name__ == "__main__":
    bad = 0
    for f in (test_uniform_category_draws_and_cooccurrence_kept, test_off_by_default):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
