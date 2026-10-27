"""clip_posteriors(all_channels=True): every head output, column 0 identical to the default track.

Production change that makes this fail: category channels stitched with a
different crop alignment than channel 0, channels reordered, or the default
(single-track) output altered by the new option.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiostore as A    # noqa: E402
import finetune as FT     # noqa: E402
import gate as G          # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))
TAG = os.environ.get("ENS_TEST_TAG", "hon_A")


def test_all_channels_column0_equals_default_track():
    arr, me = A.load_audio_store(ROOT / "audio", "val_audio")
    idx = me["clips"]
    ids = list(idx)[:3]
    net, blob = G.load(TAG, "cpu")
    kw = dict(ast=G.aux_for(blob["cfg"]))
    p1, d1, _ = FT.clip_posteriors(net, arr, idx, ids, blob["cfg"]["crop"], "cpu", **kw)
    pa, da, _ = FT.clip_posteriors(net, arr, idx, ids, blob["cfg"]["crop"], "cpu", all_channels=True, **kw)
    for c in ids:
        assert pa[c].ndim == 2 and pa[c].shape == (len(p1[c]), 8), pa[c].shape
        assert np.array_equal(pa[c][:, 0], p1[c])
        assert np.array_equal(da[c], d1[c])
        assert pa[c].min() >= 0 and pa[c].max() <= 1
        assert not np.allclose(pa[c][:, 1:], pa[c][:, :1])


if __name__ == "__main__":
    try:
        test_all_channels_column0_equals_default_track(); print("PASS test_all_channels_column0_equals_default_track")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL test_all_channels_column0_equals_default_track: {type(e).__name__}: {e}"); sys.exit(1)
