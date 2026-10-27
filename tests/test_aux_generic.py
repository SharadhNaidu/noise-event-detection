"""The second stream must work for ANY frozen encoder store (AST 768-d @ ~101 ms,
BEATs-strong 1215-d @ 40 ms), with frames aligned in time.

Production changes that make these fail: a reader using the AST hop/dim
constants instead of the store's own metadata; ASTStore recording the AST hop
for a store written at another rate; a crop taking aux frames from the wrong
time offset.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402
import finetune as FT     # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))


def test_store_records_its_own_hop():
    d = Path(tempfile.mkdtemp())
    st = AF.ASTStore(d, "t", dim=5, hop=0.04)
    st.add("c", np.ones((3, 5), np.float32))
    st.close()
    assert json.load(open(d / "t.json"))["hop"] == 0.04


def test_loader_returns_hop_and_dim():
    arr, clips, hop, dim = FT.load_ast_stores(["val_audio"], ROOT / "beats")["val_audio"]
    assert dim == 1215 and abs(hop - 0.04) < 1e-9, (hop, dim)


def test_crop_aux_is_time_aligned():
    """A crop starting at t0 must see the aux frame covering t0 first."""
    ar, me = A.load_audio_store(ROOT / "audio", "val_audio")
    stores = FT.load_ast_stores(["val_audio"], ROOT / "beats")
    barr, bclips, hop, dim = stores["val_audio"]
    cid = next(c for c, e in me["clips"].items() if e["n"] > 16000 * 8)   # a long clip
    idx = {f"val_audio:{cid}": {**me["clips"][cid], "_store": "val_audio", "_cid": cid}}
    ds = FT.CropDataset({"val_audio": ar}, idx, list(idx), 5.0, True, 0, {}, 249,
                        ast=stores)
    ds.rng = np.random.default_rng(3)
    item = ds[0]
    aux = next(np.asarray(x) for x in item if hasattr(x, "shape") and x.shape[-1:] == (dim,))
    h = np.asarray(barr[bclips[cid]["off"]: bclips[cid]["off"] + bclips[cid]["n"]], np.float32)
    # some frame of the clip's BEATs features must equal the crop's first aux frame,
    # and the crop's frame i must come from time (start + i*20ms)
    j0 = [j for j in range(len(h)) if np.allclose(h[j], aux[0])]
    assert j0, "first aux frame is not any frame of this clip's store"
    k = 100                                     # 2.0 s into the crop
    expect = h[min(j0[0] + int(round(k * 0.02 / hop)), len(h) - 1)]
    assert np.allclose(aux[k], expect, atol=1e-3), "aux frames drift in time within the crop"


if __name__ == "__main__":
    bad = 0
    for t in [test_store_records_its_own_hop, test_loader_returns_hop_and_dim, test_crop_aux_is_time_aligned]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
