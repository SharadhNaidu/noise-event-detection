"""--use-ast must feed REAL AudioSet features to training, not zeros.

Regression test for the fin_X bug: finetune.main() created `ast_stores = {}`
and never filled it, so every training crop got an all-zero aux input while
evaluation fed real AST features. The fusion layer learned "aux is always 0"
and then saw 768 unfamiliar dimensions at test time.

Run: python tests/test_ast_wiring.py   (needs $INDOML_ROOT/audio + $INDOML_ROOT/ast,
or INDOML_ROOT pointing at a tree with audio/ and ast/)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiostore as A  # noqa: E402
import finetune as FT   # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))
STORE = "gold_audio"


def test_loader_returns_features():
    stores = FT.load_ast_stores([STORE], ROOT / "ast")
    assert STORE in stores, "AST store for a training store was not loaded"
    arr, clips, hop, dim = stores[STORE]
    assert len(clips) > 0 and arr.shape[1] == 768


def test_training_crops_get_nonzero_aux():
    ar, me = A.load_audio_store(ROOT / "audio", STORE)
    idx = {f"{STORE}:{c}": {**e, "_store": STORE, "_cid": c}
           for c, e in list(me["clips"].items())[:16]}
    ds = FT.CropDataset({STORE: ar}, idx, list(idx), 5.0, True, 0, {}, 249,
                        ast=FT.load_ast_stores([STORE], ROOT / "ast"))
    nonzero = 0
    for i in range(len(ds)):
        item = ds[i]
        aux = next(x for x in item if hasattr(x, "shape") and x.shape[-1:] == (768,))
        nonzero += bool(np.abs(np.asarray(aux)).sum() > 0)
    assert nonzero == len(ds), f"only {nonzero}/{len(ds)} crops carried AST features"


def test_main_uses_loader():
    src = (Path(FT.__file__)).read_text(encoding="utf-8")
    assert "load_ast_stores(" in src.split("def main():", 1)[1], \
        "finetune.main() never loads AST stores for training"


if __name__ == "__main__":
    bad = 0
    for t in [test_loader_returns_features, test_training_crops_get_nonzero_aux,
              test_main_uses_loader]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
