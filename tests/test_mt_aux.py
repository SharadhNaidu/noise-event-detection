"""The mean teacher must feed the second stream on UNLABELLED clips too.

Bug this pins: UnlabelledCrops returned only the waveform and the MT step
called teacher(uw, um) / net(uw, um) with no aux, so a BEATs model saw
all-zero BEATs input on every unlabelled clip (the fin_X bug class).
"""
import inspect
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiostore as A    # noqa: E402
import finetune as FT     # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))


def test_unlabelled_crops_return_aux():
    ar, me = A.load_audio_store(ROOT / "audio", "val_audio")
    idx = {f"val_audio:{c}": {**e, "_store": "val_audio", "_cid": c}
           for c, e in list(me["clips"].items())[:8]}
    stores = FT.load_ast_stores(["val_audio"], ROOT / "beats")
    ds = FT.UnlabelledCrops({"val_audio": ar}, idx, list(idx), 5.0, 0,
                            ast=stores, n_frames=249)
    w, aux = ds[0]
    assert tuple(aux.shape) == (249, 1215), aux.shape
    assert float(np.abs(np.asarray(aux)).sum()) > 0, "unlabelled aux is all zeros"


def test_mt_step_passes_aux_to_teacher_and_student():
    src = inspect.getsource(FT.main)
    mt = src[src.index("if teacher is not None:"):src.index("loss = loss + w_mt * cons")]
    assert "teacher(" in mt and "aux=" in mt.split("teacher(", 1)[1].split("\n", 1)[0], \
        "teacher called without aux"
    assert "aux=" in mt.split("s_out = net(", 1)[1].split("\n", 1)[0], "student called without aux"


if __name__ == "__main__":
    bad = 0
    for t in [test_unlabelled_crops_return_aux, test_mt_step_passes_aux_to_teacher_and_student]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
