"""Conformer blocks before the BiGRU (--conformer N).

Production changes that make these fail: the flag not reaching the head, the
conformer built but bypassed in forward, or N=0 changing the old architecture
(hon_A would no longer load strictly).
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402

CKPT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir"))) / "ckpt" / "hon_A.pt"


def test_shapes_with_conformer():
    net = FT.FineTuneSED(4, aux_dim=1215, conformer=2).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 80000), aux=torch.randn(2, 249, 1215))
    assert out[0].shape == (2, 249, 8) and out[2].shape == (2, 249, 2)


def test_conformer_is_in_the_forward_path():
    torch.manual_seed(0)
    net = FT.FineTuneSED(4, aux_dim=1215, conformer=2).eval()
    w, a = torch.randn(1, 80000), torch.randn(1, 249, 1215)
    with torch.no_grad():
        y0 = net(w, aux=a)[0].clone()
        for p in net.head.conformer.parameters():
            p.add_(0.5)
        y1 = net(w, aux=a)[0]
    assert (y0 - y1).abs().max() > 1e-4, "conformer weights do not affect the output"


def test_zero_conformer_keeps_old_architecture():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    FT.FineTuneSED(4, conformer=0).load_state_dict(blob["state"])   # strict


if __name__ == "__main__":
    bad = 0
    for t in [test_shapes_with_conformer, test_conformer_is_in_the_forward_path,
              test_zero_conformer_keeps_old_architecture]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
