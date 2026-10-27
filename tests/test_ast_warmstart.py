"""A model fused with the AST stream, warm-started from hon_A, must START as hon_A.

If the widened BiGRU input or a shared LayerNorm over both streams changes the
parent's computation, the "AST vs control" comparison measures a damaged warm
start instead of the AudioSet stream. Production change that makes this fail:
normalising the concatenated [WavLM, AST] vector with one LayerNorm, or
loading the parent with the GRU input weights left fresh.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402

CKPT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir"))) / "ckpt" / "hon_A.pt"


def test_warm_start_reproduces_parent():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    base = FT.FineTuneSED(4)
    base.load_state_dict(blob["state"])
    base.eval()
    fused = FT.FineTuneSED(4, aux_dim=768)
    fused.load_state_dict(FT.widen_state(blob["state"], fused), strict=False)
    fused.eval()
    torch.manual_seed(0)
    w = torch.randn(2, 80000)
    aux = torch.randn(2, 249, 768) * 3
    with torch.no_grad():
        a = torch.sigmoid(base(w)[0])
        b = torch.sigmoid(fused(w, aux=aux)[0])
    assert torch.allclose(a, b, atol=1e-5), f"max diff {(a - b).abs().max():.3g}"


def test_aux_changes_output_once_weights_move():
    """Guards the opposite failure: the AST input silently disconnected."""
    fused = FT.FineTuneSED(4, aux_dim=768)
    fused.eval()
    torch.manual_seed(0)
    w = torch.randn(1, 80000)
    with torch.no_grad():
        a = fused(w, aux=torch.zeros(1, 249, 768))[0]
        b = fused(w, aux=torch.randn(1, 249, 768) * 3)[0]
    assert (a - b).abs().max() > 1e-4, "AST input has no effect on a fresh fused model"


if __name__ == "__main__":
    bad = 0
    for t in [test_warm_start_reproduces_parent, test_aux_changes_output_once_weights_move]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
