"""clip_posteriors(..., with_box=True): box-head outputs (B,T,6) stitched over crops like the other channels;
heatmap channels 0-1 as probabilities (sigmoid), regression channels 2-5 raw. Fails if the box output is
dropped, mis-placed across crops, or the heatmaps are left as logits."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402
from test_stitching import _Enc, HOP, L  # noqa: E402


class BoxNet(torch.nn.Module):
    """Box channel 0 = logit +20 on frames whose crop sample is loud, else -20; channel 2 = constant 1.5."""
    def __init__(self):
        super().__init__()
        self.enc = _Enc()

    def forward(self, x, aux=None):
        B, n = x.shape
        T = _Enc.frames_for(n)
        win = torch.stack([x[:, t * HOP: t * HOP + 400].abs().amax(1) for t in range(T)], 1)
        hot = win > 50.0          # absolute, on the normalised crop: the impulse is ~280 sigma, noise < 5
        box = torch.zeros(B, T, 6)
        box[..., 0] = torch.where(hot, 20.0, -20.0); box[..., 2] = 1.5
        lg = torch.zeros(B, T, 2)
        return lg, torch.zeros(B, 2), torch.zeros(B, T, 2), None, None, box


def test_box_outputs_exported_and_stitched():
    n, p = 120100, 100250
    wav = np.random.default_rng(0).normal(0, 1e-3, n).astype(np.float32); wav[p] = 0.9
    arr = (wav * 32767).astype(np.int16); idx = {"c": {"off": 0, "n": n}}
    post, dist, clipp, box = FT.clip_posteriors(BoxNet(), arr, idx, ["c"], L / FT.SR, "cpu", with_box=True)
    b = box["c"]
    assert b.shape == (len(post["c"]), 6)
    assert 0.0 <= b[:, 0].min() and b[:, 0].max() <= 1.0                    # heatmap is a probability
    assert b[p // HOP - 1: p // HOP + 1, 0].max() > 0.4 and b[10, 0] < 0.1   # placed at the impulse, not elsewhere
    assert np.allclose(b[:, 2], 1.5)                                         # regression channel raw


if __name__ == "__main__":
    try:
        test_box_outputs_exported_and_stitched(); print("PASS test_box_outputs_exported_and_stitched")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL test_box_outputs_exported_and_stitched: {type(e).__name__}: {e}"); sys.exit(1)
