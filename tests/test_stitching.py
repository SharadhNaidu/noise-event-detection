"""clip_posteriors stitching: every frame covered, every crop on the 20 ms grid.

Production change that makes this fail: placing the last crop (start n-L, not a
multiple of 320 samples) at round((n-L)/320), which shifts its posteriors by up
to +-10 ms, or leaving the final frame uncovered (posterior forced to 0) —
audit 2026-10-01: ~100% / 25% of clips > 5 s; 76% of TEST clips are > 5 s.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402

HOP, WIN, L = 320, 400, 80000


class _Enc:
    @staticmethod
    def frames_for(n):
        return (n - WIN) // HOP + 1


class FakeNet(torch.nn.Module):
    """WavLM-like framing: frame t sees samples [t*320, t*320+400) of the crop."""
    def __init__(self, const=None):
        super().__init__()
        self.enc, self.const = _Enc(), const

    def forward(self, x, aux=None):
        B, n = x.shape
        T = _Enc.frames_for(n)
        if self.const is not None:
            lg = torch.full((B, T), self.const)
        else:
            win = torch.stack([x[:, t * HOP: t * HOP + WIN].abs().amax(1) for t in range(T)], 1)
            lg = torch.where(win > 0.5 * x.abs().amax(1, keepdim=True), 20.0, -20.0)
        return lg[:, :, None].repeat(1, 1, 2), torch.zeros(B, 2), torch.zeros(B, T, 2)


def _post(net, wav, snap=True):
    arr = (wav * 32767).astype(np.int16)
    idx = {"c": {"off": 0, "n": len(arr)}}
    post, _, _ = FT.clip_posteriors(net, arr, idx, ["c"], L / FT.SR, "cpu", snap_last=snap)
    return post["c"]


def test_last_crop_on_grid_impulse_lands_on_true_frame():
    n, p = 120100, 100250                       # last crop starts at 40100: not a multiple of 320
    wav = np.random.default_rng(0).normal(0, 1e-3, n).astype(np.float32); wav[p] = 0.9
    post = _post(FakeNet(), wav)
    assert post[p // HOP] > 0.9, (post[p // HOP - 1: p // HOP + 2])
    assert post[p // HOP - 1] < 0.1, (post[p // HOP - 1: p // HOP + 2])


def test_final_frame_is_covered():
    post = _post(FakeNet(const=20.0), np.ones(80100, np.float32) * 0.1, snap=False)   # default placement still fills
    assert len(post) == _Enc.frames_for(80100)
    assert post.min() > 0.99, post[-3:]


if __name__ == "__main__":
    bad = 0
    for t in (test_last_crop_on_grid_impulse_lands_on_true_frame, test_final_frame_is_covered):
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
