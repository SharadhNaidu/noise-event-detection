"""tta_posteriors: zero-pad each clip by p seconds on both sides, run clip_posteriors, trim back, average.

Production changes that make these fail:
- trimming the wrong number of frames after padding (every event shifted by p);
- returning a different length than the unpadded posteriors (averaging misaligns);
- padding the waveform but not the precomputed aux stream (BEATs features then
  describe audio p seconds away from the frame they are fused with).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402
from test_stitching import FakeNet, _Enc, HOP, L  # noqa: E402

PADS = [0.0, 0.64, 1.28]


def _run(wav, pads, ast=None):
    arr = (wav * 32767).astype(np.int16)
    idx = {"c": {"off": 0, "n": len(arr)}}
    base, _, _ = FT.clip_posteriors(FakeNet(), arr, idx, ["c"], L / FT.SR, "cpu", ast=ast)
    tta, _, _ = FT.tta_posteriors(FakeNet(), arr, idx, ["c"], L / FT.SR, "cpu", ast=ast, pads=pads)
    return base["c"], tta["c"]


def test_impulse_stays_on_its_frame_short_and_long():
    for n, p in ((40000, 25000), (120100, 100250)):
        wav = np.random.default_rng(0).normal(0, 1e-3, n).astype(np.float32); wav[p] = 0.9
        base, tta = _run(wav, PADS)
        assert len(tta) == len(base), (len(tta), len(base))
        # the unpadded long clip already splits the impulse 0.5/0.5 (its last crop is
        # off the 20 ms grid by design); padding must reproduce the baseline exactly
        w = slice(p // HOP - 2, p // HOP + 3)
        assert np.allclose(tta[w], base[w], atol=1e-3), (tta[w], base[w])
        assert tta[p // HOP] >= 0.5 and tta[p // HOP - 2] < 0.1 and tta[p // HOP + 2] < 0.1


def test_zero_pad_only_equals_plain():
    wav = np.random.default_rng(1).normal(0, 0.1, 90000).astype(np.float32)
    base, tta = _run(wav, [0.0])
    assert np.allclose(base, tta)


class AuxEcho(torch.nn.Module):
    """Logit = +20 where the aux stream's first feature is > 0.5, else -20."""
    def __init__(self):
        super().__init__()
        self.enc = _Enc()

    def forward(self, x, aux=None):
        lg = torch.where(aux[:, :, 0] > 0.5, 20.0, -20.0)
        B, T = lg.shape
        return lg[:, :, None].repeat(1, 1, 2), torch.zeros(B, 2), torch.zeros(B, T, 2)


def test_precomputed_aux_shifts_with_audio():
    hop, n = 0.04, 60000                        # 3.75 s clip, aux at 40 ms
    na = int(np.ceil(n / FT.SR / hop)) + 1
    feat = np.zeros((na, 4), np.float32); feat[40:45, 0] = 1.0   # 1.60-1.80 s
    ast = (feat, {"c": {"off": 0, "n": na}}, hop, 4)
    arr = np.zeros(n, np.int16); idx = {"c": {"off": 0, "n": n}}
    base, _, _ = FT.clip_posteriors(AuxEcho(), arr, idx, ["c"], L / FT.SR, "cpu", ast=ast)
    tta, _, _ = FT.tta_posteriors(AuxEcho(), arr, idx, ["c"], L / FT.SR, "cpu", ast=ast, pads=PADS)
    b, t = base["c"], tta["c"]
    assert len(b) == len(t)
    on_b, on_t = np.flatnonzero(b > 0.5), np.flatnonzero(t > 0.5)
    assert abs(on_b[0] - on_t[0]) <= 1 and abs(on_b[-1] - on_t[-1]) <= 1, (on_b, on_t)


if __name__ == "__main__":
    bad = 0
    for f in (test_impulse_stays_on_its_frame_short_and_long, test_zero_pad_only_equals_plain,
              test_precomputed_aux_shifts_with_audio):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
