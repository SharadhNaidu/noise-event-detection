"""Fine-tuned Dasheng stream (aux_mode="dasheng_ft", v10).

Production changes that make these fail: feeding Dasheng normalised instead of
raw-amplitude audio (its log-mel is level-dependent); a 5 s crop not giving
125 x 40 ms frames; init_bn running stats drifting in train mode (batch 6);
Dasheng's stem landing at the TOP LLRD rate; the multi-encoder misaligning.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import beats_enc as BE      # noqa: E402
import finetune as FT       # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _wav(seconds, seed=0):
    g = np.random.default_rng(seed)
    return (0.05 * g.standard_normal(int(16000 * seconds))).astype(np.float32)


def test_matches_reference_dasheng():
    from dasheng import dasheng_base
    w = torch.from_numpy(_wav(5.0))[None]
    ref = dasheng_base().eval()(w)[0].detach().numpy()
    enc = BE.DashengEncoder().eval()
    with torch.no_grad():
        got = enc(w)[0].numpy()
    assert got.shape == ref.shape == (125, 768), (got.shape, ref.shape)
    assert np.allclose(got, ref, atol=1e-4), float(np.abs(got - ref).max())


def test_level_dependent_input():
    enc = BE.DashengEncoder().eval()
    w = torch.from_numpy(_wav(2.0))[None]
    with torch.no_grad():
        a, b = enc(w), enc(4 * w)
    assert not torch.allclose(a, b, atol=1e-3)


def test_init_bn_stats_frozen_in_train_mode():
    enc = BE.DashengEncoder().train()
    bn = enc.m.init_bn[1]
    before = bn.running_mean.clone()
    enc(torch.from_numpy(_wav(2.0))[None].repeat(2, 1))
    assert not bn.training
    assert torch.equal(before, bn.running_mean)
    assert any(p.requires_grad for p in enc.m.blocks.parameters())


def test_stem_gets_bottom_llrd_depth():
    for n in ("aux_enc.m.time_pos_embed", "aux_enc.m.freq_pos_embed", "aux_enc.m.init_bn.1.weight",
              "aux_enc.m.patch_embed.proj.weight"):
        assert FT._beats_depth(n) == 0, n
    assert FT._beats_depth("aux_enc.m.blocks.11.attn.proj.weight") == 12
    assert FT._beats_depth("aux_enc.m.norm.weight") == 13


def test_multi_encoder_beats_plus_dasheng_aligned():
    enc = BE.MultiEncoder(("BEATs", "Dasheng")).to(DEV).eval()
    assert enc.out_dim == 768 + 447 + 768
    with torch.no_grad():
        f = enc(torch.from_numpy(_wav(5.0))[None].to(DEV))
    assert f.shape == (1, 125, enc.out_dim), f.shape
