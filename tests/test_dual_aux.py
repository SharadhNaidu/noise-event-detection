"""v9 dual in-model sound encoders (--aux beats_atst_ft): BEATs-strong AND ATST-F-strong, both fine-tuned, their
40 ms features concatenated (2 x 1215) next to WavLM.

Production changes that make these fail:
- only one of the two encoders built (or the same one twice);
- the two streams' frames misaligned or not concatenated (wrong width);
- a deep-copied teacher sharing an encoder's captured embedding with the student;
- LLRD depth not seeing the per-encoder layer names (encs.N. prefix);
- gate/submit not treating the dual stream as in-model ("scale" aux input).
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import astfeat as AF  # noqa: E402
import beats_enc as BE  # noqa: E402
import finetune as FT  # noqa: E402
import gate as G  # noqa: E402


def test_registry_and_width():
    assert AF.IN_MODEL["beats_atst_ft"] == ("BEATs", "ATST-F")
    assert AF.AUX["beats_atst_ft"] == (None, 2 * (768 + 447), None)
    assert G.aux_for({"use_ast": True, "aux": "beats_atst_ft"}) == "scale"


def test_multi_encoder_concatenates_two_distinct_encoders():
    enc = BE.MultiEncoder(("BEATs", "ATST-F")).eval()
    assert [type(e.m.model).__name__ for e in enc.encs] == ["BEATsWrapper", "ATSTWrapper"]
    assert enc.out_dim == 2430
    x = torch.randn(1, 5 * 16000) * 0.1
    tea = copy.deepcopy(enc)
    with torch.no_grad():
        out = tea(x)
    assert out.shape == (1, 125, 2430)
    assert all(e._emb is not None for e in tea.encs) and all(e._emb is None for e in enc.encs)


def test_llrd_depth_with_encoder_prefix():
    assert FT._beats_depth("aux_enc.encs.0.m.model.beats.encoder.layers.5.fc1.weight") == 6
    assert FT._beats_depth("aux_enc.encs.1.m.model.atst.blocks.2.attn.qkv.weight") == 3


def test_model_forward_dual():
    net = FT.FineTuneSED(4, aux_dim=AF.AUX["beats_atst_ft"][1], aux_mode="beats_atst_ft",
                         red_head=True, red_ch=8, conv_heads=5).eval()
    with torch.no_grad():
        res = net(torch.randn(1, 5 * 16000), aux=torch.tensor([0.1]))
    assert res[0].shape[:2] == (1, 249)
