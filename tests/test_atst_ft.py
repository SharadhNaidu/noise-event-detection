"""In-model ATST-Frame (AudioSet-strong, PretrainedSED) as a second fine-tuned stream (--aux atst_ft).

Why: ensemble gains come from DIVERSE encoders (replacing members with v5 clones cost -0.015);
ATST-F is a different architecture with frame-level AudioSet-strong pre-training, unused so far.

Production changes that make these fail:
- the ATST wrapper/checkpoint not selected for atst_ft (silently loading BEATs);
- frames not at 40 ms (5 s -> 125) or a feature layout other than 768 + 447;
- a deep-copied teacher not capturing its own embedding;
- LLRD depth not recognising ATST block names (all layers dumped at one rate);
- gate / submit paths not treating atst_ft as an in-model stream ("scale" aux input).
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


def test_registry_maps_aux_names_to_pretrained_models():
    assert AF.IN_MODEL["beats_ft"] == "BEATs" and AF.IN_MODEL["atst_ft"] == "ATST-F"
    assert AF.AUX["atst_ft"] == (None, 768 + 447, None)
    assert G.aux_for({"use_ast": True, "aux": "atst_ft"}) == "scale"


def test_atst_encoder_layout_frames_and_teacher_copy():
    enc = BE.BeatsEncoder("ATST-F").eval()
    assert type(enc.m.model).__name__ == "ATSTWrapper" and enc.out_dim == 1215
    x = torch.randn(2, 5 * 16000) * 0.1
    tea = copy.deepcopy(enc)
    with torch.no_grad():
        out = tea(x)
    assert out.shape == (2, 125, 1215)
    assert tea._emb is not None and enc._emb is None


def test_llrd_depth_understands_atst_blocks():
    assert FT._beats_depth("aux_enc.m.model.atst.blocks.0.attn.qkv.weight") == 1
    assert FT._beats_depth("aux_enc.m.model.atst.blocks.11.mlp.fc1.weight") == 12
    assert FT._beats_depth("aux_enc.m.model.atst.patch_embed.patch_embed.weight") == 0
    assert FT._beats_depth("aux_enc.m.model.beats.encoder.layers.3.fc1.weight") == 4


def test_model_forward_with_atst_stream():
    net = FT.FineTuneSED(4, aux_dim=AF.AUX["atst_ft"][1], aux_mode="atst_ft",
                         red_head=True, red_ch=8).eval()
    assert type(net.aux_enc.m.model).__name__ == "ATSTWrapper"
    w = torch.randn(1, 5 * 16000)
    with torch.no_grad():
        res = net(w, aux=torch.tensor([0.1]))
    assert res[0].shape[:2] == (1, 249)
