"""v2 recipe pieces: degenerate-clip weights, BEATs layer-wise LR decay, warm-head freeze.

Production changes that make these fail:
- a clip with a >=95% event still getting clip/boundary weight 0 under --clip-w-fix
  (audit 2026-10-01: ~8% of gold/valtune ignored by EVERY loss);
- an LLRD param group with the wrong rate, or a trainable tensor in no group
  (it would silently never train) or in two groups;
- the warm-head toggle unfreezing the permanently frozen conv front-end, or
  leaving encoder tensors trainable during the warm epoch.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402


def test_degenerate_clip_keeps_clip_weight_under_fix():
    idx = {"deg": {"events": [(0.0, 4.9)], "dur": 5.0, "tier": "gold"},
           "ok": {"events": [(1.0, 2.0)], "dur": 5.0, "tier": "gold"},
           "sil": {"events": [(1.0, 2.0)], "dur": 5.0, "tier": "gold", "no_boundary": True}}
    fw, cw = FT.build_clip_weights(idx, list(idx), silver_w=0.5, clip_w_fix=True)
    assert fw["deg"] == 0.0 and cw["deg"] == 1.0
    assert fw["ok"] == 1.0 and cw["ok"] == 1.0
    assert fw["sil"] == 0.5 and cw["sil"] == 0.5
    fw0, cw0 = FT.build_clip_weights(idx, list(idx), silver_w=0.5, clip_w_fix=False)
    assert cw0 == fw0 and fw0["deg"] == 0.0                                  # old behaviour preserved


def test_keep_degenerate_frames_restores_frame_weight():
    """Whole-clip events hold 30% of natural reference frames and are covered only
    39% [MEASURED 2026-10-01]; zero frame weight never teaches whole-clip presence."""
    idx = {"deg": {"events": [(0.0, 4.9)], "dur": 5.0, "tier": "gold"},
           "sil": {"events": [(0.0, 5.0)], "dur": 5.0, "tier": "gold", "no_boundary": True}}
    fw, cw = FT.build_clip_weights(idx, list(idx), silver_w=0.5, clip_w_fix=True, keep_deg_frames=True)
    assert fw["deg"] == 1.0 and cw["deg"] == 1.0
    assert fw["sil"] == 0.5 and cw["sil"] == 0.5


def test_store_mult_oversamples_only_named_stores():
    idx = {"gold_audio:a": {"_store": "gold_audio"}, "synth_aug:b": {"_store": "synth_aug"},
           "valtune_audio:c": {"_store": "valtune_audio"}}
    out = FT.oversample(list(idx), idx, {"gold_audio": 2, "valtune_audio": 3})
    assert sorted(out) == sorted(["gold_audio:a"] * 2 + ["synth_aug:b"] + ["valtune_audio:c"] * 3)
    assert FT.oversample(list(idx), idx, {}) == list(idx)
    assert FT.parse_store_mult(["gold_audio=2", "valtune_audio=3"]) == {"gold_audio": 2, "valtune_audio": 3}


class _Beats(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Module(); self.m.model = nn.Module(); self.m.model.beats = nn.Module()
        b = self.m.model.beats
        b.patch_embedding = nn.Linear(2, 2)
        b.encoder = nn.Module(); b.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        b.encoder.layer_norm = nn.LayerNorm(2)
        self.m.strong_head = nn.Linear(2, 2)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Module(); self.enc.conv = nn.Linear(2, 2); self.enc.layers = nn.ModuleList([nn.Linear(2, 2)] * 1)
        for p in self.enc.conv.parameters():
            p.requires_grad_(False)                                            # frozen conv front-end
        self.head = nn.Linear(2, 2)
        self.aux_enc = _Beats()
        self.aux_mode = "beats_ft"


def test_llrd_rates_and_full_partition():
    net = _Net()
    groups = FT.param_groups(net, lr_enc=5e-5, lr_head=1e-3, beats_top=1e-4, llrd=0.8)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)), "a tensor is in two groups"
    trainable = {id(p) for p in net.parameters() if p.requires_grad}
    assert set(seen) == trainable, "trainable tensors missing from (or frozen ones in) the groups"
    L = net.aux_enc.m.model.beats.encoder.layers
    for i in range(12):
        assert abs(lr_of[id(L[i].weight)] - 1e-4 * 0.8 ** (11 - i)) < 1e-12, i
    assert abs(lr_of[id(net.aux_enc.m.model.beats.patch_embedding.weight)] - 1e-4 * 0.8 ** 12) < 1e-12
    assert abs(lr_of[id(net.aux_enc.m.strong_head.weight)] - 1e-4) < 1e-12
    assert abs(lr_of[id(net.enc.layers[0].weight)] - 5e-5) < 1e-12
    assert abs(lr_of[id(net.head.weight)] - 1e-3) < 1e-12


def test_llrd_off_keeps_old_two_groups():
    net = _Net()
    groups = FT.param_groups(net, lr_enc=5e-5, lr_head=1e-3)
    assert [g["lr"] for g in groups] == [5e-5, 1e-3]


def test_warm_head_toggle_never_touches_frozen_conv():
    net = _Net()
    enc = FT.encoder_params(net)
    FT.set_trainable(enc, False)
    assert not any(p.requires_grad for p in enc)
    assert net.head.weight.requires_grad
    FT.set_trainable(enc, True)
    assert all(p.requires_grad for p in enc)
    assert not any(p.requires_grad for p in net.enc.conv.parameters())


if __name__ == "__main__":
    bad = 0
    for t in (test_degenerate_clip_keeps_clip_weight_under_fix, test_keep_degenerate_frames_restores_frame_weight,
              test_store_mult_oversamples_only_named_stores,
              test_llrd_rates_and_full_partition,
              test_llrd_off_keeps_old_two_groups, test_warm_head_toggle_never_touches_frozen_conv):
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
