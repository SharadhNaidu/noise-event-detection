"""--src-heads: one shared encoder, a separate output head per LABEL SOURCE (gold/valtune, silver, synthetic,
Vaani pseudo-labels, external strong-label corpora), so each source's edge convention is learned on its own head
and never averaged into the others. Inference uses head 0 (Gold = the convention of validation and test).

Production changes that make these fail:
- a source routed to the wrong head (e.g. synth_ext counted as external, pseudo stores as gold);
- the RED head (which places our edges) not being per-source, only the frame/boundary heads;
- extra heads not starting as exact copies of the warm Gold heads (run would not begin at the parent model);
- gate/submit rebuilding a multi-head checkpoint with one head (strict load fails at submission time).
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402
import model as M  # noqa: E402


def test_store_to_source_head():
    t = lambda s: FT.source_head(s, src_heads=True)  # noqa: E731
    assert [t("gold_audio"), t("valtune_audio"), t("silver_long"), t("synth_aug"), t("synth_ext"),
            t("vaani_pl"), t("ext_strong")] == [0, 0, 1, 2, 2, 3, 4]
    # legacy --conv-heads mapping unchanged (0 gold, 1 silver, 2 synth; everything else gold)
    assert [FT.source_head(s, src_heads=False) for s in ("silver_long", "synth_aug", "vaani_pl", "ext_strong")] == [1, 2, 0, 0]


def test_head_count_from_config():
    assert FT.n_conv_heads({"src_heads": True}) == 5
    assert FT.n_conv_heads({"conv_heads": True}) == 3
    assert FT.n_conv_heads({}) == 1


def test_red_head_is_selected_per_source():
    h = M.SEDHead(in_dim=16, rnn_dim=8, red_head=True, red_ch=8, conv_heads=5).eval()
    assert len(h.red_x) == 4
    with torch.no_grad():
        for k, rx in enumerate(h.red_x, start=1):
            rx.weight.zero_(); rx.bias.fill_(float(k))
        h.red.weight.zero_(); h.red.bias.fill_(0.0)
    x = torch.randn(5, 12, 16)
    out = h(x, tier=torch.tensor([0, 1, 2, 3, 4]))
    start = out[3][0]
    assert [float(start[i, 0, 0]) for i in range(5)] == [0.0, 1.0, 2.0, 3.0, 4.0]
    plain = h(x)                                    # inference: no tier -> Gold head
    assert float(plain[3][0].abs().max()) == 0.0


def test_extra_heads_start_as_copies_of_gold():
    h = M.SEDHead(in_dim=16, rnn_dim=8, red_head=True, red_ch=8, conv_heads=5)
    FT.init_source_heads(types.SimpleNamespace(head=h))
    h.eval()                                       # dropout off: compare deterministic outputs
    x = torch.randn(3, 10, 16)
    with torch.no_grad():
        routed, gold = h(x, tier=torch.tensor([0, 3, 4])), h(x)       # same batch: exact comparison
    assert torch.equal(routed[0], gold[0])                            # frame heads identical at start
    assert torch.equal(routed[3][0], gold[3][0]) and torch.equal(routed[3][1], gold[3][1])   # RED start/end


def test_foreign_convention_stores_require_source_heads():
    import pytest
    base = dict(ool_collar=False, red=False, red_classes=True, dice_red=False)
    for st in (["gold_audio", "vaani_pl"], ["gold_audio", "ext_strong"]):
        with pytest.raises(SystemExit):
            FT.validate_objective_flags(types.SimpleNamespace(train_stores=st, src_heads=False, **base))
        FT.validate_objective_flags(types.SimpleNamespace(train_stores=st, src_heads=True, **base))
    FT.validate_objective_flags(types.SimpleNamespace(train_stores=["gold_audio", "silver_long"], src_heads=False, **base))


def test_dev_split_can_be_restricted_to_gold_like_stores():
    """With pseudo-labelled / external stores in training, checkpoint selection must judge only the
    Gold convention; their would-be dev clips go back to training (no data lost)."""
    dev = ["gold_audio:a", "vaani_pl:b", "ext_strong:c", "valtune_audio:d"]
    tr = ["gold_audio:e", "vaani_pl:f"]
    d2, t2 = FT.restrict_dev(dev, tr, ["gold_audio", "valtune_audio"])
    assert d2 == ["gold_audio:a", "valtune_audio:d"]
    assert sorted(t2) == sorted(tr + ["vaani_pl:b", "ext_strong:c"])
    assert FT.restrict_dev(dev, tr, None) == (dev, tr)
