"""soft_dice_loss: per-clip soft Dice on the any-noise channel (arXiv 2107.10471), as the metric's Dice term.

Production changes that make these fail:
- Dice pooled over the batch instead of per clip (the metric averages per clip, so one long clip
  would dominate);
- padded frames counted;
- an empty-reference clip predicted empty scoring as a loss (the metric scores it perfect).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model as M  # noqa: E402

BIG = 30.0


R = 60                                         # 4 toy frames x 60 = 240, a real 5 s crop's size


def _lg(bits):
    return torch.tensor([[BIG if b else -BIG for b in bits for _ in range(R)]], dtype=torch.float32)


def _t(rows):
    return torch.tensor([[v for v in r for _ in range(R)] for r in rows], dtype=torch.float32)


def test_perfect_and_disjoint():
    y = _t([[1, 1, 0, 0]]); m = torch.ones(1, 4 * R)
    assert M.soft_dice_loss(_lg([1, 1, 0, 0]), y, m) < 1e-2
    assert M.soft_dice_loss(_lg([0, 0, 1, 1]), y, m) > 0.99


def test_empty_reference_predicted_empty_is_perfect():
    y = torch.zeros(1, 4 * R); m = torch.ones(1, 4 * R)
    assert M.soft_dice_loss(_lg([0, 0, 0, 0]), y, m) < 1e-3


def test_padding_ignored():
    y = _t([[1, 1, 0, 0]]); m = _t([[1, 1, 0, 0]])
    assert M.soft_dice_loss(_lg([1, 1, 1, 1]), y, m) < 1e-2     # positives on padded frames don't count


def test_per_clip_mean_and_weights():
    lg = torch.cat([_lg([1, 1, 0, 0]), _lg([0, 0, 1, 1])]); y = _t([[1, 1, 0, 0], [1, 1, 0, 0]])
    m = torch.ones(2, 4 * R)
    assert abs(float(M.soft_dice_loss(lg, y, m)) - 0.5) < 1e-2
    assert float(M.soft_dice_loss(lg, y, m, sample_weight=torch.tensor([1.0, 0.0]))) < 1e-2


def test_gradient_flows():
    lg = torch.zeros(1, 4, requires_grad=True); y = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    M.soft_dice_loss(lg, y, torch.ones(1, 4)).backward()
    assert lg.grad[0, 0] < 0 and lg.grad[0, 3] > 0


def test_prob_dice_on_red_presence():
    """--dice-red: Dice on the DEPLOYED RED presence (probabilities, not logits): gradients reach the
    RED start/end logits; values match the logit version at the same probabilities."""
    import torch
    y = _t([[1, 1, 0, 0]]); m = torch.ones(1, 4 * R)
    p = torch.sigmoid(_lg([1, 1, 0, 0]))
    assert abs(float(M.soft_dice_prob(p, y, m)) - float(M.soft_dice_loss(_lg([1, 1, 0, 0]), y, m))) < 1e-6
    assert float(M.soft_dice_prob(torch.zeros(1, 4 * R), torch.zeros(1, 4 * R), m)) < 1e-3   # empty/empty = perfect
    s_ = torch.zeros(1, 4 * R, 1, requires_grad=True); e_ = torch.zeros(1, 4 * R, 1, requires_grad=True)
    pres, _, _ = M.red_presence(s_, e_)
    M.soft_dice_prob(pres[..., 0], y, m).backward()
    assert s_.grad.abs().sum() > 0 and e_.grad.abs().sum() > 0


def test_dice_red_flag_validation():
    """--dice-red without an active RED head or with dice_w <= 0 must fail at startup, not silently
    train without Dice (professor's v6 launch review)."""
    import types
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import finetune as FT
    ok = types.SimpleNamespace(dice_red=True, dice_w=0.5, red=False, red_classes=True)
    FT.validate_objective_flags(ok)
    for bad in (dict(red=False, red_classes=False), dict(dice_w=0.0)):
        ns = types.SimpleNamespace(**{**vars(ok), **bad})
        try:
            FT.validate_objective_flags(ns)
        except SystemExit:
            continue
        raise AssertionError(f"accepted inconsistent flags {bad}")


if __name__ == "__main__":
    bad = 0
    for f in (test_perfect_and_disjoint, test_empty_reference_predicted_empty_is_perfect, test_padding_ignored,
              test_per_clip_mean_and_weights, test_gradient_flows, test_prob_dice_on_red_presence,
              test_dice_red_flag_validation):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
