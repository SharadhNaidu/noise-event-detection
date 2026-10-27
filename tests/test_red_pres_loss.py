"""bts_R2: RED presence trained with plain BCE + pos_weight (as in Schmid et al.) instead of focal gamma=2."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model as M  # noqa: E402


def test_gamma0_is_plain_bce():
    p = torch.tensor([[[0.2], [0.7], [0.9]]]); t = torch.tensor([[[0.0], [1.0], [1.0]]]); m = torch.ones(1, 3)
    want = torch.nn.functional.binary_cross_entropy(p, t)
    assert torch.isclose(M.focal_bce(p, t, m, gamma=0.0), want, atol=1e-6)


def test_pos_weight_scales_positive_term_only():
    p = torch.tensor([[[0.3], [0.6]]]); m = torch.ones(1, 2)
    pos = torch.tensor([[[1.0], [1.0]]]); neg = torch.tensor([[[0.0], [0.0]]])
    assert torch.isclose(M.focal_bce(p, pos, m, gamma=0.0, pos_weight=2.0), 2 * M.focal_bce(p, pos, m, gamma=0.0))
    assert torch.isclose(M.focal_bce(p, neg, m, gamma=0.0, pos_weight=2.0), M.focal_bce(p, neg, m, gamma=0.0))


def test_default_unchanged():
    p = torch.rand(2, 5, 3) * 0.98 + 0.01; t = (torch.rand(2, 5, 3) > 0.5).float(); m = torch.ones(2, 5)
    assert torch.equal(M.focal_bce(p, t, m), M.focal_bce(p, t, m, gamma=2.0, pos_weight=None))
