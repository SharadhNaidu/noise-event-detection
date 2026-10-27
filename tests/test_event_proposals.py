import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from event_proposals import EventProposalNet, interval_targets, proposal_loss


def test_cross_category_nested_intervals_round_trip():
    target, weights, active, stats = interval_targets([(0.2, 1.8), (.8, 1.1)], ['vehicle_traffic', 'animal'], 100, 2.)
    times = (np.arange(100)+.5)*.02
    for ch, edges in [(2, (.2, 1.8)), (1, (.8, 1.1))]:
        keep = weights[:, ch]>0
        assert np.allclose(times[keep]-target[keep, ch, 0], edges[0])
        assert np.allclose(times[keep]+target[keep, ch, 1], edges[1])
        assert np.isclose(weights[:, ch].sum(), 1.)
    assert stats['complete_events'] == 3
    assert active[:, 1].sum() and active[:, 2].sum()


def test_same_category_ambiguity_and_censoring_are_masked():
    target, weights, active, stats = interval_targets([(-.5, 1.), (.5, 1.5)], ['animal', 'animal'], 100, 2.)
    times = (np.arange(100)+.5)*.02
    assert not weights[(times>=.5)&(times<1.), 1].any()
    assert not weights[:, 0].any()
    assert active[:, 1].sum() == 75
    assert stats['censored_events'] == 2
    assert stats['ambiguous_frames'] == 25


def test_padding_does_not_change_valid_recurrent_outputs():
    torch.manual_seed(13)
    net = EventProposalNet(acoustic_dim=512).eval()
    features = torch.randn(1, 11, 536)
    first, _ = net(features, torch.tensor([11]))
    padded = torch.cat([features, torch.randn(1, 9, 536)], 1)
    second, _ = net(padded, torch.tensor([11]))
    assert torch.allclose(first, second[:, :11], atol=1e-6)


def test_event_balancing_and_finite_gradients():
    target, weights, _, _ = interval_targets([(.1, .3), (.5, 1.5)], ['animal', 'animal'], 100, 2.)
    p = torch.tensor(target)[None].clone().requires_grad_()
    w = torch.tensor(weights)[None]
    truth = torch.tensor(target)[None]
    assert proposal_loss(p, truth, w).item() == 0.
    loss = proposal_loss(p+.05, truth, w, collar_weight=.1)
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert torch.allclose(w.sum(), torch.tensor(4.))


def test_zero_event_batch_has_zero_loss_and_gradients():
    pred = torch.ones(2, 10, 8, 2, requires_grad=True)
    loss = proposal_loss(pred, torch.zeros_like(pred), torch.zeros(2, 10, 8))
    loss.backward()
    assert loss.item() == 0.
    assert pred.grad.abs().sum().item() == 0.
