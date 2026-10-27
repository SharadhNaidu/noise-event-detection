import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pytest
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from event_proposals import EventProposalNet, decode_proposals
from joint_interval import JointIntervalNet, presence_loss, refined_presence


def _features(batch=2, frames=20):
    features = torch.randn(batch, frames, 536, device='cpu')
    features[..., :8] = torch.rand(batch, frames, 8, device='cpu')*.8+.1
    return features


def test_imported_parent_state_preserves_distance_path_and_clamped_presence():
    torch.manual_seed(13)
    parent = EventProposalNet(acoustic_dim=512).eval()
    joint = JointIntervalNet().eval()
    state = joint.load_state_dict(parent.state_dict(), strict=False)
    assert state.missing_keys == ['presence_residual.weight', 'presence_residual.bias']
    assert state.unexpected_keys == []
    assert torch.count_nonzero(joint.presence_residual.weight) == 0
    assert torch.count_nonzero(joint.presence_residual.bias) == 0
    features = _features()
    features[0, 0, :2] = torch.tensor([0., 1.])
    lengths = torch.tensor([20, 11])
    with torch.no_grad():
        original_distances, _ = parent(features, lengths)
        distances, logits = joint(features, lengths)
    assert torch.equal(distances, original_distances)
    expected = features[..., :8].clamp(1e-6, 1.-1e-6)
    assert torch.allclose(refined_presence(logits), expected, atol=1e-7, rtol=1e-6)
    assert refined_presence(logits)[0, 0, 0] > features[0, 0, 0]
    assert refined_presence(logits)[0, 0, 1] < features[0, 0, 1]


@pytest.mark.parametrize('threshold', [.3, .5, .7])
def test_zero_residual_preserves_decoded_events_on_controlled_intervals(threshold):
    torch.manual_seed(13)
    joint = JointIntervalNet(hidden=4).eval()
    features = _features(batch=1, frames=20)
    features[..., :8] = .08
    features[:, 3:8, 1] = torch.tensor([.42, .58, .86, .62, .44])
    features[:, 11:16, 2] = torch.tensor([.76, .83, .89, .81, .75])
    with torch.no_grad():
        _, logits = joint(features, torch.tensor([20]))
    probability = features[0, :, :8].numpy()
    refined = refined_presence(logits)[0].numpy()
    distances = np.full((20, 8, 2), .01, dtype=np.float32)
    times = (np.arange(20)+.5)*.02
    for channel, first, last in [(1, 3, 8), (2, 11, 16)]:
        distances[first:last, channel, 0] = times[first:last]-.02*first
        distances[first:last, channel, 1] = .02*last-times[first:last]
    original = decode_proposals(probability, distances, .4, threshold, channels=(1, 2))
    candidate = decode_proposals(refined, distances, .4, threshold, channels=(1, 2))
    assert [(a, b, ch) for a, b, _, ch in candidate] == [(a, b, ch) for a, b, _, ch in original]
    if threshold < .7:
        assert len(original) == 2
    else:
        assert len(original) == 1
    assert np.allclose([c for _, _, c, _ in candidate], [c for _, _, c, _ in original], atol=1e-7)


def test_padding_does_not_change_valid_joint_outputs():
    torch.manual_seed(13)
    joint = JointIntervalNet(hidden=4).eval()
    with torch.no_grad():
        joint.presence_residual.weight.normal_(0., .1)
        joint.presence_residual.bias.normal_(0., .1)
    features = _features(batch=1, frames=11)
    padded = torch.cat([features, _features(batch=1, frames=9)], dim=1)
    with torch.no_grad():
        original_distances, original_logits = joint(features, torch.tensor([11]))
        padded_distances, padded_logits = joint(padded, torch.tensor([11]))
    assert torch.allclose(original_distances, padded_distances[:, :11], atol=1e-6)
    assert torch.allclose(original_logits, padded_logits[:, :11], atol=1e-6)


def test_presence_loss_weights_clips_equally_and_ignores_padding():
    logits = torch.tensor([[[1.]*8, [-2.]*8, [float('nan')]*8], [[.4]*8, [float('nan')]*8, [float('nan')]*8]], requires_grad=True)
    labels = torch.tensor([[[0.]*8, [1.]*8, [float('nan')]*8], [[1.]*8, [float('nan')]*8, [float('nan')]*8]])
    loss = presence_loss(logits, labels, torch.tensor([2, 1]))
    expected = .5*(F.binary_cross_entropy_with_logits(logits[0, :2], labels[0, :2])+F.binary_cross_entropy_with_logits(logits[1, :1], labels[1, :1]))
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].abs().sum() == 0
    assert logits.grad[1, 1:].abs().sum() == 0
    assert logits.grad[0, 0].sum() > 0
    assert logits.grad[0, 1].sum() < 0
    assert logits.grad[1, 0].sum() < 0


def test_joint_presence_receives_real_gradient_at_initialization():
    torch.manual_seed(13)
    joint = JointIntervalNet(hidden=4)
    features = _features(batch=2, frames=9)
    lengths = torch.tensor([9, 4])
    _, logits = joint(features, lengths)
    logits.retain_grad()
    presence_loss(logits, torch.zeros_like(logits), lengths).backward()
    assert joint.presence_residual.weight.grad.abs().sum() > 0
    assert joint.presence_residual.bias.grad.abs().sum() > 0
    assert torch.isfinite(joint.presence_residual.weight.grad).all()
    assert logits.grad[1, 4:].abs().sum() == 0
    assert logits.grad[1, :4].abs().sum() > 0


@pytest.mark.parametrize('lengths', [torch.tensor([0]), torch.tensor([4]), torch.tensor([2.]), torch.tensor([[2]]), torch.tensor([2, 2])])
def test_invalid_lengths_are_rejected(lengths):
    joint = JointIntervalNet(hidden=4)
    with pytest.raises(ValueError):
        joint(_features(batch=1, frames=3), lengths)
    with pytest.raises(ValueError):
        presence_loss(torch.zeros(1, 3, 8), torch.zeros(1, 3, 8), lengths)


def test_invalid_feature_or_presence_shapes_are_rejected():
    joint = JointIntervalNet(hidden=4)
    with pytest.raises(ValueError):
        joint(torch.zeros(1, 3, 24), torch.tensor([3]))
    with pytest.raises(ValueError):
        presence_loss(torch.zeros(1, 3, 8), torch.zeros(1, 3, 7), torch.tensor([3]))
