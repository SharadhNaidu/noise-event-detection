import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from train_epn import ProposalDataset
from event_proposals import proposal_loss


def example(store, weight):
    data = ProposalDataset.__new__(ProposalDataset)
    data.ids = ['clip']
    data.items = {'clip': dict(_store=store, events=[(.1, .4)], cats=['human_non_speech'], n=8000,
                               feature=dict(off=0, n=25))}
    data.features = {store: np.zeros((25, 24), dtype=np.float16)}
    data.silver_loc_weight = weight
    return data[0]


def test_silver_zero_removes_precise_loss_gradient_and_preserves_inputs():
    original = example('silver_long', .5)
    excluded = example('silver_long', 0)
    assert torch.equal(original[0], excluded[0])
    assert torch.equal(original[1], excluded[1])
    assert original[2].sum() > 0
    assert excluded[2].sum() == 0
    prediction = torch.ones_like(excluded[1], requires_grad=True)
    loss = proposal_loss(prediction, excluded[1], excluded[2])
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(prediction.grad) == 0


def test_strong_targets_are_unchanged_by_silver_policy():
    original = example('gold_audio', .5)
    excluded = example('gold_audio', 0)
    assert all(torch.equal(a, b) for a, b in zip(original, excluded))
