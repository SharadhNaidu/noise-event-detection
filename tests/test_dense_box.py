"""Dense interval head (--box-kind dense): targets, loss and decoder.

Every frame inside an event regresses its distances to that event's onset and offset (ActionFormer-style),
in 3 lanes (0 = any event, 1 = human_non_speech, 2 = every other category) so a breath nested inside a
vehicle span keeps its own instance. Weights are per EVENT (1 / its frames): short events count fully.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dataset as DS        # noqa: E402
import model as M           # noqa: E402
import densedec as DD       # noqa: E402

FS = 0.02


def test_targets_single_event_distances_and_weights():
    tgt, msk = DS.dense_targets([(1.0, 2.0)], ["vehicle_traffic"], 250)
    assert tgt.shape == (250, 6) and msk.shape == (250, 6)
    t = 75                                     # frame centre 1.51 s
    assert np.allclose(tgt[t, 0:2], [0.51, 0.49], atol=1e-6)      # lane 0 (any)
    assert np.allclose(tgt[t, 4:6], [0.51, 0.49], atol=1e-6)      # lane 2 (non-HNS)
    assert msk[t, 2] == 0 and msk[t, 3] == 0                      # lane 1 (HNS) empty
    assert np.isclose(msk[:, 0].sum(), 1.0) and np.isclose(msk[:, 4].sum(), 1.0)   # one event = weight 1
    assert msk[49, 0] == 0 and msk[100, 0] == 0                   # outside the event (0.99 s, 2.01 s)


def test_targets_nested_breath_gets_its_own_lane():
    evs = [(0.5, 3.5), (1.0, 1.4)]
    tgt, msk = DS.dense_targets(evs, ["vehicle_traffic", "human_non_speech"], 250)
    t = 60                                     # 1.21 s: inside both
    assert np.allclose(tgt[t, 0:2], [0.71, 2.29], atol=1e-6)      # lane 0 = merged outer span
    assert np.allclose(tgt[t, 2:4], [0.21, 0.19], atol=1e-6)      # lane 1 = the breath
    assert np.allclose(tgt[t, 4:6], [0.71, 2.29], atol=1e-6)      # lane 2 = the vehicle
    assert np.isclose(msk[:, 2].sum(), 1.0)                       # short breath weighs as much as the long vehicle


def test_targets_censored_event_is_skipped():
    tgt, msk = DS.dense_targets([(-0.3, 0.6), (4.8, 5.4)], ["animal", "animal"], 250)   # crop is 5.0 s
    assert msk.sum() == 0


def test_dense_loss_zero_for_perfect_and_larger_for_worse():
    tgt, msk = DS.dense_targets([(1.0, 2.0), (1.2, 1.5)], ["animal", "human_non_speech"], 250)
    tg = torch.from_numpy(tgt)[None]; mk = torch.from_numpy(msk)[None]; m = torch.ones(1, 250)
    raw = torch.log(torch.expm1(tg.clamp(min=1e-4)))              # softplus^-1
    l0 = M.dense_box_loss(raw, tg, mk, m)
    worse = (raw + 0.5).requires_grad_(True)
    l1 = M.dense_box_loss(worse, tg, mk, m)
    assert float(l0) < 1e-3 < float(l1)
    l1.backward()
    assert worse.grad is not None and float(worse.grad.abs().sum()) > 0


def test_decoder_recovers_nested_events_from_perfect_outputs():
    evs = [(0.5, 3.5), (1.0, 1.4)]
    cats = ["vehicle_traffic", "human_non_speech"]
    tgt, _ = DS.dense_targets(evs, cats, 250)
    lanes = np.zeros((250, 3), np.float32)
    times = (np.arange(250) + 0.5) * FS
    lanes[(times >= 0.5) & (times < 3.5), 0] = 0.9
    lanes[(times >= 1.0) & (times < 1.4), 1] = 0.9
    lanes[(times >= 0.5) & (times < 3.5), 2] = 0.9
    out = DD.decode(lanes, tgt, duration=5.0, thr=(0.5, 0.5, 0.5))
    assert len(out) == 2, out
    for (a, b), (ra, rb) in zip(sorted(out), sorted(evs)):
        assert abs(a - ra) < 0.021 and abs(b - rb) < 0.021
