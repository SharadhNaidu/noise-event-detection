"""Per-class RED (--red-classes): one latched start/end track per channel, so overlapping
cross-category events are representable (natural DECIDE: 22.6% of refs merged, 73% of those
pairs overlap across categories - a single track cannot emit both).

Production changes that make these fail:
- channel-0 onset/offset impulses placed at NESTED event edges (inside an already-present
  region the latch cannot start, so the target contradicts the recurrence);
- a category channel getting edges from another category's events;
- the RED head not widening to 8 channels, or old single-channel RED checkpoints changing shape;
- the RED loss supervising only channel 0 when per-class RED is on.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as DS  # noqa: E402
import model as M  # noqa: E402

FS = DS.FRAME_SEC


def _f(t):
    return int(round(t / FS))


def test_class_impulses_union_on_ch0_and_own_edges_per_class():
    ev = [(1.0, 3.0), (1.5, 1.9), (2.5, 4.0)]
    cats = ["animal", "human_non_speech", "vehicle_traffic"]
    on, off = DS.impulse_targets_cls(ev, cats, 250)
    assert on.shape == (250, DS.N_OUT) and off.shape == (250, DS.N_OUT)
    assert np.flatnonzero(on[:, 0]).tolist() == [_f(1.0)]            # union (1.0, 4.0): nested edges excluded
    assert np.flatnonzero(off[:, 0]).tolist() == [_f(4.0)]
    a, h, v = 1 + DS.CAT2IDX["animal"], 1 + DS.CAT2IDX["human_non_speech"], 1 + DS.CAT2IDX["vehicle_traffic"]
    assert np.flatnonzero(on[:, a]).tolist() == [_f(1.0)] and np.flatnonzero(off[:, a]).tolist() == [_f(3.0)]
    assert np.flatnonzero(on[:, h]).tolist() == [_f(1.5)] and np.flatnonzero(off[:, h]).tolist() == [_f(1.9)]
    assert np.flatnonzero(on[:, v]).tolist() == [_f(2.5)] and np.flatnonzero(off[:, v]).tolist() == [_f(4.0)]
    other = [c for c in range(1, DS.N_OUT) if c not in (a, h, v)]
    assert on[:, other].sum() == 0 and off[:, other].sum() == 0


def test_same_class_overlaps_merge_within_channel():
    on, off = DS.impulse_targets_cls([(1.0, 2.0), (1.8, 3.0)], ["animal", "animal"], 250)
    a = 1 + DS.CAT2IDX["animal"]
    assert np.flatnonzero(on[:, a]).tolist() == [_f(1.0)] and np.flatnonzero(off[:, a]).tolist() == [_f(3.0)]


def test_red_head_width():
    h8 = M.SEDHead(in_dim=16, rnn_dim=8, red_head=True, red_ch=8, boundary_head=True)
    h1 = M.SEDHead(in_dim=16, rnn_dim=8, red_head=True, boundary_head=True)
    x = torch.randn(2, 30, 16)
    s8, e8 = h8(x)[3]
    s1, e1 = h1(x)[3]
    assert s8.shape == (2, 30, 8) and e8.shape == (2, 30, 8)
    assert s1.shape == (2, 30, 1) and e1.shape == (2, 30, 1)
    assert h1.red.weight.shape[0] == 2                                  # old checkpoints keep their shape


def test_censored_first_frame_onset_not_penalised():
    """A crop starting inside an event: presence target 1 at frame 0, onset target 0.
    The latch starts at 0, so onset@0 must be free to fire (external audit 20:27)."""
    import torch
    y = torch.zeros(2, 6, 8); ion = torch.zeros(2, 6, 8)
    y[0, :, 0] = 1                       # clip 0: event in progress at crop start (censored)
    y[1, 2:, 0] = 1; ion[1, 2, 0] = 1    # clip 1: genuine interior onset
    m = torch.ones(2, 6)
    mask = M.onset_censor_mask(y, ion, m)
    assert mask.shape == (2, 6, 8)
    assert mask[0, 0, 0] == 0 and mask[0, 1, 0] == 1           # only frame 0 of the censored channel
    assert mask[1, 0, 0] == 1 and mask[0, 0, 1] == 1            # uncensored channels/clips untouched
    p_on = torch.full((2, 6, 8), 0.5, requires_grad=True)
    M.focal_bce(p_on, ion, mask).backward()
    assert p_on.grad[0, 0, 0] == 0                              # no push against starting the latch
    assert p_on.grad[1, 2, 0] < 0 and p_on.grad[1, 3, 0] > 0    # real onset pulled up, negatives down


def test_scan_equals_serial_recurrence():
    """red_presence (parallel affine scan) == the serial recurrence it replaced, outputs and grads."""
    import torch
    torch.manual_seed(0)
    for B, T, C in ((2, 249, 8), (1, 1, 8), (3, 517, 1)):
        s_ = (torch.randn(B, T, C) * 3).requires_grad_(); e_ = (torch.randn(B, T, C) * 3).requires_grad_()
        new, old = M.red_presence(s_, e_), M.red_presence_serial(s_, e_)
        for x, y_ in zip(new, old):
            assert (x - y_).abs().max() < 1e-5
        g1 = torch.autograd.grad(sum(x.sum() for x in new), (s_, e_))
        g2 = torch.autograd.grad(sum(x.sum() for x in old), (s_, e_))
        assert max((a - b).abs().max() for a, b in zip(g1, g2)) < 1e-4


def test_focal_sample_weight():
    """--red-tier-w: per-clip weights reach the RED focal losses (a 0-weight clip contributes nothing)."""
    import torch
    p = torch.full((2, 4, 1), 0.5, requires_grad=True); t = torch.zeros(2, 4, 1); m = torch.ones(2, 4)
    M.focal_bce(p, t, m, sample_weight=torch.tensor([1.0, 0.0])).backward()
    assert p.grad[1].abs().sum() == 0 and p.grad[0].abs().sum() > 0


if __name__ == "__main__":
    bad = 0
    for f in (test_class_impulses_union_on_ch0_and_own_edges_per_class, test_same_class_overlaps_merge_within_channel,
              test_red_head_width, test_censored_first_frame_onset_not_penalised,
              test_scan_equals_serial_recurrence, test_focal_sample_weight):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
