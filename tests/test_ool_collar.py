"""--ool-collar: RED's onset/offset loss rewards an edge anywhere inside the metric's own
collar tau = max(0.2 * duration, 0.05 s), instead of only at the exact annotated frame.

Root cause (docs/controller/ROOT_CAUSE_NATURAL_2026_10_02.md): natural edge spread is the
lever for BOTH F1 and Dice; the exact-frame OOL charges a 1-frame miss on a 3 s event like
one on a 0.3 s event, while the scorer forgives 600 ms on the first and 60 ms on the second.

Production changes that make these fail:
- windows not sized by the event's own tau (fixed radius, or seconds/frames mixed up);
- overlapping same-kind windows sharing frames (one peak would satisfy two edges - the
  flaw a reviewer measured in collar_loss);
- edges outside the crop producing positive or negative supervision (left/right-censored);
- the loss not reducing to the existing focal OOL when every window is a single frame.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as DS  # noqa: E402
import model as M  # noqa: E402

FS = DS.FRAME_SEC


def _ids(ev, cats=None, T=250):
    cats = cats or ["animal"] * len(ev)
    return DS.collar_window_ids_cls(ev, cats, T)


def test_window_radius_is_the_events_own_tolerance():
    on, off = _ids([(1.0, 2.0)])                      # tau 0.2 s = 10 frames
    assert on.shape == (250, DS.N_OUT) and off.shape == (250, DS.N_OUT)
    assert np.flatnonzero(on[:, 0] == 1).tolist() == list(range(40, 61))
    assert np.flatnonzero(off[:, 0] == 1).tolist() == list(range(90, 111))
    on, off = _ids([(1.0, 1.1)])                      # floor tau 0.05 s = 2 frames
    assert np.flatnonzero(on[:, 0] == 1).tolist() == list(range(48, 53))


def test_category_channel_gets_its_own_windows():
    on, _ = _ids([(1.0, 2.0)], ["vehicle_traffic"])
    v = 1 + DS.CAT2IDX["vehicle_traffic"]
    assert np.flatnonzero(on[:, v] == 1).tolist() == list(range(40, 61))
    others = [c for c in range(1, DS.N_OUT) if c != v]
    assert not on[:, others].any()


def test_overlapping_same_kind_windows_are_split_disjoint():
    # onsets at frame 50 (tau 0.05 -> r 2) and frame 58 (event 1.16-3.0, tau 0.368 -> r 18)
    on, _ = _ids([(1.0, 1.1), (1.16, 3.0)], ["animal", "vehicle_traffic"])
    ch0 = on[:, 0]
    assert set(np.unique(ch0[40:77])) == {1, 2}
    for t in range(40, 77):                            # each frame belongs to its NEAREST covering edge
        cover = [k for k, (c, r) in enumerate([(50, 2), (58, 18)], 1) if abs(t - c) <= r]
        best = min(cover, key=lambda k: (abs(t - (50, 58)[k - 1]), k))
        assert ch0[t] == best, t


def test_edges_outside_the_crop_are_ignored_not_negative():
    on, off = _ids([(-0.1, 2.0)])                      # onset frame -5, tau 0.42 -> r 21: frames 0..16 inside
    assert (on[:17, 0] == -1).all() and (on[17:, 0] == 0).all()
    assert np.flatnonzero(off[:, 0] == 1).tolist() == list(range(79, 122))
    on, off = _ids([(4.0, 5.05)])                      # offset frame 252 >= T=250, r 10: frames 242..249
    assert (off[242:, 0] == -1).all() and not (off[:242, 0] == -1).any() and not (off[:, 0] == 1).any()
    assert np.flatnonzero(on[:, 0] == 1).tolist() == list(range(190, 211))


def _focal_reference(p, ids, mask):
    imp = (ids > 0).float()
    return M.focal_bce(p, imp, mask)


def test_single_frame_windows_reduce_to_focal_ool():
    torch.manual_seed(0)
    p = torch.rand(2, 30, 3) * 0.9 + 0.05
    ids = torch.zeros(2, 30, 3)
    ids[0, 5, 0] = 1; ids[0, 20, 0] = 2; ids[1, 7, 2] = 1
    mask = torch.ones(2, 30, 3)
    got = M.collar_ool_loss(p, ids, mask)
    assert torch.allclose(got, _focal_reference(p, ids, mask), atol=1e-6)


def test_position_inside_the_window_does_not_matter_outside_does():
    T = 40
    ids = torch.zeros(1, T, 1); ids[0, 15:26, 0] = 1  # one window, frames 15..25
    mask = torch.ones(1, T, 1)

    def at(t):
        p = torch.full((1, T, 1), 0.01); p[0, t, 0] = 0.9
        return float(M.collar_ool_loss(p, ids, mask))
    assert abs(at(15) - at(20)) < 1e-6 and abs(at(25) - at(20)) < 1e-6
    assert at(30) > at(20) + 0.01


def test_ignored_frames_contribute_nothing_and_one_peak_cannot_serve_two_windows():
    T = 40
    ids = torch.zeros(1, T, 1); ids[0, :5, 0] = -1
    mask = torch.ones(1, T, 1)
    p1 = torch.full((1, T, 1), 0.01); p2 = p1.clone(); p2[0, :5, 0] = 0.99
    assert abs(float(M.collar_ool_loss(p1, ids, mask)) - float(M.collar_ool_loss(p2, ids, mask))) < 1e-6
    ids2 = torch.zeros(1, T, 1); ids2[0, 10:15, 0] = 1; ids2[0, 15:20, 0] = 2
    p = torch.full((1, T, 1), 0.01); p[0, 14, 0] = 0.99
    one = float(M.collar_ool_loss(p, ids2, mask))
    p[0, 16, 0] = 0.99
    assert float(M.collar_ool_loss(p, ids2, mask)) < one - 0.01   # the second window needs its own peak


def test_gradient_flows_and_respects_mask():
    p = (torch.rand(1, 20, 1) * 0.8 + 0.1).requires_grad_(True)
    ids = torch.zeros(1, 20, 1); ids[0, 8:12, 0] = 1
    mask = torch.ones(1, 20, 1); mask[0, 15:, 0] = 0
    M.collar_ool_loss(p, ids, mask).backward()
    g = p.grad[0, :, 0]
    assert torch.isfinite(g).all() and (g[15:] == 0).all() and (g[8:12] != 0).any()


def test_flag_requires_red_and_crop_dataset_emits_collar_ids():
    import types
    import finetune as FT
    import pytest
    with pytest.raises(SystemExit):
        FT.validate_objective_flags(types.SimpleNamespace(ool_collar=True, red=False, red_classes=False))
    FT.validate_objective_flags(types.SimpleNamespace(ool_collar=True, red=False, red_classes=True))
    # the crop path must hand collar ids (not impulses) to the loss
    on, off = DS.collar_window_ids_cls([(1.0, 2.0)], ["animal"], 250)
    imp_on, _ = DS.impulse_targets_cls([(1.0, 2.0)], ["animal"], 250)
    assert (on[:, 0] == 1).sum() == 21 and imp_on[:, 0].sum() == 1
