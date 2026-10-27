"""v11 Arm L: unbiased 'centre' rasterisation and tau-unit boundary loss.

centre: a frame is positive iff its CENTRE lies inside the event, i.e. frames
[round(on/fs), round(off/fs)). That matches the RED impulses (round) and the
decoder's time = frame_index * fs, so a perfectly learned target decodes back to
the reference within half a frame instead of ~-10 / +10 ms (floor/ceil 'touch').
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset as DS  # noqa: E402
import model as M  # noqa: E402


def _span(y):
    on = np.flatnonzero(y[:, 0])
    return on[0], on[-1] + 1


def test_touch_is_default_and_unchanged():
    y = DS.rasterise([(0.105, 0.395)], ["animal"], 50)
    assert _span(y) == (5, 20)          # floor(5.25)=5, ceil(19.75)=20


def test_centre_rounds_both_edges():
    y = DS.rasterise([(0.105, 0.395)], ["animal"], 50, mode="centre")
    assert _span(y) == (5, 20)          # round(5.25)=5, round(19.75)=20
    y = DS.rasterise([(0.115, 0.385)], ["animal"], 50, mode="centre")
    assert _span(y) == (6, 19)          # round(5.75)=6, round(19.25)=19 (touch gives 5, 20)
    assert y[6:19, 1 + DS.CAT2IDX["animal"]].all()


def test_centre_decodes_unbiased():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(500):
        on = rng.uniform(0.2, 2.0); off = on + rng.uniform(0.1, 1.0)
        a, b = _span(DS.rasterise([(on, off)], [None], 200, mode="centre"))
        errs += [a * DS.FRAME_SEC - on, b * DS.FRAME_SEC - off]
    assert abs(np.mean(errs)) < 0.002 and np.max(np.abs(errs)) <= 0.0101


def test_centre_keeps_at_least_one_frame():
    y = DS.rasterise([(0.101, 0.108)], [None], 50, mode="centre")
    assert y[:, 0].sum() == 1


def test_matches_red_impulse_frames():
    evs = [(0.115, 0.385)]
    a, b = _span(DS.rasterise(evs, [None], 50, mode="centre"))
    ion, ioff = DS.impulse_targets(evs, 50)
    assert np.argmax(ion) == a and np.argmax(ioff) == b


def test_tau_valid_carries_event_tolerance():
    d, v = DS.boundary_targets([(1.0, 1.1), (2.0, 4.0)], 250, tau_valid=True)
    t = (np.arange(250) + 0.5) * DS.FRAME_SEC
    near_short = np.abs(t - 1.0) < 0.04
    near_long = np.abs(t - 2.0) < 0.04
    assert np.allclose(v[near_short, 0], 0.05)          # 0.2*0.1 -> floor 0.05
    assert np.allclose(v[near_long, 0], 0.4)            # 0.2*2.0
    d0, v0 = DS.boundary_targets([(1.0, 1.1), (2.0, 4.0)], 250)
    assert np.array_equal(d, d0) and np.array_equal(v > 0, v0 > 0)


def test_tau_unit_loss_scales_by_event_tolerance():
    pred = torch.full((1, 4, 2), 0.1); tgt = torch.zeros(1, 4, 2); m = torch.ones(1, 4)
    v_short = torch.full((1, 4, 2), 0.05); v_long = torch.full((1, 4, 2), 0.4)
    ls = M.boundary_loss(pred, tgt, v_short, m, tau_unit=True)
    ll = M.boundary_loss(pred, tgt, v_long, m, tau_unit=True)
    assert ls > 4 * ll                                   # same 100 ms error costs far more on a short event
    lf = M.boundary_loss(pred, tgt, (v_short > 0).float(), m)   # legacy fixed 50 ms unit
    assert torch.isclose(ls, lf)                         # identical at the 50 ms floor


def test_tau_unit_ignores_invalid_frames():
    pred = torch.randn(2, 6, 2); tgt = torch.zeros(2, 6, 2); m = torch.ones(2, 6)
    v = torch.zeros(2, 6, 2)
    assert M.boundary_loss(pred, tgt, v, m, tau_unit=True) == 0
