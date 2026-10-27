"""--bd-union: the ch0 boundary (distance) head is trained on MERGED events, like RED ch0, so a breath nested inside a
longer event no longer creates an edge target inside the outer event."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset as DS  # noqa: E402


def test_nested_breath_edge_not_a_target_when_merged():
    outer, inner = (1.0, 4.0), (2.0, 2.4)
    d_raw, v_raw = DS.boundary_targets([outer, inner], 250)
    d_u, v_u = DS.boundary_targets(DS._merge([outer, inner]), 250)
    t = (np.arange(250) + 0.5) * DS.FRAME_SEC
    near_inner_on = np.argmin(abs(t - 2.0))
    assert abs(d_raw[near_inner_on, 0]) < 0.02           # raw: frame next to the breath onset sees an edge right there
    assert abs(d_u[near_inner_on, 0]) > 0.5 - 1e-6 or v_u[near_inner_on, 0] == 0   # merged: nearest onset is the outer one (1 s away)


def test_merge_keeps_disjoint_events():
    evs = [(0.5, 1.0), (2.0, 2.5)]
    assert DS._merge(evs) == evs
