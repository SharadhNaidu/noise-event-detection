"""box_loss positives = the ONE exact peak frame per edge (professor 2026-10-02). With `ht > 0.999` a long
event's wide Gaussian marks 3 (5 s) / 5 (10 s) / 13 (30 s) frames as exact edges, so a prediction that
peaks exactly at the annotated onset is penalised at the neighbouring 'positives'. Fails if the positive
rule counts shoulder frames."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as DS  # noqa: E402
import model as M  # noqa: E402


def _loss_for_exact_peaks(ev, T):
    tgt, msk = DS.box_targets(ev, T)
    pred = np.zeros((T, 6), np.float32)
    exact = np.isclose(tgt[:, :2], 1.0, atol=0.0, rtol=0.0)
    pred[:, :2] = np.where(exact, 12.0, -12.0)      # confident single peak at each true edge
    pred[:, 2:] = tgt[:, 2:]                         # perfect regression
    t = lambda x: torch.tensor(x)[None]
    return float(M.box_loss(t(pred), t(tgt), t(msk), torch.ones(1, T)))


def test_single_positive_per_edge_for_long_events():
    for ev in ([(2.0, 32.0)], [(1.0, 11.0)], [(0.5, 5.5)]):
        assert _loss_for_exact_peaks(ev, 1800) < 0.05, ev


def test_short_events_unchanged():
    assert _loss_for_exact_peaks([(1.0, 1.4)], 300) < 0.05


if __name__ == "__main__":
    bad = 0
    for f in (test_single_positive_per_edge_for_long_events, test_short_events_unchanged):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
