"""Synthetic-clip template snap: a noise recording inserted into a mixture is located by normalised
cross-correlation and its exact span replaces a nearby ML event's edges (never adds events)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tmplsnap as TS  # noqa: E402


def _mixture(rng, noise, start_s, dur_s=6.0, snr_db=0.0):
    sr = TS.SR_IN
    speech = rng.standard_normal(int(dur_s * sr)) * np.hanning(int(dur_s * sr))  # stand-in background
    a = int(start_s * sr)
    gain = np.sqrt(np.mean(speech ** 2) / np.mean(noise ** 2)) * 10 ** (snr_db / 20)
    x = speech.copy(); x[a:a + len(noise)] += gain * noise
    return x.astype(np.float32)


def test_match_recovers_exact_span():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(int(1.3 * TS.SR_IN)) * np.linspace(0.3, 1.0, int(1.3 * TS.SR_IN))
    bank = TS.Bank.from_events([(noise, 0.0, 1.3)])
    x = _mixture(rng, noise, 2.17)
    ev = TS.match_clip(x, bank)
    assert len(ev) == 1
    assert abs(ev[0][0] - 2.17) <= 0.005 and abs(ev[0][1] - 3.47) <= 0.005


def test_no_match_on_unrelated_audio():
    rng = np.random.default_rng(2)
    bank = TS.Bank.from_events([(rng.standard_normal(TS.SR_IN), 0.0, 1.0)])
    x = rng.standard_normal(6 * TS.SR_IN).astype(np.float32)
    assert TS.match_clip(x, bank) == []


def test_snap_only_moves_agreeing_events_and_never_adds():
    ml = [(1.00, 2.00), (3.00, 3.50)]
    tm = [(1.03, 1.96), (5.0, 6.0), (3.20, 3.55)]
    out = TS.snap(ml, tm, agree=0.06)
    assert out == [(1.03, 1.96), (3.00, 3.50)]       # first snapped; second disagrees on onset (0.2 s) -> kept
    assert len(out) == len(ml)


def test_snap_empty_templates_is_identity():
    ml = [(0.5, 0.9)]
    assert TS.snap(ml, [], agree=0.06) == ml
