"""Identify synthetic clips at inference time.

Needed because the two populations want different treatment and the test set
ships audio only. EXP-27 measured the difference: on synthetic clips our
predicted spans are systematically ~0.075 s too NARROW with a std of only
0.069, a correctable bias on top of a precise estimate; correcting it is worth
+0.0906 combined on synthetic. On natural clips the bias is ~0 and the std is
0.275, so the same correction does nothing. One global correction therefore
leaves most of the gain on the table - it has to be routed.

Synthetic clips are clean studio speech with noise mixed in programmatically.
Natural clips are real field recordings. The two differ in ways that survive
without any reference: the noise floor between speech is far cleaner on a
synthetic clip, and the mixing leaves the speech itself unusually undistorted.

Features are deliberately cheap and interpretable rather than learned from
spectrograms - a small feature set with a linear model is much less likely to
latch onto a corpus artifact that will not hold on the test set.
"""
from __future__ import annotations

import numpy as np

SR = 16000


def clip_features(wav, sr=SR):
    """A handful of statistics that separate 'recorded in a room' from
    'clean speech plus an added signal'."""
    w = np.asarray(wav, dtype=np.float32)
    if w.size < sr // 4:
        w = np.pad(w, (0, sr // 4 - w.size))
    # frame energies at 20 ms
    n, h = 400, 320
    idx = np.arange(n)[None, :] + h * np.arange(1 + (len(w) - n) // h)[:, None]
    fr = w[np.clip(idx, 0, len(w) - 1)]
    e = np.sqrt((fr ** 2).mean(1) + 1e-12)
    le = np.log(e + 1e-9)
    q = np.percentile(le, [5, 25, 50, 75, 95])

    # the noise floor: how far below the median the quietest frames sit. A
    # studio recording's silences are much quieter relative to its speech.
    floor_gap = q[2] - q[0]
    dyn = q[4] - q[0]

    # spectral flatness of the quietest decile - room tone is broadband and
    # flat, a clean digital silence is not
    quiet = fr[np.argsort(e)[:max(1, len(e) // 10)]]
    sp = np.abs(np.fft.rfft(quiet * np.hanning(n), axis=1)) + 1e-9
    flat = float(np.mean(np.exp(np.log(sp).mean(1)) / sp.mean(1)))

    # high-frequency content of the quiet frames
    half = sp.shape[1] // 2
    hf = float(np.mean(sp[:, half:].mean(1) / (sp[:, :half].mean(1) + 1e-9)))

    # DC offset and clipping, both recording-chain artifacts
    dc = float(np.abs(w.mean()) / (w.std() + 1e-9))
    clip = float(np.mean(np.abs(w) > 0.98))
    return np.array([floor_gap, dyn, flat, hf, dc, clip,
                     float(le.std()), float(np.median(np.abs(np.diff(le))))],
                    dtype=np.float32)


class SynthClassifier:
    """Logistic regression over `clip_features`. Fit on validation, where the
    `syntheticData` flag gives us ground truth."""

    def __init__(self):
        self.clf = None
        self.mu = None
        self.sd = None

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, dtype=np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.clf = LogisticRegression(max_iter=2000, C=1.0).fit(
            (X - self.mu) / self.sd, np.asarray(y))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.clf.predict_proba((X - self.mu) / self.sd)[:, 1]

    def predict(self, X, thr=0.5):
        return self.predict_proba(X) >= thr
