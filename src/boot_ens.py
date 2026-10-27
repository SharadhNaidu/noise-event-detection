"""Paired bootstrap: ensemble vs a single baseline on DECIDE (clip resampling).

    python src/boot_ens.py --base hon_A --members hon_A bts_C --readout-json <json from ens_gate>

Posteriors are computed once per checkpoint; the baseline is decoded with its
own stored readout, the ensemble with the readout ens_gate chose on TUNE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import ens_gate as E      # noqa: E402
import finetune as FT     # noqa: E402
import postproc as PP     # noqa: E402
import scorer as S        # noqa: E402
import splits as SP       # noqa: E402
import train as T         # noqa: E402


def resample_score(refs, preds, ids):
    """Score a (with-replacement) resample: every draw gets its own key, so a
    clip drawn twice counts twice. A dict keyed by clip id would silently
    collapse duplicates and turn the bootstrap into a ~63% subsample."""
    keys = [f"{c}#{k}" for k, c in enumerate(ids)]
    return S.score({k: refs[c] for k, c in zip(keys, ids)}, {k: preds[c] for k, c in zip(keys, ids)})


def speaker_groups(ids):
    """{speaker_key: [clip ids]} - the resampling unit for group_bootstrap."""
    g = {}
    for c in ids:
        g.setdefault(SP.speaker_key(c), []).append(c)
    return g


def group_bootstrap(groups, stat_a, stat_b, n=500, seed=0):
    """Paired bootstrap of stat_a(ids) - stat_b(ids) resampling whole SPEAKER groups with
    replacement (a group drawn k times contributes its clips k times). stat_* take a clip-id
    list that may contain duplicates (use resample_score). Returns mean, 95% CI, P(>0)."""
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        ids = [c for k in rng.choice(keys, len(keys), replace=True) for c in groups[k]]
        ds.append(stat_a(ids) - stat_b(ids))
    ds = np.asarray(ds, dtype=float)
    return {"mean": float(ds.mean()), "lo": float(np.percentile(ds, 2.5)),
            "hi": float(np.percentile(ds, 97.5)), "p": float(np.mean(ds > 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--readout-json", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    saved = json.load(open(a.readout_json))
    assert sorted(saved["members"]) == sorted(a.members), "readout tuned for other members"
    refs, _ = D.load_validation()
    _, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    arr, am = A.load_audio_store(FT.AUD, "val_audio")
    idx = am["clips"]
    decide = [c for c in decide if c in idx]
    cache = {}
    for tag in sorted(set(a.members + [a.base])):
        post, dist, blob = E.posteriors(tag, decide, dev, arr, idx)
        cache[tag] = (post, dist, blob)
    bp, bd, bb = cache[a.base]
    rc = dict(bb["readout"]); rc.setdefault("kind", "global")
    base = PP.clip_to_duration(T.decode_all(bp, rc, bd, decide), {c: idx[c]['n'] / A.SR for c in decide})
    ep = E.average([cache[t][0] for t in a.members])
    ed = E.average([cache[t][1] for t in a.members])
    rc2 = dict(saved["readout"]); rc2.setdefault("kind", "global")
    ens = PP.clip_to_duration(T.decode_all(ep, rc2, ed, decide), {c: idx[c]['n'] / A.SR for c in decide})

    def comb(ids, pr):
        return resample_score(refs, pr, ids)["combined"]

    d0 = comb(decide, ens) - comb(decide, base)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(a.n):
        ids = list(rng.choice(decide, len(decide), replace=True))
        ds.append(comb(ids, ens) - comb(ids, base))
    ds = np.array(ds)
    print(f"ensemble({'+'.join(a.members)}) - {a.base}: {d0:+.4f}  bootstrap 95% CI "
          f"[{np.percentile(ds, 2.5):+.4f}, {np.percentile(ds, 97.5):+.4f}]  P(>0)={np.mean(ds > 0):.3f}  "
          f"(ens {comb(decide, ens):.4f} vs base {comb(decide, base):.4f})", flush=True)


if __name__ == "__main__":
    main()
