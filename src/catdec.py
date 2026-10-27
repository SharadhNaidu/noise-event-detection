"""Shared per-category decoder: select and evaluate category tracks for an ensemble.

    python src/catdec.py --members v5_S [--readout-json ens.json] [--out catdec_v5_S.json]

Evaluation and submission use the SAME path: members' all-channel posteriors come from
submit_ens.member_posteriors (cached per store/tag), the base track is channel 0 with the
ensemble readout, and category tracks are added by postproc.category_union - exactly what
submit_ens applies when the readout carries a "categories" key.

Why: perfect per-category tracks score natural F1 0.996 vs 0.894 for a perfect union track
(rank1_gap_probe, 2026-10-01); per-class RED (--red-classes) trains those tracks, but a
channel-0 decode throws them away.

Selection: greedy forward over channels 1..7 x thresholds, natural clips only, fit on one
speaker-disjoint DECIDE half and judged on the other (both directions), then refit on all
DECIDE for the shipped config. TUNE is training data for these models, so it is not used
for category selection.
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
import boot_ens as B      # noqa: E402
import data as D          # noqa: E402
import ens_gate as E      # noqa: E402
import finetune as FT     # noqa: E402
import postproc as PP     # noqa: E402
import splits as SP       # noqa: E402
import submit_ens as SE   # noqa: E402
import train as T         # noqa: E402

THRS = (0.5, 0.6, 0.7, 0.8)


def greedy(base, post8, refs, fit_ids, natural, dur):
    """Forward selection of {channel: thr}; a channel is kept only if it raises the
    combined score on fit_ids. Events are clipped to the real duration (short clips
    are zero-padded to the crop, so a category track can run past the end)."""
    sc = lambda cfg: B.resample_score(refs, PP.clip_to_duration(
        SE.apply_categories(base, post8, fit_ids, cfg, natural), dur), fit_ids)["combined"]
    cfg, best = {}, sc({})
    improved = True
    while improved:
        improved = False
        for ch in range(1, 8):
            if ch in cfg:
                continue
            for t in THRS:
                s = sc({**cfg, ch: t})
                if s > best + 1e-4:
                    best, pick, improved = s, (ch, t), True
        if improved:
            cfg[pick[0]] = pick[1]
    return cfg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--readout-json", default=None, help="base readout (default: choose_readout on TUNE)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    refs, meta = D.load_validation()
    syn = {c: str(meta[c]["syntheticData"]) == "True" for c in refs}
    tune, dec = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    arr, me = A.load_audio_store(FT.AUD, "val_audio")
    idx = me["clips"]
    tune = [c for c in tune if c in idx]
    dec = [c for c in dec if c in idx]
    ids = tune + dec
    dur = {c: idx[c]["n"] / A.SR for c in ids}
    post8, dist = SE.ensemble_posteriors(a.members, "val_audio", ids, dev)
    post = {c: post8[c][:, 0] for c in ids}
    if a.readout_json:
        rc = dict(json.load(open(a.readout_json))["readout"])
    else:
        rc = T.choose_readout(post, refs, tune, dist)
        rc = dict(rc[0] if isinstance(rc, tuple) else rc)
    rc.setdefault("kind", "global")
    for k in ("overlay", "categories"):
        rc.pop(k, None)
    natural = {c: not syn[c] for c in ids}
    base = PP.clip_to_duration(T.decode_all(post, rc, dist, dec), dur)
    dA, dB = SP.split_by_speaker(dec, frac=0.5, seed=7)
    nat = [c for c in dec if natural[c]]
    print(f"members {a.members} | base readout {rc}", flush=True)
    for fit, jud, nm in ((dA, dB, "A->B"), (dB, dA, "B->A")):
        cfg, _ = greedy(base, post8, refs, fit, natural, dur)
        b0 = B.resample_score(refs, base, jud)["combined"]
        b1 = B.resample_score(refs, PP.clip_to_duration(SE.apply_categories(base, post8, jud, cfg, natural), dur), jud)["combined"]
        print(f"{nm}: categories {cfg} | judged channel-0 {b0:.4f} -> category decoder {b1:.4f} ({b1 - b0:+.4f})", flush=True)
    cfg, _ = greedy(base, post8, refs, dec, natural, dur)
    full = PP.clip_to_duration(SE.apply_categories(base, post8, dec, cfg, natural), dur)
    for name, pr in (("channel-0", base), ("category decoder", full)):
        r, rn = B.resample_score(refs, pr, dec), B.resample_score(refs, pr, nat)
        print(f"{name:17s} DECIDE {r['combined']:.4f} | natural F1 {rn['event_f1']:.4f} Dice {rn['segment_dice']:.4f}", flush=True)
    out = a.out or str(FT.CKPT / f"catdec_{'+'.join(a.members)}.json")
    json.dump({"members": a.members, "readout": {**rc, "categories": {str(k): v for k, v in cfg.items()}}},
              open(out, "w"), indent=1)
    print(f"saved {out} (categories chosen on all DECIDE: {cfg})", flush=True)


if __name__ == "__main__":
    main()
