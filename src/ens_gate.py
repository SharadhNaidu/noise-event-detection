"""Posterior ensemble of checkpoints (possibly with different encoders), scored honestly.

    python src/ens_gate.py hon_A bts_S [...]

Each member runs with its own second stream (gate.load decides aux / no aux).
Posteriors and boundary distances are averaged (truncated to the common length
per clip); ONE readout is chosen for the ensemble on the TUNE half with
train.choose_readout, and the DECIDE half is scored once with it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import finetune as FT     # noqa: E402
import gate as G          # noqa: E402
import postproc as PP     # noqa: E402
import scorer as S        # noqa: E402
import splits as SP       # noqa: E402
import train as T         # noqa: E402
import astfeat as AF      # noqa: E402


def average(posts):
    """Mean over members, per clip, truncated to the shortest member."""
    out = {}
    for c in posts[0]:
        arrs = [np.asarray(p[c]) for p in posts if c in p]
        n = min(len(a) for a in arrs)
        out[c] = np.mean([a[:n] for a in arrs], axis=0)
    return out


def save_readout(path, members, readout):
    """The ensemble's TUNE-chosen readout, consumed verbatim by submit_ens.py."""
    import json
    json.dump({"members": list(members), "readout": readout}, open(path, "w"), indent=1)


def posteriors(tag, ids, dev, arr, idx):
    net, blob = G.load(tag, dev)
    net.eval()
    v_ast = G.aux_for(blob["cfg"])
    post, dist, _ = FT.clip_posteriors(net, arr, idx, ids, blob["cfg"]["crop"], dev, ast=v_ast)
    del net
    torch.cuda.empty_cache()
    return post, dist, blob


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    refs, meta = D.load_validation()
    tune, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    arr, am = A.load_audio_store(FT.AUD, "val_audio")
    idx = am["clips"]
    tune = [c for c in tune if c in idx]
    decide = [c for c in decide if c in idx]
    ids = tune + decide
    posts, dists = [], []
    for tag in sys.argv[1:]:
        p, d, blob = posteriors(tag, ids, dev, arr, idx)
        rc = dict(blob["readout"])
        rc.setdefault("kind", "global")
        r = S.score({c: refs[c] for c in decide}, PP.clip_to_duration(T.decode_all(p, rc, d, decide), {c: idx[c]['n'] / A.SR for c in decide}))
        print(f"member {tag:8s} own-readout DECIDE {r['combined']:.4f}", flush=True)
        posts.append(p)
        dists.append(d)
    post, dist = average(posts), average(dists)
    cfg = T.choose_readout(post, refs, tune, dist)
    cfg = cfg[0] if isinstance(cfg, tuple) else cfg
    cfg = dict(cfg)
    cfg.setdefault("kind", "global")
    r = S.score({c: refs[c] for c in decide}, PP.clip_to_duration(T.decode_all(post, cfg, dist, decide), {c: idx[c]['n'] / A.SR for c in decide}))
    save_readout(FT.CKPT / f"ens_{'+'.join(sys.argv[1:])}.json", sys.argv[1:], cfg)
    print(f"ENSEMBLE {'+'.join(sys.argv[1:])} DECIDE {r['combined']:.4f} "
          f"(F1 {r['event_f1']:.4f} Dice {r['segment_dice']:.4f}) readout {cfg}", flush=True)


if __name__ == "__main__":
    main()
