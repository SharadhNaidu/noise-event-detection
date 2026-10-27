"""Average the posteriors of several fine-tuned checkpoints.

Why this is worth testing carefully rather than assuming
--------------------------------------------------------
Ensembling is the most reliable gain in competitive ML, but the previous
attempt at this competition MEASURED it hurting: a 4-model average scored
1.0236 on their bench against 0.9836 for the best single model, then 1.0956 on
real validation against that model's 1.1190. Their bench had mis-ranked models
five times, so the negative result is suspect - but it is a real negative
result and it earns a careful test rather than a default.

Two things to get right, both of which they may not have:

  * AVERAGE POSTERIORS, NOT EVENT LISTS. Merging decoded spans has to resolve
    disagreements about how many events there are, which is exactly the
    one-to-one matching problem the metric punishes. Averaging before the
    readout sidesteps it entirely.
  * RE-SWEEP THE READOUT ON THE ENSEMBLE. An average is smoother and less
    confident than any member, so a threshold tuned for a single model is
    systematically too high for the mean. Reusing a member's readout would
    make a good ensemble look bad - the same transfer failure that cost 0.07
    earlier today.

The boundary-distance field is averaged too, which is sound: it is a
regression output in seconds, so the mean of several estimates is a better
estimate under squared loss.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import finetune as FT     # noqa: E402

CKPT = (P.ROOT / "ckpt")


def load(tag, dev):
    blob = torch.load(CKPT / f"{tag}.pt", map_location=dev, weights_only=False)
    cfg = blob["cfg"]
    net = FT.FineTuneSED(cfg["enc_layers"],
                         boundary_head=not cfg.get("no_boundary_head", False),
                         freeze_conv=not cfg.get("unfreeze_conv", False)).to(dev)
    net.load_state_dict(blob["state"])
    net.eval()
    return net, cfg, blob


def member_posteriors(tags, arr, idx, ids, dev, weights=None):
    """{tag: (post, dist)} for each member, on the same clips."""
    out = {}
    for t in tags:
        net, cfg, _ = load(t, dev)
        post, dist, _ = FT.clip_posteriors(net, arr, idx, ids, cfg["crop"], dev)
        out[t] = (post, dist)
        del net
        torch.cuda.empty_cache()
    return out


def average(members, ids, weights=None):
    """Mean posterior and mean distance field across members.

    Clips can differ in frame count between members when their crop lengths
    differ, so everything is truncated to the shortest - a 1-2 frame
    difference at the tail, far below the 4.3-frame tolerance.
    """
    tags = list(members)
    w = np.array([1.0] * len(tags) if weights is None
                 else [weights[t] for t in tags], dtype=np.float64)
    w = w / w.sum()
    post, dist = {}, {}
    for c in ids:
        ps = [members[t][0][c] for t in tags]
        n = min(len(p) for p in ps)
        post[c] = np.average([p[:n] for p in ps], axis=0, weights=w
                             ).astype(np.float32)
        ds = [members[t][1].get(c) for t in tags]
        if all(d is not None for d in ds):
            dist[c] = np.average([d[:n] for d in ds], axis=0, weights=w
                                 ).astype(np.float32)
    return post, dist


def diversity(members, ids):
    """Mean pairwise correlation between member posteriors.

    An ensemble only helps when its members make DIFFERENT errors. Near 1.0
    means the members agree and averaging buys nothing; lower means there is
    something to gain. Reported so a null result can be explained rather than
    just observed.
    """
    tags = list(members)
    if len(tags) < 2:
        return {}
    out = {}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            rs = []
            for c in ids[:400]:
                a = members[tags[i]][0][c]
                b = members[tags[j]][0][c]
                n = min(len(a), len(b))
                if n > 10 and a[:n].std() > 1e-6 and b[:n].std() > 1e-6:
                    rs.append(float(np.corrcoef(a[:n], b[:n])[0, 1]))
            out[(tags[i], tags[j])] = float(np.mean(rs)) if rs else float("nan")
    return out


def multiscale_posteriors(tag, arr, idx, ids, dev, crops=(5.0, 10.0)):
    """Run one model at several crop lengths and average.

    A test of the context hypothesis that costs no training. 73% of Gold clips
    are longer than the 5 s training crop and 5.2% of events exceed it
    outright, so a wrapper event is never seen whole during training - but the
    encoder is fully convolutional plus self-attention and accepts any length,
    so it CAN be given more context at inference. If the extra context helps,
    averaging across scales should show it; if the model simply cannot use a
    length it never trained on, it will not.

    Either answer is useful: a gain is free, and a null result means the crop
    hypothesis has to be tested by retraining rather than assumed.
    """
    net, cfg, _ = load(tag, dev)
    out = {}
    for cs in crops:
        post, dist, _ = FT.clip_posteriors(net, arr, idx, ids, cs, dev)
        out[f"{tag}@{cs}"] = (post, dist)
    del net
    torch.cuda.empty_cache()
    return out
