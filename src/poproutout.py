"""Per-population readout: choose a separate decode for natural and synthetic.

Measured need
-------------
After the layer-4 model reached leaderboard 1.4843 the binding constraint
flipped. Our F1 (0.7231) is already ahead of ranks 12 and 14; our Dice (0.7613)
is behind both, and behind rank 10's 0.8107 by 0.049 - twice the F1 gap.

The two populations want opposite decodes, and a single sweep has to average
them. Swept separately on the tune half:

    natural    thr 0.60 + refine    - looser, buys coverage, and Dice
    synthetic  thr 0.80, no refine  - tighter, buys precision, and F1

That is not a coincidence. EXP-23 measured natural coverage at 0.248 against a
reference 0.349, so natural clips are starved of Dice and want a lower bar;
synthetic clips already sit at 86.6% in-tolerance and want to protect it.
Routing the two apart is worth +0.0437 combined on the decide half.

Routing uses the 97.3%-accurate classifier from `synthid`, whose test-set
estimate (2,093 of 5,517 synthetic) matches the organizers' stated 2,080.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import json
from pathlib import Path

import numpy as np

CKPT = (P.ROOT / "ckpt")


def post_features(p):
    """Posterior statistics for the synthetic classifier. Must stay in sync
    with the features it was fitted on - the 5x edge-sharpness difference
    between the populations (EXP-27) is what carries the signal."""
    p = np.asarray(p, dtype=np.float32)
    d = np.abs(np.diff(p)) if len(p) > 1 else np.array([0.0], dtype=np.float32)
    return np.array([p.max(), p.mean(), float((p > 0.9).mean()),
                     float((p < 0.1).mean()), float(np.percentile(d, 95)),
                     d.mean(), float(p.std()),
                     float(((p > 0.1) & (p < 0.9)).mean())], dtype=np.float32)


def sweep_population(ids, post, dist, refs, decode_one, scorer,
                     thrs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                     meds=(1, 3, 7, 11), min_durs=(0.08, 0.12, 0.16),
                     refines=(None, 0.15, 0.3), splits=(False, True),
                     merge_gaps=(0.0, 0.06, 0.15), dilates=(0.0, 0.04, -0.04),
                     rels=(0.45, 0.6, 0.75), floors=(0.1, 0.2, 0.35)):
    """Best decode config for one population, over EVERY readout family.

    The family matters more than its parameters, and restricting this to the
    global-threshold family was a real bug: nat_val's own sweep chose the
    ADAPTIVE family, so forcing per-population decoding into the global family
    scored 1.2807 against 1.3611 for a single adaptive readout - the routing
    looked harmful when it was the family restriction doing the damage. On
    nat_L4, whose best family happened to be global, the same code gained
    +0.0437. A confound that only shows up on some models is the worst kind.

    Two stages: first the parameters that decide WHICH spans exist, then the
    two that only adjust their extent (merge gap, dilation). That keeps the
    cost near additive instead of multiplicative.
    """
    best = (-1.0, None)
    rr = {c: refs[c] for c in ids}

    def try_cfg(cfg):
        nonlocal best
        sc = scorer.score(rr, {c: decode_one(c, cfg) for c in ids})["combined"]
        if sc > best[0]:
            best = (sc, dict(cfg))

    # family 1: global threshold + median filter
    for thr in thrs:
        for med in meds:
            for md in min_durs:
                for rf in refines:
                    for sp in splits:
                        cfg = {"kind": "global", "thr": thr, "med": med,
                               "min_dur": md, "merge_gap": 0.06}
                        if rf:
                            cfg["refine"] = rf
                        if sp:
                            cfg["split"] = True
                        try_cfg(cfg)

    # family 2: per-clip adaptive threshold
    for rel in rels:
        for fl in floors:
            for md in min_durs:
                for rf in refines:
                    cfg = {"kind": "adaptive", "rel": rel, "floor": fl,
                           "min_dur": md, "merge_gap": 0.06}
                    if rf:
                        cfg["refine"] = rf
                    try_cfg(cfg)

    # family 3: asymmetric thresholds - the posterior's fall is 36% slower
    # than its rise on natural clips, so one level cannot serve both edges
    for ton in (0.35, 0.5, 0.6, 0.7):
        for toff in (0.35, 0.5, 0.6, 0.7, 0.8):
            for rf in (None, 0.3):
                cfg = {"kind": "asym", "thr_on": ton, "thr_off": toff,
                       "min_dur": 0.12, "merge_gap": 0.06}
                if rf:
                    cfg["refine"] = rf
                try_cfg(cfg)

    # stage 2: extent only, holding the detection parameters fixed
    base = dict(best[1])
    for mg in merge_gaps:
        for dl in dilates:
            cfg = {**base, "merge_gap": mg}
            if dl:
                cfg["dilate"] = dl
            else:
                cfg.pop("dilate", None)
            try_cfg(cfg)
    return best[1], best[0]


def save(tag, nat_cfg, syn_cfg):
    p = CKPT / f"{tag}_poprdt.json"
    json.dump({"nat": nat_cfg, "syn": syn_cfg}, open(p, "w"))
    return p


def load(tag):
    p = CKPT / f"{tag}_poprdt.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["nat"], d["syn"]


def fit_bias(ids, refs, preds, grid=None):
    """Fit the boundary shift for ONE population on its own clips.

    Previously hardcoded to (+0.03, -0.04), measured on ft_gold_syn in the
    early afternoon. That was that model's systematic error; a later model
    whose synthetic boundary error is 0.018s median does not have it, so
    applying the old shift introduces a bias instead of removing one - worth
    -0.056 on fin_A. Like the readout, this has to be fitted per model.
    """
    import scorer as S
    grid = grid or [round(x, 3) for x in
                    [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]]
    rr = {c: refs[c] for c in ids}
    best = (-1.0, (0.0, 0.0))
    for d_on in grid:
        for d_off in grid:
            pr = {}
            for c in ids:
                q = [(max(0.0, a + d_on), b + d_off) for a, b in preds[c]]
                pr[c] = sorted([(a, b) for a, b in q if b > a])
            sc = S.score(rr, pr)["combined"]
            if sc > best[0]:
                best = (sc, (d_on, d_off))
    return best[1], best[0]
