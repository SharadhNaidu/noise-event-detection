"""Rank every trained model on data NONE of them has seen.

Why this exists
---------------
Last night's submission was selected on a contaminated number. fin_A, fin_C
and fin_X all trained on all 2,501 validation clips, so their validation
scores measured memorisation as well as skill - and by DIFFERENT amounts,
because they saw validation a different number of times:

    fin_C   validation = 10% of its data, 10 epochs
    fin_A   validation = 10% of its data, 10 epochs
    fin_X   validation = 5.4% of its data, 5 epochs

fin_C won the selection at ALL 1.5500 and scored 1.4894 on the leaderboard -
an inflation of 0.173 against the +0.1125 offset measured on an uncontaminated
model. fin_X looked catastrophic at 1.2479, but it also memorised the least,
so I cannot currently tell whether the AudioSet fusion failed or simply
cheated least. That is the question this answers.

The holdout is 2,000 Silver clips that no model trained on. Silver's
timestamps are ~4x rounder than Gold's, so the ABSOLUTE scores here will be
low and are not comparable to validation numbers. But every model faces the
same label noise, so the RANKING is sound - and a ranking is precisely what
went wrong.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402
import finetune as FT     # noqa: E402
import readout as R       # noqa: E402
import scorer as S        # noqa: E402
import train as T         # noqa: E402

CKPT = (P.ROOT / "ckpt")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arr, meta = A.load_audio_store(str(P.ROOT / "audio"), "holdout")
    idx = meta["clips"]
    ids = list(idx)
    refs = {c: sorted(tuple(x) for x in (idx[c].get("events") or []))
            for c in ids}
    ids = [c for c in ids if refs[c]]
    print(f"clean holdout: {len(ids):,} Silver clips, "
          f"{sum(len(v) for v in refs.values()):,} events, "
          f"trained on by NO model")

    try:
        aa, am = AF.load_ast_store(str(P.ROOT / "ast"), "holdout")
        ast = (aa, am["clips"])
        print("  AST features available")
    except Exception as e:
        ast = None
        print(f"  no AST features ({type(e).__name__}) - fin_X will be skipped")

    tags = [t for t in ("fin_C", "fin_A", "fin_X", "nat_val", "nat_L4")
            if (CKPT / f"{t}.pt").exists()]
    print(f"\n  {'model':<10}{'val ALL':>9}{'LB':>9}"
          f"{'HOLDOUT F1':>12}{'Dice':>8}{'combined':>10}")
    print("  " + "-" * 60)
    known = {"fin_C": (1.5500, 1.4894), "nat_L4": (1.3718, 1.4843),
             "fin_A": (1.4345, None), "fin_X": (1.2479, None),
             "nat_val": (1.4011, None)}
    out = {}
    for t in tags:
        blob = torch.load(CKPT / f"{t}.pt", map_location=dev,
                          weights_only=False)
        cfg = blob["cfg"]
        if cfg.get("use_ast") and ast is None:
            print(f"  {t:<10} skipped (needs AST)")
            continue
        try:
            net = FT.FineTuneSED(
                cfg["enc_layers"],
                boundary_head=not cfg.get("no_boundary_head", False),
                freeze_conv=not cfg.get("unfreeze_conv", False),
                aux_dim=AF.AST_DIM if cfg.get("use_ast") else 0).to(dev)
            net.load_state_dict(blob["state"])
            net.eval()
            post, dist, _ = FT.clip_posteriors(
                net, arr, idx, ids, cfg["crop"], dev, batch=4,
                ast=ast if cfg.get("use_ast") else None)
            del net
            torch.cuda.empty_cache()
        except Exception as ex:
            print(f"  {t:<10} FAILED: {type(ex).__name__}: {str(ex)[:60]}")
            continue

        # one readout family swept on THIS holdout, identical for every model,
        # so no model is advantaged by a decode tuned elsewhere
        best = (-1.0, None)
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            for md in (0.10, 0.16, 0.25):
                for rf in (None, 0.3):
                    cfg_r = {"kind": "global", "thr": thr, "med": 7,
                             "min_dur": md, "merge_gap": 0.06}
                    if rf:
                        cfg_r["refine"] = rf
                    pr = {}
                    for c in ids:
                        ev = T.apply_readout(post[c], cfg_r)
                        if rf and c in dist:
                            ev = R.refine(ev, dist[c], radius=rf)
                        pr[c] = ev
                    sc = S.score(refs, pr)
                    if sc["combined"] > best[0]:
                        best = (sc["combined"], sc)
        sc = best[1]
        va, lb = known.get(t, (None, None))
        out[t] = sc["combined"]
        print(f"  {t:<10}{va if va else float('nan'):>9.4f}"
              f"{(f'{lb:.4f}' if lb else '—'):>9}"
              f"{sc['event_f1']:>12.4f}{sc['segment_dice']:>8.4f}"
              f"{sc['combined']:>10.4f}")

    if out:
        rank = sorted(out, key=lambda k: -out[k])
        print(f"\n  HONEST RANKING: {' > '.join(rank)}")
        print(f"  we shipped fin_C, which ranks "
              f"#{rank.index('fin_C')+1} of {len(rank)} here"
              if "fin_C" in rank else "")


if __name__ == "__main__":
    main()
