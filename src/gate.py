"""Gate for an experiment pair: DECIDE score plus natural human/background
match rates, computed identically for every checkpoint named.

    python src/gate.py hon_A ctl_C ast_A

A checkpoint trained with --use-ast is scored WITH its AST stream (fails if the
validation AST store is missing); one trained without it is scored without.
"""
from __future__ import annotations

import ast as _ast
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import postproc as PP     # noqa: E402
import finetune as FT     # noqa: E402
import scorer as S        # noqa: E402
import splits as SP       # noqa: E402
import train as T         # noqa: E402

tol = lambda d: max(0.2 * d, 0.05)  # noqa: E731
ov = lambda a, b: min(a[1], b[1]) - max(a[0], b[0]) > 0  # noqa: E731


def aux_for(cfg, store=None):
    """Second-stream input a checkpoint needs: None, "scale" (in-model BEATs),
    or the (frames, clips, hop, dim) store for `store` (default: its val store)."""
    if not cfg.get("use_ast"):
        return None
    name = cfg.get("aux", "ast")
    if name in AF.IN_MODEL:
        return "scale"
    d, _, vs = AF.AUX[name]
    store = store or vs or "val_audio"
    return FT.load_ast_stores([store], FT.AUD.parent / d)[store]


DS_N_OUT = 8   # dataset.N_OUT (any-noise + 7 categories)


def load(tag, dev):
    blob = torch.load(FT.CKPT / f"{tag}.pt", map_location=dev, weights_only=False)
    cfg = blob["cfg"]
    net = FT.FineTuneSED(cfg["enc_layers"],
                         boundary_head=not cfg.get("no_boundary_head", False),
                         freeze_conv=not cfg.get("unfreeze_conv", False),
                         layer_sum=cfg.get("layer_sum", False),
                         init_layer=cfg.get("init_layer"),
                         aux_dim=AF.AUX[cfg.get("aux", "ast")][1] if cfg.get("use_ast") else 0,
                         aux_mode=cfg.get("aux") if cfg.get("aux") in AF.IN_MODEL else None,
                         conv_heads=FT.n_conv_heads(cfg),
                          no_wavlm=cfg.get("no_wavlm", False),
                          conformer=cfg.get("conformer", 0),
                          red_head=bool(cfg.get("red") or cfg.get("red_classes")),
                          red_ch=DS_N_OUT if cfg.get("red_classes") else 1,
                          cat_boundary=7 if cfg.get("cat_boundary") else 0,
                          lane2=bool(cfg.get("lane2")),                       # B6a: (T, 9) posteriors
                          box_head=cfg.get("box_head", False), box_kind=cfg.get("box_kind", "peaks"), collar_head=cfg.get("collar_head", False),
                         enc_name=cfg.get("enc_name", "microsoft/wavlm-base-plus")).to(dev)
    net.load_state_dict(blob["state"])
    return net, blob


def group_rates(pred, refs, meta, ids):
    st = defaultdict(lambda: [0, 0])
    for c in ids:
        R, P = refs[c], pred[c]
        cands = sorted((abs(p[0] - r[0]) + abs(p[1] - r[1]), ri, pi)
                       for ri, r in enumerate(R) for pi, p in enumerate(P)
                       if abs(p[0] - r[0]) <= tol(r[1] - r[0]) and abs(p[1] - r[1]) <= tol(r[1] - r[0]))
        mr, mp = set(), set()
        for _, ri, pi in cands:
            if ri not in mr and pi not in mp:
                mr.add(ri)
                mp.add(pi)
        ev = meta[c]["NoiseSubCategoryTimeStamp"]
        ev = _ast.literal_eval(ev) if isinstance(ev, str) else ev
        for e in ev:
            r = (float(e["start"]), float(e["end"]))
            g = "human" if e["category"] == "human_non_speech" else "background"
            st[g][0] += 1
            st[g][1] += (R.index(r) in mr) if r in R else 0
    return {g: m / n for g, (n, m) in st.items()}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    refs, meta = D.load_validation()
    syn = {c: str(meta[c]["syntheticData"]) == "True" for c in refs}
    _, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    arr, am = A.load_audio_store(FT.AUD, "val_audio")
    idx = am["clips"]
    decide = [c for c in decide if c in idx]
    nat = [c for c in decide if not syn[c]]
    for tag in sys.argv[1:]:
        net, blob = load(tag, dev)
        v_ast = aux_for(blob["cfg"])
        post, dist, _ = FT.clip_posteriors(net, arr, idx, decide, blob["cfg"]["crop"], dev, ast=v_ast)
        rc = dict(blob["readout"])
        rc.setdefault("kind", "global")
        pred = PP.clip_to_duration(T.decode_all(post, rc, dist, decide), {c: idx[c]['n'] / A.SR for c in decide})
        r = S.score({c: refs[c] for c in decide}, pred)
        rn = S.score({c: refs[c] for c in nat}, {c: pred[c] for c in nat})
        g = group_rates(pred, refs, meta, nat)
        print(f"{tag:8s} DECIDE {r['combined']:.4f} (F1 {r['event_f1']:.4f} Dice {r['segment_dice']:.4f}) | "
              f"natural {rn['combined']:.4f} | matched: human {g['human']:.1%} background {g['background']:.1%} | "
              f"stored decide {blob['decide']['combined']:.4f} ast={bool(v_ast)}", flush=True)


if __name__ == "__main__":
    main()
