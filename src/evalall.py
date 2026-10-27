"""Score a checkpoint on ALL 2,501 validation clips, split natural / synthetic.

Why this exists: the 1.1835 currently on the leaderboard is OUR OWN earlier
attempt (github.com/SharadhNaidu/indoml, submission 002, "FDY-CRNN + RED"), and
that repo records its measurement as

    real validation ALL 1.1190  |  natural 1.0280  |  synthetic 1.2994
    F1 0.4154  Dice 0.7036      ->  leaderboard 1.18

which is scored over all 2,501 clips, not a held-out half. Our own numbers have
been decide-half only, so the two were never comparable and the "gate" was being
set from the wrong quantity. This reports the same three figures.

Read it with the split it is: ALL includes the tune clips the readout was
chosen on, so it is mildly optimistic in a way the decide half is not. Both are
printed. The decide half remains the honest number for deciding; ALL is the
number that compares against the incumbent.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import data as D          # noqa: E402
import model as M         # noqa: E402
import scorer as S        # noqa: E402
import splits as SP       # noqa: E402
import train as T         # noqa: E402

CKPT = (P.ROOT / "ckpt")

# The incumbent to beat, from the previous attempt's own records.
INCUMBENT = {"all": 1.1190, "natural": 1.0280, "synthetic": 1.2994,
             "f1": 0.4154, "dice": 0.7036, "leaderboard": 1.1835}


def synthetic_ids(zip_path="data/validation.zip"):
    z = zipfile.ZipFile(zip_path)
    return {os.path.basename(n)[:-4] for n in z.namelist()
            if n.endswith(".wav") and "__MACOSX" not in n
            and "syntheticNoiseAudio/" in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--val-store", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CKPT / f"{a.tag}.pt", map_location=dev, weights_only=False)
    cfg = blob["cfg"]
    store = a.val_store or ("val" if cfg.get("layer") is None
                            else f"val_L{cfg['layer']}")
    net = M.SEDHead(rnn_dim=cfg.get("rnn_dim", 256),
                    boundary_head=cfg.get("boundary_head", False)).to(dev)
    net.load_state_dict(blob["state"])
    net.eval()

    refs, _ = D.load_validation()
    res = T.posteriors_for(net, store, list(refs), device=dev)
    post, dist = res[0], (res[2] if len(res) > 2 else None)
    ids = [c for c in refs if c in post]

    rcfg = dict(blob["readout"])
    rcfg.setdefault("kind", "global")
    preds = T.decode_all(post, rcfg, dist, ids)

    syn = synthetic_ids()
    tune, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    groups = {
        "ALL": ids,
        "natural": [c for c in ids if c not in syn],
        "synthetic": [c for c in ids if c in syn],
        "tune half": [c for c in tune if c in post],
        "decide half": [c for c in decide if c in post],
    }
    print(f"{a.tag}   layer={cfg.get('layer')}   store={store}   "
          f"epochs={cfg.get('epochs')}   boundary_head="
          f"{cfg.get('boundary_head', False)}")
    print(f"readout: {rcfg}")
    print()
    print(f"  {'split':<14} {'clips':>7} {'F1':>9} {'Dice':>9} {'combined':>10}")
    print("  " + "-" * 52)
    out = {}
    for name, g in groups.items():
        if not g:
            continue
        o = S.score({c: refs[c] for c in g}, {c: preds[c] for c in g})
        out[name] = o
        print(f"  {name:<14} {len(g):>7,} {o['event_f1']:>9.4f} "
              f"{o['segment_dice']:>9.4f} {o['combined']:>10.4f}")

    print()
    print("  " + "=" * 62)
    print("  INCUMBENT (our own submission 002, leaderboard 1.1835):")
    print(f"  {'':<14} {'':>7} {INCUMBENT['f1']:>9.4f} {INCUMBENT['dice']:>9.4f} "
          f"{INCUMBENT['all']:>10.4f}   ALL")
    print(f"  {'':<14} {'':>7} {'':>9} {'':>9} {INCUMBENT['natural']:>10.4f}   "
          f"natural")
    print(f"  {'':<14} {'':>7} {'':>9} {'':>9} {INCUMBENT['synthetic']:>10.4f}   "
          f"synthetic")
    d = out["ALL"]["combined"] - INCUMBENT["all"]
    verdict = "BEATS IT" if d > 0 else "below it"
    print(f"\n  ALL {out['ALL']['combined']:.4f} vs {INCUMBENT['all']:.4f} "
          f"-> {d:+.4f}   {verdict}")
    print("  (their ALL 1.1190 scored 1.1835 on the leaderboard, offset +0.065)")


if __name__ == "__main__":
    main()
