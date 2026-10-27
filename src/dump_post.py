"""Cache each checkpoint's validation posteriors once, for CPU-side ensemble experiments.

    python src/dump_post.py hon_A bts_C mt_A   ->  <ckpt dir>/post_<tag>.pkl

Uses ens_gate.posteriors (per-member aux stream; its one-member output equals
the gate's), over ALL validation clips (tune + decide) so combiners can be
chosen on TUNE and judged on DECIDE.
"""
from __future__ import annotations

import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A    # noqa: E402
import ens_gate as E      # noqa: E402
import finetune as FT     # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arr, am = A.load_audio_store(FT.AUD, "val_audio")
    idx = am["clips"]
    ids = list(idx)
    for tag in sys.argv[1:]:
        post, dist, blob = E.posteriors(tag, ids, dev, arr, idx)
        out = FT.CKPT / f"post_{tag}.pkl"
        pickle.dump({"post": post, "dist": dist, "readout": blob["readout"]}, open(out, "wb"))
        print(f"cached {tag} -> {out} ({len(post)} clips)", flush=True)


if __name__ == "__main__":
    main()
