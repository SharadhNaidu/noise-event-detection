"""Submission from a posterior ENSEMBLE of checkpoints (each with its own second stream).

    python src/submit_ens.py --members hon_A bts_C --readout-json ens_readout.json [--fallback 0.7] --out sub.zip

The readout must be the one ens_gate.py chose for this exact member set on the
TUNE half (saved with ens_gate --save); never re-tuned here. Writing and
validation are submit_ft.write_zip, shared with single-model submissions.
"""
from __future__ import annotations

import argparse

import numpy as np
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import ens_gate as E      # noqa: E402
import finetune as FT     # noqa: E402
import gate as G          # noqa: E402
import postproc as PP     # noqa: E402
import submit_ft as SF    # noqa: E402
import train as T         # noqa: E402


def member_posteriors(tag, store, ids, dev, arr, idx):
    """One member's posteriors on a store, cached per (store, tag) so later
    ensembles on the same test set only compute new members."""
    import pickle
    cache = FT.CKPT / f"post8_{store}_{tag}.pkl"           # all channels: [:, 0] = any-noise track
    stamp = (FT.CKPT / f"{tag}.pt").stat().st_mtime        # a retrained checkpoint invalidates its cache
    if cache.exists():
        c = pickle.load(open(cache, "rb"))
        if c.get("ckpt_mtime") == stamp and all(i in c["post"] for i in ids):
            return c["post"], c["dist"]
    net, blob = G.load(tag, dev)
    net.eval()
    has_box = getattr(net.head, "box", None) is not None
    res = FT.clip_posteriors(net, arr, idx, ids, blob["cfg"]["crop"], dev,
                             ast=G.aux_for(blob["cfg"], store), all_channels=True, with_box=has_box)
    post, dist = res[0], res[1]
    del net
    if dev == "cuda":
        torch.cuda.empty_cache()
    blob_out = {"post": post, "dist": dist, "ckpt_mtime": stamp}
    if has_box:
        blob_out["box"] = res[3]          # dense head: (T, 6) softplus seconds to onset/offset per lane
        blob_out["box_kind"] = blob["cfg"].get("box_kind", "peaks")
    pickle.dump(blob_out, open(cache, "wb"))
    return post, dist


def ensemble_posteriors(tags, store, ids, dev):
    arr, me = A.load_audio_store(FT.AUD, store)
    idx = me["clips"]
    posts, dists = [], []
    for tag in tags:
        post, dist = member_posteriors(tag, store, ids, dev, arr, idx)
        posts.append(post)
        dists.append(dist)
    k = min((next(iter(p.values())).shape[-1] for p in posts if p), default=None)   # B6a: a --lane2 member is (T, 9);
    posts = [{c: v[..., :k] for c, v in p.items()} for p in posts]    # a mixed ensemble averages the shared 8
    return E.average(posts), E.average(dists)


def natural_by_peak(arr, idx, ids):
    """Natural clips peak at int16 full scale, synthetic at 0.99 [MEASURED:
    exact 3437/2080 split on test, exact on validation - DATA_DEEP_DIVE 1.2]."""
    out = {}
    for c in ids:
        e = idx[c]
        w = np.asarray(arr[e["off"]: e["off"] + e["n"]])
        # >= 0.9999 of full scale: the int16 store rounds natural full-scale peaks to
        # 32766 OR 32767 (2734 + 703 of 3437 test naturals); synthetic peak ~32440.
        # Requiring exactly 32767 mis-routed 80% of natural test clips (2026-10-01).
        out[c] = bool(len(w)) and int(np.abs(w.astype(np.int32)).max()) >= 32765
    return out


def decode_routed(post, dist, ids, readout, natural):
    """readout kind 'perpop' = {'natural': cfg, 'synthetic': cfg}, routed per clip;
    any other readout decodes every clip with that one config (unchanged)."""
    if readout.get("kind") != "perpop":
        rc = dict(readout)
        rc.setdefault("kind", "global")
        return T.decode_all(post, rc, dist, ids)
    out = {}
    for pop, want in (("natural", True), ("synthetic", False)):
        sub = [c for c in ids if natural[c] == want]
        rc = dict(readout[pop])
        rc.setdefault("kind", "global")
        out.update(T.decode_all(post, rc, dist, sub))
    return out


def apply_overlay(preds, post8, ids, ov, natural):
    """Second track nested inside base events (postproc.overlay_nested). ov = {'ch', 'thr',
    'ratio', 'natural_only'}; None = unchanged."""
    if not ov:
        return preds
    out = dict(preds)
    for c in ids:
        if ov.get("natural_only", True) and not natural[c]:
            continue
        out[c] = PP.overlay_nested(preds[c], post8[c][:, ov["ch"]], ov["thr"], ov["ratio"])
    return out


def apply_categories(preds, post8, ids, cats, natural, natural_only=True):
    """Per-category track events added where they do not duplicate a base event
    (postproc.category_union). cats = {channel: threshold|None}; None/{} = unchanged."""
    if not cats:
        return preds
    cats = {int(k): v for k, v in cats.items()}
    out = dict(preds)
    for c in ids:
        if natural_only and not natural[c]:
            continue
        out[c] = PP.category_union(preds[c], post8[c], cats)
    return out


def ensemble_preds(tags, store, ids, dev, readout, fallback=None):
    post8, dist = ensemble_posteriors(tags, store, ids, dev)
    post = {c: post8[c][:, 0] for c in ids}
    arr, me = A.load_audio_store(FT.AUD, store)
    natural = natural_by_peak(arr, me["clips"], ids)
    preds = decode_routed(post, dist, ids, readout, natural)
    preds = apply_overlay(preds, post8, ids, readout.get("overlay"), natural)
    preds = apply_categories(preds, post8, ids, readout.get("categories"), natural,
                             readout.get("categories_natural_only", True))
    if fallback:
        preds = PP.empty_clip_fallback(preds, post, q=fallback)
    # LAST: the fallback reads padded posteriors and could place an event in the padding
    return PP.clip_to_duration(preds, store_durations(store))


def store_durations(store):
    """Real clip length in seconds for every clip of an audio store."""
    _, me = A.load_audio_store(FT.AUD, store)
    return {c: e["n"] / A.SR for c, e in me["clips"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--readout-json", required=True)
    ap.add_argument("--fallback", type=float, default=None)
    ap.add_argument("--test-store", default="test_audio")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    saved = json.load(open(a.readout_json))
    if sorted(saved["members"]) != sorted(a.members):
        raise SystemExit(f"readout was tuned for {saved['members']}, not {a.members}")
    _, me = A.load_audio_store(FT.AUD, a.test_store)
    ids = list(me["clips"])
    preds = ensemble_preds(a.members, a.test_store, ids, dev, saved["readout"], a.fallback)
    SF.write_zip(preds, Path(a.out), set(D.list_test_clip_ids()))


if __name__ == "__main__":
    main()
