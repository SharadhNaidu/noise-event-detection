"""Build a Codabench submission ZIP from a trained checkpoint.

Format requirements, taken from the Evaluation page and confirmed against the
organizers' own baseline (they write `p.stem`):

  * a single predictions.jsonl at the ZIP ROOT - no enclosing folder
  * one JSON object per line: {"clip_id": str, "events": [{"onset","offset"}]}
  * EVERY evaluation clip exactly once; clips with nothing detected get [],
    not a missing line
  * clip_ids absent from the reference contribute all their events as false
    positives, and missing clips lose all their reference events

Five leaderboard entries sit at exactly 0.0000, which is the signature of
getting clip_id wrong - so this module validates hard before writing anything.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import data as D          # noqa: E402
import decode as DEC      # noqa: E402
import features as F      # noqa: E402
import model as M         # noqa: E402
import readout as R       # noqa: E402
import train as T         # noqa: E402

ART = Path(__file__).resolve().parent.parent / "artifacts"
CKPT = (P.ROOT / "ckpt")
EXPECTED_CLIPS = 5517


def write_submission(preds, out_zip, expected=EXPECTED_CLIPS):
    """Validate and write predictions.jsonl into a ZIP at the archive root."""
    ids = sorted(preds)
    if len(ids) != expected:
        raise ValueError(f"expected {expected} clips, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate clip_ids")

    n_ev = 0
    for cid, evs in preds.items():
        for on, off in evs:
            if not (np.isfinite(on) and np.isfinite(off)):
                raise ValueError(f"non-finite event in {cid}: {(on, off)}")
            if off < on:
                raise ValueError(f"offset before onset in {cid}: {(on, off)}")
            if on < 0:
                raise ValueError(f"negative onset in {cid}: {on}")
            n_ev += 1

    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / "predictions.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for cid in ids:
            rec = {"clip_id": cid,
                   "events": [{"onset": round(float(a), 3),
                               "offset": round(float(b), 3)}
                              for a, b in preds[cid]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_zip = Path(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, "predictions.jsonl")          # at ZIP ROOT

    # read the archive back and re-validate - never trust the write
    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
        if names != ["predictions.jsonl"]:
            raise ValueError(f"ZIP must contain exactly predictions.jsonl, got {names}")
        lines = z.read("predictions.jsonl").decode("utf-8").strip().split("\n")
    if len(lines) != expected:
        raise ValueError(f"ZIP holds {len(lines)} lines, expected {expected}")
    seen = set()
    for ln in lines:
        o = json.loads(ln)
        assert set(o) == {"clip_id", "events"}, o.keys()
        seen.add(o["clip_id"])
    if len(seen) != expected:
        raise ValueError("clip_id collision inside the written file")

    empty = sum(1 for cid in ids if not preds[cid])
    print(f"  wrote {out_zip}  ({out_zip.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(ids):,} clips | {n_ev:,} events | {n_ev/len(ids):.2f} per clip "
          f"| {empty} empty")
    return out_zip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="tag under $INDOML_ROOT/ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-store", default=None,
                    help="default: chosen from the checkpoint's encoder layer")
    ap.add_argument("--readout", default="auto",
                    choices=["auto", "threshold", "adaptive"],
                    help="auto = the readout this checkpoint won its sweep with")
    ap.add_argument("--rel", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--min-dur", type=float, default=0.10)
    ap.add_argument("--merge-gap", type=float, default=0.06)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CKPT / f"{a.ckpt}.pt", map_location=dev, weights_only=False)
    cfg = blob["readout"]
    print(f"checkpoint {a.ckpt}")
    print(f"  decide-half score : {blob['decide']}")
    print(f"  readout           : {cfg}")

    net = M.SEDHead(rnn_dim=blob["cfg"].get("rnn_dim", 256),
                    boundary_head=blob["cfg"].get("boundary_head", False)).to(dev)
    net.load_state_dict(blob["state"])
    net.eval()

    # The test store must come from the SAME encoder layer the head was trained
    # on. Mismatched layers give a perfectly well-formed submission of noise, so
    # this is an assertion rather than a convention.
    want = blob["cfg"].get("layer")
    store = a.test_store or ("test" if want is None else f"test_L{want}")
    have = F.store_layer(str(P.ROOT / "features"), store)
    if have != want:
        raise SystemExit(
            f"layer mismatch: checkpoint was trained on layer {want} but store "
            f"'{store}' holds layer {have}. Extract the matching test store.")
    print(f"running inference on the test set ({store}, layer {want}) ...")
    res = T.posteriors_for(net, store, device=dev)
    post, dist = res[0], (res[2] if len(res) > 2 else None)
    print(f"  posteriors for {len(post):,} clips")

    # The checkpoint carries the readout that WON the sweep on the tune half
    # for this specific model. Readout settings do not transfer between
    # checkpoints - the same gs_filtered posteriors scored 0.9130 with their own
    # swept readout and 0.8371 with parameters fitted to a different model - so
    # overriding them from the command line is opt-in, never the default.
    if a.readout == "auto":
        if "kind" not in cfg:
            cfg = {"kind": "global", **cfg}
        print(f"  readout: from the checkpoint -> {cfg}")
        preds = T.decode_all(post, cfg, dist)
    elif a.readout == "adaptive":
        print(f"  readout: per-clip adaptive (rel={a.rel} floor={a.floor} "
              f"min_dur={a.min_dur} merge_gap={a.merge_gap})")
        preds = {cid: R.adaptive_threshold(p, rel=a.rel, floor=a.floor,
                                           min_dur=a.min_dur,
                                           merge_gap=a.merge_gap)
                 for cid, p in post.items()}
    else:
        print(f"  readout: global threshold {cfg}")
        preds = {cid: DEC.prob_to_events(p, thr=cfg["thr"], med=cfg["med"],
                                         min_dur=cfg["min_dur"],
                                         merge_gap=cfg["merge_gap"])
                 for cid, p in post.items()}

    # the clip_ids we emit must be exactly the test WAV stems
    want = set(D.list_test_clip_ids())
    got = set(preds)
    if want != got:
        raise ValueError(f"clip_id mismatch: missing {len(want-got)}, "
                         f"extra {len(got-want)}; e.g. missing "
                         f"{list(want-got)[:2]} extra {list(got-want)[:2]}")
    print(f"  clip_ids match the test set exactly ({len(want):,})")

    out = a.out or (ART / f"submission_{a.ckpt}.zip")
    write_submission(preds, out)


if __name__ == "__main__":
    main()
