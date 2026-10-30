"""Build a submission from a FINE-TUNED checkpoint.

submit.py reads cached WavLM features, which only exist for a frozen encoder.
A fine-tuned model has moved the encoder, so its features must be computed from
raw audio at inference - hence a separate path rather than a flag.

Same guarantees as submit.py, because those were each bought by a real bug:
  * clip_ids must equal the test WAV stems EXACTLY, asserted not assumed;
  * the readout is the one THIS checkpoint won its own sweep with - readout
    settings do not transfer between models, and using another model's cost
    0.07 once already;
  * the finished archive is re-read from disk and re-validated.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import finetune as FT     # noqa: E402
import readout as R       # noqa: E402
import synthid as SI      # noqa: E402
import train as T         # noqa: E402

CKPT = (P.ROOT / "ckpt")
AUD = (P.ROOT / "audio")
ART = Path(__file__).resolve().parent.parent / "artifacts"


def write_zip(preds, out, want):
    """Write predictions.jsonl into `out`, refuse id mismatches, re-read and validate."""
    got = set(preds)
    if want != got:
        miss, extra = sorted(want - got)[:3], sorted(got - want)[:3]
        raise SystemExit(
            f"clip_id mismatch: {len(want)} expected, {len(got)} produced. "
            f"missing e.g. {miss}; unexpected e.g. {extra}")
    print(f"  clip_ids match the input clips exactly ({len(want):,})")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cid in sorted(preds):
        ev = [{"onset": round(float(s), 3), "offset": round(float(e), 3)}
              for s, e in preds[cid] if e > s]
        lines.append(json.dumps({"clip_id": cid, "events": ev},
                                ensure_ascii=False))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "\n".join(lines) + "\n")

    # Re-read what we actually wrote. A submission that is correct in memory and
    # wrong on disk is the failure mode this catches.
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        if names != ["predictions.jsonl"]:
            raise SystemExit(f"archive holds {names}, expected one file at root")
        rows = [json.loads(l) for l in
                z.read("predictions.jsonl").decode("utf-8").splitlines() if l]
    if {r["clip_id"] for r in rows} != want:
        raise SystemExit("re-read of the archive does not match the test ids")
    n_ev = sum(len(r["events"]) for r in rows)
    n_empty = sum(1 for r in rows if not r["events"])
    bad = [r["clip_id"] for r in rows
           for e in r["events"] if e["offset"] <= e["onset"] or e["onset"] < 0]
    if bad:
        raise SystemExit(f"{len(bad)} malformed events, e.g. {bad[:3]}")
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(rows):,} clips | {n_ev:,} events | "
          f"{n_ev/len(rows):.2f} per clip | {n_empty} empty")
    return out


def build_net(cfg, dev):
    """The network exactly as the checkpoint was trained, AST input included."""
    return FT.FineTuneSED(cfg["enc_layers"],
                          boundary_head=not cfg.get("no_boundary_head", False),
                          freeze_conv=not cfg.get("unfreeze_conv", False),
                          layer_sum=cfg.get("layer_sum", False),
                          init_layer=cfg.get("init_layer"),
                          cat_boundary=(len(FT.DS.CATS)
                                        if cfg.get("cat_boundary") else 0),
                          aux_dim=AF.AUX[cfg.get("aux", "ast")][1] if cfg.get("use_ast") else 0,
                          aux_mode=cfg.get("aux") if cfg.get("aux") in AF.IN_MODEL else None,
                          conv_heads=FT.n_conv_heads(cfg),
                          no_wavlm=cfg.get("no_wavlm", False),
                          conformer=cfg.get("conformer", 0),
                          enc_name=cfg.get("enc_name",
                                           "microsoft/wavlm-base-plus")
                          ).to(dev)


def ast_for(cfg, store, root):
    """AST features for `store` when the checkpoint was trained with them.
    Missing features raise instead of silently feeding zeros."""
    if not cfg.get("use_ast"):
        return None
    if cfg.get("aux") in AF.IN_MODEL:
        return "scale"
    return FT.load_ast_stores([store], root)[store]


def aux_root(cfg):
    d = AF.AUX[cfg.get("aux", "ast")][0]
    return AUD.parent / d if d else None


def _post_feat(p):
    """Posterior statistics for the synthetic classifier. Must match the
    features it was fitted with - EXP-27 measured a 5x difference in edge
    sharpness between the populations, which is what carries the signal."""
    p = np.asarray(p, dtype=np.float32)
    d = np.abs(np.diff(p)) if len(p) > 1 else np.array([0.0])
    return np.array([p.max(), p.mean(), float((p > 0.9).mean()),
                     float((p < 0.1).mean()), float(np.percentile(d, 95)),
                     d.mean(), float(p.std()),
                     float(((p > 0.1) & (p < 0.9)).mean())], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-store", default="test_audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bias-correct", action="store_true",
                    help="skip the per-population boundary bias correction")
    ap.add_argument("--syn-shift", type=float, nargs=2, default=(0.03, -0.04),
                    help="onset/offset shift applied to synthetic clips")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CKPT / f"{a.ckpt}.pt", map_location=dev,
                      weights_only=False)
    cfg = blob["cfg"]
    if not cfg.get("finetune"):
        raise SystemExit(f"{a.ckpt} is not a fine-tuned checkpoint; use submit.py")

    print(f"checkpoint {a.ckpt}")
    print(f"  decide : {blob['decide']['combined']:.4f}")
    if "all" in blob:
        print(f"  ALL    : {blob['all']['combined']:.4f}   "
              f"(incumbent 1.1190 -> leaderboard 1.1835)")
    print(f"  readout: {blob['readout']}")

    net = build_net(cfg, dev)
    net.load_state_dict(blob["state"])
    net.eval()

    arr, meta = A.load_audio_store(AUD, a.test_store)
    idx = meta["clips"]
    print(f"running inference on {len(idx):,} test clips "
          f"({a.test_store}, crop {cfg['crop']}s) ...", flush=True)
    post, dist, _ = FT.clip_posteriors(net, arr, idx, list(idx), cfg["crop"], dev,
                                       ast=ast_for(cfg, a.test_store, aux_root(cfg)))

    rcfg = dict(blob["readout"])
    rcfg.setdefault("kind", "global")
    preds = T.decode_all(post, rcfg, dist, list(post))

    if not a.no_bias_correct:
        # EXP-27: synthetic spans come out ~0.075 s too NARROW on top of a tight
        # std (0.069); natural spans have ~zero bias and a std of 0.275. So the
        # correction only belongs on synthetic clips, and routing beats one
        # global shift (+0.0313 vs +0.0157 on the decide half, against an oracle
        # ceiling of +0.0334).
        clf = pickle.load(open(CKPT / "synthid2.pkl", "rb"))
        ids = list(preds)
        X = np.stack([np.concatenate([
            SI.clip_features(A.get_wav(arr, idx[c])), _post_feat(post[c])])
            for c in ids])
        p_syn = clf.predict_proba(X)
        n_syn = int((p_syn >= 0.5).sum())
        print(f"  synthetic routing: {n_syn:,}/{len(ids):,} clips "
              f"({100*n_syn/len(ids):.1f}%; the organizers state 37.7%)")
        preds = {c: R.routed_correct(preds[c], 1.0 if ps >= 0.5 else 0.0,
                                     syn=tuple(a.syn_shift))
                 for c, ps in zip(ids, p_syn)}

    want = set(D.list_test_clip_ids())
    write_zip(preds, Path(a.out) if a.out else ART / f"submission_{a.ckpt}.zip", want)


if __name__ == "__main__":
    main()
