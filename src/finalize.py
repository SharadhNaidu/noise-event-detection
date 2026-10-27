"""Score a fine-tuned checkpoint with the full decode stack, and optionally
build the submission.

The stack, in the order it is applied, with what each part bought:

  1. the model's own swept readout                     (train.py)
  2. PER-POPULATION readout, natural vs synthetic      +0.0437 decide
  3. routed boundary bias correction on synthetic      (inside 2's measurement)

Steps 2 and 3 are fitted on the TUNE half only and reported on DECIDE, so the
number this prints is honest. ALL is printed too because that is the quantity
the leaderboard offset was calibrated against.

    python src/finalize.py --ckpt nat_val
    python src/finalize.py --ckpt nat_val --submit sunny/submission.zip
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
import audiostore as A       # noqa: E402
import data as D             # noqa: E402
import finetune as FT        # noqa: E402
import poproutout as PR      # noqa: E402
import readout as R          # noqa: E402
import scorer as S           # noqa: E402
import splits as SP          # noqa: E402
import synthid as SI         # noqa: E402
import train as T            # noqa: E402

CKPT = (P.ROOT / "ckpt")
AUD = (P.ROOT / "audio")
REPO = Path(__file__).resolve().parent.parent


def load_net(tag, dev):
    blob = torch.load(CKPT / f"{tag}.pt", map_location=dev, weights_only=False)
    cfg = blob["cfg"]
    net = FT.FineTuneSED(cfg["enc_layers"],
                         boundary_head=not cfg.get("no_boundary_head", False),
                         freeze_conv=not cfg.get("unfreeze_conv", False)).to(dev)
    net.load_state_dict(blob["state"])
    net.eval()
    return net, cfg, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--submit", default=None, help="path for the submission zip")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg, blob = load_net(a.ckpt, dev)
    print(f"{a.ckpt}: enc_layers={cfg['enc_layers']} crop={cfg['crop']}s "
          f"epochs={cfg['epochs']} stores={cfg.get('train_stores')}")
    print(f"  swept readout: {blob['readout']}")

    refs, _ = D.load_validation()
    tune, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    varr, vmeta = A.load_audio_store(AUD, cfg["val_store"])
    vidx = vmeta["clips"]
    tune = [c for c in tune if c in vidx]
    decide = [c for c in decide if c in vidx]
    post, dist, _ = FT.clip_posteriors(net, varr, vidx, tune + decide,
                                       cfg["crop"], dev)

    z = zipfile.ZipFile(REPO / "data" / "validation.zip")
    syn = {os.path.basename(n)[:-4] for n in z.namelist()
           if n.endswith(".wav") and "__MACOSX" not in n
           and "syntheticNoiseAudio/" in n}

    def decode_one(c, rc):
        ev = T.apply_readout(post[c], rc)
        if rc.get("split") and c in dist:
            ev = R.split_on_crossings(ev, dist[c])
        if rc.get("refine"):
            ev = R.refine(ev, dist[c], radius=rc["refine"])
        if rc.get("dilate"):
            ev = R.bias_correct(ev, -rc["dilate"] / 2, rc["dilate"] / 2)
        return ev

    # ---- baseline: the model's own single swept readout --------------------
    rc0 = dict(blob["readout"])
    rc0.setdefault("kind", "global")
    base = S.score({c: refs[c] for c in decide},
                   {c: decode_one(c, rc0) for c in decide})

    # ---- fit the router and the per-population readouts on TUNE ONLY -------
    Xt = np.stack([np.concatenate([SI.clip_features(A.get_wav(varr, vidx[c])),
                                   PR.post_features(post[c])]) for c in tune])
    clf = SI.SynthClassifier().fit(
        Xt, np.array([1 if c in syn else 0 for c in tune]))
    tn = [c for c in tune if c not in syn]
    ts = [c for c in tune if c in syn]
    cfn, _ = PR.sweep_population(tn, post, dist, refs, decode_one, S)
    cfs, _ = PR.sweep_population(ts, post, dist, refs, decode_one, S)
    print(f"  natural readout  : {cfn}")
    print(f"  synthetic readout: {cfs}")
    PR.save(a.ckpt, cfn, cfs)

    def full(ids, probs):
        out = {}
        for c in ids:
            is_syn = probs[c] >= 0.5
            ev = decode_one(c, cfs if is_syn else cfn)
            out[c] = R.routed_correct(ev, 1.0 if is_syn else 0.0)
        return out

    Xd = np.stack([np.concatenate([SI.clip_features(A.get_wav(varr, vidx[c])),
                                   PR.post_features(post[c])]) for c in decide])
    pd_ = dict(zip(decide, clf.predict_proba(Xd)))
    held = S.score({c: refs[c] for c in decide}, full(decide, pd_))
    allp = dict(pd_)
    allp.update(dict(zip(tune, clf.predict_proba(Xt))))
    alls = S.score({c: refs[c] for c in tune + decide},
                   full(tune + decide, allp))

    print()
    print("=" * 70)
    print(f"  {'decode':<32}{'F1':>9}{'Dice':>9}{'combined':>11}")
    print("  " + "-" * 61)
    print(f"  {'single swept readout':<32}{base['event_f1']:>9.4f}"
          f"{base['segment_dice']:>9.4f}{base['combined']:>11.4f}   DECIDE")
    print(f"  {'+ per-population + bias':<32}{held['event_f1']:>9.4f}"
          f"{held['segment_dice']:>9.4f}{held['combined']:>11.4f}   DECIDE "
          f"({held['combined']-base['combined']:+.4f})")
    print(f"  {'+ per-population + bias':<32}{alls['event_f1']:>9.4f}"
          f"{alls['segment_dice']:>9.4f}{alls['combined']:>11.4f}   ALL")
    print("=" * 70)
    print("  reference points: incumbent ALL 1.1190 -> leaderboard 1.1835;")
    print("  nat_L4 ALL 1.3718 -> leaderboard 1.4843; rank 10 is 1.5576")

    # SAFETY NET, matching build_final.py. The per-population routing has now
    # lost three times in a row and the pattern is clear: it helps a weak model
    # and hurts a strong one (+0.0437 on nat_L4 at decide 1.3161, -0.0804 on
    # nat_val at 1.3611, -0.0213 on hon_A at 1.3921). As the base model
    # improves the single swept readout approaches optimal and the extra
    # machinery only adds variance. Never ship a decode that loses on the
    # honest half.
    plain = S.score({c: refs[c] for c in decide},
                    {c: dec_one(c, rc0) for c in decide})
    if plain["combined"] > held["combined"]:
        print(f"  !! routing LOSES by {plain['combined']-held['combined']:.4f}"
              f" on DECIDE - falling back to the plain swept readout")
        use_plain = True
    else:
        use_plain = False

    if not a.submit:
        return

    # ---------------------------------------------------------------- submit
    import json
    tarr, tmeta = A.load_audio_store(AUD, "test_audio")
    tidx = tmeta["clips"]
    print(f"\nrunning inference on {len(tidx):,} test clips ...", flush=True)
    tpost, tdist, _ = FT.clip_posteriors(net, tarr, tidx, list(tidx),
                                         cfg["crop"], dev)
    # refit the router on ALL validation for deployment
    Xa = np.concatenate([Xt, Xd])
    ya = np.array([1 if c in syn else 0 for c in tune + decide])
    clf_all = SI.SynthClassifier().fit(Xa, ya)
    ids = list(tpost)
    Xtest = np.stack([np.concatenate([
        SI.clip_features(A.get_wav(tarr, tidx[c])),
        PR.post_features(tpost[c])]) for c in ids])
    p_syn = clf_all.predict_proba(Xtest)
    print(f"  synthetic routing: {int((p_syn>=0.5).sum()):,}/{len(ids):,} "
          f"({100*(p_syn>=0.5).mean():.1f}%; organizers state 37.7%)")

    def decode_test(c, rc):
        ev = T.apply_readout(tpost[c], rc)
        if rc.get("split") and c in tdist:
            ev = R.split_on_crossings(ev, tdist[c])
        if rc.get("refine"):
            ev = R.refine(ev, tdist[c], radius=rc["refine"])
        if rc.get("dilate"):
            ev = R.bias_correct(ev, -rc["dilate"] / 2, rc["dilate"] / 2)
        return ev

    preds = {}
    for c, ps in zip(ids, p_syn):
        if use_plain:
            preds[c] = decode_test(c, rc0)
            continue
        is_syn = ps >= 0.5
        preds[c] = R.routed_correct(decode_test(c, cfs if is_syn else cfn),
                                    1.0 if is_syn else 0.0)

    want = set(D.list_test_clip_ids())
    if want != set(preds):
        raise SystemExit(f"clip_id mismatch: {len(want)} vs {len(preds)}")
    out = REPO / "artifacts" / a.submit
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"clip_id": c,
                         "events": [{"onset": round(float(s), 3),
                                     "offset": round(float(e), 3)}
                                    for s, e in preds[c] if e > s]},
                        ensure_ascii=False) for c in sorted(preds)]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.jsonl", "\n".join(lines) + "\n")
    with zipfile.ZipFile(out) as zf:
        rows = [json.loads(l) for l in
                zf.read("predictions.jsonl").decode("utf-8").splitlines() if l]
    if {r["clip_id"] for r in rows} != want:
        raise SystemExit("re-read of the archive does not match the test ids")
    n_ev = sum(len(r["events"]) for r in rows)
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(rows):,} clips | {n_ev:,} events | {n_ev/len(rows):.2f} "
          f"per clip | {sum(1 for r in rows if not r['events'])} empty")


if __name__ == "__main__":
    main()
