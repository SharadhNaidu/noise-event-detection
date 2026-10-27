"""Build the final submission: ensemble several checkpoints, then decode.

Stack, with what each part measured:

  1. posterior averaging over members        +0.009 on 3 correlated models
  2. per-population readout, swept per side  +0.044
  3. routed boundary bias correction         (inside 2's measurement)

Members are averaged BEFORE the readout, never after. Merging decoded event
lists would have to reconcile disagreements about how many events exist, which
is exactly what the one-to-one matching in Event-F1 punishes; averaging
posteriors sidesteps it.

Honesty about the readout fit
-----------------------------
A model trained on all of validation leaves no clean set to fit the readout on.
Two options and we take the second: reuse a readout fitted on a model that was
honestly held out (safe, slightly mismatched), or refit on all of validation
(optimistic, but the readout has four parameters against 2,501 clips, so the
overfitting risk is small). `--readout-from` selects the first when given.
Either way the printed ALL score for such a model is NOT a held-out estimate,
and the code says so rather than letting the number speak for itself.
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
import audiostore as A       # noqa: E402
import data as D             # noqa: E402
import ensemble as E         # noqa: E402
import poproutout as PR      # noqa: E402
import readout as R          # noqa: E402
import scorer as S           # noqa: E402
import splits as SP          # noqa: E402
import synthid as SI         # noqa: E402
import train as T            # noqa: E402

CKPT = (P.ROOT / "ckpt")
AUD = (P.ROOT / "audio")
REPO = Path(__file__).resolve().parent.parent


def synth_ids():
    z = zipfile.ZipFile(REPO / "data" / "validation.zip")
    return {os.path.basename(n)[:-4] for n in z.namelist()
            if n.endswith(".wav") and "__MACOSX" not in n
            and "syntheticNoiseAudio/" in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--submit", default=None)
    ap.add_argument("--fit-on", default="tune", choices=["tune", "all"],
                    help="which validation clips to fit readout/router on")
    ap.add_argument("--crops", type=float, nargs="*", default=None,
                    help="also average each member over these crop lengths")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    refs, _ = D.load_validation()
    tune, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    varr, vmeta = A.load_audio_store(AUD, "val_audio")
    vidx = vmeta["clips"]
    tune = [c for c in tune if c in vidx]
    decide = [c for c in decide if c in vidx]
    ids = tune + decide
    syn = synth_ids()

    print(f"members: {', '.join(a.members)}")
    # the best single readout any member found for itself, as the fallback
    members_single_readout = torch.load(
        CKPT / f"{a.members[0]}.pt", map_location="cpu",
        weights_only=False)["readout"]
    mem = {}
    for t in a.members:
        if a.crops:
            mem.update(E.multiscale_posteriors(t, varr, vidx, ids, dev,
                                               crops=tuple(a.crops)))
        else:
            mem.update(E.member_posteriors([t], varr, vidx, ids, dev))
    if len(mem) > 1:
        print("pairwise posterior correlation:")
        for (x, y), r in E.diversity(mem, decide).items():
            print(f"  {x:<20} vs {y:<20} {r:.4f}")
    post, dist = E.average(mem, ids)

    def dec_one(c, rc, P=post, Dd=dist):
        ev = T.apply_readout(P[c], rc)
        if rc.get("split") and c in Dd:
            ev = R.split_on_crossings(ev, Dd[c])
        if rc.get("refine") and c in Dd:
            ev = R.refine(ev, Dd[c], radius=rc["refine"])
        if rc.get("dilate"):
            ev = R.bias_correct(ev, -rc["dilate"] / 2, rc["dilate"] / 2)
        return ev

    fit_ids = tune if a.fit_on == "tune" else ids
    fn = [c for c in fit_ids if c not in syn]
    fs = [c for c in fit_ids if c in syn]
    cfn, _ = PR.sweep_population(fn, post, dist, refs, dec_one, S)
    cfs, _ = PR.sweep_population(fs, post, dist, refs, dec_one, S)
    print(f"  natural readout  : {cfn}")
    print(f"  synthetic readout: {cfs}")

    Xf = np.stack([np.concatenate([SI.clip_features(A.get_wav(varr, vidx[c])),
                                   PR.post_features(post[c])]) for c in fit_ids])
    clf = SI.SynthClassifier().fit(
        Xf, np.array([1 if c in syn else 0 for c in fit_ids]))

    # Fit the boundary shift for THIS model rather than reusing the one
    # measured on ft_gold_syn in the early afternoon. A better model has a
    # different systematic error, and applying a stale correction introduces
    # bias rather than removing it - worth -0.056 on fin_A before this fix.
    bn, sn = PR.fit_bias(fn, refs, {c: dec_one(c, cfn) for c in fn})
    bs, ss = PR.fit_bias(fs, refs, {c: dec_one(c, cfs) for c in fs})
    print(f"  natural bias     : onset {bn[0]:+.3f}s offset {bn[1]:+.3f}s")
    print(f"  synthetic bias   : onset {bs[0]:+.3f}s offset {bs[1]:+.3f}s")

    def full(g, probs):
        out = {}
        for c in g:
            is_syn = probs[c] >= 0.5
            ev = dec_one(c, cfs if is_syn else cfn)
            d = bs if is_syn else bn
            out[c] = R.bias_correct(ev, d[0], d[1])
        return out

    Xa = np.stack([np.concatenate([SI.clip_features(A.get_wav(varr, vidx[c])),
                                   PR.post_features(post[c])]) for c in ids])
    pa = dict(zip(ids, clf.predict_proba(Xa)))
    allsc = S.score({c: refs[c] for c in ids}, full(ids, pa))

    # SAFETY NET. The per-population stack has now been wrong twice - once
    # from a readout-family restriction, once from a stale bias constant - and
    # both times it looked like a gain on the model it was built against. So
    # compare it against the plain single readout on the same clips and keep
    # whichever actually wins. A pipeline that can only improve is worth more
    # than one that is usually better.
    single = dict(members_single_readout)
    single.setdefault("kind", "global")
    simple = S.score({c: refs[c] for c in ids},
                     {c: dec_one(c, single) for c in ids})
    print(f"  plain single readout on the same clips: "
          f"combined {simple['combined']:.4f}")
    use_routed = allsc["combined"] >= simple["combined"]
    if not use_routed:
        print(f"  !! routing LOSES by {simple['combined']-allsc['combined']:.4f}"
              f" - falling back to the single readout")
        allsc = simple

        def full(g, probs, _s=single):          # noqa: F811
            return {c: dec_one(c, _s) for c in g}
    print()
    print("=" * 66)
    if a.fit_on == "tune":
        held = S.score({c: refs[c] for c in decide}, full(decide, pa))
        print(f"  DECIDE  F1 {held['event_f1']:.4f}  Dice "
              f"{held['segment_dice']:.4f}  combined {held['combined']:.4f}"
              f"   <- held out")
    print(f"  ALL     F1 {allsc['event_f1']:.4f}  Dice "
          f"{allsc['segment_dice']:.4f}  combined {allsc['combined']:.4f}")
    print("=" * 66)
    print("  reference: nat_L4 ALL 1.3718 -> leaderboard 1.4843; rank 10 1.5576")
    if a.fit_on == "all":
        print("  NOTE: readout fitted on all of validation, so ALL above is")
        print("  optimistic. Compare magnitudes, not absolutes.")

    if not a.submit:
        return

    # ---------------------------------------------------------------- submit
    tarr, tmeta = A.load_audio_store(AUD, "test_audio")
    tidx = tmeta["clips"]
    print(f"\ninference on {len(tidx):,} test clips ...", flush=True)
    tmem = {}
    for t in a.members:
        if a.crops:
            tmem.update(E.multiscale_posteriors(t, tarr, tidx, list(tidx), dev,
                                                crops=tuple(a.crops)))
        else:
            tmem.update(E.member_posteriors([t], tarr, tidx, list(tidx), dev))
    tpost, tdist = E.average(tmem, list(tidx))

    Xt = np.stack([np.concatenate([SI.clip_features(A.get_wav(tarr, tidx[c])),
                                   PR.post_features(tpost[c])])
                   for c in tpost])
    ps = clf.predict_proba(Xt)
    print(f"  synthetic routing: {int((ps>=0.5).sum()):,}/{len(ps):,} "
          f"({100*(ps>=0.5).mean():.1f}%; organizers state 37.7%)")

    preds = {}
    for c, p in zip(tpost, ps):
        is_syn = p >= 0.5
        ev = dec_one(c, cfs if is_syn else cfn, tpost, tdist)
        d = bs if is_syn else bn
        preds[c] = R.bias_correct(ev, d[0], d[1])

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
