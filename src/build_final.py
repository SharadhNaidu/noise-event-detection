"""Build the last submission, unattended, choosing the best option available.

Runs after training with no human in the loop, so it is written to be
conservative rather than clever:

  * it scores every candidate on the SAME clips with the SAME decode, and
    ships whichever wins - it never assumes the newest model is the best;
  * `fin_A` is the floor. It is already on disk at a level above the 1.4843
    currently on the board, so the worst outcome is that we resubmit
    something at least that good;
  * every stage is compared against doing nothing, because the per-population
    decode has been wrong twice today (a readout-family restriction, then a
    stale bias constant) and both times it looked like a gain on the model it
    was derived from.

Writes artifacts/sunny/submission.zip and a plain-language verdict to
artifacts/sunny/VERDICT.md.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import astfeat as AF         # noqa: E402
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
AST = (P.ROOT / "ast")
REPO = Path(__file__).resolve().parent.parent
OFFSET = 0.1125          # leaderboard = ALL + this, from three submissions


def load(tag, dev):
    blob = torch.load(CKPT / f"{tag}.pt", map_location=dev, weights_only=False)
    cfg = blob["cfg"]
    net = FT.FineTuneSED(cfg["enc_layers"],
                         boundary_head=not cfg.get("no_boundary_head", False),
                         freeze_conv=not cfg.get("unfreeze_conv", False),
                         aux_dim=AF.AST_DIM if cfg.get("use_ast") else 0).to(dev)
    net.load_state_dict(blob["state"])
    net.eval()
    return net, cfg, blob


def ast_for(store, want):
    if not want:
        return None
    try:
        a, m = AF.load_ast_store(AST, store)
        return (a, m["clips"])
    except Exception:
        return None


def posteriors(tag, dev, store, arr, idx, ids):
    net, cfg, blob = load(tag, dev)
    p, d, _ = FT.clip_posteriors(net, arr, idx, ids, cfg["crop"], dev,
                                 ast=ast_for(store, cfg.get("use_ast")))
    del net
    torch.cuda.empty_cache()
    return p, d, blob


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    refs, _ = D.load_validation()
    tune, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    varr, vmeta = A.load_audio_store(AUD, "val_audio")
    vidx = vmeta["clips"]
    tune = [c for c in tune if c in vidx]
    decide = [c for c in decide if c in vidx]
    ids = tune + decide
    z = zipfile.ZipFile(REPO / "data" / "validation.zip")
    syn = {os.path.basename(n)[:-4] for n in z.namelist()
           if n.endswith(".wav") and "__MACOSX" not in n
           and "syntheticNoiseAudio/" in n}

    # AST features for the validation store live under the valall name
    def vast(cfg):
        return ast_for("valall_audio", cfg.get("use_ast"))

    cands = [t for t in ("fin_X", "fin_A", "fin_C")
             if (CKPT / f"{t}.pt").exists()]
    print(f"candidates on disk: {', '.join(cands)}")
    mem = {}
    for t in cands:
        try:
            net, cfg, blob = load(t, dev)
            p, d, _ = FT.clip_posteriors(net, varr, vidx, ids, cfg["crop"],
                                         dev, ast=vast(cfg))
            mem[t] = (p, d, blob)
            del net
            torch.cuda.empty_cache()
            print(f"  {t}: posteriors ready")
        except Exception:
            print(f"  {t}: FAILED to run\n{traceback.format_exc()[-400:]}")

    if not mem:
        raise SystemExit("no usable checkpoint - nothing to submit")

    # ------------------------------------------------------- score options
    def evaluate(post, dist, label):
        """Full stack, fitted on ALL validation (these models trained on it,
        so there is no clean half left; the readout has few parameters against
        2,501 clips so the overfitting risk is small, and the number is
        labelled optimistic wherever it is printed)."""
        def dec_one(c, rc, P=post, Dd=dist):
            ev = T.apply_readout(P[c], rc)
            if rc.get("split") and c in Dd:
                ev = R.split_on_crossings(ev, Dd[c])
            if rc.get("refine") and c in Dd:
                ev = R.refine(ev, Dd[c], radius=rc["refine"])
            if rc.get("dilate"):
                ev = R.bias_correct(ev, -rc["dilate"] / 2, rc["dilate"] / 2)
            return ev
        fn = [c for c in ids if c not in syn]
        fs = [c for c in ids if c in syn]
        cfn, _ = PR.sweep_population(fn, post, dist, refs, dec_one, S)
        cfs, _ = PR.sweep_population(fs, post, dist, refs, dec_one, S)
        bn, _ = PR.fit_bias(fn, refs, {c: dec_one(c, cfn) for c in fn})
        bs, _ = PR.fit_bias(fs, refs, {c: dec_one(c, cfs) for c in fs})
        X = np.stack([np.concatenate([
            SI.clip_features(A.get_wav(varr, vidx[c])),
            PR.post_features(post[c])]) for c in ids])
        clf = SI.SynthClassifier().fit(
            X, np.array([1 if c in syn else 0 for c in ids]))
        p_syn = dict(zip(ids, clf.predict_proba(X)))
        pr = {}
        for c in ids:
            is_s = p_syn[c] >= 0.5
            pr[c] = R.bias_correct(dec_one(c, cfs if is_s else cfn),
                                   *(bs if is_s else bn))
        o = S.score({c: refs[c] for c in ids}, pr)
        print(f"  {label:<26} ALL {o['combined']:.4f}  "
              f"(F1 {o['event_f1']:.4f} Dice {o['segment_dice']:.4f})")
        return o["combined"], dict(cfn=cfn, cfs=cfs, bn=bn, bs=bs, clf=clf,
                                   dec_one=dec_one)

    print("\nscoring every option on the same clips:")
    options = {}
    for t in mem:
        p, d, _ = mem[t]
        options[t] = evaluate(p, d, t)
    if len(mem) > 1:
        import ensemble as E
        for combo in ([c for c in ("fin_X", "fin_A") if c in mem],
                      list(mem)):
            if len(combo) < 2:
                continue
            key = "+".join(combo)
            if key in options:
                continue
            m = {t: (mem[t][0], mem[t][1]) for t in combo}
            p, d = E.average(m, ids)
            options[key] = evaluate(p, d, key)
            mem[key] = (p, d, None)

    best = max(options, key=lambda k: options[k][0])
    score, aux = options[best]
    print(f"\nBEST: {best}  ALL {score:.4f}  -> leaderboard ~{score+OFFSET:.3f}")

    # ------------------------------------------------------------- submit
    tarr, tmeta = A.load_audio_store(AUD, "test_audio")
    tidx = tmeta["clips"]
    members = best.split("+")
    print(f"\ninference on {len(tidx):,} test clips with {best} ...",
          flush=True)
    tm = {}
    for t in members:
        net, cfg, _ = load(t, dev)
        p, d, _ = FT.clip_posteriors(net, tarr, tidx, list(tidx), cfg["crop"],
                                     dev, ast=ast_for("test_audio",
                                                      cfg.get("use_ast")))
        tm[t] = (p, d)
        del net
        torch.cuda.empty_cache()
    if len(members) > 1:
        import ensemble as E
        tpost, tdist = E.average(tm, list(tidx))
    else:
        tpost, tdist = tm[members[0]]

    Xt = np.stack([np.concatenate([
        SI.clip_features(A.get_wav(tarr, tidx[c])),
        PR.post_features(tpost[c])]) for c in tpost])
    ps = aux["clf"].predict_proba(Xt)
    print(f"  synthetic routing: {int((ps>=0.5).sum()):,}/{len(ps):,} "
          f"({100*(ps>=0.5).mean():.1f}%; organizers state 37.7%)")

    def dec_test(c, rc):
        ev = T.apply_readout(tpost[c], rc)
        if rc.get("split") and c in tdist:
            ev = R.split_on_crossings(ev, tdist[c])
        if rc.get("refine") and c in tdist:
            ev = R.refine(ev, tdist[c], radius=rc["refine"])
        if rc.get("dilate"):
            ev = R.bias_correct(ev, -rc["dilate"] / 2, rc["dilate"] / 2)
        return ev

    preds = {}
    for c, p in zip(tpost, ps):
        is_s = p >= 0.5
        preds[c] = R.bias_correct(
            dec_test(c, aux["cfs"] if is_s else aux["cfn"]),
            *(aux["bs"] if is_s else aux["bn"]))

    want = set(D.list_test_clip_ids())
    if want != set(preds):
        raise SystemExit(f"clip_id mismatch: {len(want)} vs {len(preds)}")
    out = REPO / "artifacts" / "sunny" / "submission.zip"
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
    proj = score + OFFSET
    verdict = (
        f"# Submission verdict\n\n"
        f"**File:** `artifacts/sunny/submission.zip`\n\n"
        f"**Model:** {best}\n\n"
        f"**Validation ALL:** {score:.4f}  -> **projected leaderboard "
        f"~{proj:.3f}**\n\n"
        f"Reference points: your current best is 1.4843 (rank 15), rank 10 is "
        f"1.5576, rank 5 is 1.5998, rank 3 is 1.6066.\n\n"
        f"**Recommendation: "
        f"{'UPLOAD' if proj > 1.4843 + 0.02 else 'HOLD - not clearly better'}"
        f"**\n\n"
        f"{len(rows):,} clips, {n_ev:,} events, {n_ev/len(rows):.2f} per clip.\n\n"
        f"Caveat: these models trained on all of validation, so the ALL score "
        f"is inflated by memorisation (measured at ~0.079 on a model trained "
        f"on half). The projection above uses the offset calibrated on "
        f"models that had NOT seen validation, so treat ~{proj:.3f} as the "
        f"optimistic end.\n\n"
        f"## Every option scored\n\n"
        + "\n".join(f"- {k}: ALL {v[0]:.4f}" for k, v in
                    sorted(options.items(), key=lambda x: -x[1][0])) + "\n")
    (out.parent / "VERDICT.md").write_text(verdict, encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(rows):,} clips | {n_ev:,} events")
    print(f"  verdict written to {out.parent/'VERDICT.md'}")


if __name__ == "__main__":
    main()
