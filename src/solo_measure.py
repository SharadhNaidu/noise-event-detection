"""Solo (and small-ensemble) natural-clip measurement on the held-out DECIDE half.

One self-contained instrument, so every run is judged the same way:
  * DECIDE = the speaker-disjoint half of validation that no TUNE-trained member saw
    (splits.split_by_speaker, frac 0.5, seed 42; the other half is valtune_audio).
  * Decode = the champion natural route (md02 readout: asym 0.6/0.7, min_dur 0.2, refine 0.2,
    ch7 overlay 0.55 / ratio 4, empty-clip fallback q 0.7, clip to duration).
  * Two populations are reported, because the 10-09 instrument check showed they disagree:
      plain      = every DECIDE natural clip once
      matched    = DECIDE resampled to the TEST clip-length mix (test_cells.json)
  * Paired speaker-group bootstrap (boot_ens.group_bootstrap) for A - B.

    python src/solo_measure.py v12bn_ep2 v12_ep2            # A vs B, solo
    python src/solo_measure.py hon_A+bts_C+v12 v12 --n 300   # ensemble vs solo
Posteriors are read from $INDOML_ROOT/ckpt/post8_val_audio_<tag>.pkl (then srvcache/).
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiostore as A        # noqa: E402
import boot_ens as B          # noqa: E402
import data as D              # noqa: E402
import ens_gate as E          # noqa: E402
import postproc as PP         # noqa: E402
import scorer as S            # noqa: E402
import splits as SP           # noqa: E402
import submit_ens as SE       # noqa: E402

ROOT = Path(str(P.ROOT))
MD02 = {
    "kind": "perpop",
    "natural": {"kind": "asym", "thr_on": 0.6, "thr_off": 0.7, "min_dur": 0.2, "merge_gap": 0.06, "refine": 0.2},
    "synthetic": {"kind": "asym", "thr_on": 0.6, "thr_off": 0.7, "min_dur": 0.08, "merge_gap": 0.06, "refine": 0.3},
    "overlay": {"ch": 7, "thr": 0.55, "ratio": 4.0, "natural_only": True},
}
BINS = ((0, 5), (5, 10), (10, 1e9))


def _len_bin(d):
    return next(i for i, (a, b) in enumerate(BINS) if a <= d < b)


def test_cells():
    """Share of TEST clips per (natural, length-bin) cell; cached by the 10-01 session."""
    p = ROOT / "ckpt" / "test_cells.json"
    return {tuple(json.loads(k)): v for k, v in json.load(open(p)).items()}


def matched_ids(draw, syn, dur, seed):
    sh = test_cells(); rng = np.random.default_rng(seed); out = []
    cells = {}
    for c in draw:
        cells.setdefault((not syn[c], _len_bin(dur[c])), []).append(c)
    for k, share in sh.items():
        pool = cells.get(k)
        if pool:
            out += list(rng.choice(pool, int(round(len(draw) * share)), replace=True))
    return out


class Decide:
    def __init__(self):
        self.refs, self.meta = D.load_validation()
        _tune, decide = SP.split_by_speaker(list(self.refs), frac=0.5, seed=42)
        self.ids = list(decide)
        arr, me = A.load_audio_store(ROOT / "audio", "val_audio")
        self.nat = SE.natural_by_peak(arr, me["clips"], self.ids)
        self.dur = {c: me["clips"][c]["n"] / A.SR for c in self.ids}
        self.syn = {c: bool(self.meta[c]["syntheticData"]) for c in self.ids}
        self.nids = [c for c in self.ids if not self.syn[c]]
        self.groups = B.speaker_groups(self.nids)
        self.cache = {}
        # the peak rule and the metadata must agree, or the route is wrong
        bad = [c for c in self.ids if self.nat[c] == self.syn[c]]
        assert not bad, f"peak routing disagrees with metadata on {len(bad)} clips"

    def post(self, tag):
        if tag not in self.cache:
            for d in ("ckpt", "srvcache"):
                p = ROOT / d / f"post8_val_audio_{tag}.pkl"
                if p.exists():
                    self.cache[tag] = pickle.load(open(p, "rb")); break
            else:
                raise FileNotFoundError(f"no val posteriors for {tag}")
        return self.cache[tag]

    def natural_preds(self, members, readout=MD02):
        """Champion natural route on the DECIDE natural clips for a member list."""
        M = [self.post(t) for t in members]
        ids = self.nids
        p8 = E.average([{c: m["post"][c] for c in ids} for m in M])
        dd = E.average([{c: m["dist"][c] for c in ids} for m in M])
        return self.natural_preds_from(p8, dd, readout)

    def natural_preds_from(self, p8, dd, readout=MD02):
        """Same route from already-averaged (T, 8) posteriors and (T, 2) distances, keyed by clip."""
        ids = self.nids
        post = {c: p8[c][:, 0] for c in ids}
        p = SE.decode_routed(post, dd, ids, readout, self.nat)
        p = SE.apply_overlay(p, p8, ids, readout.get("overlay"), self.nat)
        p = PP.clip_to_duration(PP.empty_clip_fallback(p, post, q=0.7), self.dur)
        return {c: [tuple(e[:2]) for e in p[c] if e[1] > e[0]] for c in ids}

    def point(self, preds, ids=None):
        ids = ids or self.nids
        s = S.score({c: self.refs[c] for c in ids}, {c: preds[c] for c in ids})
        return s["event_f1"], s["segment_dice"]

    def compare(self, pa, pb, n=300, seed=0):
        """Paired speaker-group bootstrap of A - B on natural F1 and Dice, plain and test-matched."""
        out = {}
        for pop in ("plain", "matched"):
            def stat(pr, key):
                def f(draw):
                    t = draw if pop == "plain" else [c for c in matched_ids(draw, self.syn, self.dur, len(draw)) if not self.syn[c]]
                    return B.resample_score(self.refs, pr, t)[key]
                return f
            for key in ("event_f1", "segment_dice"):
                out[(pop, key)] = B.group_bootstrap(self.groups, stat(pa, key), stat(pb, key), n=n, seed=seed)
        return out


def fmt(r):
    return f"{r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] P{r['p']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="member tag, or tags joined with '+' for a mean ensemble")
    ap.add_argument("b", nargs="?", help="paired reference (same syntax); omit for a point estimate only")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    dec = Decide()
    print(f"DECIDE: {len(dec.ids)} clips, {len(dec.nids)} natural, {len(dec.groups)} speaker groups", flush=True)
    pa = dec.natural_preds(a.a.split("+"))
    fa, da = dec.point(pa)
    print(f"{a.a:28s} natural F1 {fa:.4f}  Dice {da:.4f}", flush=True)
    if not a.b:
        return
    pb = dec.natural_preds(a.b.split("+"))
    fb, db = dec.point(pb)
    print(f"{a.b:28s} natural F1 {fb:.4f}  Dice {db:.4f}", flush=True)
    r = dec.compare(pa, pb, n=a.n)
    for pop in ("plain", "matched"):
        print(f"  {pop:8s} A-B: natural F1 {fmt(r[(pop, 'event_f1')])} | natural Dice {fmt(r[(pop, 'segment_dice')])}", flush=True)


if __name__ == "__main__":
    main()
