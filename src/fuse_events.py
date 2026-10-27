"""Event-level fusion of per-member decodes (2026-10-09). Motivation (diagnose.py): the union of the six members hits 55.6% of
DECIDE natural refs, the averaged ensemble 38.7%. Averaging posteriors blurs edges and fills the dips between events.

Pool = events decoded from every member alone (+ optionally the ensemble's events). Greedy clustering by IoU; a cluster is kept
if it has >= `support` distinct sources. Its edges come from a selector:
  median  weighted median of the cluster's onsets/offsets (weights = source count)
  sharp   the source whose own ch0 posterior is steepest around its two edges (sum |dp/dt| within +-0.1 s)
  conf    the source with the highest mean ch0 posterior inside its event
  base    the ensemble event when it is in the cluster, else median
Scored with the official metric on the half named by --half (decide | tune): fit knobs on TUNE, confirm on DECIDE.
    python src/fuse_events.py --half decide --support 2,3,4 --iou 0.3,0.5 --sel median,sharp,conf,base
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer as S                      # noqa: E402
import splits as SP                     # noqa: E402
from solo_measure import Decide         # noqa: E402

FS = 0.02


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def wmedian(x, w):
    o = np.argsort(x); c = np.cumsum(np.asarray(w)[o]); return float(np.asarray(x)[o][np.searchsorted(c, 0.5 * c[-1])])


def sharpness(p, ev):
    """sum of |dp/dt| in +-0.1 s around onset and offset of ev on posterior track p (20 ms frames)."""
    d = np.abs(np.diff(p)); T = len(d); s = 0.0
    for t in ev[:2]:
        i = int(round(t / FS)); s += float(d[max(0, i - 5): min(T, i + 5)].sum())
    return s


def fuse_clip(sources, posts, support, thr_iou, sel, min_dur=0.06):
    """sources: {name: [(on, off), ...]}; posts: {name: ch0 track}. Returns fused [(on, off)]."""
    pool = [(name, ev) for name, evs in sources.items() for ev in evs]
    # score each candidate for greedy ordering: mean posterior inside its own track
    def conf(name, ev):
        p = posts[name]; a, b = int(ev[0] / FS), max(int(ev[0] / FS) + 1, int(ev[1] / FS))
        return float(p[a:b].mean()) if b <= len(p) else float(p[a:].mean() if len(p[a:]) else 0)
    pool.sort(key=lambda x: -conf(*x))
    used = [False] * len(pool); out = []
    for i, (ni, ei) in enumerate(pool):
        if used[i]:
            continue
        members = [(ni, ei)]; used[i] = True
        for j in range(i + 1, len(pool)):
            if not used[j] and pool[j][0] not in {m[0] for m in members} and iou(ei, pool[j][1]) >= thr_iou:
                members.append(pool[j]); used[j] = True
        if len(members) < support:
            continue
        if sel == "median":
            on = wmedian([m[1][0] for m in members], [1] * len(members)); off = wmedian([m[1][1] for m in members], [1] * len(members))
        elif sel == "sharp":
            on, off = max(members, key=lambda m: sharpness(posts[m[0]], m[1]))[1][:2]
        elif sel == "conf":
            on, off = max(members, key=lambda m: conf(*m))[1][:2]
        elif sel == "base":
            b = [m for m in members if m[0] == "ens"]
            if b: on, off = b[0][1][:2]
            else: on = wmedian([m[1][0] for m in members], [1] * len(members)); off = wmedian([m[1][1] for m in members], [1] * len(members))
        else:
            raise ValueError(sel)
        if off - on >= min_dur:
            out.append((on, off))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="hon_A+bts_C+bft_S+bts_R+v12+v5_S")
    ap.add_argument("--half", default="decide", choices=["decide", "tune"])
    ap.add_argument("--support", default="2,3,4"); ap.add_argument("--iou", default="0.3,0.5")
    ap.add_argument("--sel", default="median,sharp,conf,base"); ap.add_argument("--with-ens", default="1")
    a = ap.parse_args()
    dec = Decide()
    if a.half == "tune":
        tune, _ = SP.split_by_speaker(list(dec.refs), frac=0.5, seed=42)
        dec.ids = list(tune); dec.nids = [c for c in dec.ids if not dec.meta[c]["syntheticData"]]
        dec.nat = {c: not dec.meta[c]["syntheticData"] for c in dec.ids}
        dec.syn = {c: bool(dec.meta[c]["syntheticData"]) for c in dec.ids}
        import audiostore as A; from solo_measure import ROOT
        _, me = A.load_audio_store(ROOT / "audio", "val_audio"); dec.dur = {c: me["clips"][c]["n"] / A.SR for c in dec.ids}
    ids = dec.nids; mem = a.members.split("+")
    base = dec.natural_preds(mem)
    Pm = {t: dec.natural_preds([t]) for t in mem}
    posts = {t: {c: dec.post(t)["post"][c][:, 0] for c in ids} for t in mem}
    import ens_gate as E
    posts["ens"] = E.average([posts[t] for t in mem])
    refs = {c: dec.refs[c] for c in ids}
    s0 = S.score(refs, {c: base[c] for c in ids})
    print(f"{a.half.upper()} natural {len(ids)} clips | ensemble route: F1 {s0['event_f1']:.4f} Dice {s0['segment_dice']:.4f}", flush=True)
    for t in mem:
        s = S.score(refs, {c: Pm[t][c] for c in ids}); print(f"  {t:8s} solo F1 {s['event_f1']:.4f} Dice {s['segment_dice']:.4f}")
    res = []
    for sup, thr, sel, we in itertools.product([int(x) for x in a.support.split(",")], [float(x) for x in a.iou.split(",")],
                                               a.sel.split(","), [bool(int(x)) for x in a.with_ens.split(",")]):
        fused = {}
        for c in ids:
            src = {t: Pm[t][c] for t in mem}
            if we: src["ens"] = base[c]
            fused[c] = fuse_clip(src, {t: posts[t][c] for t in posts}, sup, thr, sel)
        s = S.score(refs, fused)
        res.append((s["event_f1"] + s["segment_dice"], sup, thr, sel, we, s["event_f1"], s["segment_dice"], sum(len(v) for v in fused.values())))
        print(f"  support {sup} iou {thr:.1f} sel {sel:6s} ens {int(we)}: F1 {s['event_f1']:.4f} ({s['event_f1']-s0['event_f1']:+.4f}) "
              f"Dice {s['segment_dice']:.4f} ({s['segment_dice']-s0['segment_dice']:+.4f}) preds {res[-1][-1]}", flush=True)
    res.sort(reverse=True)
    print("best by F1+Dice:", res[0])


if __name__ == "__main__":
    main()
