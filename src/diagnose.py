"""Fresh error anatomy of the champion natural route on DECIDE natural (2026-10-09, new lead; nothing inherited).

For every reference event: hit | edge (a pred overlaps it >= 0.5 IoU but an edge is outside tau) | merged (its best pred
also covers another ref) | split (>= 2 preds inside it) | missed (no pred overlaps). For every prediction: TP | FP-kind.
Then the mechanism questions that decide what to build:
  Q1 which edge fails (onset / offset / both), signed error in ms and in units of tau, by family and duration bin
  Q2 posterior anatomy: ch0 peak inside missed refs; ramp width (0.3->0.7 crossing) at hit vs edge-miss refs
  Q3 nesting: refs inside a longer ref: hit rate, and whether the outer was predicted
  Q4 member diversity: per-member hit on each ref; union-of-members (any member hits) vs best member vs ensemble
  Q5 fix-one-edge ceilings: F1 if only the failing onset / only the failing offset were within tau
  Q6 Dice anatomy: per-clip Dice bins, share of loss, and what the worst clips look like
    python src/diagnose.py  [--members hon_A+bts_C+bft_S+bts_R+v12+v5_S]
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ens_gate as E          # noqa: E402
import scorer as S            # noqa: E402
from solo_measure import Decide, MD02  # noqa: E402

BR = ("breathing", "inhaling", "lip smacking", "lip_smacking", "exhaling")


def tol(r):
    return max(0.2 * (r[1] - r[0]), 0.05)


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def match(rs, ps):
    cand = [(abs(p[0] - r[0]) + abs(p[1] - r[1]), ri, pi) for ri, r in enumerate(rs) for pi, p in enumerate(ps)
            if abs(p[0] - r[0]) <= tol(r) and abs(p[1] - r[1]) <= tol(r)]
    mr, mp = {}, {}
    for _, ri, pi in sorted(cand):
        if ri not in mr and pi not in mp:
            mr[ri] = pi; mp[pi] = ri
    return mr, mp


def f1(tp, npred, nref):
    p = tp / max(npred, 1); r = tp / max(nref, 1)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--members", default="hon_A+bts_C+bft_S+bts_R+v12+v5_S"); a = ap.parse_args()
    dec = Decide(); ids = dec.nids; refs = dec.refs; meta = dec.meta
    mem = a.members.split("+")
    P = dec.natural_preds(mem)
    Pm = {t: dec.natural_preds([t]) for t in mem}
    p8 = E.average([{c: dec.post(t)["post"][c] for c in ids} for t in mem])
    srt = lambda c: sorted(meta[c]["NoiseSubCategoryTimeStamp"], key=lambda e: (float(e["start"]), float(e["end"])))
    fam = {c: ["breath" if e["tag"].strip("<>[]") in BR else e["category"] for e in srt(c)] for c in ids}
    R = {c: [tuple(r[:2]) for r in refs[c]] for c in ids}
    for c in ids:
        assert len(R[c]) == len(fam[c]), c

    # ---- taxonomy ----
    rows = []                      # one per ref
    pred_kind = collections.Counter()
    for c in ids:
        rs, ps = R[c], P[c]
        mr, mp = match(rs, ps)
        # best-overlap pred per ref
        for ri, r in enumerate(rs):
            best = max(range(len(ps)), key=lambda pi: iou(r, ps[pi]), default=None)
            bio = iou(r, ps[best]) if best is not None else 0.0
            inside = [pi for pi, p in enumerate(ps) if p[0] >= r[0] - tol(r) and p[1] <= r[1] + tol(r)]
            covers = [rj for rj, r2 in enumerate(rs) if best is not None and rj != ri and iou(r2, ps[best]) > 0]
            if ri in mr: kind = "hit"
            elif best is None or bio == 0.0: kind = "missed"
            elif len(inside) >= 2: kind = "split"
            elif covers: kind = "merged"
            else: kind = "edge"
            L = r[1] - r[0]
            d_on = d_off = np.nan
            if best is not None and bio > 0:
                d_on, d_off = ps[best][0] - r[0], ps[best][1] - r[1]
            nested = any(rj != ri and r2[0] <= r[0] and r[1] <= r2[1] and (r2[1] - r2[0]) > L for rj, r2 in enumerate(rs))
            outer_hit = None
            if nested:
                outs = [rj for rj, r2 in enumerate(rs) if rj != ri and r2[0] <= r[0] and r[1] <= r2[1] and (r2[1] - r2[0]) > L]
                outer_hit = any(rj in mr for rj in outs)
            # posterior anatomy
            t = np.arange(len(p8[c])) * 0.02 + 0.01
            seg = p8[c][(t >= r[0]) & (t <= r[1]), 0]
            peak = float(seg.max()) if len(seg) else 0.0
            rows.append(dict(c=c, ri=ri, kind=kind, fam=fam[c][ri], L=L, tau=tol(r), d_on=d_on, d_off=d_off,
                             nested=nested, outer_hit=outer_hit, peak=peak, dur=dec.dur[c],
                             mhit={tm: ri in match(rs, Pm[tm][c])[0] for tm in mem}))
        for pi, p in enumerate(ps):
            if pi in mp: pred_kind["TP"] += 1
            elif not any(iou(p, r) > 0 for r in rs): pred_kind["FP_nothing"] += 1
            else: pred_kind["FP_overlaps_ref"] += 1
    n = len(rows); kinds = collections.Counter(r["kind"] for r in rows)
    tp = kinds["hit"]; npred = sum(len(P[c]) for c in ids)
    print(f"DECIDE natural: {len(ids)} clips, {n} refs, {npred} preds | F1 {f1(tp, npred, n):.4f} (TP {tp})")
    print("REF kinds:", {k: f"{v} ({v/n:.1%})" for k, v in kinds.most_common()})
    print("PRED kinds:", dict(pred_kind))
    fams = ["breath", "animal", "vehicle_traffic", "baby_child", "phone_signal_alarm", "singing_music", "appliance_machine"]
    print("\nby family: n | hit | edge | merged | split | missed")
    for f in fams:
        g = [r for r in rows if r["fam"] == f]
        if not g: continue
        k = collections.Counter(r["kind"] for r in g); m = len(g)
        print(f"  {f:20s} {m:5d} | " + " | ".join(f"{k[x]/m:.2f}" for x in ("hit", "edge", "merged", "split", "missed")))
    print("by ref duration bin:")
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
        g = [r for r in rows if lo <= r["L"] < hi]
        k = collections.Counter(r["kind"] for r in g); m = max(len(g), 1)
        print(f"  [{lo:4.2f},{hi:4.2f}) {len(g):5d} | " + " | ".join(f"{k[x]/m:.2f}" for x in ("hit", "edge", "merged", "split", "missed")))

    # ---- Q1 which edge fails ----
    print("\nQ1 edge-kind refs: which edge is outside tau, signed error (ms) median [IQR], and in tau units")
    for f in ("breath", "animal", "vehicle_traffic", "baby_child"):
        g = [r for r in rows if r["kind"] == "edge" and r["fam"] == f]
        if len(g) < 8: continue
        on_bad = [abs(r["d_on"]) > r["tau"] for r in g]; off_bad = [abs(r["d_off"]) > r["tau"] for r in g]
        both = sum(a and b for a, b in zip(on_bad, off_bad))
        q = lambda x: f"{np.median(x)*1000:+5.0f} [{np.percentile(x,25)*1000:+5.0f},{np.percentile(x,75)*1000:+5.0f}]"
        print(f"  {f:16s} n={len(g):4d} onset-bad {np.mean(on_bad):.2f} offset-bad {np.mean(off_bad):.2f} both {both/len(g):.2f} | "
              f"d_on {q([r['d_on'] for r in g])} d_off {q([r['d_off'] for r in g])} | |d_off|/tau med {np.median([abs(r['d_off'])/r['tau'] for r in g]):.2f}")
    print("hit refs, same stats (bias check):")
    for f in ("breath", "animal", "vehicle_traffic", "baby_child"):
        g = [r for r in rows if r["kind"] == "hit" and r["fam"] == f]
        if len(g) < 8: continue
        q = lambda x: f"{np.median(x)*1000:+5.0f} [{np.percentile(x,25)*1000:+5.0f},{np.percentile(x,75)*1000:+5.0f}]"
        print(f"  {f:16s} n={len(g):4d} d_on {q([r['d_on'] for r in g])} d_off {q([r['d_off'] for r in g])}")

    # ---- Q2 posterior anatomy ----
    print("\nQ2 ch0 peak inside ref (median): " + ", ".join(
        f"{k} {np.median([r['peak'] for r in rows if r['kind']==k]):.2f}" for k in ("hit", "edge", "merged", "missed")))
    miss = [r for r in rows if r["kind"] == "missed"]
    print(f"   missed refs with peak >= 0.6: {np.mean([r['peak']>=0.6 for r in miss]):.2f}, >= 0.3: {np.mean([r['peak']>=0.3 for r in miss]):.2f} (n={len(miss)})")

    # ---- Q3 nesting ----
    nest = [r for r in rows if r["nested"]]
    print(f"\nQ3 nested refs: {len(nest)} ({len(nest)/n:.1%}); hit {np.mean([r['kind']=='hit' for r in nest]):.2f}; "
          f"kinds {dict(collections.Counter(r['kind'] for r in nest))}; outer hit {np.mean([bool(r['outer_hit']) for r in nest]):.2f}")
    print("   nested by family:", {f: (sum(r["fam"] == f for r in nest), round(np.mean([r["kind"] == "hit" for r in nest if r["fam"] == f] or [0]), 2)) for f in fams if any(r["fam"] == f for r in nest)})

    # ---- Q4 member diversity ----
    print("\nQ4 member hit rates on the same refs: " + ", ".join(f"{t} {np.mean([r['mhit'][t] for r in rows]):.3f}" for t in mem))
    anyhit = np.mean([any(r["mhit"].values()) for r in rows]); allhit = np.mean([all(r["mhit"].values()) for r in rows])
    print(f"   ensemble {tp/n:.3f} | union of members {anyhit:.3f} | all members {allhit:.3f}")
    print(f"   refs hit by exactly k members: {dict(sorted(collections.Counter(sum(r['mhit'].values()) for r in rows).items()))}")
    # oracle: F1 if we could pick, per ref, any member that hits it (preds = union of all member TPs + ensemble FPs)
    print(f"   union-oracle recall ceiling {anyhit:.3f} vs ensemble recall {tp/n:.3f}")

    # ---- Q5 fix-one-edge ceilings ----
    def ceiling(fix):
        tp2 = 0
        for c in ids:
            rs, ps = R[c], list(P[c])
            mr, mp = match(rs, ps)
            for ri, r in enumerate(rs):
                if ri in mr: continue
                best = max(range(len(ps)), key=lambda pi: iou(r, ps[pi]), default=None)
                if best is None or iou(r, ps[best]) < 0.3 or best in mp: continue
                p = ps[best]; q = list(p)
                if fix in ("on", "both") and abs(p[0] - r[0]) > tol(r): q[0] = r[0]
                if fix in ("off", "both") and abs(p[1] - r[1]) > tol(r): q[1] = r[1]
                if abs(q[0] - r[0]) <= tol(r) and abs(q[1] - r[1]) <= tol(r):
                    mp[best] = ri; mr[ri] = best
            tp2 += len(mr)
        return f1(tp2, npred, n), tp2
    for fx in ("on", "off", "both"):
        v, t2 = ceiling(fx); print(f"Q5 fix failing {fx:4s} edge of the best-overlap pred (IoU>=0.3): F1 {v:.4f} (+{v - f1(tp, npred, n):.4f}, TP {t2})")

    # ---- Q6 Dice anatomy ----
    dice = {}
    for c in ids:
        s = S.score({c: refs[c]}, {c: P[c]}); dice[c] = s["segment_dice"]
    d = np.array(list(dice.values())); loss = 1 - d
    print(f"\nQ6 natural Dice {d.mean():.4f}; clips with Dice<0.3: {np.mean(d<0.3):.3f} carrying {loss[d<0.3].sum()/loss.sum():.2f} of the loss; "
          f"Dice==0: {np.sum(d==0)} clips")
    worst = sorted(dice, key=dice.get)[:12]
    for c in worst:
        print(f"   {dice[c]:.2f} dur {dec.dur[c]:.1f}s refs {[(round(a,2), round(b,2)) for a,b in R[c]]} preds {[(round(a,2), round(b,2)) for a,b in P[c]]} fams {fam[c]}")


if __name__ == "__main__":
    main()
