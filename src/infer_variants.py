"""Inference-physics test (2026-10-09): does the way overlapping 5 s windows are combined move natural F1?

Variants on ONE member, DECIDE natural clips, champion natural decode, paired bootstrap vs the shipped plain average:
  plain      shipped: equal-weight mean of overlapping crops
  taper0.3   crop frames within 0.3 s of a crop edge are down-weighted (linear ramp)
  taper0.6   same, 0.6 s
  whole      one crop = the whole clip (no windows, no padding)
    python src/infer_variants.py hon_A [--device cpu] [--n 300]
"""
import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiostore as A, finetune as FT, gate as G
from solo_measure import Decide, ROOT, fmt

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--variants", default="plain,taper0.3,taper0.6,whole")
a = ap.parse_args()
dec = Decide(); ids = dec.nids
arr, me = A.load_audio_store(ROOT / "audio", "val_audio"); idx = me["clips"]
net, blob = G.load(a.tag, a.device); crop = blob["cfg"]["crop"]; aux = G.aux_for(blob["cfg"])
print(f"{a.tag}: crop {crop}s, {len(ids)} DECIDE natural clips, device {a.device}", flush=True)
out = {}
for v in a.variants.split(","):
    t0 = time.time()
    kw = dict(all_channels=True)
    if v.startswith("taper"): kw["taper_s"] = float(v[5:]); cs = crop
    elif v == "whole": cs = 0.0
    else: cs = crop
    if v.startswith("tta"):      # tta or tta+taper0.3: zero-pad shifts of the crop grid, averaged
        if "+taper" in v: kw["taper_s"] = float(v.split("+taper")[1])
        post, dist, _ = FT.tta_posteriors(net, arr, idx, ids, cs, a.device, ast=aux, **kw)
    else:
        post, dist, _ = FT.clip_posteriors(net, arr, idx, ids, cs, a.device, ast=aux, **kw)
    out[v] = (post, dist)
    pickle.dump({"post": post, "dist": dist}, open(ROOT / "ckpt" / f"variant_{a.tag}_{v}.pkl", "wb"))
    print(f"  {v}: {time.time()-t0:.0f}s", flush=True)
# sanity: plain must equal the cached member posteriors
if "plain" in out:
    C = dec.post(a.tag); d = max(float(np.abs(out["plain"][0][c] - C["post"][c]).max()) for c in ids[:50])
    print(f"sanity plain vs cache: max |diff| {d:.4f} on 50 clips", flush=True)
P = {v: dec.natural_preds_from(*out[v]) for v in out}
for v in P:
    f, d = dec.point(P[v]); print(f"{v:10s} natural F1 {f:.4f} Dice {d:.4f}", flush=True)
base = P.get("plain") or P[list(P)[0]]
for v in P:
    if P[v] is base: continue
    r = dec.compare(P[v], base, n=a.n)
    for pop in ("plain", "matched"):
        print(f"  {v:10s} vs plain [{pop}]: F1 {fmt(r[(pop,'event_f1')])} | Dice {fmt(r[(pop,'segment_dice')])}", flush=True)
