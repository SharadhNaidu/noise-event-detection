"""Train the labelling-habit model (src/habit.py) with the recipe shipped on 2026-10-15.

The original pickle was lost; checkpoints/habit_model.pkl is this script's retrain, which agrees
with the shipped test decisions on 98.8 % of natural clips (reproduce/habit_test_shipped.json).

    python train/train_habit.py [--out checkpoints/habit_model.pkl] [--device cuda]

Needs $INDOML_ROOT/audio/{gold_audio,val_audio}.{i16,json} and $INDOML_DATA/validation.zip.
Train = gold (11,111 clips, minus a 20 % hold-out used only for the threshold check) + TUNE natural (the speaker
half `split_by_speaker(val, 0.5, seed=42)[0]`). DECIDE (the other half) is reported, never trained on.
Clip-pooled BEATs tags are cached in $INDOML_ROOT/tags/<store>.npz.
Shipped run: gold 11111 (15.0 % long-bg) | TUNE 894 | DECIDE 912; gold hold-out AUC 0.785, DECIDE AUC 0.734."""
import argparse
import ast
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import audiostore as A   # noqa: E402
import data as D         # noqa: E402
import habit as H        # noqa: E402
import paths as P        # noqa: E402
import splits as SP      # noqa: E402


def tags(store, device):
    f = P.ROOT / "tags" / f"{store}.npz"
    if f.exists():
        z = np.load(f); return z["ids"].tolist(), z["X"]
    arr, me = A.load_audio_store(P.ROOT / "audio", store); ids = list(me["clips"])
    print(f"tagging {store} ({len(ids)} clips)", flush=True)
    X = H.tag_store(arr, me["clips"], ids, device)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, X=X, ids=np.array(ids))
    return ids, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "checkpoints" / "habit_model.pkl"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    refs, meta = D.load_validation()
    STATES = sorted({H.state_of(c) for c in refs} | {"?"})
    gm = json.load(open(P.ROOT / "audio" / "gold_audio.json")); gcl = gm.get("clips", gm)
    gids, gX = tags("gold_audio", a.device); gT = H.tagfeat(gX)
    Xg, Yg = [], []
    for c, tf in zip(gids, gT):
        e = gcl[c]
        evs = ast.literal_eval(e["events"]) if isinstance(e["events"], str) else e["events"]
        cats = ast.literal_eval(e["cats"]) if isinstance(e["cats"], str) else e["cats"]
        if len(evs) != len(cats): continue
        d = float(e["dur"])
        Xg.append(H.features(tf, e.get("state", "?"), d, STATES)); Yg.append(H.long_bg(evs, cats, d))
    Xg, Yg = np.array(Xg), np.array(Yg)
    vids, vX = tags("val_audio", a.device); VT = dict(zip(vids, H.tagfeat(vX)))
    def vlab(c):
        d = meta[c]["duration"]
        return any(abs(float(x["end"]) - float(x["start"])) >= 0.5 * d and x["category"] != "human_non_speech" for x in meta[c]["NoiseSubCategoryTimeStamp"])
    half_a, half_b = SP.split_by_speaker(list(refs), 0.5, 42)
    tune = [c for c in half_a if not meta[c]["syntheticData"]]; decide = [c for c in half_b if not meta[c]["syntheticData"]]
    row = lambda c: H.features(VT[c], H.state_of(c), meta[c]["duration"], STATES)
    Xt, Yt = np.array([row(c) for c in tune]), np.array([vlab(c) for c in tune])
    Xd, Yd = np.array([row(c) for c in decide]), np.array([vlab(c) for c in decide])
    print(f"gold {len(Yg)} clips, long-bg labelled {Yg.mean():.1%} | TUNE {len(Yt)} {Yt.mean():.1%} | DECIDE {len(Yd)} {Yd.mean():.1%}")
    hold = np.random.default_rng(0).random(len(Yg)) < 0.2
    m = H.make_model().fit(np.concatenate([Xg[~hold], Xt]), np.concatenate([Yg[~hold], Yt]))
    print(f"gold hold-out AUC {roc_auc_score(Yg[hold], m.predict_proba(Xg[hold])[:, 1]):.3f} | "
          f"DECIDE AUC {roc_auc_score(Yd, m.predict_proba(Xd)[:, 1]):.3f}  (shipped: 0.785 / 0.734)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(model=m, STATES=STATES, thr=0.5), open(a.out, "wb"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
