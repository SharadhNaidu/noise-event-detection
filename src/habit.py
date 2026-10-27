"""Labelling-habit model (shipped 2026-10-15, LB +0.0050).

P(annotators labelled a long background, i.e. one non-breath event >= 50 % of the clip | audible evidence, state).
Audible evidence = the frozen AudioSet-strong tagger BEATs_strong_1 (PretrainedSED), pooled per clip and per
sound group; habit = the recording state (parsed from the clip id) and clip duration.
Ported verbatim from the scripts that built the uploaded file (tag_clips.py, p1_habit_model.py)."""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ext" / "PretrainedSED"))
from data_util.audioset_classes import as_strong_train_classes as CLS  # noqa: E402

N_CLS = 447
WIN = 160000                      # 10 s windows (BEATs-strong training length)
GR = {"insect": r"insect|cricket|mosquito|bee, wasp|fly, housefly", "bird": r"bird|chirp|crow|caw|squawk|pigeon|rooster|chicken|cluck|duck",
      "animal": r"animal|dog|bark|howl|growl|livestock|cattle|goat|sheep|frog|cat\b|meow",
      "machine": r"television|mechanical fan|air conditioning|engine|generator|sewing|vacuum|blender|electric|motor|hum\b|buzz|whir|mechanisms",
      "vehicle": r"traffic|vehicle|car\b|truck|bus\b|motorcycle|horn|train|aircraft|siren|accelerat", "music": r"music|singing|song|choir|instrument|chant|radio",
      "bg": r"^background noise$|^wind|^noise$|static|hiss|rumble", "speech": r"speech|speaking|conversation|narration|babbling", "child": r"child|baby|infant|kid"}
GI = [np.array([bool(re.search(p, c, re.I)) for c in CLS]) for p in GR.values()]
NS = lambda x: re.sub(r"[^a-z]", "", str(x).lower())


def state_of(clip_id):
    """Recording state from a Vaani clip id (4th underscore field), lower-case letters only; "" if absent."""
    parts = clip_id.split("_")
    return NS(parts[3]) if len(parts) > 3 else ""


def tagfeat(X):
    """(n, 447*3+768) [mean | max | active-frac(p>=0.3) | emb] -> (n, 27): per sound group, max over its classes."""
    mean, mx, fr = X[:, :N_CLS], X[:, N_CLS:2 * N_CLS], X[:, 2 * N_CLS:3 * N_CLS]
    return np.concatenate([np.stack([b[:, g].max(1) for g in GI], 1) for b in (mean, mx, fr)], 1)


def features(tf_row, state, dur, STATES):
    state = NS(state)
    return np.concatenate([tf_row, [dur], [1.0 if state == s else 0.0 for s in STATES]])


def tag_store(arr, clips, ids, device="cuda", log_every=2000):
    """Clip-pooled BEATs-strong tags for every clip id -> (n, 2109) float32 (same windowing as shipped)."""
    import time
    import torch
    import beats_enc as BE
    enc = BE.BeatsEncoder("BEATs").to(device).eval()
    use_amp = device.startswith("cuda")
    X = np.zeros((len(ids), N_CLS * 3 + 768), np.float32); t0 = time.time()
    with torch.no_grad():
        for i, c in enumerate(ids):
            e = clips[c]; y = np.asarray(arr[e["off"]:e["off"] + e["n"]], np.float32) / 32768
            segs = [y[k:k + WIN] for k in range(0, len(y), WIN)]
            segs = [s for s in segs if len(s) >= 8000] or [y]
            out = []
            for s in segs:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    f = enc(torch.from_numpy(s)[None].to(device)).float()[0]
                out.append(f)
            f = torch.cat(out); emb, pr = f[:, :768], f[:, 768:]
            if pr.min() < 0 or pr.max() > 1: pr = torch.sigmoid(pr)
            X[i] = torch.cat([pr.mean(0), pr.max(0).values, (pr >= 0.3).float().mean(0), emb.mean(0)]).cpu().numpy()
            if log_every and i % log_every == 0: print(f"  tagged {i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    return X


def long_bg(evs, cats, d):
    return any((b - a) >= 0.5 * d and cat != "human_non_speech" for (a, b), cat in zip(evs, cats))


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)


def predict_habit(blob, rows):
    """blob = {"model", "STATES"} from train/train_habit.py; rows = features(...) stacked -> P(long background labelled)."""
    return blob["model"].predict_proba(np.asarray(rows))[:, 1]
