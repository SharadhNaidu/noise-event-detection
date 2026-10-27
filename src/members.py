"""Per-member frame posteriors on an audio store, plain or with test-time augmentation (TTA)."""
import json
from pathlib import Path

import numpy as np

import audiostore as A
import finetune as FT
import gate as G

REPO = Path(__file__).resolve().parent.parent
TTA_PADS = (0.0, 0.64, 1.28)


def load_members(path=None):
    return json.load(open(path or REPO / "configs" / "members.json"))


def posteriors(tag, store, tta=False, device="cuda", batch=8, ids=None):
    """Checkpoint $INDOML_ROOT/ckpt/<tag>.pt on store $INDOML_ROOT/audio/<store> -> ({cid: (T, 8)}, {cid: dist})."""
    net, blob = G.load(tag, device); net.eval(); crop = blob["cfg"]["crop"]
    arr, me = A.load_audio_store(FT.AUD, store); idx = me["clips"]; ids = ids or list(idx)
    aux = G.aux_for(blob["cfg"], store)
    if tta:
        post, dist, _ = FT.tta_posteriors(net, arr, idx, ids, crop, device, batch=batch, ast=aux, all_channels=True, pads=TTA_PADS)
    else:
        post, dist, _ = FT.clip_posteriors(net, arr, idx, ids, crop, device, batch=batch, ast=aux, all_channels=True)
    del net
    if device.startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    return {c: np.asarray(v)[:, :8] for c, v in post.items()}, dist
