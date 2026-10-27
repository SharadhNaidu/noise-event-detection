"""Member posteriors on an audio store, plain or with test-time augmentation (TTA).

    python scripts/tta_posteriors.py TAG STORE [--tta] [--out DIR] [--batch 8] [--device cuda] [--check N --ref PKL]

--tta  : finetune.tta_posteriors with pads 0 / 0.64 / 1.28 s averaged (shipped for hon_A, bft_S, v5_S)
--check: plain posteriors on the first N clips vs a reference cache -> max |diff| (environment fidelity)
Writes <out>/post8_<STORE>_<TAG>[_tta].pkl = {"post": {cid: (T, 8)}, "dist": {...}}, the format postpass expects.
Checkpoints are read from $INDOML_ROOT/ckpt/<TAG>.pt, audio from $INDOML_ROOT/audio/<STORE>.{i16,json}."""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import audiostore as A   # noqa: E402
import finetune as FT    # noqa: E402
import members as M      # noqa: E402

PADS = M.TTA_PADS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("store")
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--out", default=str(FT.CKPT))
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", type=int, default=0); ap.add_argument("--ref")
    a = ap.parse_args()
    if a.check:
        arr, me = A.load_audio_store(FT.AUD, a.store); ids = list(me["clips"])[:a.check]
        post, dist = M.posteriors(a.tag, a.store, False, a.device, a.batch, ids)
        ref = pickle.load(open(a.ref, "rb"))
        d = max(float(np.abs(post[c] - np.asarray(ref["post"][c])[:len(post[c]), :8]).max()) for c in ids)
        print(f"CHECK {a.tag} {a.store} {a.check} clips: max |post diff| {d:.2e}")
        return
    t0 = time.time(); post, dist = M.posteriors(a.tag, a.store, a.tta, a.device, a.batch)
    out = Path(a.out) / f"post8_{a.store}_{a.tag}{'_tta' if a.tta else ''}.pkl"; out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"post": post, "dist": dist, "tta_pads": PADS if a.tta else None}, open(out, "wb"))
    print(f"{a.tag} {a.store}: {len(post)} clips -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
