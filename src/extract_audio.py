"""Decode audio once into a flat int16 store, for encoder fine-tuning.

Reuses extract.py's iterators verbatim, which matters: they carry the
clip-id collision guard, the `syntheticCleanRefAudio` exclusion and the count
assertions that caught a silent validation-set corruption once already.

    python src/extract_audio.py hf  --name gold_audio --tier verified_timestamps
    python src/extract_audio.py val --name val_audio
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A  # noqa: E402
import extract as X     # noqa: E402
import features as F    # noqa: E402

OUT = (P.ROOT / "audio")


def run(source, name, tier=None, limit=None, workers=14):
    if source == "val":
        it = X._iter_zip(X.REPO / "data" / "validation.zip", limit=limit,
                         expect=2501)
    elif source == "test":
        it = X._iter_zip(X.REPO / "data" / "input_data.zip", limit=limit,
                         expect=5517)
    elif source == "hf":
        it = X._iter_parquet(tier=tier, limit=limit)
    else:
        raise ValueError(source)

    print(f"[audio] source={source} name={name} tier={tier}")
    store = A.AudioStore(OUT, name)
    t0 = time.time()
    n_ok = n_err = 0
    audio_s = 0.0
    batch, metas = [], []
    CHUNK = 256

    def flush(pool):
        nonlocal n_ok, n_err, audio_s
        if not batch:
            return
        for (cid, wav), meta in zip(pool.map(X._decode_worker, batch), metas):
            if isinstance(wav, str):
                n_err += 1
                continue
            store.add(cid, wav, {**meta,
                                 "dur_measured": len(wav) / F.TARGET_SR})
            audio_s += len(wav) / F.TARGET_SR
            n_ok += 1
        batch.clear()
        metas.clear()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for payload, meta in it:
            batch.append(payload)
            metas.append(meta)
            if len(batch) >= CHUNK:
                flush(pool)
                if n_ok % 4096 < CHUNK:
                    el = time.time() - t0
                    print(f"    {n_ok:>6} clips | {audio_s/3600:5.2f} h | "
                          f"{el/60:5.1f} min", flush=True)
        flush(pool)

    p = store.close()
    el = time.time() - t0
    print(f"[audio] done: {n_ok} clips ({n_err} failed), {audio_s/3600:.2f} h, "
          f"{el/60:.1f} min")
    print(f"        -> {p}  "
          f"({os.path.getsize(OUT/(name+'.i16'))/2**30:.2f} GiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["val", "test", "hf"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tier", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    run(a.source, a.name, a.tier, a.limit, a.workers)
