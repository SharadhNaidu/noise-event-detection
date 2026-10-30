"""Driver: decode audio on CPU workers, encode with WavLM on GPU, cache to B:.

Pipeline shape is dictated by two measurements:
  * EXP-06/08 - clips must be encoded SOLO (group-norm front-end; any padding
    injects ~7%+ per-frame noise). Solo runs at ~180x realtime, so this is fine.
  * EXP-04    - the GPU is not the bottleneck, CPU audio decode is. We decode in
    a process pool (32 cores available) and hand waveforms to the GPU in order.

Sources:
  zip  - validation.zip / input_data.zip (competition downloads)
  hf   - the parquet shards in $INDOML_ROOT/raw/data

Output per source: <name>.f16 + <name>.json in $INDOML_ROOT/features (see features.py).
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import features as F  # noqa: E402

RAW = (P.ROOT / "raw/data")
FEAT = (P.ROOT / "features")
REPO = Path(__file__).resolve().parent.parent


def load_token():
    """HF token for gated/private downloads; public models need none. HF_TOKEN env, else an optional .env, else None."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO / ".env"
    if not env.exists():
        return None
    for line in open(env, encoding="utf-8"):
        if line.strip().startswith(("HF_TOKEN", "HUGGING_FACE_PAT")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# --------------------------------------------------------------------------
# CPU worker: bytes -> waveform
# --------------------------------------------------------------------------
def _decode_worker(payload):
    cid, blob = payload
    try:
        return cid, F.decode_audio(blob)
    except Exception as e:  # never let one bad row kill a 90k pass
        return cid, f"ERROR {type(e).__name__}: {e}"


# validation.zip holds THREE audio folders and only two of them are model input:
#   naturalNoisyAudio/      1806  <- input
#   syntheticNoiseAudio/     695  <- input
#   syntheticCleanRefAudio/  695  <- Track 2 clean reference, NOT input
# The clean references share filenames byte-for-byte with syntheticNoiseAudio,
# so keying on the stem silently overwrites the noisy clip with its clean twin -
# i.e. the model would be validated on clean audio carrying noise labels, with
# no error raised anywhere. Exclude it explicitly and assert the count.
EXCLUDE_DIRS = ("syntheticCleanRefAudio",)


def _iter_zip(zip_path, prefix_filter=None, limit=None, expect=None):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist()
             if n.endswith(".wav") and "__MACOSX" not in n
             and not any(d in n for d in EXCLUDE_DIRS)
             and (prefix_filter is None or prefix_filter in n)]
    names.sort()

    stems = [os.path.basename(n)[:-4] for n in names]
    if len(set(stems)) != len(stems):
        dupes = [s for s in set(stems) if stems.count(s) > 1][:3]
        raise ValueError(f"colliding clip_ids in {zip_path}: {dupes} "
                         f"({len(stems)} files, {len(set(stems))} distinct stems)")
    if expect is not None and len(names) != expect:
        raise ValueError(f"expected {expect} clips in {zip_path}, found {len(names)}")

    if limit:
        names = names[:limit]
    for n in names:
        yield (os.path.basename(n)[:-4], z.read(n)), {
            "src": os.path.basename(os.path.dirname(n))}


def _iter_parquet(tier=None, limit=None):
    """Yield ((clip_id, audio_bytes), meta) from the local parquet shards."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(RAW / "*.parquet")))
    n = 0
    for fi, f in enumerate(files):
        t = pq.ParquetFile(f).read().to_pylist()
        for ri, r in enumerate(t):
            if tier and r["annotationQuality"] != tier:
                continue
            a = r["audio"]
            blob = a.get("bytes") if isinstance(a, dict) else None
            if blob is None:
                continue
            evs, cats = F.parse_events(r.get("NoiseSubCategoryTimeStamp"))
            cid = f"s{fi:03d}r{ri:05d}"          # stable synthetic id
            yield (cid, blob), {
                "tier": F.TIER_SHORT.get(r["annotationQuality"], r["annotationQuality"]),
                "dur": float(r["duration"]),
                "lang": r.get("language"),
                "state": r.get("state"),
                "district": r.get("district"),
                "events": evs,
                "cats": cats,
                "transcript_hash": hash(r.get("transcript") or "") & 0xFFFFFFFF,
            }
            n += 1
            if limit and n >= limit:
                return


def run(source, name, tier=None, limit=None, workers=12, device="cuda", layer=None):
    import data as D
    FEAT.mkdir(parents=True, exist_ok=True)
    if source == "val":
        it = _iter_zip(D.VAL_ZIP, limit=limit, expect=2501)
    elif source == "test":
        it = _iter_zip(D.TEST_ZIP, limit=limit, expect=5517)
    elif source == "hf":
        it = _iter_parquet(tier=tier, limit=limit)
    else:
        raise ValueError(source)

    print(f"[extract] source={source} name={name} tier={tier} limit={limit}")
    ex = F.WavLMExtractor(device=device, token=load_token(), layer=layer)
    print(f'[extract] encoder layer = {layer if layer is not None else "last (10)"}')
    store = F.FeatureStore(FEAT, name, layer=layer)

    t0 = time.time()
    n_ok = n_err = 0
    audio_s = 0.0
    batch, metas = [], []
    CHUNK = 64          # how many clips we decode in parallel before encoding

    def flush(pool):
        nonlocal n_ok, n_err, audio_s
        if not batch:
            return
        for (cid, wav), meta in zip(pool.map(_decode_worker, batch), metas):
            if isinstance(wav, str):
                n_err += 1
                continue
            feats = ex([wav])[0]          # SOLO - see EXP-08
            store.add(cid, feats, {**meta, "dur_measured": len(wav) / F.TARGET_SR})
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
                if n_ok % 2048 == 0:
                    el = time.time() - t0
                    print(f"    {n_ok:>6} clips | {audio_s/3600:5.2f} h | "
                          f"{el/60:5.1f} min | {audio_s/el:6.0f}x RT", flush=True)
        flush(pool)

    p = store.close()
    el = time.time() - t0
    print(f"[extract] done: {n_ok} clips ({n_err} failed), {audio_s/3600:.2f} h audio, "
          f"{el/60:.1f} min, {audio_s/el:.0f}x realtime")
    print(f"          -> {p}  ({os.path.getsize(FEAT/(name+'.f16'))/2**30:.2f} GiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["val", "test", "hf"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tier", default=None,
                    choices=[None, "verified_timestamps", "unverified_timestamps",
                             "no_timestamps"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    run(a.source, a.name, a.tier, a.limit, a.workers, layer=a.layer)
