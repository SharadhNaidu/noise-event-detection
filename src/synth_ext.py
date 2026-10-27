"""Synthetic store from EXTERNAL long background recordings (open data policy; disclosed in METHOD.md).

The gap, measured 2026-10-01: natural Dice is lost on long sustained backgrounds (appliance 16%,
animal 55% frame coverage; whole-clip events 39%). Every existing synthetic store is built from the
competition's own isolated events (median 0.56 s, looped to make long ones) and has 0% whole-clip
events. Here the noise comes from genuinely long real recordings with a known category:
  * DEMAND (CC-BY 4.0, zenodo 1227121), 16 kHz, channel 1, speech-free environments only;
  * DESED synthetic soundbank (CC-BY 4.0, zenodo 6026841), labelled foreground classes only;
  * FSD50K dev (per-clip CC0/CC-BY/CC-BY-NC/Sampling+; organisers confirmed use 2026-10-01), clips picked by
    src/fsd_select.py: one category per clip, no speech-like labels, >= 1.5 s, longest first.
Placement: a fraction of events span the WHOLE host clip; the rest are long (>=1.5 s) segments.
Hosts are all Gold clips >= 3 s (real speech present to mask the source), exactly as build_masked.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A  # noqa: E402

SR = 16000
AUD = (P.ROOT / "audio")
EXT = P.EXT_ROOT

DEMAND = {"DWASHING": "appliance_machine",
          "TCAR": "vehicle_traffic", "TBUS": "vehicle_traffic",
          "TMETRO": "vehicle_traffic", "STRAFFIC": "vehicle_traffic"}
DESED = {"Dog": "animal", "Cat": "animal",
         "Blender": "appliance_machine", "Vacuum_cleaner": "appliance_machine",
         "Electric_shaver_toothbrush": "appliance_machine",
         "Alarm_bell_ringing": "phone_signal_alarm"}


def category(source, name):
    """Competition category for an external source class, or None = do not use."""
    return (DEMAND if source == "demand" else DESED if source == "desed" else {}).get(name)


def place(L, rng, p_whole=0.3, min_s=1.5):
    """Sample span [a, b) in samples inside a host of L samples."""
    if rng.random() < p_whole or L < int(min_s * SR) + 1:
        return 0, L
    n = int(rng.uniform(min_s * SR, 0.9 * L)) if 0.9 * L > min_s * SR else int(min_s * SR)
    n = min(n, L)
    a = int(rng.integers(0, L - n + 1))
    return a, a + n


def _resample(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def load_bank(root=EXT):
    """[(path, category, source_name, seconds)] - audio is read lazily, a window at a time,
    so a 20+ h bank costs no RAM (the laptop also trains)."""
    import csv
    import soundfile as sf
    bank = []
    for env, cat in DEMAND.items():
        for p in sorted((root / "demand").glob(f"{env}/ch01.wav")):
            bank.append((p, cat, f"demand/{env}", sf.info(str(p)).duration))
    for cls, cat in DESED.items():
        for p in sorted((root / "desed").glob(f"**/foreground/{cls}/*.wav")):
            if "eval" in p.parts or "validation" in p.parts:
                continue
            d = sf.info(str(p)).duration
            if d >= 0.5:
                bank.append((p, cat, f"desed/{cls}", d))
    man = root / "fsd50k" / "manifest.tsv"
    if man.exists():
        for r in csv.DictReader(open(man), delimiter="	"):
            p = root / "fsd50k" / "clips" / f"{r['fname']}.wav"
            if p.exists() and p.stat().st_size > 1000:
                bank.append((p, r["category"], f"fsd50k/{r['fname']}", float(r["dur"])))
    return bank


def _bed(path, n, rng):
    """n samples at 16 kHz from a source file: a random window of a long file, or the whole
    (short) file looped with crossfades. Only the needed window is read from disk."""
    import soundfile as sf
    import synth as S
    info = sf.info(str(path))
    need = int(np.ceil(n * info.samplerate / SR)) + 64
    a = int(rng.integers(0, info.frames - need)) if info.frames > need else 0
    x, sr = sf.read(str(path), start=a, frames=min(need, info.frames), dtype="float32", always_2d=True)
    x = _resample(x[:, 0], sr)
    if x.std() < 1e-5:
        return None
    if len(x) >= n:
        return x[:n].copy()
    return S.extend(x, n / SR, rng)[:n]


def _tilt(x, rng):
    from scipy.signal import lfilter
    a = float(rng.uniform(-0.6, 0.6))
    if abs(a) < 0.05:
        return x
    y = lfilter([1.0], [1.0, -a], x).astype(np.float32)
    return y * (np.abs(x).max() + 1e-9) / (np.abs(y).max() + 1e-9)


def build(name="synth_ext", n_clips=6000, seed=11, snr_range=(-10.0, 10.0), p_whole=0.3):
    import synth as S
    rng = np.random.default_rng(seed)
    bank = load_bank()
    if not bank:
        raise SystemExit("empty external bank - extract DEMAND/DESED under $INDOML_EXT_ROOT first")
    by_cat = {}
    for w, c, s, _ in bank:
        by_cat.setdefault(c, []).append((w, c, s))
    print("[ext] bank:", {c: len(v) for c, v in by_cat.items()})
    cats = sorted(by_cat)
    arr, meta = A.load_audio_store(AUD, "gold_audio")
    idx = meta["clips"]
    hosts = [c for c, e in idx.items() if (e.get("dur_measured") or e.get("dur") or 0) >= 3.0]
    rng.shuffle(hosts)
    store = A.AudioStore(AUD, name)
    made, whole = 0, 0
    for k in range(n_clips):
        cid = hosts[k % len(hosts)]
        e = idx[cid]
        host = A.get_wav(arr, e).copy()
        L = len(host); dur = L / SR
        events = [(a, b) for a, b in e.get("events", []) if b <= dur]
        ecats = list(e.get("cats", []))[:len(events)]
        ecats += [None] * (len(events) - len(ecats))
        c = cats[int(rng.integers(0, len(cats)))]                     # balance categories, not files
        w, cat, src = by_cat[c][int(rng.integers(0, len(by_cat[c])))]
        a, b = place(L, rng, p_whole)
        try:
            seg = _bed(w, b - a, rng)
        except Exception:                 # e.g. a file still being downloaded: skip, don't crash the build
            seg = None
        if seg is None:
            continue
        seg = _tilt(seg, rng)
        span = S.mix(host, seg, a, float(rng.uniform(*snr_range)), rng)
        if span is None:
            continue
        whole += (a == 0 and b == L)
        events.append(span); ecats.append(cat)
        peak = float(np.abs(host).max())
        if peak > 0.99:
            host = host * (0.99 / peak)
        order = np.argsort([x for x, _ in events])
        store.add(f"ext{made:06d}", host, {
            "tier": "gold", "dur": dur, "dur_measured": dur,
            "lang": e.get("lang"), "state": e.get("state"), "district": e.get("district"),
            "events": [events[i] for i in order], "cats": [ecats[i] for i in order],
            "synthetic": True, "host": cid, "ext_source": src})
        made += 1
        if made % 1000 == 0:
            print(f"    {made:,} clips", flush=True)
    p = store.close()
    print(f"[ext] wrote {made:,} clips ({whole / max(made, 1):.0%} with a whole-clip event) -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synth_ext")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--p-whole", type=float, default=0.3)
    a = ap.parse_args()
    build(a.name, a.n, a.seed, p_whole=a.p_whole)
