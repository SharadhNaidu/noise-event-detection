"""Build synthetic training clips with EXACT boundaries.

The case for this, measured
---------------------------
EXP-18 put the model's per-event posterior rise at 0.36 s against an 0.086 s
tolerance, and EXP-16 put the median boundary error at 1.05x the tolerance -
we miss by a hair, on half the events. Every boundary the model has ever been
trained on is an annotator's estimate of when a breath started, and a model
fitted to jittered targets learns the jitter. Against the incumbent our
natural-clip score is 0.8880 vs 1.0280 while synthetic is 1.2379 vs 1.2994:
the gap is concentrated exactly where the labels are human-made.

Synthetic mixtures have boundaries that are programmatic. We know when the
noise starts because we put it there. That is the only boundary supervision
available to us that carries no annotation noise at all.

Where the noise comes from
--------------------------
`validation.zip` ships 695 matched pairs: `syntheticCleanRefAudio/X.wav` and
`syntheticNoiseAudio/X.wav`. Their difference IS the added noise, in isolation.
Verified on a sample: the residual's energy envelope spans 0.285-2.957 s on a
clip whose labelled events run 0.265-2.969 s.

CONTAMINATION DISCIPLINE. Those pairs live in validation, so the noise bank is
harvested ONLY from clips in the TUNE half. The DECIDE half is never touched,
by any part of this, and remains the honest measurement. The speaker-disjoint
tune/decide split is the same one every other experiment uses.

Where the speech comes from
---------------------------
Gold training clips whose own noise coverage is lowest - the quietest speech we
have. They are not literally clean, so a mixture's "true" boundaries are those
of the ADDED events plus whatever the clip already carried; we keep both, with
the added ones exact.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import splits as SP       # noqa: E402

SR = 16000
REPO = Path(__file__).resolve().parent.parent
AUD = (P.ROOT / "audio")


def harvest_noise(tune_only=True, min_dur=0.15, pad=0.02):
    """Isolated noise events from the clean/noisy twins, with exact times.

    Returns [(waveform, category)] - one entry per labelled event, cut from the
    residual `noisy - clean` at the reference boundaries. No envelope detection
    is involved: the label IS the boundary, which is the whole point.
    """
    import soundfile as sf

    refs, meta = D.load_validation()
    tune, _ = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    keep = set(tune) if tune_only else set(refs)

    z = zipfile.ZipFile(D.VAL_ZIP)
    names = [n for n in z.namelist()
             if n.endswith(".wav") and "__MACOSX" not in n]
    clean = {n.split("/")[-1][:-4]: n for n in names
             if "syntheticCleanRefAudio/" in n}
    noisy = {n.split("/")[-1][:-4]: n for n in names
             if "syntheticNoiseAudio/" in n}
    pairs = sorted(set(clean) & set(noisy) & keep)

    bank = []
    for cid in pairs:
        c, _ = sf.read(io.BytesIO(z.read(clean[cid])), dtype="float32")
        n, _ = sf.read(io.BytesIO(z.read(noisy[cid])), dtype="float32")
        L = min(len(c), len(n))
        resid = n[:L] - c[:L]
        cats = [e.get("category") for e in
                meta[cid].get("NoiseSubCategoryTimeStamp", [])]
        for i, (a, b) in enumerate(sorted(refs[cid])):
            if b - a < min_dur:
                continue
            ia = max(0, int((a - pad) * SR))
            ib = min(L, int((b + pad) * SR))
            seg = resid[ia:ib]
            if len(seg) < int(min_dur * SR) or seg.std() < 1e-5:
                continue
            bank.append((np.ascontiguousarray(seg),
                         cats[i] if i < len(cats) else None))
    return bank, len(pairs)


def quiet_speech(store="gold_audio", n=4000, seed=0):
    """Gold clips with the least existing noise coverage, as mixing hosts."""
    arr, meta = A.load_audio_store(AUD, store)
    idx = meta["clips"]
    scored = []
    for cid, e in idx.items():
        dur = e.get("dur_measured") or e.get("dur") or 0
        if dur < 3.0:
            continue
        cov = sum(b - a for a, b in e.get("events", [])) / max(dur, 1e-6)
        scored.append((cov, cid))
    scored.sort()
    picked = [c for _, c in scored[:n]]
    rng = np.random.default_rng(seed)
    rng.shuffle(picked)
    return arr, idx, picked


def augment(seg, rng):
    """Perturb one noise event so the model cannot memorise the recording.

    MEASURED NEED: 84% of decide-half synthetic clips contain a noise segment
    matching the tune-half bank at cosine >= 0.99 - the whole validation
    synthetic set is drawn from one pool of ~1,161 recordings. Training on the
    raw bank therefore teaches "I recognise this bird", which transfers to the
    test set only if it shares the pool. Perturbation forces the model to learn
    "a transient was added here" instead, which transfers either way.

    Each transform preserves event EXTENT in time except the resample, whose
    factor is returned so the caller can keep the boundary exact.
    """
    x = seg.astype(np.float32).copy()
    # resample: changes pitch and duration together, like a different source
    rate = float(rng.uniform(0.85, 1.18))
    if abs(rate - 1.0) > 0.02:
        n = max(8, int(round(len(x) / rate)))
        x = np.interp(np.linspace(0, len(x) - 1, n),
                      np.arange(len(x)), x).astype(np.float32)
    # one-pole tilt: stands in for a different room or microphone
    a = float(rng.uniform(-0.6, 0.6))
    if abs(a) > 0.05:
        y = np.empty_like(x)
        prev = 0.0
        for i in range(len(x)):          # cheap 1-pole, deliberate
            prev = x[i] + a * prev
            y[i] = prev
        x = y / (np.abs(y).max() + 1e-9) * (np.abs(seg).max() + 1e-9)
    if rng.random() < 0.35:              # occasional reversal
        x = x[::-1].copy()
    return np.ascontiguousarray(x, dtype=np.float32)


def mix(host, noise, at, snr_db, rng):
    """Add `noise` into `host` starting at sample `at`, at a target SNR.

    SNR is measured against the host's own RMS over the SAME span, not the whole
    clip, so a quiet passage does not silently receive a deafening event.
    """
    L = len(noise)
    end = min(len(host), at + L)
    if end - at < int(0.05 * SR):
        return None
    seg = noise[: end - at]
    h_rms = float(np.sqrt(np.mean(host[at:end] ** 2))) + 1e-9
    n_rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
    g = (h_rms / n_rms) * (10.0 ** (-snr_db / 20.0))
    host[at:end] += g * seg
    return (at / SR, end / SR)


def build(name="synth_audio", n_clips=6000, seed=0, snr_range=(-5.0, 15.0),
          max_events=4, tune_only=True):
    """tune_only=False harvests from ALL validation pairs instead of just the
    tune half. Only for a FINAL model that trains on all validation anyway -
    it makes the decide half unusable as a measurement, which is a fair trade
    once the recipe has already been validated with tune_only=True."""
    rng = np.random.default_rng(seed)
    bank, n_pairs = harvest_noise(tune_only=tune_only)
    print(f"[synth] noise bank: {len(bank):,} isolated events from {n_pairs} "
          f"TUNE-half pairs")
    if not bank:
        raise SystemExit("empty noise bank")
    durs = np.array([len(w) / SR for w, _ in bank])
    print(f"[synth] event durations: p25 {np.percentile(durs,25):.2f}s  "
          f"median {np.median(durs):.2f}s  p75 {np.percentile(durs,75):.2f}s")

    arr, idx, hosts = quiet_speech(n=max(n_clips, 4000), seed=seed)
    print(f"[synth] {len(hosts):,} quiet Gold clips available as hosts")

    store = A.AudioStore(AUD, name)
    made = 0
    for k in range(n_clips):
        cid = hosts[k % len(hosts)]
        e = idx[cid]
        host = A.get_wav(arr, e).copy()
        dur = len(host) / SR
        if dur < 3.0:
            continue
        # keep whatever the host already carried, then add exact-boundary events
        events = [(a, b) for a, b in e.get("events", []) if b <= dur]
        cats = list(e.get("cats", []))[:len(events)]
        while len(cats) < len(events):
            cats.append(None)

        n_add = int(rng.integers(1, max_events + 1))
        added = 0
        for _ in range(n_add):
            w, cat = bank[int(rng.integers(0, len(bank)))]
            w = augment(w, rng)
            if len(w) / SR > 0.8 * dur:
                continue
            at = int(rng.uniform(0, len(host) - len(w)))
            # do not start an added event inside one the host already has
            t = at / SR
            if any(a - 0.05 <= t <= b + 0.05 for a, b in events):
                continue
            span = mix(host, w, at, float(rng.uniform(*snr_range)), rng)
            if span is None:
                continue
            events.append(span)
            cats.append(cat)
            added += 1
        if not added:
            continue

        peak = float(np.abs(host).max())
        if peak > 0.99:
            host = host * (0.99 / peak)
        order = np.argsort([a for a, _ in events])
        store.add(f"syn{made:06d}", host, {
            "tier": "gold", "dur": dur, "dur_measured": dur,
            "lang": e.get("lang"), "state": e.get("state"),
            "district": e.get("district"),
            "events": [events[i] for i in order],
            "cats": [cats[i] for i in order],
            "synthetic": True, "host": cid,
        })
        made += 1
        if made % 1000 == 0:
            print(f"    {made:,} clips", flush=True)

    p = store.close()
    print(f"[synth] wrote {made:,} clips -> {p}  "
          f"({os.path.getsize(AUD/(name+'.i16'))/2**30:.2f} GiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synth_audio")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-pairs", action="store_true",
                    help="harvest noise from ALL validation pairs, not just "
                         "the tune half; only for a final untuned model")
    a = ap.parse_args()
    build(a.name, a.n, a.seed, tune_only=not a.all_pairs)


def extend(seg, target_s, rng, xf=0.05):
    """Lengthen a noise event to `target_s` by looping with crossfades.

    The bank's events have a median duration of 0.56 s, so a long event has to
    be built rather than found. Looping is acoustically honest for the sources
    that actually produce long events - a machine hum, traffic, an engine are
    stationary, and a bird calling repeatedly over five seconds is a real
    thing. The crossfade stops the seam from becoming a click, which would be
    a spurious transient the model could learn to key on.
    """
    L = int(target_s * SR)
    if len(seg) >= L:
        return seg[:L].copy()
    k = max(1, int(xf * SR))
    out = seg.astype(np.float32).copy()
    while len(out) < L:
        nxt = seg.astype(np.float32).copy()
        if rng.random() < 0.5:
            nxt = nxt[::-1].copy()          # avoid an obvious period
        k2 = min(k, len(out), len(nxt))
        ramp = np.linspace(0.0, 1.0, k2, dtype=np.float32)
        tail = out[-k2:] * (1 - ramp) + nxt[:k2] * ramp
        out = np.concatenate([out[:-k2], tail, nxt[k2:]])
    return out[:L]


def build_masked(name="synth_mask", n_clips=9000, seed=5,
                 snr_range=(-12.0, 2.0), dur_range=(2.0, 8.0),
                 tune_only=True):
    """Long events that are PRESENT throughout but only intermittently AUDIBLE.

    This manufactures the exact situation EXP-32 identified as our failure and
    that no existing store contains. Measured: synth_full is 8.3% long events
    at a 0.49 s median, Gold 16.6% - neither teaches a model to hold presence
    across a gap where speech masks the source.

    Three deliberate departures from `build`:
      * events are EXTENDED to 2-8 s, the range where frame recall collapses
        (57.7% at 1.5-4 s, 34.6% above 4 s);
      * hosts are drawn from ALL Gold clips, not the quietest, so there is
        real speech to do the masking;
      * the SNR is LOW (-12..+2 dB against the host's local RMS), so the
        source genuinely disappears under loud passages.

    The label still covers the full extent, because the source is still there.
    That is the whole point: exact supervision for 'present but inaudible',
    which is the one thing we cannot get from annotated data.
    """
    rng = np.random.default_rng(seed)
    # tune_only=True by default: a noise bank drawn from the decide half
    # leaks its audio into training and destroys the only instrument that
    # ranks models honestly. That mistake cost a submission once already.
    bank, n_pairs = harvest_noise(tune_only=tune_only)
    print(f"[mask] bank: {len(bank):,} events from {n_pairs} pairs "
          f"({'TUNE half only' if tune_only else 'ALL pairs - CONTAMINATED'})")

    arr, meta = A.load_audio_store(AUD, "gold_audio")
    idx = meta["clips"]
    hosts = [c for c, e in idx.items()
             if (e.get("dur_measured") or e.get("dur") or 0) >= 4.0]
    rng.shuffle(hosts)
    print(f"[mask] {len(hosts):,} Gold hosts of >=4s (ALL of them, not the "
          f"quietest - we need speech present to do the masking)")

    store = A.AudioStore(AUD, name)
    made = 0
    masked_frac = []
    for k in range(n_clips):
        cid = hosts[k % len(hosts)]
        e = idx[cid]
        host = A.get_wav(arr, e).copy()
        dur = len(host) / SR
        if dur < 4.0:
            continue
        events = [(a, b) for a, b in e.get("events", []) if b <= dur]
        cats = list(e.get("cats", []))[:len(events)]
        while len(cats) < len(events):
            cats.append(None)

        w, cat = bank[int(rng.integers(0, len(bank)))]
        tgt = float(rng.uniform(dur_range[0], min(dur_range[1], dur * 0.85)))
        if tgt < 1.5:
            continue
        seg = augment(extend(w, tgt, rng), rng)
        at = int(rng.uniform(0, max(1, len(host) - len(seg))))
        span = mix(host, seg, at, float(rng.uniform(*snr_range)), rng)
        if span is None:
            continue
        events.append(span)
        cats.append(cat)

        # how much of the event is actually buried? recorded so the store can
        # be checked rather than assumed to contain masked frames.
        a0, b0 = span
        i0, i1 = int(a0 * SR), int(b0 * SR)
        hs = host[i0:i1]
        fr = max(1, len(hs) // 320)
        he = np.array([np.sqrt((hs[t*320:(t+1)*320] ** 2).mean() + 1e-12)
                       for t in range(fr)])
        masked_frac.append(float(np.mean(he > 2 * np.median(he))))

        peak = float(np.abs(host).max())
        if peak > 0.99:
            host = host * (0.99 / peak)
        order = np.argsort([a for a, _ in events])
        store.add(f"msk{made:06d}", host, {
            "tier": "gold", "dur": dur, "dur_measured": dur,
            "lang": e.get("lang"), "state": e.get("state"),
            "district": e.get("district"),
            "events": [events[i] for i in order],
            "cats": [cats[i] for i in order],
            "synthetic": True, "host": cid, "masked_long": True,
        })
        made += 1
        if made % 2000 == 0:
            print(f"    {made:,} clips", flush=True)

    p = store.close()
    print(f"[mask] wrote {made:,} clips -> {p}  "
          f"({os.path.getsize(AUD/(name+'.i16'))/2**30:.2f} GiB)")
    if masked_frac:
        print(f"[mask] on average {100*np.mean(masked_frac):.1f}% of each "
              f"event's span sits under a loud host passage")
