"""Real-real nested-breath store (round 2, R2_07).

Merges are the largest natural error bucket (22% of refs on the test mix), mostly a ~0.36 s breath nested inside a
~2.87 s animal/vehicle/child sound. Gold contains 1,218 natural nested breaths, but their outer/inner length ratio has a
median of 15.8, against 4.7 for the merges we fail on; synth.py never nests events at all (it skips inserts inside
existing events). This builder pastes an ISOLATED real Gold breath (human_non_speech, 0.15-0.8 s) INSIDE an isolated
real Gold long background event (animal / vehicle_traffic / baby_child, >= 1.5 s) of another clip, so that both
events keep their human labels and the outer/inner ratio covers the failing 2-10x range.

Level: natural nested breaths sit at a median -18.5 dB (IQR -23.1..-13.6) relative to the local mixture; the paste gain
is sampled from that distribution, measured the same way (breath RMS vs host RMS over the paste window).
The store must NOT be named *synth* (finetune.source_head routes such names to head 2).
"""
from __future__ import annotations

import argparse

import numpy as np

SR = 16000
DONOR_CAT, HOST_CATS = "human_non_speech", ("animal", "vehicle_traffic", "baby_child")
MARGIN, FADE = 0.03, 0.015


def _isolated(evs, i, pad=0.1):
    a, b = evs[i]
    return all(j == i or e[1] <= a - pad or e[0] >= b + pad for j, e in enumerate(evs))


def pools(clips):
    """clips: {id: {"wav", "events", "cats"}} -> (donors [(id, i)], hosts [(id, i)])."""
    donors, hosts = [], []
    for cid, c in clips.items():
        evs, cats = c["events"], c["cats"]
        for i, (a, b) in enumerate(evs):
            k = cats[i] if i < len(cats) else None
            if not _isolated(evs, i):
                continue
            if k == DONOR_CAT and 0.15 <= b - a <= 0.8 and a >= MARGIN and b + MARGIN <= len(c["wav"]) / SR:
                donors.append((cid, i))
            elif k in HOST_CATS and b - a >= 1.5:
                hosts.append((cid, i))
    return donors, hosts


def _cut(wav, a, b):
    s, e = int((a - MARGIN) * SR), int((b + MARGIN) * SR)
    seg = wav[s:e].astype(np.float64).copy()
    f = int(FADE * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
    seg[:f] *= ramp; seg[-f:] *= ramp[::-1]
    return seg


def make_one(clips, host, donor, rng, gain_db=None):
    """Return (wav, events, cats) for one nested clip built from host (id, i) and donor (id, i)."""
    hid, hi = host; did, di = donor
    hw = clips[hid]["wav"].astype(np.float64) * clips[hid].get("scale", 1.0)
    ha, hb = clips[hid]["events"][hi]
    da, db = clips[did]["events"][di]
    seg = _cut(clips[did]["wav"].astype(np.float64) * clips[did].get("scale", 1.0), da, db)
    L = db - da
    lo, hi_t = ha + 0.1 + MARGIN, hb - 0.1 - MARGIN - L          # breath label strictly inside the host event
    if hi_t <= lo:
        return None
    on = float(rng.uniform(lo, hi_t))
    s = int(round((on - MARGIN) * SR))
    seg = seg[: max(0, min(len(seg), len(hw) - s))]
    core = slice(int(MARGIN * SR), int(MARGIN * SR) + int(L * SR))
    host_rms = np.sqrt(np.mean(hw[s + core.start: s + core.stop] ** 2) + 1e-12)
    don_rms = np.sqrt(np.mean(seg[core] ** 2) + 1e-12)
    g_db = float(np.clip(rng.normal(-18.5, 4.0), -28.0, -8.0)) if gain_db is None else gain_db
    out = hw.copy()
    out[s: s + len(seg)] += seg * (host_rms / don_rms) * 10 ** (g_db / 20)
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out *= 0.99 / peak
    evs = [list(map(float, e)) for e in clips[hid]["events"]] + [[round(on, 3), round(on + L, 3)]]
    cats = list(clips[hid]["cats"][: len(clips[hid]["events"])]) + [DONOR_CAT]
    order = np.argsort([e[0] for e in evs])
    return out.astype(np.float32), [evs[k] for k in order], [cats[k] for k in order]


def build(n=3000, src="gold_audio", name="nest_rr_gold", seed=0):
    import audiostore as A
    import finetune as FT
    arr, me = A.load_audio_store(FT.AUD, src)
    raw = me["clips"]
    clips = {}
    for cid, e in raw.items():
        if not e.get("events"):
            continue
        clips[cid] = {"wav": arr[e["off"]: e["off"] + e["n"]], "scale": 1.0 / 32768.0,   # lazy int16 memmap view
                      "events": e["events"], "cats": e.get("cats", []), "meta": e}
    donors, hosts = pools(clips)
    print(f"[nest] donors {len(donors):,} | hosts {len(hosts):,}", flush=True)
    rng = np.random.default_rng(seed)
    st = A.AudioStore(FT.AUD, name); made = 0; tries = 0
    while made < n and tries < 20 * n:
        tries += 1
        h = hosts[rng.integers(len(hosts))]; d = donors[rng.integers(len(donors))]
        if h[0] == d[0]:
            continue
        r = make_one(clips, h, d, rng)
        if r is None:
            continue
        wav, evs, cats = r
        m = {k: v for k, v in clips[h[0]]["meta"].items() if k not in ("off", "n", "events", "cats")}
        st.add(f"{h[0]}__nest{made:05d}", wav, {**m, "events": evs, "cats": cats, "nest_donor": d[0]})
        made += 1
    print(f"[nest] wrote {made:,} clips -> {st.close()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(n=a.n, seed=a.seed)
