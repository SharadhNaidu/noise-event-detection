"""WavLM feature extraction and caching.

Design notes
------------
EXP benchmarks put the laptop's RTX 4070 at ~4,100x realtime on a WavLM-shaped
forward, so the GPU is not the bottleneck - CPU audio decode is. We therefore
decode in a process pool (32 cores available) and keep the GPU fed with large
batches.

Storage: one flat fp16 binary per tier plus a JSON index, rather than 90k small
files. This gives O(1) random access via np.memmap during training (essential -
streaming can only buffer-shuffle, and EXP-04 showed shards are not tier-uniform
so we must shuffle globally).

  features/<tier>.f16     raw float16, shape (total_frames, 768), C-order
  features/<tier>.json    {clip_id: {off, n, dur, tier, events, cats, lang, ...}}

Sizing at 768-d / 50 Hz / fp16 = 276 MB per audio-hour:
  gold 6.0 GB · silver 27.7 GB · bronze 9.0 GB · val 1.0 GB · test 3.0 GB
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

FRAME_SEC = 0.02          # WavLM-base+ emits 50 Hz
FEAT_DIM = 768
TARGET_SR = 16000

TIER_SHORT = {
    "verified_timestamps": "gold",
    "unverified_timestamps": "silver",
    "no_timestamps": "bronze",
}


# ---------------------------------------------------------------------------
# audio decode (CPU, runs in worker processes)
# ---------------------------------------------------------------------------
def decode_audio(blob, target_sr=TARGET_SR):
    """Decode arbitrary audio bytes to mono float32 at target_sr.

    The HF `Audio` feature has changed representation twice; we always pass
    decode=False and handle raw bytes ourselves so this stays version-proof.
    """
    import soundfile as sf

    if isinstance(blob, dict):
        if blob.get("bytes"):
            blob = blob["bytes"]
        elif blob.get("path"):
            blob = open(blob["path"], "rb").read()
        else:
            raise ValueError("audio dict has neither bytes nor path")

    wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def parse_events(raw):
    """NoiseSubCategoryTimeStamp -> (list[(on, off)], list[category]).

    start/end are STRINGS in this dataset despite the README example showing
    floats. Malformed entries are dropped rather than crashing a 90k-row pass.
    """
    evs, cats = [], []
    for e in (raw or []):
        try:
            on, off = float(e["start"]), float(e["end"])
        except (TypeError, ValueError, KeyError):
            continue
        if off < on:
            on, off = off, on
        if on < 0:
            continue
        evs.append((on, off))
        cats.append(e.get("category"))
    order = sorted(range(len(evs)), key=lambda i: evs[i])
    return [evs[i] for i in order], [cats[i] for i in order]


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
class WavLMExtractor:
    """Frozen WavLM-base+ truncated to (n_layers - drop_last) transformer blocks.

    Two details that are easy to get wrong and silently corrupt everything:

    1. ATTENTION MASK. wavlm-base-plus sets return_attention_mask=True. Batching
       clips of different lengths without a mask lets self-attention read the
       zero padding; EXP-06 measured max|solo - batched| = 3.79 on real clips
       when the mask was omitted. The mask is mandatory, not an optimisation.
    2. NORMALISATION. The checkpoint sets do_normalize=True (zero-mean/unit-var
       per utterance). Feeding raw waveforms shifts the whole feature space.

    Both are handled by delegating to the checkpoint's own feature extractor
    rather than hand-rolling the batching.
    """

    def __init__(self, name="microsoft/wavlm-base-plus", drop_last=2,
                 device="cuda", dtype=None, token=None, layer=None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch = torch
        self.device = device
        self.dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
        self.fe = AutoFeatureExtractor.from_pretrained(name, token=token)
        # EXP-15: a linear probe over every layer found layer 1 beats layer 10
        # by +0.057 AP for noise detection. WavLM's upper layers specialise for
        # phonetic/semantic content and discard the acoustic detail a horn or a
        # breath is made of. `layer` selects which hidden state to cache;
        # None keeps the old behaviour (final state of the truncated encoder).
        self.layer = layer
        self.model = AutoModel.from_pretrained(
            name, token=token, output_hidden_states=layer is not None)
        if drop_last > 0:
            self.model.encoder.layers = self.model.encoder.layers[:-drop_last]
        self.model = self.model.to(device=device, dtype=self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def n_layers(self):
        return len(self.model.encoder.layers)

    def frames_for(self, n_samples):
        """Exact encoder output length for an input of n_samples, from the
        model's own conv-stack arithmetic (never guessed from duration)."""
        t = self.model._get_feat_extract_output_lengths(
            self.torch.tensor([int(n_samples)]))
        return int(t[0])

    def __call__(self, wavs):
        """wavs: list of 1-D float32 arrays -> list of (T_i, 768) float16 arrays.

        Refuses ragged batches. See EXP-07/EXP-08: this checkpoint's group-norm
        front-end means even 1% zero padding injects ~7% relative error, and
        EXP-08 showed the error is per-frame noise (|mean|/std ~ 0.002), not a
        bias a downstream model could learn around. Equal-length batches only.
        """
        torch = self.torch
        lens = {len(w) for w in wavs}
        if len(lens) > 1:
            raise ValueError(
                f"ragged batch ({min(lens)}..{max(lens)} samples). WavLM-base+ uses "
                f"feat_extract_norm='group', so padding corrupts the whole clip. "
                f"Use extract_all() (solo, 180x realtime) or batch equal lengths.")
        enc = self.fe([np.asarray(w, dtype=np.float32) for w in wavs],
                      sampling_rate=TARGET_SR, return_tensors="pt",
                      padding=False, return_attention_mask=True)
        x = enc["input_values"].to(self.device, self.dtype)
        am = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            out = self.model(x, attention_mask=am)
            h = (out.hidden_states[self.layer] if self.layer is not None
                 else out.last_hidden_state)                          # (B, T, 768)
        h = h.float().cpu().numpy().astype(np.float16)
        out = []
        for i, w in enumerate(wavs):
            t = min(self.frames_for(len(w)), h.shape[1])
            out.append(h[i, : max(1, t)])
        return out

    def extract_all(self, wavs):
        """Encode a list of variable-length clips, one at a time.

        Measured at ~180x realtime on an RTX 4070 -> ~56 min for the whole
        169 h (train + validation + test). Batching would be faster but is not
        available to us at any padding level, so solo it is.
        """
        return [self([w])[0] for w in wavs]


# ---------------------------------------------------------------------------
# cache writer
# ---------------------------------------------------------------------------
class FeatureStore:
    """Append-only fp16 feature blob + JSON index."""

    def __init__(self, root, name, dim=FEAT_DIM, layer=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.root / f"{name}.f16"
        self.idx_path = self.root / f"{name}.json"
        self.dim = dim
        # Which encoder hidden state these features came from. Recorded so a
        # model can never be evaluated against a store from a different layer -
        # that failure is silent (well-shaped features, meaningless values).
        self.layer = layer
        # Remove any existing index FIRST. Opening the blob 'wb' truncates it, so
        # a crash mid-write would otherwise leave a short .f16 beside a complete
        # .json from a previous run - which load_store would happily accept and
        # silently feed garbage into training. Seen for real on the gold store.
        if self.idx_path.exists():
            self.idx_path.unlink()
        self._fh = open(self.bin_path, "wb")
        self.index = {}
        self._offset = 0

    def add(self, clip_id, feats, meta):
        feats = np.ascontiguousarray(feats, dtype=np.float16)
        assert feats.ndim == 2 and feats.shape[1] == self.dim, feats.shape
        self._fh.write(feats.tobytes())
        self.index[clip_id] = {"off": self._offset, "n": int(feats.shape[0]), **meta}
        self._offset += int(feats.shape[0])

    def close(self):
        self._fh.close()
        with open(self.idx_path, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "frame_sec": FRAME_SEC,
                       "layer": self.layer,
                       "total_frames": self._offset, "clips": self.index},
                      f, ensure_ascii=False)
        return self.idx_path


def load_store(root, name):
    """Return (memmap (total_frames, dim), index dict).

    Refuses to open a store whose blob size disagrees with its index - that
    combination means an interrupted write, and reading it would hand training
    quietly-wrong features rather than raising.
    """
    root = Path(root)
    idx_path, bin_path = root / f"{name}.json", root / f"{name}.f16"
    with open(idx_path, encoding="utf-8") as f:
        meta = json.load(f)
    expect = meta["total_frames"] * meta["dim"] * 2
    actual = bin_path.stat().st_size
    if actual != expect:
        raise ValueError(
            f"feature store '{name}' is inconsistent: index expects "
            f"{expect:,} bytes ({meta['total_frames']:,} frames) but the blob is "
            f"{actual:,} bytes. Re-extract it.")
    arr = np.memmap(bin_path, dtype=np.float16, mode="r",
                    shape=(meta["total_frames"], meta["dim"]))
    return arr, meta


def store_layer(root, name):
    """Which encoder layer a store was built from. None = the truncated
    encoder's last hidden state, i.e. the original layer-10 stores."""
    with open(Path(root) / f"{name}.json", encoding="utf-8") as f:
        return json.load(f).get("layer")


def assert_same_layer(root, names):
    """Refuse to mix stores built from different encoder layers.

    Training on layer 1 and evaluating on layer 10 yields predictions that pass
    every structural check and are numerically meaningless. Making that an
    exception rather than a silent wrong answer is the whole point.
    """
    seen = {n: store_layer(root, n) for n in names}
    distinct = set(seen.values())
    if len(distinct) > 1:
        raise ValueError(
            "feature stores come from different encoder layers: "
            + ", ".join(f"{n}=layer {v}" for n, v in seen.items()))
    return distinct.pop() if distinct else None


def get(arr, index_entry):
    off, n = index_entry["off"], index_entry["n"]
    return arr[off: off + n]
