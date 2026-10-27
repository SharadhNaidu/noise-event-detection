"""Flat 16 kHz int16 audio store, so fine-tuning can read waveforms at random.

Fine-tuning the encoder means features can no longer be cached - every epoch
needs the raw waveform again. Decoding 11,111 FLAC/WAV blobs out of parquet
every epoch would make the CPU the bottleneck by a wide margin, so we decode
once into one flat binary and memmap it.

  audio/<name>.i16     int16 PCM at 16 kHz, concatenated, C-order
  audio/<name>.json    {clip_id: {off, n, dur, events, cats, ...}}

Sizing: 16 kHz x 2 bytes = 32 kB per audio-second.
  gold 21.85 h -> 2.5 GB      validation 3.46 h -> 0.40 GB

int16 rather than float32 is a 2x saving with no real loss: the source audio is
16-bit PCM anyway, and WavLM's own feature extractor renormalises per utterance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 16000


class AudioStore:
    def __init__(self, root, name):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.root / f"{name}.i16"
        self.idx_path = self.root / f"{name}.json"
        # Same ordering discipline as FeatureStore: kill the index before
        # truncating the blob, so an interrupted write cannot leave a short
        # blob beside a complete index that a loader would accept.
        if self.idx_path.exists():
            self.idx_path.unlink()
        self._fh = open(self.bin_path, "wb")
        self.index = {}
        self._offset = 0

    def add(self, clip_id, wav, meta):
        w = np.asarray(wav, dtype=np.float32)
        pcm = np.clip(w * 32767.0, -32768, 32767).astype(np.int16)
        self._fh.write(pcm.tobytes())
        self.index[clip_id] = {"off": self._offset, "n": int(len(pcm)), **meta}
        self._offset += int(len(pcm))

    def close(self):
        self._fh.close()
        with open(self.idx_path, "w", encoding="utf-8") as f:
            json.dump({"sr": SR, "total_samples": self._offset,
                       "clips": self.index}, f, ensure_ascii=False)
        return self.idx_path


def load_audio_store(root, name):
    root = Path(root)
    with open(root / f"{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    expect = meta["total_samples"] * 2
    actual = (root / f"{name}.i16").stat().st_size
    if actual != expect:
        raise ValueError(
            f"audio store '{name}' is inconsistent: index expects {expect:,} "
            f"bytes ({meta['total_samples']:,} samples) but the blob is "
            f"{actual:,}. Re-extract it.")
    arr = np.memmap(root / f"{name}.i16", dtype=np.int16, mode="r",
                    shape=(meta["total_samples"],))
    return arr, meta


def get_wav(arr, entry, start=0, length=None):
    """Float32 waveform in [-1, 1] for one clip, optionally a crop."""
    off, n = entry["off"], entry["n"]
    a = off + start
    b = off + n if length is None else min(off + n, a + length)
    return np.asarray(arr[a:b], dtype=np.float32) / 32768.0
