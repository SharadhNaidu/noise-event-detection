"""Frozen AudioSet (AST) features, cached at their native resolution.

Why a second stream at all
--------------------------
EXP-33, with a matched-resolution control: at an identical 101 ms grid, AST is
WORSE than WavLM overall (AP 0.4496 vs 0.5250) and BETTER on events of >=1.5 s
(0.3799 vs 0.3511). That is complementarity, not a resolution artifact - AST
was trained on 2M AudioSet clips across 527 classes including Bird, Vehicle,
Animal and Machinery, and WavLM's objective never once rewarded representing
any of them.

It matters because of EXP-32: the model reports AUDIBILITY where the reference
marks PRESENCE, and long events carry 80.6% of the Dice loss. A coarse stream
is a BETTER presence detector precisely because it pools over 100 ms and
survives a momentary masking that flattens a 20 ms frame.

Storage and resolution
----------------------
AST tokenises a 1024x128 mel spectrogram into 16x16 patches at stride 10,
giving 101 time positions x 12 frequency positions. We average over frequency
and keep time, so one clip becomes (n_time, 768) at a hop of 10.24/101 s ~=
101 ms. Cached at that native rate - upsampling to the 20 ms grid before
storage would cost 5x the disk for no information.

A clip longer than 10.24 s is covered by overlapping windows and averaged
where they overlap, the same scheme `clip_posteriors` uses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 16000
AST_WIN_S = 10.24
AST_T = 101
AST_HOP_S = AST_WIN_S / AST_T          # ~0.1014 s
AST_DIM = 768


class ASTExtractor:
    def __init__(self, name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 device="cuda", token=None):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
        self.torch = torch
        self.device = device
        self.fe = ASTFeatureExtractor.from_pretrained(name, token=token)
        self.m = ASTModel.from_pretrained(name, token=token).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def _window(self, w):
        """(n_time, 768) for one <=10.24 s chunk."""
        torch = self.torch
        inp = self.fe(w, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            out = self.m(inp["input_values"].to(self.device)).last_hidden_state[0]
        tok = out[2:]                                   # drop CLS + distill
        nf = tok.shape[0] // AST_T
        h = tok[: AST_T * nf].reshape(AST_T, nf, -1).mean(1)
        return h.float().cpu().numpy()

    def __call__(self, wav):
        """(n_frames, 768) at AST_HOP_S, covering the whole clip."""
        n = len(wav)
        W = int(AST_WIN_S * SR)
        n_out = max(1, int(np.ceil(n / SR / AST_HOP_S)))
        if n <= W:
            h = self._window(wav)
            keep = max(1, min(AST_T, int(round(n / SR / AST_HOP_S))))
            return h[:keep].astype(np.float16)
        # overlapping windows, averaged where they overlap
        acc = np.zeros((n_out, AST_DIM), np.float32)
        cnt = np.zeros((n_out, 1), np.float32)
        step = W // 2
        starts = list(range(0, n - W + 1, step))
        if starts[-1] + W < n:
            starts.append(n - W)
        for s in starts:
            h = self._window(wav[s:s + W])
            a = int(round(s / SR / AST_HOP_S))
            take = min(AST_T, n_out - a)
            if take <= 0:
                continue
            acc[a:a + take] += h[:take]
            cnt[a:a + take] += 1
        return (acc / np.maximum(cnt, 1)).astype(np.float16)


class ASTStore:
    """Same flat-blob-plus-index discipline as the other stores, including the
    delete-index-before-truncating order that stops an interrupted write from
    being silently accepted."""

    def __init__(self, root, name, dim=AST_DIM, hop=AST_HOP_S):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.root / f"{name}.f16"
        self.idx_path = self.root / f"{name}.json"
        self.dim = dim
        self.hop = hop
        if self.idx_path.exists():
            self.idx_path.unlink()
        self._fh = open(self.bin_path, "wb")
        self.index = {}
        self._off = 0

    def add(self, clip_id, feats):
        f = np.ascontiguousarray(feats, dtype=np.float16)
        assert f.ndim == 2 and f.shape[1] == self.dim, f.shape
        self._fh.write(f.tobytes())
        self.index[clip_id] = {"off": self._off, "n": int(f.shape[0])}
        self._off += int(f.shape[0])

    def close(self):
        self._fh.close()
        json.dump({"dim": self.dim, "hop": self.hop,
                   "total_frames": self._off, "clips": self.index},
                  open(self.idx_path, "w"))
        return self.idx_path


def load_ast_store(root, name):
    root = Path(root)
    meta = json.load(open(root / f"{name}.json"))
    expect = meta["total_frames"] * meta["dim"] * 2
    actual = (root / f"{name}.f16").stat().st_size
    if actual != expect:
        raise ValueError(f"AST store '{name}' inconsistent: index expects "
                         f"{expect:,} bytes, blob is {actual:,}. Re-extract.")
    arr = np.memmap(root / f"{name}.f16", dtype=np.float16, mode="r",
                    shape=(meta["total_frames"], meta["dim"]))
    return arr, meta


# second-stream encoders: store directory, feature dim, validation store name
AUX = {"ast": ("ast", AST_DIM, "valall_audio"), "beats": ("beats", 768 + 447, "val_audio"),
       "beats_ft": (None, 768 + 447, None),   # computed in-model from raw audio
       "atst_ft": (None, 768 + 447, None),    # same layout: ATST-Frame embedding + 447 strong logits
       "beats_atst_ft": (None, 2 * (768 + 447), None),   # v9: BEATs + ATST-F, concatenated
       "dasheng_ft": (None, 768, None),                  # v10a: Dasheng-base tokens (no strong logits)
       "beats_dasheng_ft": (None, 768 + 447 + 768, None)}  # v10b: BEATs + Dasheng, concatenated
# second streams computed INSIDE the model (trainable), keyed to their PretrainedSED model name(s)
IN_MODEL = {"beats_ft": "BEATs", "atst_ft": "ATST-F", "beats_atst_ft": ("BEATs", "ATST-F"),
            "dasheng_ft": "Dasheng", "beats_dasheng_ft": ("BEATs", "Dasheng")}


def upsample(h, n_out, hop_in=AST_HOP_S, hop_out=0.02):
    """AST's ~101 ms grid onto WavLM's 20 ms grid, by nearest neighbour.

    Nearest rather than linear on purpose: this stream carries a presence
    decision, and interpolating between two windows would invent a soft ramp
    at every boundary - exactly the smearing the fine stream exists to avoid.
    """
    if len(h) == 0:
        return np.zeros((n_out, h.shape[1] if h.ndim > 1 else AST_DIM),
                        np.float32)
    idx = np.clip((np.arange(n_out) * hop_out / hop_in).astype(np.int64),
                  0, len(h) - 1)
    return np.asarray(h, dtype=np.float32)[idx]
