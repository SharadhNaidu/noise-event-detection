"""Trainable BEATs-strong encoder (PretrainedSED, Schmid et al. ICASSP 2025).

Same feature layout as extract_beats.BeatsStrong: per 40 ms frame, the 768-d
embedding entering the strong head + 447 AudioSet-strong logits = 1215 dims.
Input is RAW-amplitude audio (BEATs' fbank preprocessing is level-dependent).
PredictionsWrapper resamples every input to `seq_len` frames, so seq_len is
set per call from the input length to keep 40 ms frames (5 s -> 125).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ext" / "PretrainedSED"))
HOP = 0.04
SR = 16000


def _wrapper(model):
    """PretrainedSED base model + AudioSet-strong checkpoint name. Both take RAW-amplitude
    audio (level-dependent mel front-ends), emit 40 ms frames and 768-d embeddings."""
    if model == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        return BEATsWrapper(), "BEATs_strong_1"
    if model == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return ATSTWrapper(), "ATST-F_strong_1"
    raise ValueError(f"unknown in-model encoder {model!r}")


class BeatsEncoder(nn.Module):
    """Trainable PretrainedSED encoder (BEATs by default; ATST-F for --aux atst_ft)."""

    def __init__(self, model="BEATs"):
        super().__init__()
        cwd = os.getcwd()
        os.chdir(REPO / "ext" / "PretrainedSED")     # checkpoints live in ./resources
        try:
            from models.prediction_wrapper import PredictionsWrapper
            base, ckpt = _wrapper(model)
            self.m = PredictionsWrapper(base, checkpoint=ckpt)
        finally:
            os.chdir(cwd)
        self.model_name = model
        self.out_dim = 768 + 447
        self._emb = None
        # a BOUND method, not a lambda closing over `self`: copy.deepcopy (mean teacher)
        # re-binds bound methods to the copy, so each copy captures its own embedding
        self.m.strong_head.register_forward_hook(self._capture)

    def _capture(self, mod, inp, out):
        self._emb = inp[0]

    def forward(self, raw):
        """raw: (B, N) float audio at 16 kHz -> (B, ceil(N/640), 1215)."""
        self.m.seq_len = max(1, int(np.ceil(raw.shape[1] / SR / HOP)))
        y, _ = self.m(self.m.mel_forward(raw.float()))          # (B, 447, T)
        return torch.cat([self._emb, y.transpose(1, 2)], -1)


class DashengEncoder(nn.Module):
    """Trainable Dasheng-base (v10): RAW-amplitude 16 kHz audio -> 768-d tokens on a 40 ms grid
    (64x4 patches over a 10 ms log-mel; a 5 s crop gives 125). No strong logits, so out_dim = 768.
    init_bn is a BatchNorm over mel bins: its running stats stay at the pretrained values
    (batch 6 would drift them); its affine weights still train."""

    def __init__(self, model="Dasheng"):
        super().__init__()
        from dasheng import dasheng_base
        self.m = dasheng_base()
        self.model_name = model
        self.out_dim = 768

    def train(self, mode=True):
        super().train(mode)
        self.m.init_bn.eval()
        return self

    def forward(self, raw):
        """raw: (B, N) float audio at 16 kHz -> (B, ceil(N/640), 768)."""
        return self.m(raw.float())


def make_encoder(model):
    return DashengEncoder(model) if model == "Dasheng" else BeatsEncoder(model)


class MultiEncoder(nn.Module):
    """Several in-model encoders on the same raw audio, features concatenated (all emit 40 ms
    frames, so their sequences align). v9: ("BEATs", "ATST-F"); v10b: ("BEATs", "Dasheng")."""

    def __init__(self, models):
        super().__init__()
        self.encs = nn.ModuleList([make_encoder(m) for m in models])
        self.out_dim = sum(e.out_dim for e in self.encs)

    def forward(self, raw):
        fs = [e(raw) for e in self.encs]
        t = min(f.shape[1] for f in fs)
        return torch.cat([f[:, :t] for f in fs], -1)
