"""Fine-tuned BEATs stream (aux_mode="beats_ft").

Production changes that make these fail: feeding BEATs normalised instead of
raw-amplitude audio; letting PredictionsWrapper stretch a 5 s crop to 250
frames (20 ms instead of 40 ms); BEATs weights frozen by accident.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiostore as A      # noqa: E402
import beats_enc as BE      # noqa: E402
import extract_beats as XB  # noqa: E402
import finetune as FT       # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _clip(seconds):
    ar, me = A.load_audio_store(ROOT / "audio", "val_audio")
    e = next(e for e in me["clips"].values() if e["n"] >= int(16000 * seconds))
    return A.get_wav(ar, e)[: int(16000 * seconds)].astype(np.float32)


def test_matches_frozen_extractor_on_10s():
    w = _clip(10.0)
    ref = XB.BeatsStrong(DEV)(w)                                  # (250, 1215)
    enc = BE.BeatsEncoder().to(DEV).eval()
    with torch.no_grad():
        got = enc(torch.from_numpy(w)[None].to(DEV))[0].float().cpu().numpy()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.allclose(got, ref, atol=2e-2), float(np.abs(got - ref).max())


def test_5s_crop_keeps_40ms_frames():
    enc = BE.BeatsEncoder().to(DEV).eval()
    with torch.no_grad():
        got = enc(torch.from_numpy(_clip(5.0))[None].to(DEV))
    assert got.shape[1] == 125, got.shape


def test_hybrid_restores_raw_scale_and_trains_beats():
    w = _clip(5.0)
    std = float(w.std() + 1e-7)
    wn = (w - w.mean()) / std
    net = FT.FineTuneSED(4, aux_dim=1215, aux_mode="beats_ft").to(DEV).eval()
    assert all(p.requires_grad for p in net.aux_enc.parameters())
    with torch.no_grad():
        feats = net.beats_features(torch.from_numpy(wn)[None].to(DEV),
                                   torch.tensor([[std]], device=DEV))
        direct = net.aux_enc(torch.from_numpy(w - w.mean())[None].to(DEV))
        out = net(torch.from_numpy(wn)[None].to(DEV), aux=torch.tensor([[std]], device=DEV))
    assert torch.allclose(feats, direct, atol=2e-2)
    assert out[0].shape == (1, 249, 8)


def test_optimizer_trains_beats():
    """Behavioural (was a source-text check): every trainable in-model BEATs tensor
    must be in an optimizer group, under the old recipe AND under --llrd."""
    import inspect
    import torch.nn as nn
    assert "param_groups(" in inspect.getsource(FT.main), "main() no longer builds groups via param_groups"
    net = nn.Module()
    net.enc = nn.Linear(2, 2); net.head = nn.Linear(2, 2); net.aux_mode = "beats_ft"
    net.aux_enc = nn.Module(); net.aux_enc.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    for kw in ({}, {"beats_top": 1e-4, "llrd": 0.8}):
        got = {id(p) for g in FT.param_groups(net, 5e-5, 1e-3, **kw) for p in g["params"]}
        assert all(id(p) in got for p in net.aux_enc.parameters()), \
            f"fine-tuned BEATs params never reach the optimizer ({kw or 'old recipe'})"


def test_submit_and_gate_build_beats_ft_nets():
    import gate as G
    import submit_ft as SF
    cfg = {"finetune": True, "enc_layers": 4, "crop": 5.0, "use_ast": True, "aux": "beats_ft"}
    assert SF.build_net(cfg, "cpu").aux_mode == "beats_ft"
    assert SF.ast_for(cfg, "test_audio", SF.aux_root(cfg)) == "scale"
    assert G.aux_for(cfg) == "scale"


if __name__ == "__main__":
    bad = 0
    for t in [test_matches_frozen_extractor_on_10s, test_5s_crop_keeps_40ms_frames,
              test_hybrid_restores_raw_scale_and_trains_beats, test_optimizer_trains_beats,
              test_submit_and_gate_build_beats_ft_nets]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)


def test_deepcopy_teacher_gets_its_own_embedding():
    """Mean teacher deep-copies the student. The strong-head hook must write the
    embedding onto the COPY, not the original (a lambda capturing `self` did the
    latter: the teacher's _emb stayed None and the MT run crashed at step 0)."""
    import copy
    enc = BE.BeatsEncoder().eval()
    tea = copy.deepcopy(enc)
    x = torch.randn(1, 16000) * 0.1
    with torch.no_grad():
        enc._emb = None
        out = tea(x)
    assert tea._emb is not None and out.shape[-1] == 1215
    assert enc._emb is None          # the student was not touched by the teacher's forward
