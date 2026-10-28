"""Frozen BEATs (AudioSet-STRONG, frame-level; Schmid et al. ICASSP 2025) features.

    python src/extract_beats.py --check                 # sanity check on the repo's metro clip
    python src/extract_beats.py [--device=cpu|cuda] val_audio [store ...]    # -> <root>/beats/<store>.f16/.json

Per 40 ms frame: 768-d embedding (input of the strong head) + 447 AudioSet-strong
class logits = 1215 dims, stored with the same layout as the AST stores so
astfeat.load_ast_store() reads them.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(REPO / "ext" / "PretrainedSED"))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402

ROOT = Path(str(P.ROOT))
SR, SEG, HOP = 16000, 160000, 0.04
DIM = 768 + 447


def build(model):
    """PretrainedSED strong checkpoint -> (PredictionsWrapper, embedding dim)."""
    from models.prediction_wrapper import PredictionsWrapper
    if model == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        return PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1"), 768
    if model == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return PredictionsWrapper(ATSTWrapper(), checkpoint="ATST-F_strong_1"), 768
    if model == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        m2d = M2DWrapper()
        d = m2d.m2d.cfg.feature_d
        return PredictionsWrapper(m2d, checkpoint="M2D_strong_1", embed_dim=d), d
    raise ValueError(model)


class BeatsStrong:
    def __init__(self, dev, model="BEATs"):
        cwd = os.getcwd()
        os.chdir(REPO / "ext" / "PretrainedSED")      # checkpoints live in ./resources
        self.m, self.emb_dim = build(model)
        self.m = self.m.to(dev).eval()
        os.chdir(cwd)
        self.dev = dev
        self._emb = None
        self.m.strong_head.register_forward_hook(lambda mod, inp, out: setattr(self, "_emb", inp[0]))

    @torch.no_grad()
    def __call__(self, wav):
        n = len(wav)
        chunks = []
        for s in range(0, max(n, 1), SEG):
            c = wav[s:s + SEG]
            if len(c) < SEG:
                c = np.pad(c, (0, SEG - len(c)))
            x = torch.from_numpy(c.astype(np.float32))[None].to(self.dev)
            y, _ = self.m(self.m.mel_forward(x))              # (1, 447, 250)
            e = self._emb                                       # (1, 250, 768)
            chunks.append(torch.cat([e[0], y[0].transpose(0, 1)], -1).float().cpu().numpy())
        f = np.concatenate(chunks, 0)
        return f[: max(1, int(np.ceil(n / SR / HOP)))]


def check(dev):
    import librosa
    sys.path.insert(0, str(REPO / "ext" / "PretrainedSED"))
    from data_util import audioset_classes
    w, _ = librosa.load(str(REPO / "ext" / "PretrainedSED" / "test_files" /
                            "752547__iscence__milan_metro_coming_in_station.wav"), sr=SR, mono=True)
    f = BeatsStrong(dev)(w)
    assert f.shape[1] == DIM, f.shape
    top = np.argsort(-f[:, 768:].mean(0))[:5]
    names = [audioset_classes.as_strong_train_classes[i] for i in top]
    print("frames", f.shape, "top classes:", names)
    return names


def main():
    args = sys.argv[1:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args and args[0].startswith("--device="):              # inference/infer.py passes its --device through
        dev = args.pop(0).split("=", 1)[1]
    if args == ["--check"]:
        check(dev)
        return
    model = "BEATs"
    if args and args[0].startswith("--model="):
        model = args.pop(0).split("=", 1)[1]
    ex = BeatsStrong(dev, model)
    out = ROOT / ("beats" if model == "BEATs" else model.lower().replace("-", ""))
    for name in args:
        arr, meta = A.load_audio_store(ROOT / "audio", name)
        st = AF.ASTStore(out, name, dim=ex.emb_dim + 447, hop=HOP)
        t0 = time.time()
        for i, (cid, e) in enumerate(meta["clips"].items()):
            st.add(cid, ex(A.get_wav(arr, e)))
            if (i + 1) % 1000 == 0:
                print(f"    {name} {i+1:,}/{len(meta['clips']):,} {(i+1)/(time.time()-t0):.0f} clips/s", flush=True)
        st.close()
        print(f"[beats] {name}: {len(meta['clips']):,} clips {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
