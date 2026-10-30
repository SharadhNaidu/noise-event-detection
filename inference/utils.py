"""Helpers of inference/infer.py: wav input (folder or zip) -> 16 kHz audio store, checkpoint linking."""
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np


def _wav_sources(src):
    """[(clip_id, name, reader)] sorted by name; rejects colliding stems."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"--audio {src} does not exist")
    if src.suffix.lower() == ".zip":
        z = zipfile.ZipFile(src)
        names = sorted(n for n in z.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n and not os.path.basename(n).startswith("._"))
        items = [(os.path.basename(n)[:-4], n, (lambda n=n: z.read(n))) for n in names]
    else:
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() == ".wav" and not p.name.startswith("._"))
        items = [(p.stem, str(p), (lambda p=p: p.read_bytes())) for p in files]
    stems = [i[0] for i in items]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        raise ValueError(f"colliding clip ids (same file stem in several places): {dupes[:5]}")
    if not items:
        raise ValueError(f"no .wav files found in {src}")
    return items


def build_store(src, root, name, limit=None):
    """Decode every wav exactly as the shipped stores were built (features.decode_audio -> AudioStore.add)."""
    import audiostore as A
    import features as F
    items = _wav_sources(src)[:limit] if limit else _wav_sources(src)
    store = A.AudioStore(Path(root) / "audio", name)
    for cid, name, read in items:
        try:
            wav = F.decode_audio(read())
        except Exception as e:                                   # one unreadable file must not sink the whole run
            print(f"  WARNING: could not decode {name} ({type(e).__name__}: {e}); clip {cid} gets no events", flush=True)
            wav = np.zeros(0, np.float32)
        store.add(cid, wav, {"dur_measured": len(wav) / F.TARGET_SR})
    store.close()
    return [i[0] for i in items]


def link_checkpoints(ckpt_dir, work):
    """Mirror <ckpt_dir>/*.pt into <work>/ckpt (src modules read $INDOML_ROOT/ckpt/<tag>.pt). Always rebuilt, so a
    re-downloaded or removed checkpoint is never shadowed by a stale link from an earlier run."""
    dst_dir = Path(work) / "ckpt"
    if dst_dir.resolve() == Path(ckpt_dir).resolve():           # checkpoints already live in the work dir: use in place
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    for f in Path(ckpt_dir).glob("*.pt"):
        try:
            os.link(f, dst_dir / f.name)
        except OSError:
            shutil.copy2(f, dst_dir / f.name)
