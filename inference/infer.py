"""Track 1 inference: a folder (or zip) of .wav files -> submission zip holding predictions.jsonl.

    python inference/infer.py --audio PATH --out predictions.zip [--work DIR] [--ckpt-dir checkpoints]
                              [--device cuda|cpu] [--batch 8] [--limit N] [--keep-work]

PATH: a folder searched recursively for *.wav, or a .zip (e.g. the organisers' input_data.zip). clip_id = file stem.
Any sample rate / channel count is accepted (down-mixed to mono, resampled to 16 kHz).
Pipeline (configs/model_config.yaml; helpers in inference/utils.py):
  1. audio store (16 kHz int16) and natural/synthetic routing by peak level
  2. frozen BEATs-strong features for the bts_* members
  3. member posteriors (TTA for hon_A, bft_S, v5_S); lost members substituted, see the banner
  4. habit model on clip-pooled BEATs tags (natural clips)
  5. decode chain: champion readouts -> level filter / bgext -> habit span -> type-C -> synthetic edge shift
Output: zip with one predictions.jsonl, one line per input clip, {"clip_id", "events": [{"onset", "offset"}]}."""
import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from inference.utils import build_store, link_checkpoints  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True); ap.add_argument("--out", default="predictions.zip")
    ap.add_argument("--work", default=str(REPO / "workdir" / "infer"))
    ap.add_argument("--ckpt-dir", default=str(REPO / "checkpoints"))
    ap.add_argument("--device", default=None); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    work = Path(a.work); (work / "ckpt").mkdir(parents=True, exist_ok=True)
    os.environ["INDOML_ROOT"] = str(work)                       # every src module reads paths from here
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    link_checkpoints(a.ckpt_dir, work)
    import audiostore as A
    import habit as H
    import members as M
    import postpass as PP
    import submit_ens as SE
    cfg, mem = PP.load_cfg(), M.load_members()
    missing = [t for t in mem["checkpoints"] if not (work / "ckpt" / f"{t}.pt").exists()]
    hab_pkl = Path(a.ckpt_dir) / "habit_model.pkl"
    if missing or not hab_pkl.exists():
        raise SystemExit(f"missing checkpoints {missing + ([] if hab_pkl.exists() else ['habit_model.pkl'])} in {a.ckpt_dir}: run scripts/download_assets.py")
    sub = mem["substitute"]
    print("== live inference: lost members substituted:", ", ".join(f"{k} -> {v}" for k, v in sub.items()), f"| device {dev}", flush=True)
    t0 = time.time(); store = "infer_audio"
    ids = build_store(a.audio, work, store, a.limit)
    arr, me = A.load_audio_store(work / "audio", store); clips = me["clips"]
    nat = SE.natural_by_peak(arr, clips, ids); dur = {c: clips[c]["n"] / A.SR for c in ids}
    print(f"[1] store: {len(ids)} clips | natural {sum(nat.values())} | synthetic {len(ids) - sum(nat.values())} ({time.time() - t0:.0f}s)", flush=True)
    import subprocess
    subprocess.run([sys.executable, str(REPO / "src" / "extract_beats.py"), f"--device={dev}", store], check=True, env={**os.environ, "INDOML_ROOT": str(work)}, cwd=str(REPO / "src"))
    print(f"[2] BEATs features ({time.time() - t0:.0f}s)", flush=True)
    need = {}
    for t in mem["natural"] + mem["synthetic"]:
        base, tta = (t[:-4], True) if t.endswith("_tta") else (t, False)
        need[t] = (sub.get(base, base), tta)
    C = {}
    for t, (ck, tta) in need.items():
        key = (ck, tta)
        if key not in C:
            post, dist = M.posteriors(ck, store, tta, dev, a.batch, ids)
            C[key] = {"post": post, "dist": dist}
            print(f"[3] {ck}{' TTA' if tta else ''} ({time.time() - t0:.0f}s)", flush=True)
    CC = {t: C[need[t]] for t in need}
    natc = [c for c in ids if nat[c]]
    habit_p = {}
    if natc:
        blob = pickle.load(open(hab_pkl, "rb"))
        X = H.tag_store(arr, clips, natc, dev, log_every=0)
        rows = [H.features(tf, H.state_of(c), dur[c], blob["STATES"]) for c, tf in zip(natc, H.tagfeat(X))]
        habit_p = dict(zip(natc, H.predict_habit(blob, rows).tolist()))
    print(f"[4] habit model: {sum(p >= cfg['habit']['thr'] for p in habit_p.values())} natural clips flagged ({time.time() - t0:.0f}s)", flush=True)
    final = PP.run_chain(CC, ids, nat, dur, arr, clips, habit_p, mem["natural"], mem["synthetic"], cfg)
    md5 = PP.write_predictions(final, a.out, ids)
    print(f"[5] wrote {a.out}: {len(ids)} clips, {sum(len(v) for v in final.values())} events, inner md5 {md5} ({time.time() - t0:.0f}s)")
    if not a.keep_work:
        # the predictions are written; scratch cleanup must never turn a good run into a failure. Drop the memmaps
        # first (Windows refuses to delete a mapped file), then delete best-effort.
        del arr, me, clips, C, CC
        import gc
        gc.collect()
        for sub_dir in ("audio", "beats"):
            for f in (work / sub_dir).glob(f"{store}.*"):
                try:
                    f.unlink()
                except OSError as e:
                    print(f"  note: could not remove scratch file {f} ({e}); safe to delete by hand", flush=True)
        if (work / "ckpt").resolve() != Path(a.ckpt_dir).resolve():   # links/copies of --ckpt-dir, never the originals
            shutil.rmtree(work / "ckpt", ignore_errors=True)


if __name__ == "__main__":
    main()
