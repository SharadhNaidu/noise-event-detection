"""Rebuild the final leaderboard submission (LB 1.5926) from cached TEST posteriors.

    python reproduce/rebuild_lb.py [--posteriors DIR] [--audio-root DIR] [--out rebuilt.zip]
                                   [--habit-model checkpoints/habit_model.pkl] [--stage typec_syn86|tta3]

--posteriors : folder with post8_test_audio_<tag>.pkl for the 8 members (+ <tag>_tta for hon_A, bft_S, v5_S);
               scripts/download_assets.py --what posteriors puts them in posteriors/
--audio-root : folder holding test_audio.{i16,json} (needed for routing and the synthetic level filter)
--stage      : tta3 (default) = the final file (LB 1.5926); typec_syn86 = the same chain on non-TTA posteriors
               (LB 1.5907 file `typec` + the synthetic +8/-6 ms edge shift; bit-exact from the shipped caches)
Habit decisions: by default the shipped ones (reproduce/habit_test_shipped.json); --habit-model recomputes them.
Prints the inner predictions.jsonl md5 and a per-clip comparison with submissions/sub_1016_tta3.zip."""
import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import audiostore as A   # noqa: E402
import members as M      # noqa: E402
import postpass as PP    # noqa: E402
import submit_ens as SE  # noqa: E402

# inner predictions.jsonl md5 when run on the published posteriors. typec_syn86 equals the file built during the
# competition byte for byte. tta3: the uploaded file (628e807d...) was built from TTA caches that were lost; the
# published *_tta caches were recomputed on an RTX 4070 (bf16 kernel noise), so tta3 is exact against THESE caches and
# is compared with the uploaded file by agreement below (event F1 0.9996, Dice 0.9989).
EXPECT = {"tta3": "1315f264bb9ed3864db26530ad50d1f9", "typec_syn86": "f820fdfd4a94aa0c0fa203fb1b7f1023"}
UPLOADED = REPO / "submissions" / "sub_1016_tta3.zip"   # the LB 1.5926 file, inner md5 628e807db067dcc10825b11d5e6f1650


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posteriors", default=str(REPO / "posteriors"))
    ap.add_argument("--audio-root", default=str(REPO / "workdir" / "audio"))
    ap.add_argument("--out", default=str(REPO / "workdir" / "rebuilt_sub.zip"))
    ap.add_argument("--habit-model", default=None)
    ap.add_argument("--stage", choices=["tta3", "typec_syn86"], default="tta3")
    a = ap.parse_args()
    arr, me = A.load_audio_store(a.audio_root, "test_audio"); clips = me["clips"]; ids = list(clips)
    nat = SE.natural_by_peak(arr, clips, ids); dur = {c: clips[c]["n"] / A.SR for c in ids}
    mem = M.load_members()
    nat_m = mem["natural"] if a.stage == "tta3" else [t.replace("_tta", "") for t in mem["natural"]]
    C = {t: pickle.load(open(Path(a.posteriors) / f"post8_test_audio_{t}.pkl", "rb")) for t in set(nat_m + mem["synthetic"])}
    if a.habit_model:
        import habit as H
        blob = pickle.load(open(a.habit_model, "rb")); natc = [c for c in ids if nat[c]]
        import torch
        X = H.tag_store(arr, clips, natc, "cuda" if torch.cuda.is_available() else "cpu", log_every=0)
        rows = [H.features(tf, H.state_of(c), dur[c], blob["STATES"]) for c, tf in zip(natc, H.tagfeat(X))]
        habit_p = dict(zip(natc, H.predict_habit(blob, rows).tolist()))
    else:
        habit_p = {c: 1.0 for c in json.load(open(REPO / "reproduce" / "habit_test_shipped.json"))["clips"]}
    cfg = PP.load_cfg()
    final = PP.run_chain(C, ids, nat, dur, arr, clips, habit_p, nat_m, mem["synthetic"], cfg)
    md5 = PP.write_predictions(final, a.out, ids)
    ok = md5 == EXPECT[a.stage] if not a.habit_model else None
    print(f"wrote {a.out}")
    print(f"inner md5 {md5} | expected for {a.stage} from the published posteriors: {EXPECT[a.stage]} -> "
          f"{'EXACT' if ok else 'n/a (habit recomputed)' if ok is None else 'DIFFERS'}")
    r = PP.agreement(final, PP.read_predictions(UPLOADED))
    print(f"vs uploaded LB file {UPLOADED.name}: clips differing {r['clips_differing']}/{r['clips']}, events {r['events'][0]} vs "
          f"{r['events'][1]}, event F1 (0.2 s collar) {r['event_f1']:.4f}, segment Dice {r['dice']:.4f}")
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
