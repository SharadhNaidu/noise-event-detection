"""Build a training store from the TUNE half of the validation release.

Why this is the most in-distribution natural data we have
---------------------------------------------------------
`dataset_info.txt` says the evaluation set is "derived from the same Vaani
corpus recordings, split by state/district/speaker, with noise-event timestamps
verified by annotators". The validation release comes from the same process.
So validation labels are drawn from exactly the annotation procedure that
produced the test labels - which the Gold training tier may not match, since
the natural gap has been our whole problem and Gold is what we train on.

EXP-28 localised that gap precisely: natural events reach 52.8% onset-in-
tolerance but only 43.7% offset, and 88% of our false positives sit on real
events we mislocalised. Adding ~900 natural clips annotated by the same people
who annotated the test set targets that directly.

CONTAMINATION DISCIPLINE, unchanged
-----------------------------------
Only the TUNE half goes in. The DECIDE half is never trained on, by anything,
and remains the honest measurement - as it has all day. The split is the same
speaker-disjoint one every experiment uses (`splits.split_by_speaker`,
frac=0.5, seed=42), so a speaker cannot appear on both sides.

Using the validation release for training is explicitly permitted by the user,
who confirmed it with the organizers' open data policy in mind. It will be
disclosed in the system description.
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import features as F      # noqa: E402
import splits as SP       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
AUD = (P.ROOT / "audio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="valtune_audio")
    ap.add_argument("--half", default="tune", choices=["tune", "decide", "all"])
    a = ap.parse_args()

    refs, meta = D.load_validation()
    tune, decide = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    keep = set({"tune": tune, "decide": decide, "all": list(refs)}[a.half])

    z = zipfile.ZipFile(D.VAL_ZIP)
    paths = {os.path.basename(n)[:-4]: n for n in z.namelist()
             if n.endswith(".wav") and "__MACOSX" not in n
             and "syntheticCleanRefAudio/" not in n}
    if len(paths) != 2501:
        raise SystemExit(f"expected 2501 input clips, found {len(paths)}")

    store = A.AudioStore(AUD, a.name)
    n_ok = 0
    n_syn = 0
    for cid in sorted(keep):
        if cid not in paths:
            continue
        rec = meta[cid]
        evs, cats = F.parse_events(rec.get("NoiseSubCategoryTimeStamp"))
        wav = F.decode_audio(z.read(paths[cid]))
        is_syn = bool(rec.get("syntheticData"))
        n_syn += is_syn
        store.add(cid, wav, {
            "tier": "gold",                     # verified timestamps
            "dur": float(rec.get("duration") or len(wav) / F.TARGET_SR),
            "dur_measured": len(wav) / F.TARGET_SR,
            "lang": rec.get("language"),
            "state": rec.get("state"),
            "district": rec.get("district"),
            "events": evs,
            "cats": cats,
            "from_validation": True,
            "syntheticData": is_syn,
        })
        n_ok += 1

    p = store.close()
    print(f"[valtune] {a.half} half: {n_ok:,} clips "
          f"({n_ok - n_syn:,} natural, {n_syn:,} synthetic)")
    print(f"          -> {p}  "
          f"({os.path.getsize(AUD / (a.name + '.i16')) / 2**30:.2f} GiB)")
    if a.half == "tune":
        print(f"          the {len(decide):,}-clip DECIDE half is untouched and "
              f"remains the measurement")


if __name__ == "__main__":
    main()
