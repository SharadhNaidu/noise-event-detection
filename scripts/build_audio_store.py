"""Build a 16 kHz int16 audio store from a folder or zip of .wav files (the format every src/ module reads).

    python scripts/build_audio_store.py SRC NAME [--root DIR]      # -> DIR/audio/NAME.{i16,json}

e.g. the exact rebuild needs the TEST store:  python scripts/build_audio_store.py data/input_data.zip test_audio
(sample-identical to the store used for the leaderboard runs, verified on all 5,517 clips)."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "inference"))
import infer as I  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("name"); ap.add_argument("--root", default=str(REPO / "workdir"))
a = ap.parse_args()
ids = I.build_store(a.src, a.root, a.name)
print(f"{len(ids)} clips -> {Path(a.root) / 'audio' / a.name}.i16/.json")
