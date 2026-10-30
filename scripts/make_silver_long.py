"""Build the silver_long training store from silver_audio: keep clips whose longest event lasts >= 1.5 s.

    python scripts/make_silver_long.py [--src silver_audio] [--name silver_long] [--min-len 1.5]

Copies the int16 samples unchanged (no float round trip) into $INDOML_ROOT/audio/<name>.{i16,json}.
The competition store also dropped 216 of the 12,230 candidates with a filter that was not recorded,
so this script should keep about 12,230 clips where ours kept 12,014."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiostore as A   # noqa: E402
import paths as P        # noqa: E402


def longest(entry):
    return max((e - s for s, e in entry.get("events") or []), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="silver_audio")
    ap.add_argument("--name", default="silver_long")
    ap.add_argument("--min-len", type=float, default=1.5)
    a = ap.parse_args()
    aud = Path(P.ROOT) / "audio"
    arr, meta = A.load_audio_store(aud, a.src)
    keep = {c: e for c, e in meta["clips"].items() if longest(e) >= a.min_len}
    idx, off = {}, 0
    with open(aud / f"{a.name}.i16", "wb") as f:
        for cid, e in keep.items():
            f.write(arr[e["off"]:e["off"] + e["n"]].tobytes())
            idx[cid] = {**e, "off": off}
            off += e["n"]
    with open(aud / f"{a.name}.json", "w", encoding="utf-8") as f:
        json.dump({"sr": A.SR, "total_samples": off, "clips": idx}, f, ensure_ascii=False)
    print(f"[silver_long] {len(keep):,} of {len(meta['clips']):,} clips have an event >= {a.min_len} s -> {aud / a.name}.i16")


if __name__ == "__main__":
    main()
