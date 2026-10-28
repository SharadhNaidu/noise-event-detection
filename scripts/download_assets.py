"""Download and md5-verify everything the code needs that is too large for git.

    python scripts/download_assets.py [--what checkpoints|posteriors|beats|all] [--dest REPO_ROOT]

checkpoints : 6 member checkpoints + habit_model.pkl   -> checkpoints/          (GitHub release assets)
posteriors  : cached TEST posteriors of all members    -> posteriors/           (for reproduce/rebuild_lb.py)
beats       : BEATs_strong_1.pt (PretrainedSED v0.0.1) -> ext/PretrainedSED/resources/
No token is needed: every source is public. Files already present with the right md5 are skipped."""
import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASE = "https://github.com/SharadhNaidu/noise-event-detection/releases/download/v1.0"
BEATS_URL = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/BEATs_strong_1.pt"
BEATS_MD5 = "0b3d1253749058df5551beeb0a8f5e87"
TIMEOUT = 120   # seconds without data before a stalled connection is abandoned


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, p):
    """Download url to p via a .part file, so an interrupted download never looks complete."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    print("  downloading", url)
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    tmp.replace(p)


def get(url, p, want):
    p = Path(p)
    if not (p.exists() and md5(p) == want):
        fetch(url, p)
    ok = md5(p) == want
    print(f"  {'OK ' if ok else 'BAD'} {p}")
    if not ok:
        p.unlink()                      # never leave a corrupt file where the code would load it; rerun to retry
    return [] if ok else [str(p)]


def release_files(prefix, dest):
    """md5.txt (a release asset) lists '<md5>  <repo path>'; the asset name is the file's basename."""
    try:
        table = urllib.request.urlopen(f"{RELEASE}/md5.txt", timeout=TIMEOUT).read().decode().splitlines()
    except urllib.error.URLError as e:
        sys.exit(f"cannot read {RELEASE}/md5.txt ({e}). Is the GitHub release v1.0 reachable from this machine?")
    bad, n = [], 0
    for line in table:
        if not line.strip():
            continue
        want, name = line.split(maxsplit=1)
        if name.startswith(prefix):
            n += 1
            bad += get(f"{RELEASE}/{Path(name).name}", Path(dest) / name, want)
    if not n:
        sys.exit(f"md5.txt of the release lists no entries under {prefix}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["checkpoints", "posteriors", "beats", "all"], default="all")
    ap.add_argument("--dest", default=str(REPO))
    a = ap.parse_args()
    bad = []
    if a.what in ("beats", "all"):
        bad += get(BEATS_URL, Path(a.dest) / "ext" / "PretrainedSED" / "resources" / "BEATs_strong_1.pt", BEATS_MD5)
    if a.what in ("checkpoints", "all"): bad += release_files("checkpoints/", a.dest)
    if a.what in ("posteriors", "all"): bad += release_files("posteriors/", a.dest)
    if bad:
        sys.exit(f"md5 mismatch: {bad}")
    print("all assets verified")


if __name__ == "__main__":
    main()
