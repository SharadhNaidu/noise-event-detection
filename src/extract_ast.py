"""Cache frozen AST features for an audio store.

    python src/extract_ast.py gold_audio valall_audio silver_long ...
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402

AUD = (P.ROOT / "audio")
OUT = (P.ROOT / "ast")
REPO = Path(__file__).resolve().parent.parent


def token():
    """HF token for gated/private downloads; public models need none. HF_TOKEN env, else an optional .env, else None."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO / ".env"
    if not env.exists():
        return None
    for line in open(env, encoding="utf-8"):
        if line.strip().startswith(("HF_TOKEN", "HUGGING_FACE_PAT")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main():
    names = sys.argv[1:]
    if not names:
        raise SystemExit("usage: extract_ast.py <store> [<store> ...]")
    ex = AF.ASTExtractor(token=token())
    for name in names:
        arr, meta = A.load_audio_store(AUD, name)
        clips = meta["clips"]
        store = AF.ASTStore(OUT, name)
        t0 = time.time()
        for i, (cid, e) in enumerate(clips.items()):
            store.add(cid, ex(A.get_wav(arr, e)))
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"    {name} {i+1:,}/{len(clips):,}  "
                      f"{el/60:.1f} min  {(i+1)/el:.0f} clips/s", flush=True)
        p = store.close()
        print(f"[ast] {name}: {len(clips):,} clips, "
              f"{(time.time()-t0)/60:.1f} min -> {p} "
              f"({os.path.getsize(OUT/(name+'.f16'))/2**30:.2f} GiB)",
              flush=True)


if __name__ == "__main__":
    main()
