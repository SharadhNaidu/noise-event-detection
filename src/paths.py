"""Single source of every data path. Set INDOML_ROOT to the folder holding audio/, ckpt/, posteriors/, ...
(default: <repo>/workdir). INDOML_EXT_ROOT holds optional external noise banks (default: <INDOML_ROOT>_ext)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("INDOML_ROOT", str(REPO / "workdir")))
EXT_ROOT = Path(os.environ.get("INDOML_EXT_ROOT", str(ROOT) + "_ext"))
