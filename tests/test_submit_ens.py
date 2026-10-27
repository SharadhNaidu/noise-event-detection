"""Ensemble submission: a one-member ensemble must reproduce that member's own predictions.

Production change that makes this fail: members decoded with a different
readout, aux streams dropped at inference, or averaging that alters a single member.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import audiostore as A    # noqa: E402
import finetune as FT     # noqa: E402
import gate as G          # noqa: E402
import postproc as PP     # noqa: E402
import submit_ens as SE   # noqa: E402
import train as T         # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))
TAG = os.environ.get("ENS_TEST_TAG", "hon_A")


def test_single_member_matches_member():
    arr, me = A.load_audio_store(ROOT / "audio", "val_audio")
    idx = me["clips"]
    ids = list(idx)[:12]
    net, blob = G.load(TAG, "cpu")
    post, dist, _ = FT.clip_posteriors(net, arr, idx, ids, blob["cfg"]["crop"], "cpu",
                                       ast=G.aux_for(blob["cfg"]))
    rc = dict(blob["readout"]); rc.setdefault("kind", "global")
    dur = {c: idx[c]["n"] / 16000 for c in ids}
    want = PP.clip_to_duration(T.decode_all(post, rc, dist, ids), dur)
    got = SE.ensemble_preds([TAG], "val_audio", ids, "cpu", rc)
    assert got == want
    assert all(b <= dur[c] + 1e-9 for c in ids for _, b in got[c])   # nothing past the real clip end


if __name__ == "__main__":
    try:
        test_single_member_matches_member(); print("PASS test_single_member_matches_member")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL test_single_member_matches_member: {type(e).__name__}: {e}"); sys.exit(1)
