"""submit_ft must run an AST-trained checkpoint WITH its AST stream.

Production change that makes this fail: building the net without aux_dim, or
calling clip_posteriors without ast= — the model would see all-zero AudioSet
input on the test set, the mirror image of the fin_X training bug.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import submit_ft as SF  # noqa: E402

ROOT = Path(os.environ.get("INDOML_ROOT", str(Path(__file__).resolve().parents[1] / "workdir")))
CFG = {"finetune": True, "enc_layers": 4, "crop": 5.0}


def test_net_gets_aux_dim_only_when_trained_with_ast():
    assert SF.build_net({**CFG, "use_ast": True}, "cpu").aux_dim == 768
    assert SF.build_net(CFG, "cpu").aux_dim == 0


def test_ast_features_loaded_for_ast_checkpoint():
    feats = SF.ast_for({**CFG, "use_ast": True}, "valall_audio", ROOT / "ast")
    arr, clips, hop, dim = feats
    assert arr.shape[1] == 768 and len(clips) > 0
    assert SF.ast_for(CFG, "valall_audio", ROOT / "ast") is None


def test_beats_checkpoint_builds_and_loads_beats_stream():
    cfg = {**CFG, "use_ast": True, "aux": "beats"}
    assert SF.build_net(cfg, "cpu").aux_dim == 1215
    arr, clips, hop, dim = SF.ast_for(cfg, "val_audio", SF.aux_root(cfg))
    assert dim == 1215 and abs(hop - 0.04) < 1e-9 and len(clips) > 0


def test_missing_ast_store_fails_loudly():
    try:
        SF.ast_for({**CFG, "use_ast": True}, "no_such_store", ROOT / "ast")
    except FileNotFoundError:
        return
    raise AssertionError("an AST checkpoint without test AST features must not run")


if __name__ == "__main__":
    bad = 0
    for t in [test_net_gets_aux_dim_only_when_trained_with_ast,
              test_ast_features_loaded_for_ast_checkpoint, test_beats_checkpoint_builds_and_loads_beats_stream,
              test_missing_ast_store_fails_loudly]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
