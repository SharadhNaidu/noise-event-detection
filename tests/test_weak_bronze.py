"""Bronze weak supervision: weak clips (e['weak']) train ONLY the clip head, with per-category targets from their
parquet NoiseCategory tags; frame / RED / Dice weight is 0."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import finetune as FT  # noqa: E402
import dataset as DS  # noqa: E402


def test_weights_for_weak_clips():
    idx = {"g": {"events": [[0.5, 1.0]], "cats": ["animal"], "tier": "gold", "dur": 5.0},
           "w": {"events": [], "cats": ["animal"], "tier": "bronze", "dur": 4.0, "weak": True}}
    fw, cw = FT.build_clip_weights(idx, ["g", "w"], 0.5, clip_w_fix=True, keep_deg_frames=True, weak_w=0.3)
    assert fw["w"] == 0.0 and cw["w"] == 0.3
    fw0, cw0 = FT.build_clip_weights(idx, ["g"], 0.5, clip_w_fix=True, keep_deg_frames=True)
    assert fw0["g"] == fw["g"] and cw0["g"] == cw["g"]          # non-weak clips unchanged


def test_weak_clip_target():
    t = FT.weak_clip_target(["animal", "human_non_speech"])
    assert t.shape == (DS.N_OUT,) and t[0] == 1
    assert t[1 + DS.CAT2IDX["animal"]] == 1 and t[1 + DS.CAT2IDX["human_non_speech"]] == 1 and t.sum() == 3
