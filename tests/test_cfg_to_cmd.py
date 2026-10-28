import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "train"))
import cfg_to_cmd as C  # noqa: E402


def test_every_member_cfg_round_trips_through_finetune_parser():
    cfgs = json.load(open(REPO / "configs" / "members_cfg.json"))
    ap = C.finetune_parser()
    for tag, cfg in cfgs.items():
        argv = C.cfg_to_args(cfg, ap)
        assert C.check(cfg, ap, argv) == {}, tag
        assert argv[argv.index("--tag") + 1] == cfg["tag"]


def test_non_default_flag_detected():
    ap = C.finetune_parser()
    cfg = {"seed": 7, "epochs": ap.get_default("epochs")}
    assert C.cfg_to_args(cfg, ap) == ["--seed", "7"]
