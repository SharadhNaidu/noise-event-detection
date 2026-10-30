import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# loaded by file path: putting train/ on sys.path would shadow src/train.py, which finetune.py imports as `train`
_spec = importlib.util.spec_from_file_location("train_utils", REPO / "train" / "utils.py")
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)


def test_every_member_cfg_round_trips_through_finetune_parser():
    conf = U.load_config()
    ap = U.finetune_parser()
    for name in list(conf["members"]) + list(conf["derived"]):
        cfg = U.member_cfg(name, conf)
        argv = U.command(name, conf, ap)
        assert U.check(cfg, ap, argv) == {}, name
        assert argv[argv.index("--tag") + 1] == cfg["tag"]


def test_derived_members():
    conf = U.load_config()
    assert U.member_cfg("v12s62", conf)["seed"] == 62
    assert U.member_cfg("v12", conf) == conf["members"]["v12_ep4"]
    v5all = U.member_cfg("v5all", conf)
    assert "valall_audio" in v5all["train_stores"] and "valtune_audio" not in v5all["train_stores"]
    assert {k: v for k, v in v5all.items() if k not in ("tag", "train_stores")} == \
        {k: v for k, v in conf["members"]["v5_S"].items() if k not in ("tag", "train_stores")}


def test_non_default_flag_detected():
    ap = U.finetune_parser()
    cfg = {"seed": 7, "epochs": ap.get_default("epochs")}
    assert U.cfg_to_args(cfg, ap) == ["--seed", "7"]


def test_model_config_has_every_constant():
    import yaml
    conf = yaml.safe_load(open(REPO / "configs" / "model_config.yaml", encoding="utf-8"))
    assert set(conf) == {"ensemble", "postpass", "readouts"}
    assert json.dumps(conf["readouts"]["natural"]["natural"]) == json.dumps(
        {"kind": "asym", "thr_on": 0.6, "thr_off": 0.7, "min_dur": 0.2, "merge_gap": 0.06, "refine": 0.2})
