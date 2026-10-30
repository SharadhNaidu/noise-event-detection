"""Helpers of train/train.py: training config (train/config.yaml) -> exact, verified src/finetune.py command."""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / "config.yaml"
sys.path.insert(0, str(REPO / "src"))


class _Captured(Exception):
    pass


def load_config(path=CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def member_cfg(name, conf):
    """Stored cfg of a shipped member, or a lost member's derived cfg (stored cfg + the fields that differ)."""
    if name in conf["members"]:
        return dict(conf["members"][name])
    if name in conf["derived"]:
        d = conf["derived"][name]
        return {**conf["members"][d["from"]], **d["set"]}
    raise SystemExit(f"unknown member {name!r}; known: {sorted(conf['members']) + sorted(conf['derived'])}")


def finetune_parser():
    """finetune.py's own argparse parser, captured without training anything."""
    import finetune as FT
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = grab
    try:
        FT.main()
    except _Captured as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError("finetune.main() never called parse_args")


def cfg_to_args(cfg, ap):
    """argv list reproducing cfg under parser ap; keys the parser does not know are skipped (runtime-only fields)."""
    out = []
    for act in ap._actions:
        if not act.option_strings or act.dest not in cfg or act.dest == "help":
            continue
        v, flag = cfg[act.dest], act.option_strings[-1]
        if v == act.default:
            continue
        if isinstance(act, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            out.append(flag)
        elif isinstance(v, (list, tuple)):
            out += [flag, *map(str, v)]
        elif v is None:
            raise ValueError(f"{act.dest}=None differs from default {act.default!r}; cannot express on the command line")
        else:
            out += [flag, str(v)]
    return out


def check(cfg, ap, argv):
    """{field: (cfg value, parsed value)} for every field the command line fails to reproduce; {} when exact."""
    ns = vars(ap.parse_args(argv))
    return {k: (cfg[k], ns[k]) for k in cfg if k in ns and ns[k] != cfg[k]
            and not (isinstance(cfg[k], list) and list(ns[k] or []) == cfg[k])}


def command(name, conf=None, ap=None):
    """Verified finetune.py argv for member `name` (run from src/)."""
    conf, ap = conf or load_config(), ap or finetune_parser()
    cfg = member_cfg(name, conf)
    argv = cfg_to_args(cfg, ap)
    bad = check(cfg, ap, argv)
    if bad:
        raise SystemExit(f"{name}: command does not reproduce the config: {bad}")
    return argv
