"""Print the exact `finetune.py` launch line of a shipped member, rebuilt from the cfg stored in its checkpoint.

    python train/cfg_to_cmd.py [TAG ...]        (default: every member in configs/members_cfg.json)

configs/members_cfg.json holds the full argparse namespace each checkpoint was trained with. This script captures
finetune.py's own parser (without training anything), emits every argument whose value differs from the parser
default, and re-parses the line to prove it reproduces the stored cfg exactly. Run the printed line from src/."""
import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _Captured(Exception):
    pass


def finetune_parser():
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
        if isinstance(act, argparse._StoreTrueAction):
            out.append(flag)
        elif isinstance(act, argparse._StoreFalseAction):
            out.append(flag)
        elif isinstance(v, (list, tuple)):
            out += [flag, *map(str, v)]
        elif v is None:
            raise ValueError(f"{act.dest}=None differs from default {act.default!r}; cannot express on the command line")
        else:
            out += [flag, str(v)]
    return out


def check(cfg, ap, argv):
    ns = vars(ap.parse_args(argv))
    bad = {k: (cfg[k], ns[k]) for k in cfg if k in ns and ns[k] != cfg[k]
           and not (isinstance(cfg[k], list) and list(ns[k] or []) == cfg[k])}
    return bad


def main():
    cfgs = json.load(open(REPO / "configs" / "members_cfg.json"))
    tags = sys.argv[1:] or list(cfgs)
    ap = finetune_parser()
    for t in tags:
        argv = cfg_to_args(cfgs[t], ap)
        bad = check(cfgs[t], ap, argv)
        if bad:
            sys.exit(f"{t}: round trip mismatch {bad}")
        unknown = sorted(k for k in cfgs[t] if k not in {a.dest for a in ap._actions})
        print(f"# {t}  (round trip OK; runtime-only cfg fields not on the command line: {unknown})")
        print("python finetune.py " + " ".join(shlex.quote(x) for x in argv) + "\n")


if __name__ == "__main__":
    main()
