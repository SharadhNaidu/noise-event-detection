"""Train a Track 1 ensemble member (or the habit model) with the exact configuration of the submitted system.

    python train/train.py --member hon_A                 # train one member (runs src/finetune.py)
    python train/train.py --member hon_A --dry-run       # only print the verified command
    python train/train.py --print                        # print the commands of every member
    python train/train.py --member v5_S -- --workers 4   # extra finetune.py flags after `--` (the last value wins)
    python train/train.py --habit [--out checkpoints/habit_model.pkl]

Members: hon_A bts_C bft_S bts_R v5_S v12_ep4 (shipped) and v12 v12s62 v5all (lost, derived recipes), see
train/config.yaml. Needs the audio stores (and, for bts_*, BEATs features) described in README "Training".
Checkpoints are written to $INDOML_ROOT/ckpt/<tag>.pt."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# src/ first: finetune.py imports src/train.py as `train`, which must not resolve to this script
sys.path.insert(0, str(REPO / "src"))
import utils as U  # noqa: E402  (train/utils.py, next to this script; src/ has no utils.py)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", help="member to train")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not train")
    ap.add_argument("--print", dest="print_all", action="store_true", help="print the command of every member")
    ap.add_argument("--habit", action="store_true", help="train the habit model (train/train_habit.py)")
    ap.add_argument("--out", default=str(REPO / "checkpoints" / "habit_model.pkl"), help="habit model output")
    ap.add_argument("--config", default=str(U.CONFIG))
    ap.add_argument("extra", nargs="*", help="extra finetune.py flags, after --")
    a = ap.parse_args()
    if a.habit:
        sys.exit(subprocess.call([sys.executable, str(REPO / "train" / "train_habit.py"), "--out", a.out]))
    conf = U.load_config(a.config)
    names = list(conf["members"]) + list(conf["derived"]) if a.print_all else [a.member]
    if names == [None]:
        ap.error("give --member NAME, --print or --habit")
    parser = U.finetune_parser()
    for name in names:
        argv = U.command(name, conf, parser) + a.extra
        line = "cd src && python finetune.py " + " ".join(shlex.quote(x) for x in argv)
        if a.print_all or a.dry_run:
            print(f"# {name} (re-parses to train/config.yaml exactly)\n{line}\n")
            continue
        print(line, flush=True)
        sys.exit(subprocess.call([sys.executable, "finetune.py", *argv], cwd=str(REPO / "src"), env=os.environ.copy()))


if __name__ == "__main__":
    main()
