"""Build an upload = the 1.5808 champion with a new NATURAL-route member list; synthetic route = champion rows, unchanged.

Safety first: the champion natural route is rebuilt from the same TEST posterior caches and must match the uploaded zip
on every natural clip, or nothing is written (a changed cache, readout or decode would silently corrupt an upload).

    python src/build_upload.py OUTNAME.zip m1 m2 ...        # natural members (mean ensemble)
    python src/build_upload.py --check                      # safety check only

Ported 2026-10-09 from the 191c3e9f scratchpad (champ/build_upload.py) so it survives the Temp folder.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiostore as A        # noqa: E402
import ens_gate as E          # noqa: E402
import finetune as FT         # noqa: E402
import postproc as PP         # noqa: E402
import submit_ens as SE       # noqa: E402
import submit_ft as SF        # noqa: E402
from solo_measure import MD02, ROOT  # noqa: E402

SUB = Path(__file__).resolve().parent.parent / "artifacts" / "submissions"
CHAMP = SUB / "sub_1008_hyb_v12ro2_synv5all.zip"
CHAMP_MD5 = "041dfec8ac7aa0d048448aedb4040626"
CHAMP_NAT = ["hon_A", "bts_C", "bft_S", "bts_R", "v12", "v5_S"]


def rows(zp):
    with zipfile.ZipFile(zp) as z:
        return {json.loads(l)["clip_id"]: [(e["onset"], e["offset"]) for e in json.loads(l)["events"]]
                for l in z.read("predictions.jsonl").decode().splitlines() if l}


def cache(t):
    for d in ("srvcache", "ckpt"):
        p = ROOT / d / f"post8_test_audio_{t}.pkl"
        if p.exists():
            return pickle.load(open(p, "rb"))
    raise SystemExit(f"no TEST posterior cache for {t}")


def r3(evs):
    return [tuple(round(x, 3) for x in e[:2]) for e in evs if e[1] > e[0]]


class Builder:
    def __init__(self):
        assert hashlib.md5(open(CHAMP, "rb").read()).hexdigest() == CHAMP_MD5, "champion zip changed on disk"
        arr, me = A.load_audio_store(FT.AUD, "test_audio")
        self.ids = list(me["clips"])
        self.nat = SE.natural_by_peak(arr, me["clips"], self.ids)
        self.dur = SE.store_durations("test_audio")
        self.champ = rows(CHAMP)
        self.C = {}

    def natroute(self, members, readout=MD02):
        for t in members:
            self.C.setdefault(t, cache(t))
        ids = self.ids
        post8 = E.average([self.C[t]["post"] for t in members]); dist = E.average([self.C[t]["dist"] for t in members])
        post = {c: post8[c][:, 0] for c in ids}
        p = SE.decode_routed(post, dist, ids, readout, self.nat)
        p = SE.apply_overlay(p, post8, ids, readout.get("overlay"), self.nat)
        return PP.clip_to_duration(PP.empty_clip_fallback(p, post, q=0.7), self.dur)

    def check(self):
        ref = self.natroute(CHAMP_NAT)
        n_nat = sum(self.nat.values())
        same = sum(r3(ref[c]) == self.champ[c] for c in self.ids if self.nat[c])
        print(f"safety: champion natural route rebuilt {same}/{n_nat}", flush=True)
        if same != n_nat:
            raise SystemExit("cannot reproduce the champion; refusing to build")

    def build(self, out, members, readout=MD02):
        self.check()
        p = self.natroute(members, readout)
        new = {c: (r3(p[c]) if self.nat[c] else self.champ[c]) for c in self.ids}
        chg = sum(new[c] != self.champ[c] for c in self.ids)
        print(f"members {members}: natural clips changed {chg}; synthetic changed "
              f"{sum(new[c] != self.champ[c] for c in self.ids if not self.nat[c])}")
        print("natural events", sum(len(new[c]) for c in self.ids if self.nat[c]),
              "vs champion", sum(len(self.champ[c]) for c in self.ids if self.nat[c]))
        path = SUB / out
        SF.write_zip(new, str(path), set(self.ids))
        z = rows(path)
        assert set(z) == set(self.ids) and all(z[c] == new[c] for c in self.ids), "written zip does not read back identically"
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        print("md5", md5, path)
        return path, md5


def main():
    b = Builder()
    if len(sys.argv) < 2 or sys.argv[1] == "--check":
        b.check(); return
    b.build(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
