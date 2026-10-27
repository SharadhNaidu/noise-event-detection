"""Write a Codabench submission from saved test posteriors + a decoder.

    python submit_box.py <test_posteriors.pkl> <decoder.pkl> <out.zip>

`test_posteriors.pkl` maps clip_id -> (frame (T,8), dist (T,16), box (T,6)),
as written by the post8 inference script. `decoder.pkl` is a dict
{"kind": "box", "prm": (...)} or {"kind": "hybrid", ...} chosen on the TUNE
half only. Every check from submit_ft.py is repeated on the archive as read
back from disk.
"""
from __future__ import annotations

import json
import pickle
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D       # noqa: E402
import decoders as DC  # noqa: E402


def main():
    post_p, dec_p, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    P = pickle.load(open(post_p, "rb"))
    spec = pickle.load(open(dec_p, "rb"))
    preds = {c: DC.apply(spec, P[c]) for c in P}

    want = set(D.list_test_clip_ids())
    if set(preds) != want:
        raise SystemExit(f"clip_id mismatch: {len(want)} expected, "
                         f"{len(preds)} produced")
    lines = []
    for cid in sorted(preds):
        ev = [{"onset": round(float(s), 3), "offset": round(float(e), 3)}
              for s, e in preds[cid] if e > s]
        lines.append(json.dumps({"clip_id": cid, "events": ev}))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "\n".join(lines) + "\n")

    with zipfile.ZipFile(out) as z:
        if z.namelist() != ["predictions.jsonl"]:
            raise SystemExit(f"archive holds {z.namelist()}")
        rows = [json.loads(l) for l in
                z.read("predictions.jsonl").decode("utf-8").splitlines() if l]
    if {r["clip_id"] for r in rows} != want:
        raise SystemExit("re-read of the archive does not match the test ids")
    bad = [r["clip_id"] for r in rows for e in r["events"]
           if e["offset"] <= e["onset"] or e["onset"] < 0]
    if bad:
        raise SystemExit(f"{len(bad)} malformed events, e.g. {bad[:3]}")
    n_ev = sum(len(r["events"]) for r in rows)
    print(f"wrote {out}  {len(rows):,} clips | {n_ev:,} events | "
          f"{n_ev/len(rows):.2f}/clip | "
          f"{sum(1 for r in rows if not r['events'])} empty")


if __name__ == "__main__":
    main()
