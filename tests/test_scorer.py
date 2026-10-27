"""Prove src/scorer.py is bit-identical to the organizers' reference code.

Run:  python tests/test_scorer.py
Exits non-zero on any disagreement. Everything downstream depends on this
passing, so it is deliberately paranoid: hand-built edge cases first, then
thousands of randomised comparisons including adversarial tie/boundary cases.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reference_scorer as REF  # noqa: E402
import scorer as FAST  # noqa: E402

FAILURES = []
CHECKS = 0


def cmp(name, ref_data, pred_data):
    """Compare both implementations on one (ref, pred) pair."""
    global CHECKS
    CHECKS += 1
    r_f1 = REF.event_based_f1(ref_data, pred_data)
    f_f1 = FAST.event_based_f1(ref_data, pred_data)
    r_di = REF.segment_dice(ref_data, pred_data)
    f_di = FAST.segment_dice(ref_data, pred_data)
    ok = abs(r_f1 - f_f1) < 1e-12 and abs(r_di - f_di) < 1e-12
    if not ok:
        FAILURES.append((name, r_f1, f_f1, r_di, f_di, ref_data, pred_data))
    return ok


# ---------------------------------------------------------------- edge cases
print("=" * 70)
print("HAND-BUILT EDGE CASES")
print("=" * 70)

EDGE = {
    "both empty":
        ({"a": []}, {"a": []}),
    "ref empty, pred has events":
        ({"a": []}, {"a": [(1.0, 2.0)]}),
    "pred empty, ref has events":
        ({"a": [(1.0, 2.0)]}, {"a": []}),
    "pred clip missing entirely":
        ({"a": [(1.0, 2.0)]}, {}),
    "extra clip_id in pred (must count as FP)":
        ({"a": [(1.0, 2.0)]}, {"a": [(1.0, 2.0)], "ghost": [(0.5, 0.9), (2.0, 3.0)]}),
    "perfect match":
        ({"a": [(1.0, 2.0), (3.0, 4.0)]}, {"a": [(1.0, 2.0), (3.0, 4.0)]}),
    "exactly at tolerance (0.2*1.0 = 0.2)":
        ({"a": [(1.0, 2.0)]}, {"a": [(1.2, 2.2)]}),
    "just outside tolerance":
        ({"a": [(1.0, 2.0)]}, {"a": [(1.2001, 2.2001)]}),
    "short event hits the 50 ms floor":
        ({"a": [(1.00, 1.10)]}, {"a": [(1.05, 1.15)]}),
    "short event just outside the floor":
        ({"a": [(1.00, 1.10)]}, {"a": [(1.0501, 1.1501)]}),
    "overlapping reference events":
        ({"a": [(1.0, 3.0), (2.0, 4.0)]}, {"a": [(1.0, 3.0), (2.0, 4.0)]}),
    "two preds compete for one ref (tie-break)":
        ({"a": [(1.0, 2.0)]}, {"a": [(1.05, 2.05), (0.95, 1.95)]}),
    "two refs compete for one pred":
        ({"a": [(1.0, 2.0), (1.02, 2.02)]}, {"a": [(1.01, 2.01)]}),
    "exact distance tie (ri/pi ordering decides)":
        ({"a": [(1.0, 2.0), (1.0, 2.0)]}, {"a": [(1.01, 2.01), (0.99, 1.99)]}),
    "zero-duration reference event":
        ({"a": [(1.0, 1.0)]}, {"a": [(1.0, 1.0)]}),
    "zero-duration, slight offset":
        ({"a": [(1.0, 1.0)]}, {"a": [(1.03, 1.03)]}),
    "event starting at t=0":
        ({"a": [(0.0, 0.5)]}, {"a": [(0.0, 0.5)]}),
    "blob merge: 2 refs vs 1 wide pred":
        ({"a": [(1.0, 1.4), (1.6, 2.0)]}, {"a": [(1.0, 2.0)]}),
    "over-split: 1 ref vs 2 narrow preds":
        ({"a": [(1.0, 2.0)]}, {"a": [(1.0, 1.4), (1.6, 2.0)]}),
    "many clips, mixed":
        ({"a": [(1.0, 2.0)], "b": [], "c": [(0.1, 0.2), (0.5, 0.9)]},
         {"a": [(1.0, 2.0)], "b": [(5.0, 6.0)], "c": [(0.1, 0.2)]}),
    "long event, wide tolerance":
        ({"a": [(0.0, 10.0)]}, {"a": [(1.5, 11.5)]}),
    "identical onsets, different offsets":
        ({"a": [(1.0, 2.0), (1.0, 5.0)]}, {"a": [(1.0, 2.0), (1.0, 5.0)]}),
}

for name, (rd, pd) in EDGE.items():
    ok = cmp(name, rd, pd)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")

# ---------------------------------------------------------------- randomised
print()
print("=" * 70)
print("RANDOMISED PROPERTY TESTS")
print("=" * 70)


def rand_events(rng, n_max=6, dur_max=12.0, short_bias=False):
    n = rng.randint(0, n_max)
    ev = []
    for _ in range(n):
        on = round(rng.uniform(0.0, dur_max), 3)
        if short_bias:
            length = round(rng.uniform(0.01, 0.4), 3)
        else:
            length = round(rng.uniform(0.02, 4.0), 3)
        ev.append((on, round(on + length, 3)))
    return ev


SUITES = [
    ("general",            dict(short_bias=False, jitter=0.35, n_clips=8)),
    ("short events (breath-like)", dict(short_bias=True, jitter=0.06, n_clips=8)),
    ("tight jitter (near-tolerance)", dict(short_bias=True, jitter=0.05, n_clips=6)),
    ("coarse jitter (mostly misses)", dict(short_bias=False, jitter=2.0, n_clips=6)),
    ("dense clips", dict(short_bias=True, jitter=0.1, n_clips=3, n_max=14)),
]

for suite_name, cfg in SUITES:
    rng = random.Random(hash(suite_name) & 0xFFFF)
    fails_before = len(FAILURES)
    N = 400
    for it in range(N):
        ref_data, pred_data = {}, {}
        for c in range(cfg["n_clips"]):
            cid = f"clip{c}"
            ev = rand_events(rng, n_max=cfg.get("n_max", 6), short_bias=cfg["short_bias"])
            ref_data[cid] = ev
            # perturb to make predictions
            pred = []
            for on, off in ev:
                if rng.random() < 0.15:
                    continue  # drop -> FN
                j1 = rng.uniform(-cfg["jitter"], cfg["jitter"])
                j2 = rng.uniform(-cfg["jitter"], cfg["jitter"])
                p_on = max(0.0, round(on + j1, 4))
                p_off = max(p_on, round(off + j2, 4))
                pred.append((p_on, p_off))
            if rng.random() < 0.2:  # spurious -> FP
                on = round(rng.uniform(0, 12), 3)
                pred.append((on, round(on + rng.uniform(0.05, 1.0), 3)))
            pred_data[cid] = pred
        if rng.random() < 0.1:
            pred_data["phantom_clip"] = rand_events(rng)
        if rng.random() < 0.1 and pred_data:
            pred_data.pop(next(iter(pred_data)))
        cmp(f"{suite_name}#{it}", ref_data, pred_data)
    new_fails = len(FAILURES) - fails_before
    print(f"  {'PASS' if new_fails == 0 else 'FAIL'}  {suite_name:<32} {N} cases, {new_fails} mismatches")

# ---------------------------------------------------------------- report
print()
print("=" * 70)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} MISMATCHES out of {CHECKS} comparisons")
    for name, r1, f1, rd, fd, ref, pred in FAILURES[:5]:
        print(f"\n  case: {name}")
        print(f"    ref  F1={r1!r} dice={rd!r}")
        print(f"    fast F1={f1!r} dice={fd!r}")
        print(f"    ref_data ={ref}")
        print(f"    pred_data={pred}")
    sys.exit(1)
print(f"RESULT: ALL {CHECKS} COMPARISONS IDENTICAL  (scorer.py == reference_scorer.py)")
print("=" * 70)
