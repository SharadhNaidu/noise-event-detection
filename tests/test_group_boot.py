"""boot_ens.group_bootstrap: paired bootstrap that resamples SPEAKER groups (with replacement,
duplicates kept), not clips. Clips of one speaker are correlated; clip resampling understates
the interval (professor's report, 2026-10-01).

Production changes that make these fail: resampling clips instead of groups; collapsing a group
drawn twice; returning an unpaired difference.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boot_ens as B  # noqa: E402


def test_groups_drawn_whole_and_with_multiplicity():
    groups = {"s1": ["a", "b"], "s2": ["c"], "s3": ["d", "e", "f"]}
    seen = []
    def stat(ids):
        seen.append(list(ids)); return float(len(ids))
    B.group_bootstrap(groups, stat, stat, n=50, seed=0)
    for ids in seen:
        for g in groups.values():                       # every group is all-or-nothing, repeated k times
            k = sum(ids.count(c) for c in g)
            assert k % len(g) == 0
    assert any(len(ids) != len(set(ids)) for ids in seen)   # duplicates preserved


def test_paired_difference_and_interval():
    groups = {f"s{i}": [f"c{i}"] for i in range(40)}
    better = lambda ids: 1.0 + 0.01 * len(ids)
    worse = lambda ids: 1.0
    r = B.group_bootstrap(groups, better, worse, n=200, seed=1)
    assert r["lo"] > 0 and r["mean"] > 0 and 0 <= r["p"] <= 1
    r0 = B.group_bootstrap(groups, worse, worse, n=50, seed=1)
    assert r0["mean"] == 0 and r0["lo"] == 0 and r0["hi"] == 0


if __name__ == "__main__":
    bad = 0
    for f in (test_groups_drawn_whole_and_with_multiplicity, test_paired_difference_and_interval):
        try:
            f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
