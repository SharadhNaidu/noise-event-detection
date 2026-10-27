"""nest_store: a real Gold breath pasted INSIDE a real Gold long background event (R2_07). The output keeps the host's
own events, adds the breath with its original span, and nests it strictly inside the host event."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nest_store as NS  # noqa: E402

SR = 16000


def _fake():
    rng = np.random.default_rng(0)
    clips = {
        "hostA": {"wav": rng.standard_normal(6 * SR).astype(np.float32) * 0.1,
                  "events": [[1.0, 4.0]], "cats": ["animal"]},
        "donB": {"wav": rng.standard_normal(4 * SR).astype(np.float32) * 0.1,
                 "events": [[2.0, 2.35]], "cats": ["human_non_speech"]},
    }
    return clips


def test_pools():
    c = _fake()
    donors, hosts = NS.pools(c)
    assert [d[0] for d in donors] == ["donB"] and [h[0] for h in hosts] == ["hostA"]


def test_make_one_nests_inside_and_keeps_host_events():
    c = _fake()
    donors, hosts = NS.pools(c)
    rng = np.random.default_rng(3)
    wav, evs, cats = NS.make_one(c, hosts[0], donors[0], rng, gain_db=-18.0)
    assert len(wav) == len(c["hostA"]["wav"])
    assert [1.0, 4.0] in evs and "animal" in cats
    br = [e for e, k in zip(evs, cats) if k == "human_non_speech"][0]
    assert 1.0 < br[0] < br[1] < 4.0 and abs((br[1] - br[0]) - 0.35) < 1e-6
    assert not np.allclose(wav, c["hostA"]["wav"])                     # audio changed
    a = int((br[0] - 0.2) * SR)
    assert np.allclose(wav[:a], c["hostA"]["wav"][:a])                 # only near the paste


def test_gain_is_relative_to_local_mixture():
    c = _fake()
    donors, hosts = NS.pools(c)
    rng = np.random.default_rng(5)
    wav, evs, cats = NS.make_one(c, hosts[0], donors[0], rng, gain_db=-10.0)
    br = [e for e, k in zip(evs, cats) if k == "human_non_speech"][0]
    a, b = int(br[0] * SR), int(br[1] * SR)
    added = wav[a:b] - c["hostA"]["wav"][a:b]
    rel = 10 * np.log10(np.mean(added ** 2) / np.mean(c["hostA"]["wav"][a:b] ** 2))
    assert abs(rel - (-10.0)) < 1.5
