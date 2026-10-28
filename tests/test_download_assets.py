import hashlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import download_assets as D  # noqa: E402

BLOB = {"a.pt": b"weights-a", "p.pkl": b"post-p"}
MD5 = {k: hashlib.md5(v).hexdigest() for k, v in BLOB.items()}


def fake_urlopen(calls):
    def op(url):
        calls.append(url)
        name = url.rsplit("/", 1)[1]
        if name == "md5.txt":
            return io.BytesIO(f"{MD5['a.pt']}  checkpoints/a.pt\n{MD5['p.pkl']}  posteriors/p.pkl\n\n".encode())
        return io.BytesIO(BLOB[name])
    return op


def test_release_files_downloads_by_basename_into_repo_path_and_skips_verified(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen(calls))
    assert D.release_files("checkpoints/", tmp_path) == []
    assert (tmp_path / "checkpoints" / "a.pt").read_bytes() == BLOB["a.pt"]
    assert not (tmp_path / "posteriors").exists()                    # prefix filter
    assert calls == [f"{D.RELEASE}/md5.txt", f"{D.RELEASE}/a.pt"]
    assert not list(tmp_path.rglob("*.part"))
    calls.clear()
    assert D.release_files("checkpoints/", tmp_path) == []           # already verified -> no download
    assert calls == [f"{D.RELEASE}/md5.txt"]


def test_corrupt_download_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen([]))
    monkeypatch.setitem(MD5, "p.pkl", "0" * 32)
    assert D.release_files("posteriors/", tmp_path) == [str(tmp_path / "posteriors" / "p.pkl")]
