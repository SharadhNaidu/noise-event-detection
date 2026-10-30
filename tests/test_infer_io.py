"""Input handling of inference/infer.py: clip ids, wav decoding, zip/folder input, member substitution table."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "inference"))
import audiostore as A  # noqa: E402
import infer as I        # noqa: E402
sys.path.insert(0, str(ROOT))
from inference import utils as U  # noqa: E402


def _wav(path, sr=16000, ch=1, sec=1.0, amp=0.5):
    t = np.arange(int(sr * sec)) / sr
    y = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), np.stack([y] * ch, 1) if ch > 1 else y, sr, subtype="PCM_16")


def test_clip_ids_are_file_stems(tmp_path):
    d = tmp_path / "x"; d.mkdir()
    _wav(d / "a.wav"); _wav(d / "b.wav")
    ids = I.build_store(d, tmp_path / "root", "t")
    assert ids == ["a", "b"]
    arr, me = A.load_audio_store(tmp_path / "root" / "audio", "t")
    assert sorted(me["clips"]) == ["a", "b"] and me["clips"]["a"]["n"] == 16000


def test_load_wav_resamples_stereo_48k(tmp_path):
    d = tmp_path / "x"; d.mkdir()
    _wav(d / "s.wav", sr=48000, ch=2)
    I.build_store(d, tmp_path / "root", "t")
    arr, me = A.load_audio_store(tmp_path / "root" / "audio", "t")
    assert me["clips"]["s"]["n"] == 16000


def test_zip_input_is_read(tmp_path):
    _wav(tmp_path / "a.wav")
    z = tmp_path / "in.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(tmp_path / "a.wav", "audio/a.wav")
        f.writestr("__MACOSX/audio/._a.wav", b"junk")
    assert I.build_store(z, tmp_path / "root", "t") == ["a"]


def test_duplicate_stems_rejected(tmp_path):
    (tmp_path / "x" / "p").mkdir(parents=True); (tmp_path / "x" / "q").mkdir()
    _wav(tmp_path / "x" / "p" / "a.wav"); _wav(tmp_path / "x" / "q" / "a.wav")
    try:
        I.build_store(tmp_path / "x", tmp_path / "root", "t")
    except ValueError as e:
        assert "a" in str(e)
    else:
        raise AssertionError("duplicate clip ids must raise")


def test_full_scale_natural_routes_natural(tmp_path):
    """natural_by_peak needs the int16 store to keep full-scale peaks at >= 32765 (as the shipped test store does)."""
    import submit_ens as SE
    d = tmp_path / "x"; d.mkdir()
    _wav(d / "nat.wav", amp=1.0); _wav(d / "syn.wav", amp=0.99)
    I.build_store(d, tmp_path / "root", "t")
    arr, me = A.load_audio_store(tmp_path / "root" / "audio", "t")
    r = SE.natural_by_peak(arr, me["clips"], ["nat", "syn"])
    assert r == {"nat": True, "syn": False}


def test_member_table_is_consistent():
    import members as M
    m = M.load_members()
    assert m["natural"] == ["hon_A_tta", "bts_C", "bft_S_tta", "bts_R", "v12", "v5_S_tta", "v12s62"]
    assert m["synthetic"] == ["hon_A", "bts_C", "bft_S", "bts_R", "v5all", "v5all"]
    assert m["substitute"] == {"v12": "v12_ep4", "v12s62": "v12_ep4", "v5all": "v5_S"}
    assert set(m["tta"]) == {"hon_A", "bft_S", "v5_S"}


def test_missing_input_path_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        U._wav_sources(tmp_path / "nope")


def test_empty_folder_rejected(tmp_path):
    with pytest.raises(ValueError, match="no .wav files"):
        U._wav_sources(tmp_path)


def test_corrupt_wav_becomes_empty_clip_not_a_crash(tmp_path):
    d = tmp_path / "in"; d.mkdir()
    _wav(d / "good.wav")
    (d / "bad.wav").write_bytes(b"this is not a wav file at all" * 10)
    ids = I.build_store(d, tmp_path / "w", "s")
    assert ids == ["bad", "good"]
    arr, me = A.load_audio_store(tmp_path / "w" / "audio", "s")
    assert me["clips"]["bad"]["n"] == 0 and me["clips"]["good"]["n"] == 16000


def test_checkpoint_links_are_refreshed(tmp_path):
    src = tmp_path / "ck"; src.mkdir(); work = tmp_path / "w"
    (src / "a.pt").write_bytes(b"old")
    I.link_checkpoints(src, work)
    (src / "a.pt").unlink(); (src / "a.pt").write_bytes(b"new")      # re-download replaces the file
    I.link_checkpoints(src, work)
    assert (work / "ckpt" / "a.pt").read_bytes() == b"new"
    (src / "a.pt").unlink()
    I.link_checkpoints(src, work)                                     # removed from --ckpt-dir -> gone from work too
    assert not (work / "ckpt" / "a.pt").exists()


def test_ckpt_dir_inside_work_is_never_deleted(tmp_path):
    work = tmp_path / "w"; (work / "ckpt").mkdir(parents=True)
    (work / "ckpt" / "a.pt").write_bytes(b"real")
    I.link_checkpoints(work / "ckpt", work)
    assert (work / "ckpt" / "a.pt").read_bytes() == b"real"
