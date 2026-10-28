"""Input handling of inference/infer.py: clip ids, wav decoding, zip/folder input, member substitution table."""
import json
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
    m = json.load(open(ROOT / "configs" / "members.json"))
    assert m["natural"] == ["hon_A_tta", "bts_C", "bft_S_tta", "bts_R", "v12", "v5_S_tta", "v12s62"]
    assert m["synthetic"] == ["hon_A", "bts_C", "bft_S", "bts_R", "v5all", "v5all"]
    assert m["substitute"] == {"v12": "v12_ep4", "v12s62": "v12_ep4", "v5all": "v5_S"}
    assert set(m["tta"]) == {"hon_A", "bft_S", "v5_S"}


def test_missing_input_path_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        I._wav_sources(tmp_path / "nope")


def test_empty_folder_rejected(tmp_path):
    with pytest.raises(ValueError, match="no .wav files"):
        I._wav_sources(tmp_path)
