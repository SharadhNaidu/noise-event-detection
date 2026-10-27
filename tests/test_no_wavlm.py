"""--no-wavlm: the detector runs on the second-stream (BEATs) features alone.

Production change that makes this fail: the WavLM encoder still feeding the
head (output would change with the waveform), or the aux stream disconnected.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as FT  # noqa: E402


def _net():
    torch.manual_seed(0)
    net = FT.FineTuneSED(4, aux_dim=1215, no_wavlm=True)
    net.eval()
    return net


def test_output_ignores_waveform():
    net = _net()
    aux = torch.randn(2, 249, 1215)
    with torch.no_grad():
        a = net(torch.randn(2, 80000), aux=aux)[0]
        b = net(torch.randn(2, 80000) * 5, aux=aux)[0]
    assert a.shape == (2, 249, 8), a.shape
    assert torch.equal(a, b), "waveform (WavLM path) still influences the output"


def test_output_follows_aux():
    net = _net()
    w = torch.randn(1, 80000)
    with torch.no_grad():
        a = net(w, aux=torch.randn(1, 249, 1215))[0]
        b = net(w, aux=torch.randn(1, 249, 1215))[0]
    assert (a - b).abs().max() > 1e-4, "aux stream has no effect"


def test_no_encoder_parameters_are_trained():
    net = _net()
    trainable = [n for n, p in net.named_parameters() if p.requires_grad and n.startswith("enc.")]
    assert not trainable, f"encoder params still trainable: {trainable[:3]}"


if __name__ == "__main__":
    bad = 0
    for t in [test_output_ignores_waveform, test_output_follows_aux, test_no_encoder_parameters_are_trained]:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
