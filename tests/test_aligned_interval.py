import os
import sys
import json
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from aligned_interval import AlignedIntervalNet
from event_proposals import EventProposalNet, decode_proposals
from joint_interval import JointIntervalNet, refined_presence


def _features(batch=2, frames=20):
    features = torch.randn(batch, frames, 1304, device='cpu')
    features[..., :8] = torch.rand(batch, frames, 8, device='cpu')*.8+.1
    return features


@pytest.mark.parametrize('source_kind', ['proposal', 'joint'])
def test_complete_warm_start_preserves_original_heads_and_silences_skip(source_kind):
    torch.manual_seed(13)
    original = (JointIntervalNet(hidden=4) if source_kind == 'joint' else EventProposalNet(hidden=4, acoustic_dim=512)).eval()
    if source_kind == 'joint':
        with torch.no_grad():
            original.presence_residual.weight.normal_(0., .1)
            original.presence_residual.bias.normal_(0., .1)
    aligned = AlignedIntervalNet(hidden=4, skip_mode='real').eval()
    report = aligned.warm_start(original.state_dict())
    assert report['source_kind'] == source_kind
    for name, value in original.state_dict().items():
        imported = aligned.state_dict()[name]
        if name in ('rnn.weight_ih_l0', 'rnn.weight_ih_l0_reverse'):
            assert torch.equal(imported[:, :88], value)
            assert torch.count_nonzero(imported[:, 88:]) == 0
        else:
            assert torch.equal(imported, value)
    if source_kind == 'proposal':
        assert torch.count_nonzero(aligned.presence_residual.weight) == 0
        assert torch.count_nonzero(aligned.presence_residual.bias) == 0
    features, lengths = _features(), torch.tensor([20, 11])
    with torch.no_grad():
        distances, logits = aligned(features, lengths)
        old_distances, old_logits = original(features[..., :536], lengths)
    # Widening the GEMM can change floating-point reduction order despite zero
    # added coefficients, so mathematical parity is checked to float32 precision.
    assert torch.allclose(distances, old_distances, atol=1e-6, rtol=1e-6)
    expected = refined_presence(old_logits) if source_kind == 'joint' else features[..., :8].clamp(1e-6, 1.-1e-6)
    assert torch.allclose(refined_presence(logits), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('threshold', [.3, .5, .7])
def test_warm_started_skip_preserves_nonempty_decoding(threshold):
    torch.manual_seed(13)
    original = JointIntervalNet(hidden=4).eval()
    with torch.no_grad():
        original.distances.weight.zero_()
        original.distances.bias.fill_(float(torch.log(torch.expm1(torch.tensor(.04)))))
    aligned = AlignedIntervalNet(hidden=4).eval()
    aligned.warm_start(original.state_dict())
    features = _features(batch=1)
    features[..., :8] = .08
    features[:, 3:8, 1] = torch.tensor([.42, .58, .86, .62, .44])
    features[:, 11:16, 2] = torch.tensor([.76, .83, .89, .81, .75])
    with torch.no_grad():
        d0, p0 = original(features[..., :536], torch.tensor([20]))
        d1, p1 = aligned(features, torch.tensor([20]))
    first = decode_proposals(refined_presence(p0)[0].numpy(), d0[0].numpy(), .4, threshold, channels=(1, 2))
    second = decode_proposals(refined_presence(p1)[0].numpy(), d1[0].numpy(), .4, threshold, channels=(1, 2))
    assert first
    assert np.allclose(first, second, atol=1e-6, rtol=1e-6)


def _active_skip_pair():
    torch.manual_seed(13)
    real = AlignedIntervalNet(hidden=4, skip_mode='real').eval()
    real.warm_start(JointIntervalNet(hidden=4).state_dict())
    with torch.no_grad():
        real.rnn.weight_ih_l0[:, 88:].normal_(0., .1)
        real.rnn.weight_ih_l0_reverse[:, 88:].normal_(0., .1)
        real.presence_residual.weight.normal_(0., .1)
    zero = AlignedIntervalNet(hidden=4, skip_mode='zero').eval()
    zero.load_state_dict(real.state_dict(), strict=True)
    return real, zero


def test_zero_skip_is_identical_to_zero_coarse_control_and_has_zero_fine_gradients():
    real, zero = _active_skip_pair()
    features = _features().requires_grad_()
    zero_features = features.detach().clone()
    zero_features[..., 536:] = 0
    lengths = torch.tensor([20, 11])
    d0, p0 = zero(features, lengths)
    with torch.no_grad():
        dr, pr = real(zero_features, lengths)
    assert torch.equal(d0, dr)
    assert torch.equal(p0, pr)
    assert torch.count_nonzero(zero.fine(torch.zeros(1, 768))) == 0
    (d0.sum()+p0.sum()).backward()
    assert torch.count_nonzero(zero.fine[1].weight.grad) == 0
    assert torch.count_nonzero(features.grad[..., 536:]) == 0
    assert torch.count_nonzero(zero.rnn.weight_ih_l0.grad[:, 88:]) == 0
    assert torch.count_nonzero(zero.rnn.weight_ih_l0_reverse.grad[:, 88:]) == 0
    assert torch.count_nonzero(features.grad[..., 24:536]) > 0


def test_real_skip_receives_gradient_after_extra_gru_inputs_activate():
    real, _ = _active_skip_pair()
    features = _features().requires_grad_()
    distances, logits = real(features, torch.tensor([20, 11]))
    (distances.sum()+logits.sum()).backward()
    assert torch.isfinite(real.fine[1].weight.grad).all()
    assert torch.count_nonzero(real.fine[1].weight.grad) > 0
    assert torch.count_nonzero(features.grad[..., 536:]) > 0
    assert torch.count_nonzero(real.rnn.weight_ih_l0.grad[:, 88:]) > 0


def test_packed_padding_does_not_change_active_skip_outputs():
    real, _ = _active_skip_pair()
    features = _features(batch=1, frames=11)
    padded = torch.cat([features, _features(batch=1, frames=9)], dim=1)
    with torch.no_grad():
        first_d, first_p = real(features, torch.tensor([11]))
        second_d, second_p = real(padded, torch.tensor([11]))
    assert torch.allclose(first_d, second_d[:, :11], atol=1e-6)
    assert torch.allclose(first_p, second_p[:, :11], atol=1e-6)


@pytest.mark.parametrize('defect', ['missing_distance', 'partial_presence', 'extra_key', 'wrong_gru_width'])
def test_warm_start_rejects_incomplete_or_incompatible_state_without_mutation(defect):
    original = JointIntervalNet(hidden=4)
    state = {name: value.clone() for name, value in original.state_dict().items()}
    if defect == 'missing_distance':
        del state['distances.bias']
    elif defect == 'partial_presence':
        del state['presence_residual.bias']
    elif defect == 'extra_key':
        state['unrecognized.weight'] = torch.zeros(1)
    elif defect == 'wrong_gru_width':
        state['rnn.weight_ih_l0'] = torch.zeros(12, 89)
    aligned = AlignedIntervalNet(hidden=4)
    before = {name: value.clone() for name, value in aligned.state_dict().items()}
    with pytest.raises(ValueError):
        aligned.warm_start(state)
    assert all(torch.equal(value, aligned.state_dict()[name]) for name, value in before.items())


def test_invalid_skip_mode_and_feature_width_are_rejected():
    with pytest.raises(ValueError):
        AlignedIntervalNet(skip_mode='other')
    aligned = AlignedIntervalNet(hidden=4)
    with pytest.raises(ValueError):
        aligned(torch.zeros(1, 3, 536), torch.tensor([3]))
    aligned.skip_mode = 'other'
    with pytest.raises(ValueError):
        aligned(_features(batch=1, frames=3), torch.tensor([3]))


def _small_aligned_cache(tmp_path, monkeypatch):
    from train_aligned_interval import digest
    import train_epn
    monkeypatch.setattr(train_epn.D, 'load_validation', lambda: (None, {}))
    coarse_root, fine_root, audio_root = [tmp_path/name for name in ['coarse', 'fine', 'audio']]
    for root in [coarse_root, fine_root, audio_root]:
        root.mkdir()
    for store in ['gold_audio', 'silver_long']:
        counts = [4, 5, 6]
        clips, audio, offset = {}, {}, 0
        for i, n in enumerate(counts):
            cid = f'{store}_clip{i}'
            clips[cid] = dict(off=offset, n=n)
            audio[cid] = dict(n=n*320, events=[(.01, .05)], cats=['animal'])
            offset += n
        audio_path = audio_root/f'{store}.json'
        audio_path.write_text(json.dumps(dict(sr=16000, clips=audio)))
        provenance = dict(dimension=536, may_train=True, checkpoint_sha256='frozen_v5',
                          backbone_cfg=dict(crop=5), audio_index_sha256=digest(audio_path), frame_seconds=.02)
        coarse_path = coarse_root/f'{store}.json'
        coarse_path.write_text(json.dumps(dict(provenance=provenance, clips=clips, total_frames=offset, complete=True)))
        coarse = np.arange(offset*536, dtype=np.float16).reshape(offset, 536)/100
        coarse.tofile(coarse_root/f'{store}.f16')
        fine = np.arange(offset*768, dtype=np.float16).reshape(offset, 768)/200
        fine.tofile(fine_root/f'{store}.f16')
        fine_provenance = dict(provenance, dimension=768, reference_index_sha256=digest(coarse_path),
                               source={'export_localization_skip.py': 'one_exporter_source'})
        (fine_root/f'{store}.json').write_text(json.dumps(dict(provenance=fine_provenance, clips=clips,
                                                              total_frames=offset, complete=True,
                                                              payload_sha256=digest(fine_root/f'{store}.f16'))))
    return coarse_root, fine_root, audio_root


def test_aligned_cache_joins_exact_offsets_and_preserves_split_and_silver_targets(tmp_path, monkeypatch):
    from train_aligned_interval import AlignedDataset
    from train_joint_interval import JointDataset
    coarse, fine, audio = _small_aligned_cache(tmp_path, monkeypatch)
    stores = ['gold_audio', 'silver_long']
    base = JointDataset(coarse, audio, stores, 13)
    aligned = AlignedDataset(coarse, fine, audio, stores, 13)
    assert aligned.train_ids == base.train_ids and aligned.dev_ids == base.dev_ids
    for key in sorted(aligned.items):
        aligned.ids, base.ids = [key], [key]
        x, target, weights, labels = aligned[0]
        original_x, original_target, original_weights, original_labels = base[0]
        entry = aligned.items[key]
        f = entry['feature']
        expected_fine = torch.from_numpy(aligned.skip_features[entry['_store']][f['off']:f['off']+f['n']].astype(np.float32))
        assert x.shape == (f['n'], 1304)
        assert torch.equal(x[:, :536], original_x)
        assert torch.equal(x[:, 536:], expected_fine)
        assert torch.equal(target, original_target)
        assert torch.equal(weights, original_weights)
        assert torch.equal(labels, original_labels)
        assert weights.sum() == (1. if entry['_store'] == 'silver_long' else 2.)


@pytest.mark.parametrize('defect', ['offset', 'payload', 'mixed_source', 'incomplete'])
def test_aligned_cache_rejects_misaligned_corrupt_or_mixed_export(tmp_path, monkeypatch, defect):
    from train_aligned_interval import AlignedDataset
    coarse, fine, audio = _small_aligned_cache(tmp_path, monkeypatch)
    path = fine/'silver_long.json'
    index = json.loads(path.read_text())
    if defect == 'offset':
        index['clips']['silver_long_clip1']['off'] += 1
    elif defect == 'payload':
        payload = fine/'silver_long.f16'
        with payload.open('r+b') as f:
            f.write(b'xx')
    elif defect == 'mixed_source':
        index['provenance']['source']['export_localization_skip.py'] = 'different_exporter'
    elif defect == 'incomplete':
        index['complete'] = False
    path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError):
        AlignedDataset(coarse, fine, audio, ['gold_audio', 'silver_long'], 13)
