"""Conditionally add the frozen pre-fusion WavLM stream to the interval head.

This branch preserves the supplied stream's information. Its 20-ms sampling
grid does not establish the neural encoder's effective temporal resolution.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from joint_interval import JointIntervalNet, _valid_lengths


class AlignedIntervalNet(JointIntervalNet):
    """Joint interval/presence head with a controlled 768-column acoustic skip."""

    def __init__(self, hidden=64, acoustic_dim=512, skip_mode='real'):
        if skip_mode not in ('real', 'zero'):
            raise ValueError('skip_mode must be real or zero')
        super().__init__(hidden=hidden, acoustic_dim=acoustic_dim)
        self.skip_mode = skip_mode
        self.fine = nn.Sequential(nn.LayerNorm(768, elementwise_affine=False),
                                  nn.Linear(768, 64, bias=False), nn.GELU())
        self.rnn = nn.GRU(152, hidden, 2, bidirectional=True,
                          batch_first=True, dropout=.1)

    def warm_start(self, state):
        """Import a complete original EPN or joint state, with a silent skip.

        Validate every key and shape before changing model state. Both new GRU
        input blocks start at zero; a missing original-EPN presence residual
        also starts at zero. The fine projection retains its initialization.
        """
        if not isinstance(state, Mapping):
            raise ValueError('warm_start requires a model state mapping')
        destination = self.state_dict()
        presence_keys = {'presence_residual.weight', 'presence_residual.bias'}
        fine_keys = {name for name in destination if name.startswith('fine.')}
        required = set(destination)-fine_keys-presence_keys
        provided = set(state)
        has_presence = presence_keys <= provided
        expected = required|presence_keys if has_presence else required
        if provided != expected:
            missing, unexpected = sorted(expected-provided), sorted(provided-expected)
            raise ValueError(f'Incomplete or incompatible warm-start state: missing={missing}, unexpected={unexpected}')
        widened = {'rnn.weight_ih_l0', 'rnn.weight_ih_l0_reverse'}
        for name in expected:
            value = state[name]
            shape = (destination[name].shape[0], 88) if name in widened else destination[name].shape
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
                raise ValueError(f'Incompatible warm-start tensor: {name}, expected shape {tuple(shape)}')
        imported = {name: value.detach().clone() for name, value in destination.items()}
        for name in expected:
            if name in widened:
                imported[name].zero_()
                imported[name][:, :88].copy_(state[name])
            else:
                imported[name] = state[name].detach().clone()
        if not has_presence:
            for name in presence_keys:
                imported[name].zero_()
        self.load_state_dict(imported, strict=True)
        return dict(source_kind='joint' if has_presence else 'proposal',
                    copied_input_columns=88, zero_input_columns=64)

    def forward(self, features, lengths):
        if features.ndim != 3 or features.shape[-1] != 1304:
            raise ValueError('features must have shape (batch, frames, 1304)')
        if not features.is_floating_point():
            raise ValueError('features must be floating point')
        if self.skip_mode not in ('real', 'zero'):
            raise ValueError('skip_mode must be real or zero')
        _valid_lengths(lengths, features.shape[0], features.shape[1])
        fine = features[..., 536:1304]
        if self.skip_mode == 'zero':
            fine = torch.zeros_like(fine)
        x = torch.cat([features[..., :24], self.acoustic(features[..., 24:536]),
                       self.fine(fine)], dim=-1)
        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=features.shape[1])
        distances = F.softplus(self.distances(h).float()).reshape(features.shape[0], features.shape[1], 8, 2)
        base_logits = torch.logit(features[..., :8].float().clamp(1e-6, 1.-1e-6))
        presence_logits = base_logits+self.presence_residual(h).float()
        return distances, presence_logits
