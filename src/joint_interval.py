"""Joint presence and interval refinement of the measured acoustic head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from event_proposals import EventProposalNet


def _valid_lengths(lengths, batch, frames):
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise ValueError('lengths must be a one-dimensional tensor')
    if len(lengths) != batch or batch < 1 or frames < 1:
        raise ValueError('lengths must describe a nonempty batch of frames')
    if lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('lengths must contain integers')
    if bool((lengths < 1).any()) or bool((lengths > frames).any()):
        raise ValueError('lengths must be between one and the padded frame count')


class JointIntervalNet(EventProposalNet):
    """Refine V5 presence while jointly adapting the existing interval head.

    A zero residual reproduces the clamped input probability, rather than
    mathematically exact probabilities at zero and one. The acoustic projection,
    recurrent encoder and distance layer retain their original state names.
    """

    def __init__(self, hidden=64, acoustic_dim=512):
        if acoustic_dim != 512:
            raise ValueError('JointIntervalNet requires 512 acoustic columns')
        super().__init__(hidden=hidden, acoustic_dim=acoustic_dim, quality=False)
        self.presence_residual = nn.Linear(hidden*2, 8)
        nn.init.zeros_(self.presence_residual.weight)
        nn.init.zeros_(self.presence_residual.bias)

    def forward(self, features, lengths):
        if features.ndim != 3 or features.shape[-1] != 536:
            raise ValueError('features must have shape (batch, frames, 536)')
        if not features.is_floating_point():
            raise ValueError('features must be floating point')
        _valid_lengths(lengths, features.shape[0], features.shape[1])
        x = features[..., :24]
        x = torch.cat([x, self.acoustic(features[..., 24:536])], dim=-1)
        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=features.shape[1])
        distances = F.softplus(self.distances(h).float()).reshape(features.shape[0], features.shape[1], 8, 2)
        base_logits = torch.logit(features[..., :8].float().clamp(1e-6, 1.-1e-6))
        presence_logits = base_logits+self.presence_residual(h).float()
        return distances, presence_logits


def refined_presence(logits):
    return torch.sigmoid(logits)


def presence_loss(logits, labels, lengths):
    """Unweighted BCE, averaged over real frames and lanes within each clip."""
    if logits.ndim != 3 or logits.shape[-1] != 8 or labels.shape != logits.shape:
        raise ValueError('logits and labels must both have shape (batch, frames, 8)')
    if not logits.is_floating_point() or not labels.is_floating_point():
        raise ValueError('logits and labels must be floating point')
    _valid_lengths(lengths, logits.shape[0], logits.shape[1])
    valid = torch.arange(logits.shape[1], device=logits.device)[None, :, None] < lengths.to(logits.device)[:, None, None]
    # Mask before BCE as well as after it: invalid padding may contain arbitrary
    # values, including NaNs, and must never alter the loss or its gradients.
    real_logits = torch.where(valid, logits, torch.zeros_like(logits))
    real_labels = torch.where(valid, labels, torch.zeros_like(labels))
    losses = F.binary_cross_entropy_with_logits(real_logits, real_labels, reduction='none')
    per_clip = (losses*valid).sum((1, 2))/(lengths.to(logits.device, logits.dtype)*8.)
    return per_clip.mean()
