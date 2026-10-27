import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import dataset as DS


def merged(events):
    out = []
    for a, b in sorted(events):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def interval_targets(events, cats, n_frames, duration, frame_sec=.02):
    times = (np.arange(n_frames)+.5)*frame_sec
    distances = np.zeros((n_frames, 8, 2), np.float32)
    weights = np.zeros((n_frames, 8), np.float32)
    active = np.zeros((n_frames, 8), np.float32)
    counts = dict(complete_events=0, censored_events=0, no_anchor_events=0, ambiguous_frames=0)
    lanes = [merged(events)]+[[e for e, cat in zip(events, cats) if cat == k] for k in DS.CATS]
    for ch, evs in enumerate(lanes):
        ownership = np.zeros(n_frames, np.int32)
        for a, b in evs:
            ownership += ((times >= a)&(times < b)).astype(np.int32)
        active[:, ch] = ownership > 0
        counts['ambiguous_frames'] += int(np.count_nonzero(ownership>1))
        for a, b in evs:
            if a < 0 or b > duration+1e-6:
                counts['censored_events'] += 1
                continue
            own = (times >= a)&(times < b)&(ownership == 1)&(times < duration)
            if not own.any():
                counts['no_anchor_events'] += 1
                continue
            counts['complete_events'] += 1
            distances[own, ch, 0] = times[own]-a
            distances[own, ch, 1] = b-times[own]
            weights[own, ch] = 1./np.count_nonzero(own)
    active[times >= duration] = 0
    return distances, weights, active, counts


class EventProposalNet(nn.Module):
    def __init__(self, hidden=64, acoustic_dim=0, quality=False):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.quality_enabled = quality
        self.acoustic = nn.Sequential(nn.LayerNorm(acoustic_dim), nn.Linear(acoustic_dim, 64), nn.GELU()) if acoustic_dim else None
        self.rnn = nn.GRU(24+(64 if acoustic_dim else 0), hidden, 2,
                          bidirectional=True, batch_first=True, dropout=.1)
        self.distances = nn.Linear(hidden*2, 16)
        self.quality = nn.Linear(hidden*2, 8) if quality else None

    def forward(self, features, lengths):
        x = features[..., :24]
        if self.acoustic is not None:
            x = torch.cat([x, self.acoustic(features[..., 24:24+self.acoustic_dim])], dim=-1)
        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=features.shape[1])
        d = F.softplus(self.distances(h).float()).reshape(features.shape[0], features.shape[1], 8, 2)
        return d, self.quality(h).float() if self.quality is not None else None


def interval_iou(pred, target):
    intersection = torch.minimum(pred, target).sum(-1)
    union = torch.maximum(pred, target).sum(-1)
    return intersection/union.clamp_min(1e-7)


def proposal_loss(pred, target, weights, collar_weight=0.):
    iou = interval_iou(pred, target)
    denominator = weights.sum().clamp_min(1.)
    loss = ((1.-iou)*weights).sum()/denominator
    if collar_weight:
        tolerance = torch.maximum(.2*target.sum(-1), target.new_tensor(.05))
        edge_error = (pred-target)/tolerance[..., None]
        edges = F.smooth_l1_loss(edge_error, torch.zeros_like(edge_error), reduction='none').mean(-1)
        loss = loss+collar_weight*(edges*weights).sum()/denominator
    return loss


def decode_proposals(probability, distances, duration, threshold=.5, channels=(0,), frame_sec=.02, max_events=15):
    events = []
    times = (np.arange(len(probability))+.5)*frame_sec
    for ch in channels:
        candidates = [(max(0., times[i]-distances[i, ch, 0]), min(duration, times[i]+distances[i, ch, 1]), float(probability[i, ch]), ch)
                      for i in range(len(times)) if times[i] < duration and probability[i, ch] >= threshold]
        selected = []
        for a, b, _, _ in sorted(candidates, key=lambda e: -e[2]):
            if b-a < .04 or any(min(b, d)>max(a, c) for c, d, _, _ in selected):
                continue
            own = (times >= a)&(times < b)&(times < duration)
            confidence = float(probability[own, ch].mean()) if own.any() else 0.
            if confidence >= threshold:
                selected.append((a, b, confidence, ch))
            if len(selected) >= max_events:
                break
        events.extend(selected)
    return sorted(events)
