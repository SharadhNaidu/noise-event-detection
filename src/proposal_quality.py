"""Collar-success confidence for fixed event geometry.

This is our binary metric-aware adaptation, not a reproduction of GFL.
"""
import torch
import torch.nn.functional as F


def quality_targets(distances, target, weights, active, lengths):
    valid = torch.arange(target.shape[1], device=target.device)[None, :, None] < lengths[:, None, None]
    positive = (weights > 0) & valid
    negative = (active == 0) & valid
    tolerance = torch.maximum(.2*target.sum(-1), target.new_tensor(.05))
    good = ((distances.detach()-target).abs() <= tolerance[..., None]).all(-1)
    return (good & positive).float(), weights*positive, negative.float()


def quality_loss(logits, labels, positive_weights, negative_weights):
    values = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none')
    # Equal mass for event-balanced occupied anchors and genuine background.
    # Censored/ambiguous active anchors and padding contribute to neither group.
    terms = []
    for weights in (positive_weights, negative_weights):
        mass = weights.sum()
        if float(mass) > 0:
            terms.append((values*weights).sum()/mass)
    return torch.stack(terms).mean() if terms else logits.sum()*0.
