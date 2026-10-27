"""Detection head over cached WavLM features.

The encoder is frozen and its output is cached to disk (see features.py /
extract.py), so this module only ever sees (T, 768) tensors. That keeps each
experiment to minutes instead of hours, which is what makes a 22-day schedule
with many hypotheses feasible on one laptop GPU.

Two heads, following the organizers' baseline:
  frame  - per-frame probabilities. Channel 0 is "any noise", the only thing
           scored; channels 1..7 are the categories, used purely as auxiliary
           supervision (the submission format carries no labels).
  clip   - attention-pooled clip probabilities. This is Multiple-Instance
           Learning and is the ONLY route by which Bronze clips can contribute
           a gradient - EXP-04 confirmed Bronze rows carry zero timestamped
           events, so they have no frame-level target at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CAT = 7
N_OUT = 1 + N_CAT          # channel 0 = any-noise
REG_HORIZON = 0.5          # must match dataset.REG_HORIZON


class SEDHead(nn.Module):
    """BiGRU + frame head + attention-pooling clip head, over frozen features."""

    def __init__(self, in_dim=768, rnn_dim=256, rnn_layers=2, n_out=N_OUT,
                 dropout=0.1, input_dropout=0.0, boundary_head=False,
                 red_head=False, cat_boundary=0, box_head=False, red_ch=1, box_kind="peaks",
                 conv_heads=1, collar_head=False, aux_dim=0, conformer=0, lane2=False):
        super().__init__()
        # B6a --lane2: one extra class-free channel (lane B) on the frame head, the RED latch (per-class RED only)
        # and the per-channel distance head (catb, +1 on/off pair after the category pairs)
        self.lane2 = lane2
        if lane2:
            n_out, cat_boundary = n_out + 1, cat_boundary + 1
            red_ch = red_ch + 1 if red_ch > 1 else red_ch
        # each stream is normalised on its own, so adding the AST stream does
        # not change the statistics the WavLM stream was trained under
        self.aux_dim = aux_dim
        self.in_norm = nn.LayerNorm(in_dim)
        self.aux_norm = nn.LayerNorm(aux_dim) if aux_dim else None
        self.in_drop = nn.Dropout(input_dropout)
        # optional Conformer blocks before the BiGRU: attention for long-range
        # context + depthwise conv for local detail (DCASE-style CRNN/Conformer)
        self.conformer = None
        gru_in = in_dim + aux_dim
        if conformer:
            from torchaudio.models import Conformer
            self.conf_proj = nn.Linear(gru_in, 256)
            self.conformer = Conformer(input_dim=256, num_heads=4, ffn_dim=1024,
                                       num_layers=conformer, depthwise_conv_kernel_size=31,
                                       dropout=0.1)
            gru_in = 256
        self.rnn = nn.GRU(gru_in, rnn_dim, rnn_layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        self.frame = nn.Linear(2 * rnn_dim, n_out)
        self.att = nn.Linear(2 * rnn_dim, n_out)
        # EXP-18/20: the frame posterior transitions over 0.36 s against an
        # 0.086 s tolerance, and that ramp is BCE's calibrated uncertainty about
        # edge POSITION - not something frame reweighting can remove, because
        # BCE is already a proper scoring rule. This head asks the answerable
        # question instead: "how far is the nearest onset/offset from here?"
        # Output is tanh-squashed to +-REG_HORIZON seconds, so it is bounded,
        # continuous, and independent of the posterior's amplitude.
        self.boundary = (nn.Linear(2 * rnn_dim, 2) if boundary_head else None)
        # RED head: per-frame conditional start / end logits for channel 0.
        # Presence is derived from these by recurrence rather than predicted
        # directly, so it latches through masked frames (EXP-32).
        # red_ch=1: channel 0 only (shape of every existing RED checkpoint);
        # red_ch=N_OUT: one latched track per channel (--red-classes), so
        # overlapping cross-category events are representable.
        self.red_ch = red_ch
        self.red = (nn.Linear(2 * rnn_dim, 2 * red_ch) if red_head else None)
        # per-source RED heads (--src-heads): RED places our edges, so each label source gets its own
        # start/end head; head 0 (self.red) is Gold and is the only one used at inference
        self.red_x = nn.ModuleList(
            [nn.Linear(2 * rnn_dim, 2 * red_ch) for _ in range(conv_heads - 1)] if red_head else [])
        # Per-category onset/offset distances (2 per category). Separate from
        # `boundary` so a checkpoint without it still loads that head intact.
        self.catb = (nn.Linear(2 * rnn_dim, 2 * cat_boundary)
                     if cat_boundary else None)
        # Instance box head (dataset.box_targets): onset/offset heatmaps,
        # forward/backward log-durations, sub-frame edge residuals.
        self.box = nn.Linear(2 * rnn_dim, 6) if box_head else None
        # collar-aware onset/offset peak heads (Kalda & Alumae 2022)
        self.collar = nn.Linear(2 * rnn_dim, 2) if collar_head else None
        if self.collar is not None:
            nn.init.constant_(self.collar.bias, -4.0)
        # LABEL-CONVENTION HEADS. Gold, Silver and synthetic mark edges by
        # different conventions (Silver events 2x longer, 30% start at 0.000;
        # synthetic edges machine-exact). One shared head learns their
        # average - too wide for Gold. Head 0 (= self.frame / self.boundary)
        # is Gold and is the one decoded; extra heads absorb the others.
        self.frame_x = nn.ModuleList(
            [nn.Linear(2 * rnn_dim, n_out) for _ in range(conv_heads - 1)])
        self.boundary_x = nn.ModuleList(
            [nn.Linear(2 * rnn_dim, 2) for _ in range(conv_heads - 1)]
            if boundary_head else [])
        self.box_kind = box_kind
        if self.box is not None and box_kind == "dense":
            # dense interval head: softplus(-1.05) = 0.30 s, a neutral starting distance
            nn.init.constant_(self.box.bias, -1.05)
        elif self.box is not None:
            # CenterNet prior: heatmap logits start near p=0.01
            nn.init.constant_(self.box.bias[:2], -4.6)

    def forward(self, x, mask=None, tier=None):
        """x: (B, T, in_dim) float. mask: (B, T) 1 for real frames, 0 for padding.

        Returns (frame_logits (B, T, n_out), clip_logits (B, n_out)) and, when
        the boundary head is enabled, a third tensor (B, T, 2) of signed
        distances in SECONDS to the nearest onset and offset.

        LOGITS, not probabilities. Under autocast(bfloat16) a probability cannot
        be clamped away from 1.0 - bf16 has ~3 decimal digits, so 1 - 1e-7
        rounds to exactly 1.0 and log(1 - p) becomes log(0) = -inf. That produced
        NaN from epoch 8 of the first real run. Everything downstream therefore
        works in logit space and uses binary_cross_entropy_with_logits.

        The mask here is over CACHED FEATURES, which is safe - unlike the
        encoder's group-norm front-end (EXP-07), a GRU plus masked softmax is
        genuinely unaffected by padded frames, provided we mask the attention
        and ignore padded frames in the loss.
        """
        if self.aux_dim:
            x = torch.cat([self.in_norm(x[..., :-self.aux_dim]),
                           self.aux_norm(x[..., -self.aux_dim:])], -1)
        else:
            x = self.in_norm(x)
        h = self.in_drop(x)
        if self.conformer is not None:
            h = self.conf_proj(h)
            lengths = (mask.sum(1) if mask is not None
                       else torch.full((h.shape[0],), h.shape[1], device=h.device)).long()
            h, _ = self.conformer(h, lengths)
        h, _ = self.rnn(h)
        frame_logits = self.frame(h)                      # (B, T, n_out)
        if tier is not None and len(self.frame_x):
            sel = tier.view(-1, 1, 1)
            for k, fx in enumerate(self.frame_x, start=1):
                frame_logits = torch.where(sel == k, fx(h), frame_logits)

        a = self.att(h)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        a = torch.softmax(a.float(), dim=1)

        # Attention-pooled clip probability, then back to a logit. Computed in
        # fp32 so the round trip is well-conditioned.
        p = torch.sigmoid(frame_logits.float())
        clip_p = (p * a).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        clip_logits = torch.log(clip_p) - torch.log1p(-clip_p)
        red = None
        if self.red is not None:
            r = self.red(h).float()
            if tier is not None and len(self.red_x):
                sel = tier.view(-1, 1, 1)
                for k, rx in enumerate(self.red_x, start=1):
                    r = torch.where(sel == k, rx(h).float(), r)
            red = (r[..., :self.red_ch], r[..., self.red_ch:])   # (start_logit, end_logit)
        if self.boundary is None:
            return (frame_logits, clip_logits) if red is None else                 (frame_logits, clip_logits, None, red)
        braw = self.boundary(h)
        if tier is not None and len(self.boundary_x):
            sel = tier.view(-1, 1, 1)
            for k, bx in enumerate(self.boundary_x, start=1):
                braw = torch.where(sel == k, bx(h), braw)
        dist = REG_HORIZON * torch.tanh(braw.float())
        if self.catb is not None or self.box is not None or self.collar is not None:
            catd = (REG_HORIZON * torch.tanh(self.catb(h).float())
                    if self.catb is not None else None)
            box = self.box(h).float() if self.box is not None else None
            col = self.collar(h).float() if self.collar is not None else None
            return (frame_logits, clip_logits, dist, red, catd, box, col)
        return (frame_logits, clip_logits, dist) if red is None else             (frame_logits, clip_logits, dist, red)


def red_presence(start_logit, end_logit):
    """Parallel (log-depth) evaluation of the RED recurrence - identical maths to
    red_presence_serial: p_t = q_t * p_{t-1} + a_t with q_t = 1 - a_t - b_t, an
    affine map; compositions are associative, so prefixes come from a doubling scan
    (O(T log T) work, log2(T) sequential steps instead of T). Verified against the
    serial loop to ~2e-7 (outputs) / ~3e-7 (grads) [professor's red_scan_verified.py
    + tests/test_red_classes.py]. Returns (p_pres, p_on, p_off), all (B, T, C)."""
    a = torch.sigmoid(start_logit.float())
    b = torch.sigmoid(end_logit.float())
    q, r = 1 - a - b, a
    step = 1
    while step < a.shape[1]:
        q, r = (torch.cat((q[:, :step], q[:, step:] * q[:, :-step]), 1),
                torch.cat((r[:, :step], r[:, step:] + q[:, step:] * r[:, :-step]), 1))
        step *= 2
    pres = r                                     # initial state p_0 = 0
    prev = torch.cat((torch.zeros_like(pres[:, :1]), pres[:, :-1]), 1)
    return pres, a * (1 - prev), b * prev


def red_presence_serial(start_logit, end_logit):
    """Recurrent Event Detection: derive presence from start/end logits.

    Schmid et al., "Sound Event Detection with Boundary-Aware Optimization and
    Inference" (arXiv:2601.04178). Instead of asking "is the source audible in
    this frame", the model emits two conditional probabilities per frame:

        a_t = P(event starts at t | it was not present at t-1)
        b_t = P(event ends   at t | it WAS present at t-1)

    and presence follows by recurrence:

        p_t = a_t * (1 - p_{t-1}) + (1 - b_t) * p_{t-1}

    Once an onset fires, presence LATCHES until an offset fires. It cannot
    sag mid-event, which is exactly the failure EXP-32 measured: on natural
    clips our frame posterior anti-correlated with clip loudness inside long
    events (r = -0.178), because a speech-masked frame looks like silence.
    A latch cannot do that - the model would have to actively emit an offset.

    Returns (p_pres, p_on, p_off), all (B, T, C). p_on and p_off are the
    per-frame probabilities that an event actually starts / ends here, which
    is what the onset-offset loss supervises.
    """
    a = torch.sigmoid(start_logit.float())
    b = torch.sigmoid(end_logit.float())
    B, T, C = a.shape
    p = torch.zeros(B, C, device=a.device, dtype=a.dtype)
    pres, on, off = [], [], []
    for t in range(T):
        at, bt = a[:, t], b[:, t]
        on.append(at * (1 - p))
        off.append(bt * p)
        p = at * (1 - p) + (1 - bt) * p
        pres.append(p)
    return (torch.stack(pres, 1), torch.stack(on, 1), torch.stack(off, 1))


def focal_bce(p, target, mask=None, gamma=2.0, eps=1e-6, sample_weight=None, pos_weight=None):
    """Focal loss on a probability, for impulse-sparse targets.

    The onset/offset targets are 1 at a single frame and 0 everywhere else, so
    plain BCE is swamped by negatives. Schmid et al. use gamma=2, which is
    what makes the loss attend to the handful of frames that matter.
    """
    p = p.float().clamp(eps, 1 - eps)
    t = target.float()
    pw = 1.0 if pos_weight is None else float(pos_weight)   # gamma=0 + pos_weight = the RED paper's presence BCE
    loss = -(pw * t * (1 - p) ** gamma * torch.log(p)
             + (1 - t) * p ** gamma * torch.log(1 - p))
    if sample_weight is not None:                    # per-clip weights (B,), e.g. tier weights
        w = sample_weight.float().view(-1, *([1] * (loss.dim() - 1)))
        m = (mask.float() if mask is not None and mask.dim() == loss.dim()
             else (mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(loss)))
        m = m.expand_as(loss) * w
        return (loss * m).sum() / m.sum().clamp(min=1e-6)
    if mask is not None:
        if mask.dim() == loss.dim():                     # per-channel mask (B, T, C)
            m = mask.float()
            return (loss * m).sum() / m.sum().clamp(min=1)
        m = mask.float().unsqueeze(-1)
        return (loss * m).sum() / m.sum().clamp(min=1) / max(loss.shape[-1], 1)
    return loss.mean()


def collar_ool_loss(p, ids, mask, gamma=2.0, eps=1e-6, sample_weight=None):
    """Onset-offset loss with the scorer's collar (--ool-collar).

    p: (B, T, C) RED onset (or offset) probabilities. ids: (B, T, C) collar ids from
    dataset.collar_window_ids_cls (k>0 collar of edge k, 0 background, -1 ignored).
    Each collar W counts as ONE positive whose probability is that the edge fires
    somewhere inside it, M_W = 1 - prod_{t in W}(1 - p_t); background frames are
    ordinary focal negatives; ignored frames contribute nothing. Same focal form and
    the same normalisation (over mask frames) as focal_bce, to which it reduces
    exactly when every collar is a single frame - so ool_w keeps its meaning."""
    p = p.float().clamp(eps, 1 - eps)
    ids = ids.round().long()
    m = mask.float()
    if m.dim() < p.dim():
        m = m.unsqueeze(-1).expand_as(p)
    if sample_weight is not None:
        m = m * sample_weight.float().view(-1, *([1] * (p.dim() - 1)))
    neg = -(p ** gamma) * torch.log(1 - p) * (ids == 0).float() * m
    B, T, C = p.shape
    K = int(ids.max().clamp(min=0)) + 1
    inwin = ((ids > 0) & (m > 0)).float()
    key = ((torch.arange(B, device=p.device).view(B, 1, 1) * C
            + torch.arange(C, device=p.device).view(1, 1, C)) * K + ids.clamp(min=0))
    s = torch.zeros(B * C * K, device=p.device).index_add(
        0, key.reshape(-1), (torch.log(1 - p) * inwin).reshape(-1))
    n = torch.zeros(B * C * K, device=p.device).index_add(0, key.reshape(-1), inwin.reshape(-1))
    wsum = torch.zeros(B * C * K, device=p.device).index_add(
        0, key.reshape(-1), (m * inwin).reshape(-1))
    has = n > 0
    mass = (1 - torch.exp(s[has])).clamp(eps, 1 - eps)
    w = wsum[has] / n[has]                                  # the window's (mean) clip weight
    pos = (-(1 - mass) ** gamma * torch.log(mass) * w).sum()
    return (neg.sum() + pos) / m.sum().clamp(min=1e-6)


def onset_censor_mask(y, ion, mask):
    """(B, T, K) onset-loss mask: 0 at frame 0 for every channel whose event is
    already in progress when the crop starts (presence target 1, no observed onset).
    The RED latch starts at p=0, so presence at frame 0 can only come from an onset;
    penalising that onset contradicts the presence target (left-censored event).
    Genuine frame-0 onsets and every interior frame stay supervised."""
    K = ion.shape[-1]
    out = mask.float().unsqueeze(-1).expand(-1, -1, K).clone()
    cens = (y[:, 0, :K] > 0.5) & (ion[:, 0, :K] < 0.5)
    out[:, 0, :][cens] = 0.0
    return out


def soft_dice_prob(prob, target, mask, sample_weight=None, eps=1.0):
    """soft_dice_loss on PROBABILITIES (e.g. RED latched presence). Do not pass RED
    presence to soft_dice_loss: it would apply a second sigmoid."""
    p = prob.float() * mask
    y = target.float() * mask
    dice = (2 * (p * y).sum(1) + eps) / (p.sum(1) + y.sum(1) + eps)
    per_clip = 1.0 - dice
    if sample_weight is not None:
        sw = sample_weight.float()
        return (per_clip * sw).sum() / sw.sum().clamp(min=1e-6)
    return per_clip.mean()


def soft_dice_loss(logits, target, mask, sample_weight=None, eps=1.0):
    """1 - soft Dice per clip on one channel (B, T), averaged over clips like the
    metric's per-clip Dice (arXiv 2107.10471). eps=1 frame makes an empty
    reference predicted empty a perfect clip, as the metric scores it."""
    p = torch.sigmoid(logits.float()) * mask
    y = target.float() * mask
    inter = (p * y).sum(1)
    dice = (2 * inter + eps) / (p.sum(1) + y.sum(1) + eps)
    per_clip = 1.0 - dice
    if sample_weight is not None:
        sw = sample_weight.float()
        return (per_clip * sw).sum() / sw.sum().clamp(min=1e-6)
    return per_clip.mean()


def masked_bce(logits, target, mask, pos_weight=None, sample_weight=None):
    """BCE-with-logits over (B, T, C), ignoring padded frames.

    Args:
        logits:       (B, T, C) raw scores.
        target:       (B, T, C) in {0, 1}.
        mask:         (B, T) 1 for real frames.
        pos_weight:   scalar, up-weights positive frames (noise frames are the
                      minority, so an unweighted objective is dominated by
                      negatives).
        sample_weight: (B,) per-clip weight - where tier weighting enters
                      (EXP-05: Silver's events are 2.5x longer than Gold's and
                      teach event merging, so it should probably count for less
                      than the baseline's 0.4).
    """
    logits = logits.float()
    target = target.float()
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(float(pos_weight), device=logits.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none")
    m = mask.float().unsqueeze(-1)
    loss = loss * m
    per_clip = loss.sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1)
    if sample_weight is not None:
        sw = sample_weight.float()
        return (per_clip * sw).sum() / sw.sum().clamp(min=1e-6)
    return per_clip.mean()


TOL_FLOOR = 0.05           # the metric's tolerance floor, in seconds


def boundary_loss(pred, target, valid, mask, sample_weight=None, delta=1.0,
                  unit=TOL_FLOOR, offset_w=1.0, tau_unit=False):
    """Huber on the signed distance-to-edge, over frames near a real edge.

    MEASURED IN TOLERANCES, NOT SECONDS. This is not cosmetic. With targets in
    seconds and delta=0.05, a typical 0.1 s error gives Huber 0.0038, against a
    BCE term of order 1.0 - the regression contributed about 1% of the gradient
    even at loss weight 2, and the head duly learned a heavily shrunk estimate
    (predicted std 0.229 s against a true 0.424 s; EXP-22). Dividing by the
    0.05 s tolerance floor puts the two terms on the same scale AND expresses
    the error in the only unit the metric cares about: an error of 1.0 is
    exactly one tolerance, the point where a match is won or lost.

    Args:
        pred:   (B, T, 2) predicted seconds to the nearest onset / offset.
        target: (B, T, 2) true signed distance, clipped to the horizon.
        valid:  (B, T, 2) 1 where the nearest edge is inside the horizon.
        mask:   (B, T) 1 for real (non-padded) frames.

    delta=1.0 (one tolerance) keeps errors the metric cannot notice in the
    quadratic region and errors that would cost a match in the linear one.
    Frames with no edge within the horizon contribute nothing - predicting
    "0.5 s away" everywhere is not a signal we want to learn.
    """
    if tau_unit:
        # v11 Arm L: valid holds each frame's nearest-edge tolerance tau = max(0.2 L, 0.05) s
        # (dataset.boundary_targets(tau_valid=True)); errors are measured in THAT event's tau,
        # the metric's own unit, so short events get their share of the gradient.
        unit = valid.float().clamp(min=1e-3)
        valid = (valid > 0).float()
    v = valid.float() * mask.float().unsqueeze(-1)
    if offset_w != 1.0:
        # EXP-28: on natural clips 52.8% of ONSETS land inside tolerance
        # against 43.7% of offsets, and a match needs both - so offsets alone
        # cap matches at 31.6%. Were offsets as good as onsets, natural F1
        # would go from 0.38 to ~0.64, past the leader's ~0.61. The cause is
        # physical (the posterior's fall is 36% slower than its rise, 0.300s
        # vs 0.220s, because sounds decay more slowly than they attack), so the
        # loss should spend more of its budget on the harder edge.
        v = v * torch.tensor([1.0, offset_w], device=v.device,
                             dtype=v.dtype).view(1, 1, 2)
    loss = torch.nn.functional.huber_loss(
        pred.float() / unit, target.float() / unit,
        reduction="none", delta=delta) * v
    per_clip = loss.sum(dim=(1, 2)) / v.sum(dim=(1, 2)).clamp(min=1)
    if sample_weight is not None:
        sw = sample_weight.float()
        return (per_clip * sw).sum() / sw.sum().clamp(min=1e-6)
    return per_clip.mean()


def dense_box_loss(raw, tgt, msk, mask, sample_weight=None):
    """--box-kind dense: 1-D DIoU between the predicted and the true interval at every owned frame.

    raw/tgt/msk: (B, T, 2L) with (onset, offset) distances per lane; distances = softplus(raw) seconds.
    Both intervals contain the anchor frame, so intersection = sum of min, enclosing = sum of max, and the
    centre offset is half the difference of (offset - onset). IoU is scale-invariant, like the metric's
    tolerance (0.2 x event length): a 20 ms error costs a 0.2 s event what 200 ms costs a 2 s event.
    msk is per-event normalised (dataset.dense_targets), so the mean is over EVENTS, not frames."""
    d = F.softplus(raw.float())
    B_, T_, C_ = d.shape
    d = d.view(B_, T_, C_ // 2, 2)
    t = tgt.float().view(B_, T_, C_ // 2, 2)
    w = msk.float()[..., 0::2] * mask.float().unsqueeze(-1)
    if sample_weight is not None:
        w = w * sample_weight.float().view(-1, 1, 1)
    inter = torch.minimum(d, t).sum(-1)
    encl = torch.maximum(d, t).sum(-1).clamp_min(1e-6)
    centre = ((d[..., 1] - d[..., 0]) - (t[..., 1] - t[..., 0])) / 2
    loss = 1 - inter / encl + (centre / encl) ** 2
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def box_loss(pred, tgt, msk, mask, sample_weight=None, reg_w=1.0):
    """CenterNet penalty-reduced focal loss on the two edge heatmaps, plus
    L1 on log-duration and sub-frame residual at the frames next to each edge.
    pred/tgt/msk: (B, T, 6); mask: (B, T)."""
    m = mask.float()
    hl = pred[..., :2].float()
    ht = tgt[..., :2].float()
    p = torch.sigmoid(hl).clamp(1e-4, 1 - 1e-4)
    # positives = the ONE exact peak per edge (target exactly 1.0 = exp(0) at the edge frame); `> 0.999`
    # also caught the shoulders of long events' wide Gaussians (3/5/13 frames at 5/10/30 s)
    pos = (ht >= 1.0).float()
    neg = 1.0 - pos
    lpos = -torch.log(p) * (1 - p) ** 2 * pos
    lneg = -torch.log(1 - p) * p ** 2 * (1 - ht) ** 4 * neg
    hm = ((lpos + lneg) * m.unsqueeze(-1)).sum(dim=(1, 2)) /         pos.sum(dim=(1, 2)).clamp(min=1)
    rm = msk[..., 2:].float() * m.unsqueeze(-1)
    rp = pred[..., 2:].float()
    rt = tgt[..., 2:].float()
    reg = ((rp - rt).abs() * rm).sum(dim=(1, 2)) / rm.sum(dim=(1, 2)).clamp(min=1)
    per = hm + reg_w * reg
    if sample_weight is not None:
        sw = sample_weight.float()
        return (per * sw).sum() / sw.sum().clamp(min=1e-6)
    return per.mean()


def collar_loss(z, win, mask, sample_weight=None):
    """Collar-aware edge loss (Kalda & Alumae, Odyssey 2022).

    z: (B, T, 2) onset/offset logits. win: (B, T, 2K) 0/1 tolerance windows,
    [:K] onsets, [K:] offsets. For each window W the likelihood of 'exactly
    one positive frame inside W, all else negative' is, in logits,
        -sum_t softplus(z_t) + logsumexp_{t in W} z_t
    so the loss is sum_t softplus(z_t) - sum_W logsumexp_{t in W} z_t,
    normalised by frames. Rewards one confident peak anywhere the metric
    would accept the edge, not the annotator's exact frame."""
    z = z.float()
    m = mask.float()
    K = win.shape[-1] // 2
    per = []
    for e in (0, 1):
        ze = z[..., e]                                   # (B, T)
        neg = (torch.nn.functional.softplus(ze) * m).sum(1)
        w = win[..., e * K:(e + 1) * K].float()          # (B, T, K)
        has = (w.sum(1) > 0).float()                     # (B, K)
        lse = torch.logsumexp(ze.unsqueeze(-1) + torch.log(w.clamp_min(1e-12)), dim=1)
        pos = (lse * has).sum(1)
        per.append((neg - pos) / m.sum(1).clamp(min=1))
    per = per[0] + per[1]
    if sample_weight is not None:
        sw = sample_weight.float()
        return (per * sw).sum() / sw.sum().clamp(min=1e-6)
    return per.mean()


def clip_bce(logits, target, sample_weight=None):
    """BCE-with-logits on attention-pooled clip scores (the Bronze pathway)."""
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none").mean(dim=-1)
    if sample_weight is not None:
        sw = sample_weight.float()
        return (loss * sw).sum() / sw.sum().clamp(min=1e-6)
    return loss.mean()
