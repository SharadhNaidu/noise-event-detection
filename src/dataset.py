"""Windowed dataset over cached WavLM features.

Training on fixed-length windows rather than whole clips, for three reasons:
  * uniform shapes -> real batching (the encoder had to run solo, but the head
    does not: EXP-07's group-norm problem lives in the conv front-end, and we
    are past that now);
  * clips vary 1.4-21.2 s, so whole-clip batching would be mostly padding;
  * it matches how inference will window long clips.

Labels are rasterised onto the same 20 ms grid the features use, from the
reference (onset, offset) pairs. Channel 0 is "any noise" - the only scored
target; channels 1..7 are categories, auxiliary supervision only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CATS = ["animal", "vehicle_traffic", "baby_child", "singing_music",
        "phone_signal_alarm", "appliance_machine", "human_non_speech"]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + len(CATS)
FRAME_SEC = 0.02


def rasterise(events, cats, n_frames, frame_sec=FRAME_SEC, offset_frames=0, mode="touch"):
    """(onset, offset) pairs -> (n_frames, N_OUT) float32 target.

    mode="touch" (default, legacy): a frame is positive if the event covers any part of it
    (floor(on) / ceil(off)), so every target is ~half a frame too long at each end.
    mode="centre": frames [round(on), round(off)) - a frame is positive iff its centre is
    inside the event. Matches the RED impulses (round) and the decoder's time = index * fs,
    so a perfectly learned target decodes back to the reference unbiased (v11 Arm L).
    """
    y = np.zeros((n_frames, N_OUT), dtype=np.float32)
    for i, (on, off) in enumerate(events):
        if mode == "centre":
            a = int(round(on / frame_sec)) - offset_frames
            b = int(round(off / frame_sec)) - offset_frames
        else:
            a = int(np.floor(on / frame_sec)) - offset_frames
            b = int(np.ceil(off / frame_sec)) - offset_frames
        a, b = max(0, a), min(n_frames, max(b, a + 1))
        if b <= a:
            continue
        y[a:b, 0] = 1.0
        c = CAT2IDX.get(cats[i] if i < len(cats) else None)
        if c is not None:
            y[a:b, 1 + c] = 1.0
    return y


# Beyond this distance a frame carries no usable information about an edge, so
# the regression loss is masked off there. 0.5 s = 25 frames, chosen to be
# comfortably wider than the 0.36 s posterior ramp we are trying to sharpen.
REG_HORIZON = 0.5


def boundary_targets(events, n_frames, frame_sec=FRAME_SEC,
                     horizon=REG_HORIZON, tau_valid=False):
    """Signed distance from each frame to the nearest onset and offset.

    Returns (dist (n_frames, 2), valid (n_frames, 2)), both float32, with
    distances in SECONDS and clipped to +-horizon. valid is 0 where the nearest
    edge is further away than the horizon.

    Why this exists: a frame-wise BCE posterior is a calibrated statement about
    'is there an event here', and EXP-18 measured its transition at 0.36 s
    against an 0.086 s tolerance. That ramp is the model's honest uncertainty
    about WHERE the edge is, so reweighting frames near edges cannot remove it.
    Regression asks a different question - 'how far is the edge from here' -
    which is answerable to sub-frame precision, does not depend on the
    posterior's amplitude, and lets many frames vote on one boundary.
    """
    t = (np.arange(n_frames, dtype=np.float32) + 0.5) * frame_sec
    dist = np.zeros((n_frames, 2), dtype=np.float32)
    valid = np.zeros((n_frames, 2), dtype=np.float32)
    taus = np.asarray([max(0.2 * (off - on), 0.05) for on, off in events], dtype=np.float32)
    for k, edges in enumerate(([on for on, _ in events], [off for _, off in events])):
        if not edges:
            continue
        e = np.asarray(edges, dtype=np.float32)
        d = t[:, None] - e[None, :]              # (T, n_edges), + = edge is behind us
        j = np.argmin(np.abs(d), axis=1)
        nearest = d[np.arange(n_frames), j]
        inside = (np.abs(nearest) <= horizon).astype(np.float32)
        # tau_valid: valid carries the nearest edge's scorer tolerance (seconds) instead of 1,
        # so boundary_loss(tau_unit=True) measures each error in that event's own tolerance.
        valid[:, k] = inside * taus[j] if tau_valid else inside
        dist[:, k] = np.clip(nearest, -horizon, horizon)
    return dist, valid


def impulse_targets(events, n_frames, frame_sec=FRAME_SEC):
    """1 at the exact frame of each onset / offset, 0 everywhere else.

    These are what the onset-offset loss supervises (Schmid et al.,
    arXiv:2601.04178). They are the only targets in the whole pipeline that
    consume our annotation precision directly: EXP-34 measured the annotators
    at 2-3 ms median error against the true mix boundary, far finer than the
    20 ms frame, so an impulse at the rounded frame is a faithful target
    rather than a coarse one.
    """
    on = np.zeros((n_frames, 1), dtype=np.float32)
    off = np.zeros((n_frames, 1), dtype=np.float32)
    for a, b in events:
        i = int(round(a / frame_sec))
        j = int(round(b / frame_sec))
        if 0 <= i < n_frames:
            on[i, 0] = 1.0
        if 0 <= j < n_frames:
            off[j, 0] = 1.0
    return on, off


def _merge(evs):
    out = []
    for a, b in sorted(evs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def impulse_targets_cls(events, cats, n_frames, frame_sec=FRAME_SEC):
    """Per-channel onset / offset impulses for per-class RED: channel 0 from the
    UNION of all events (nested edges excluded - inside a present region the latch
    cannot start), channel 1+k from the merged events of category k only."""
    on = np.zeros((n_frames, N_OUT), dtype=np.float32)
    off = np.zeros((n_frames, N_OUT), dtype=np.float32)
    groups = {0: list(events)}
    for i, ev in enumerate(events):
        k = CAT2IDX.get(cats[i] if i < len(cats) else None)
        if k is not None:
            groups.setdefault(1 + k, []).append(ev)
    for ch, evs in groups.items():
        for a, b in _merge(evs):
            i, j = int(round(a / frame_sec)), int(round(b / frame_sec))
            if 0 <= i < n_frames:
                on[i, ch] = 1.0
            if 0 <= j < n_frames:
                off[j, ch] = 1.0
    return on, off


LANE_GAP = 0.1   # B6a: two events closer than this (or overlapping) go to different lanes


def lane_b_events(events, gap=LANE_GAP):
    """B6a lane B = events the greedy interval colouring cannot place in lane A (board/overlap/lanes_oracle.additive):
    sorted by onset, an event joins lane A iff it starts >= gap after the last lane-A offset (only lane-A events move
    that offset), otherwise it goes to lane B. Lane A itself is not returned: ch0 keeps the union of ALL events."""
    last, out = -1e9, []
    for on, off in sorted((float(e[0]), float(e[1])) for e in events):
        if last + gap <= on + 1e-9:
            last = max(last, off)
        else:
            out.append((on, off))
    return out


def _collar_window_ids(edges, n_frames, frame_sec):
    """edges: [(time, tau)]. Returns (T,) float ids: k>0 = frame in edge k's collar,
    -1 = collar of an edge outside the crop (ignored), 0 = background. Overlapping
    collars are split disjoint: a frame belongs to its NEAREST covering edge (ties to
    the earlier), so one peak can never satisfy two edges."""
    ids = np.zeros(n_frames, dtype=np.float32)
    best = np.full(n_frames, np.inf)
    for k, (t, tau) in enumerate(sorted(edges), 1):
        c = int(round(t / frame_sec))
        r = int(np.floor(tau / frame_sec + 1e-9))
        lo, hi = max(c - r, 0), min(c + r + 1, n_frames)
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        d = np.abs(idx - c).astype(float)
        win = d < best[idx]
        best[idx[win]] = d[win]
        ids[idx[win]] = k if 0 <= c < n_frames else -1
    return ids


def collar_window_ids_cls(events, cats, n_frames, frame_sec=FRAME_SEC):
    """--ool-collar targets: per-channel onset / offset COLLAR ids (T, N_OUT), same
    channel grouping as impulse_targets_cls (ch0 = merged union, 1+k = category k).
    Each edge's collar is the scorer's own tolerance max(0.2 * duration, 0.05 s) of
    the (merged) event, so short events demand precise edges and long ones do not."""
    on = np.zeros((n_frames, N_OUT), dtype=np.float32)
    off = np.zeros((n_frames, N_OUT), dtype=np.float32)
    groups = {0: list(events)}
    for i, ev in enumerate(events):
        k = CAT2IDX.get(cats[i] if i < len(cats) else None)
        if k is not None:
            groups.setdefault(1 + k, []).append(ev)
    for ch, evs in groups.items():
        merged = _merge(evs)
        tau = [max(0.2 * (b - a), 0.05) for a, b in merged]
        on[:, ch] = _collar_window_ids([(a, t) for (a, _), t in zip(merged, tau)], n_frames, frame_sec)
        off[:, ch] = _collar_window_ids([(b, t) for (_, b), t in zip(merged, tau)], n_frames, frame_sec)
    return on, off


COLLAR_K = 8  # max edges of each kind per crop for the collar-aware loss


def collar_targets(events, n_frames, frame_sec=FRAME_SEC, k=COLLAR_K):
    """Window masks for the collar-aware edge loss (Kalda & Alumae 2022).

    For every onset (and offset) a 0/1 mask over the frames within the
    metric's own tolerance max(0.2*dur, 50 ms) of it. The loss then only asks
    for ONE confident frame somewhere inside each window - exactly what the
    scorer accepts - instead of the annotator's exact frame, which the overfit
    probe showed is memorisable but not generalisable.
    Returns (T, 2k): columns [0:k] onset windows, [k:2k] offset windows.
    """
    m = np.zeros((n_frames, 2 * k), dtype=np.float32)
    slots = [0, 0]
    for a, b in sorted(events):
        tol = max(0.2 * (b - a), 0.05)
        r = max(int(tol / frame_sec), 1)
        for e, t in ((0, a), (1, b)):
            i = int(round(t / frame_sec - 0.5))
            lo, hi = max(i - r, 0), min(i + r + 1, n_frames)
            if hi <= lo or slots[e] >= k:
                continue
            m[lo:hi, e * k + slots[e]] = 1.0
            slots[e] += 1
    return m


BOX_CH = 6   # heat_on, heat_off, log fwd dur, log bwd dur, res_on, res_off


DENSE_LANES = 3   # --box-kind dense: 0 = any event (union), 1 = human_non_speech, 2 = every other category


def dense_targets(events, cats, n_frames, frame_sec=FRAME_SEC):
    """Dense interval targets for --box-kind dense (ActionFormer-style), BOX_CH = 2 * DENSE_LANES wide.

    Every frame inside an event regresses its distances in SECONDS to that event's onset and offset
    (columns 2k, 2k+1 for lane k). Lanes may overlap one another, so a breath nested inside a vehicle
    span keeps its own instance (27% of natural refs overlap another, 96% cross-category); within a lane
    events are merged. The weight of each frame is 1 / (frames of its event): every EVENT weighs 1, so a
    0.3 s breath counts as much as a 6 s background (frame BCE gives short events 16% of positive frames).
    Events cut by the crop (onset < 0 or offset > crop end) are skipped: their true edge is unknown."""
    times = (np.arange(n_frames) + 0.5) * frame_sec
    span = n_frames * frame_sec
    tgt = np.zeros((n_frames, 2 * DENSE_LANES), np.float32)
    msk = np.zeros_like(tgt)
    hns = [e for e, c in zip(events, cats) if c == "human_non_speech"]
    oth = [e for e, c in zip(events, cats) if c != "human_non_speech"]
    for k, evs in enumerate((list(events), hns, oth)):
        for a, b in _merge([tuple(e[:2]) for e in evs]):
            if a < 0 or b > span + 1e-6:
                continue
            own = (times >= a) & (times < b)
            if not own.any():
                continue
            tgt[own, 2 * k] = times[own] - a
            tgt[own, 2 * k + 1] = b - times[own]
            msk[own, 2 * k] = msk[own, 2 * k + 1] = 1.0 / own.sum()
    return tgt, msk


def box_targets(events, n_frames, frame_sec=FRAME_SEC):
    """Instance-level targets: every event is its own box (Voxaboxen-style,
    arXiv 2503.02389), so nested/overlapping events are representable.

    Returns (tgt (T, 6), mask (T, 6)):
      0 heat_on   Gaussian at each onset, sigma = half the metric tolerance
      1 heat_off  same at each offset (peak frame exactly 1.0)
      2 log(dur)  at frames within +-1 of an onset  (forward box)
      3 log(dur)  at frames within +-1 of an offset (backward box)
      4 res_on    exact onset position minus that frame, in frames
      5 res_off   same for the offset
    Frame k is centred at (k + 0.5) * frame_sec, as in boundary_targets.
    Durations are of the FULL event even when it leaves the crop.
    """
    tgt = np.zeros((n_frames, BOX_CH), dtype=np.float32)
    msk = np.zeros((n_frames, BOX_CH), dtype=np.float32)
    msk[:, 0:2] = 1.0
    k = np.arange(n_frames, dtype=np.float32)
    # regression frames shared by nearby edges go to the edge CLOSEST to the frame centre, ties to the
    # SHORTER event - deterministic, independent of annotation order (professor 2026-10-02)
    best = np.full((n_frames, 2), np.inf, dtype=np.float32)
    bdur = np.full((n_frames, 2), np.inf, dtype=np.float32)
    for a, b in events:
        dur = max(b - a, frame_sec)
        sig = max(0.5 * max(0.2 * dur, 0.05) / frame_sec, 1.0)
        for ch, t in ((0, a), (1, b)):
            u = t / frame_sec - 0.5
            i = int(round(u))
            if not (-3 <= i < n_frames + 3):
                continue
            g = np.exp(-0.5 * ((k - i) / sig) ** 2)
            tgt[:, ch] = np.maximum(tgt[:, ch], g)
            for j in (i - 1, i, i + 1):
                if 0 <= j < n_frames:
                    d = abs(u - j)
                    if d < best[j, ch] - 1e-6 or (abs(d - best[j, ch]) <= 1e-6 and dur < bdur[j, ch]):
                        best[j, ch], bdur[j, ch] = d, dur
                        tgt[j, 2 + ch] = np.log(dur)
                        tgt[j, 4 + ch] = u - j
                        msk[j, 2 + ch] = msk[j, 4 + ch] = 1.0
    return tgt, msk


class WindowDataset(Dataset):
    """Fixed-length windows sampled from cached features.

    Args:
        arr:        memmap (total_frames, dim) from features.load_store
        index:      {clip_id: {off, n, events, cats, tier, ...}}
        clip_ids:   which clips this split uses
        win:        window length in frames (250 = 5 s at 20 ms)
        stride:     hop between windows; None -> win (no overlap)
        weights:    {tier: float} per-clip loss weight, e.g. {"gold":1.0,"silver":0.4}
        train:      if True, sample a random window per clip each epoch;
                    if False, enumerate all windows deterministically.
    """

    def __init__(self, arr, index, clip_ids, win=250, stride=None,
                 weights=None, train=True, seed=0, clip_weights=None):
        self.arr = arr
        self.index = index
        self.clip_ids = [c for c in clip_ids if c in index]
        self.win = win
        self.stride = stride or win
        self.weights = weights or {}
        # per-clip FRAME-loss weight, from tierfilter. A clip whose only event
        # spans the whole clip gets 0 here (it cannot teach boundaries) while
        # still contributing through the clip/MIL head.
        self.clip_weights = clip_weights or {}
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.items = self._build()

    def _build(self):
        if self.train:
            return list(self.clip_ids)              # one random window per clip
        items = []
        for cid in self.clip_ids:
            n = self.index[cid]["n"]
            if n <= self.win:
                items.append((cid, 0))
            else:
                starts = list(range(0, n - self.win + 1, self.stride))
                if starts[-1] + self.win < n:
                    starts.append(n - self.win)     # cover the tail
                items.extend((cid, s) for s in starts)
        return items

    def __len__(self):
        return len(self.items)

    def _window(self, cid, start):
        e = self.index[cid]
        n = e["n"]
        feats = np.asarray(self.arr[e["off"]: e["off"] + n], dtype=np.float32)
        y = rasterise(e.get("events", []), e.get("cats", []), n)
        bd, bv = boundary_targets(e.get("events", []), n)
        # Silver's timestamps are ~4x rounder than Gold's and its events are
        # 2.5x longer, so they would teach the boundary head exactly the smear
        # we spent the day removing. Masking `valid` to zero excludes the clip
        # from the boundary loss entirely while leaving its frame and clip
        # supervision intact - which is the part of Silver worth having, since
        # it is 100 hours of real Indian acoustic scenes against our 22 of Gold.
        # This is the targeted version of the exclusion that was measured as a
        # blanket -0.042 back when there was no boundary head to protect.
        if e.get("no_boundary"):
            bv = np.zeros_like(bv)

        x = np.zeros((self.win, feats.shape[1]), dtype=np.float32)
        t = np.zeros((self.win, N_OUT), dtype=np.float32)
        m = np.zeros((self.win,), dtype=np.float32)
        d = np.zeros((self.win, 2), dtype=np.float32)
        v = np.zeros((self.win, 2), dtype=np.float32)
        take = min(self.win, n - start)
        x[:take] = feats[start: start + take]
        t[:take] = y[start: start + take]
        d[:take] = bd[start: start + take]
        v[:take] = bv[start: start + take]
        m[:take] = 1.0

        clip_t = t[:take].max(axis=0) if take else np.zeros(N_OUT, dtype=np.float32)
        w = self.clip_weights.get(cid, self.weights.get(e.get("tier", "gold"), 1.0))
        return x, t, m, clip_t.astype(np.float32), np.float32(w), d, v, cid, start

    def __getitem__(self, i):
        if self.train:
            cid = self.items[i]
            n = self.index[cid]["n"]
            start = 0 if n <= self.win else int(self.rng.integers(0, n - self.win + 1))
        else:
            cid, start = self.items[i]
        x, t, m, c, w, d, v, cid, start = self._window(cid, start)
        return (torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(m),
                torch.from_numpy(c), torch.tensor(w), torch.from_numpy(d),
                torch.from_numpy(v), cid, start)


def collate(batch):
    xs, ts, ms, cs, ws, ds, vs, cids, starts = zip(*batch)
    return (torch.stack(xs), torch.stack(ts), torch.stack(ms),
            torch.stack(cs), torch.stack(ws), torch.stack(ds),
            torch.stack(vs), list(cids), list(starts))
