"""Fine-tune the WavLM front-end end-to-end with the detection head.

Why this is the biggest lever we have left
------------------------------------------
EXP-18 measured the posterior's per-event 10-90% rise at 0.36 s against an
0.086 s tolerance, and the synthetic control (machine-exact labels, same 0.36 s
rise) proved the blur is the MODEL's, not the labels'. Every model in both of
our attempts has trained a small head on a FROZEN encoder, so that blur has
never been touched at its source. WavLM was pre-trained for masked prediction,
not for temporal precision; nothing in its objective ever rewarded a sharp
representation of an edge. Only fine-tuning can change that.

Why it fits on an 8 GB laptop GPU
---------------------------------
EXP-15 found layer 1 beats layer 10 by +0.057 AP, and the L1-vs-L10 training
A/B confirmed it (dev_AP 0.7149 vs 0.6639). If the useful signal lives at layer
1, we do not need the other eleven blocks AT ALL - not for the forward pass,
not for the backward pass, not in memory. Truncating to `--enc-layers 2` leaves
the conv front-end plus two transformer blocks: about 18M trainable parameters
instead of 94M, and roughly a 5x cheaper step. Fine-tuning goes from an
overnight job to a coffee break.

Two things that are easy to get wrong here
------------------------------------------
1. CROPS ARE EQUAL-LENGTH. EXP-07/08 showed this checkpoint's group-norm conv
   front-end corrupts a whole clip when padded - 1% padding gives 7% relative
   error. Fixed-length crops sidestep it entirely: every item in a batch is
   exactly `--crop` seconds, so there is no padding anywhere.
2. THE ENCODER NEEDS A MUCH SMALLER LR THAN THE HEAD. Parameter groups, as in
   the organizers' own stage-2 recipe (LR_ENC 5e-5, LR_HEAD 1e-3).
"""
from __future__ import annotations
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import astfeat as AF      # noqa: E402
import audiostore as A    # noqa: E402
import data as D          # noqa: E402
import dataset as DS      # noqa: E402
import model as M         # noqa: E402
import scorer as S        # noqa: E402
import splits as SP       # noqa: E402
import tierfilter as TF   # noqa: E402
import train as T         # noqa: E402

AUD = (P.ROOT / "audio")
CKPT = (P.ROOT / "ckpt")
SR = 16000
FS = 0.02                 # WavLM stride
HOP = int(SR * FS)        # 320 samples per frame


def load_ast_stores(stores, root):
    """{store: (frames memmap, clip index)} for the AudioSet stream.

    Fails loudly on a missing store: silently training on all-zero aux while
    evaluating on real features is exactly how fin_X broke."""
    out = {}
    for sn in stores:
        ar, me = AF.load_ast_store(root, sn)
        out[sn] = (ar, me["clips"], me["hop"], me["dim"])
    return out


def widen_state(state, net):
    """Zero-pad parent tensors that grew only in their last dim (the BiGRU
    input weights gaining AST columns), so the child starts as the parent."""
    own = net.state_dict()
    out = {}
    for k, v in state.items():
        t = own.get(k)
        if t is not None and t.shape != v.shape and t.shape[:-1] == v.shape[:-1]:
            pad = torch.zeros_like(t)
            pad[..., :v.shape[-1]] = v
            v = pad
        out[k] = v
    return out


def token():
    """HF token for gated/private downloads; public models need none. HF_TOKEN env, else an optional .env, else None."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return None
    for line in open(env, encoding="utf-8"):
        if line.strip().startswith(("HF_TOKEN", "HUGGING_FACE_PAT")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class Encoder(nn.Module):
    """WavLM-base+ truncated to `n_layers` transformer blocks, trainable.

    `freeze_conv` keeps the convolutional front-end fixed. That front-end is a
    generic 25 ms/20 ms filterbank; it is also the part whose group-norm makes
    padding dangerous, and the part most likely to be damaged by a small,
    noisy gradient. Freezing it is the default.
    """

    def __init__(self, n_layers=2, freeze_conv=True, name="microsoft/wavlm-base-plus",
                 layer_sum=False, init_layer=None, grad_ckpt=False):
        super().__init__()
        from transformers import AutoModel
        m = AutoModel.from_pretrained(name, token=token())
        m.encoder.layers = m.encoder.layers[:n_layers]
        self.m = m
        self.out_dim = m.config.hidden_size
        # LEARNED LAYER MIXTURE. Truncating to 4 blocks kept the acoustic
        # layers (3-5) and discarded the phonetic ones (9-11). A softmax over
        # all hidden states lets training choose. Initialised with almost all
        # mass on `init_layer`, so a warm start from a depth-4 checkpoint
        # begins at that checkpoint's own behaviour and can only move on
        # evidence.
        self.layer_sum = layer_sum
        if layer_sum:
            w0 = torch.zeros(n_layers + 1)
            if init_layer is not None:
                w0[init_layer] = 6.0
            self.layer_w = nn.Parameter(w0)
        if grad_ckpt:
            m.gradient_checkpointing_enable()
        if freeze_conv:
            for p in m.feature_extractor.parameters():
                p.requires_grad_(False)
            m.feature_extractor._freeze_parameters()

    def frames_for(self, n_samples):
        return int(self.m._get_feat_extract_output_lengths(
            torch.tensor([int(n_samples)]))[0])

    def forward(self, wav):
        """wav: (B, L) float32, already zero-mean/unit-var per item."""
        if not self.layer_sum:
            return self.m(wav).last_hidden_state
        hs = self.m(wav, output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_w.float(), 0).to(hs[0].dtype)
        return sum(w[i] * h for i, h in enumerate(hs))


def source_head(store, src_heads=False):
    """Output head for a training store. Legacy --conv-heads: 0 gold, 1 silver, 2 synth. --src-heads adds
    3 = Vaani pseudo-labels (*_pl / *pseudo*), 4 = external strong-label corpora (ext_strong)."""
    sn = str(store or "")
    if "silver" in sn:
        return 1
    if "synth" in sn:
        return 2
    if src_heads and ("_pl" in sn or "pseudo" in sn):
        return 3
    if src_heads and "ext_strong" in sn:
        return 4
    return 0


def is_foreign_store(store):
    """Stores whose labels follow another convention and must sit on their own head."""
    return source_head(store, src_heads=True) in (3, 4)


def restrict_dev(dev_ids, tr_ids, dev_stores):
    """Keep only dev clips from `dev_stores` (checkpoint selection judges the Gold convention, not pseudo or
    foreign labels); the other would-be dev clips are returned to training. None = unchanged."""
    if not dev_stores:
        return dev_ids, tr_ids
    keep = set(dev_stores)
    d = [c for c in dev_ids if c.split(":", 1)[0] in keep]
    return d, tr_ids + [c for c in dev_ids if c.split(":", 1)[0] not in keep]


def n_conv_heads(cfg):
    """Head count a checkpoint was trained with (gate/submit must rebuild it to load strictly)."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return 5 if get("src_heads") else 3 if get("conv_heads") else 1


def init_source_heads(net):
    """Extra heads start as exact copies of the warm Gold heads, so a run begins at the parent model."""
    h = net.head
    with torch.no_grad():
        for fx in getattr(h, "frame_x", []):
            fx.load_state_dict(h.frame.state_dict())
        if getattr(h, "boundary", None) is not None:
            for bx in getattr(h, "boundary_x", []):
                bx.load_state_dict(h.boundary.state_dict())
        if getattr(h, "red", None) is not None:
            for rx in getattr(h, "red_x", []):
                rx.load_state_dict(h.red.state_dict())


class CropDataset(Dataset):
    """Fixed-length waveform crops with frame and boundary targets.

    Every item is exactly `crop_s` seconds, so batches need no padding and the
    group-norm front-end sees a distribution it was trained on.
    """

    def __init__(self, arrs, index, clip_ids, crop_s=5.0, train=True, seed=0,
                 clip_weights=None, n_frames=None, p_center=0.5, ast=None,
                 cat_bd=False, mask_nb=False, box=False, collar=False, box_kind="peaks",
                 clip_loss_weights=None, red_classes=False, p_catbal=0.0,
                 ool_collar=False, src_heads=False, raster="touch", edge_tau=False, bd_union=False,
                 long_cov_w=1.0, long_cov_min=2.0, lane2=False, shoulder_w=1.0, shoulder_max=0.5):
        self.lane2 = lane2                                     # B6a lane-B targets
        self.long_cov_w, self.long_cov_min = long_cov_w, long_cov_min   # B8: weight on frames inside long refs
        self.shoulder_w, self.shoulder_max = shoulder_w, shoulder_max   # L3
        self.bd_union = bd_union      # R2: ch0 distance head on MERGED events (like RED ch0)
        self.raster = raster          # v11 Arm L: "centre" = unbiased presence targets
        self.edge_tau = edge_tau      # v11 Arm L: boundary valid carries each event's tau
        self.red_classes = red_classes
        self.ool_collar = ool_collar
        self.src_heads = src_heads
        # event-aware category balancing (professor 2026-10-02): with prob p_catbal a crop is centred on
        # an event of a UNIFORMLY chosen category (all co-occurring labels kept); else the old sampler
        self.p_catbal = p_catbal
        self.cat_events = {}
        self.arrs = arrs if isinstance(arrs, dict) else {None: arrs}
        self.index = index
        self.ids = [c for c in clip_ids if c in index]
        self.L = int(crop_s * SR)
        self.T = n_frames
        self.train = train
        self.clip_weights = clip_weights or {}
        # clip/boundary-loss weight; defaults to the frame weight (old recipe)
        self.clip_loss_weights = clip_loss_weights
        self.p_center = p_center
        self.ast = ast
        self.cat_bd = cat_bd
        self.mask_nb = mask_nb
        self.box = box
        self.box_kind = box_kind
        self.collar = collar
        self.rng = np.random.default_rng(seed)
        if p_catbal:
            for c_ in self.ids:
                ev_ = index[c_].get("events") or []
                ct_ = list(index[c_].get("cats") or [])
                for (a_, b_), k_ in zip(ev_, ct_ + [None] * (len(ev_) - len(ct_))):
                    if DS.CAT2IDX.get(k_) is not None:
                        self.cat_events.setdefault(k_, []).append((c_, a_, b_))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        forced = None
        if self.train and self.p_catbal and self.cat_events and self.rng.random() < self.p_catbal:
            ks = sorted(self.cat_events)
            pool = self.cat_events[ks[int(self.rng.integers(0, len(ks)))]]
            cid, a_, b_ = pool[int(self.rng.integers(0, len(pool)))]
            forced = (a_, b_)
        e = self.index[cid]
        n = e["n"]
        if n <= self.L:
            start = 0
        elif self.train:
            # EVENT-CENTRED CROPPING. EXP-32 showed the model reports
            # audibility where the reference marks presence: inside a long
            # event the posterior moves AGAINST the clip's loudness
            # (r = -0.178), because at a speech-masked frame the features
            # look exactly like silence. The only way to tell the two apart
            # is context - the model has to see that the source was there
            # before and after. A uniformly random crop shows both ends of a
            # long event only by luck, so half the time we centre the crop on
            # a real event instead.
            #
            # The jitter is not decoration. Always centring would let the
            # model learn 'the event is in the middle of the window', which
            # is true in training and false at inference; +-30% of the crop
            # keeps the position uninformative.
            evs = e.get("events") or []
            start = None
            if forced is not None or (evs and self.rng.random() < self.p_center):
                a, b = forced if forced is not None else evs[int(self.rng.integers(0, len(evs)))]
                mid = int((a + b) / 2 * SR)
                jitter = int(self.rng.uniform(-0.3, 0.3) * self.L)
                start = mid - self.L // 2 + jitter
                start = int(np.clip(start, 0, n - self.L))
            if start is None:
                start = int(self.rng.integers(0, n - self.L + 1))
        else:
            start = max(0, (n - self.L) // 2)
        w = A.get_wav(self.arrs[e.get("_store")], e, start, self.L)
        aux = None
        if self.ast == "scale":
            ww = np.pad(w, (0, max(0, self.L - len(w))))
            aux = np.array([ww.std() + 1e-7], np.float32)
        elif self.ast is not None:
            sn = e.get("_store")
            ar, ix, hop, _ = self.ast.get(sn, (None, None, AF.AST_HOP_S, None))
            ae = ix.get(e["_cid"]) if ix else None
            if ae is not None:
                h = np.asarray(ar[ae["off"]: ae["off"] + ae["n"]],
                               dtype=np.float32)
                # slice the AST frames covering this crop, then put them on
                # the 20 ms grid the fine stream uses
                a0 = int(start / SR / hop)
                a1 = a0 + int(np.ceil(self.L / SR / hop)) + 1
                aux = AF.upsample(h[a0:a1], self.T, hop_in=hop)
        if aux is None:
            dim = next(iter(self.ast.values()))[3] if self.ast else AF.AST_DIM
            aux = np.zeros((self.T, dim), np.float32)
        if len(w) < self.L:                       # pad ONLY the tail of short
            w = np.pad(w, (0, self.L - len(w)))   # clips; still equal length
        # per-utterance normalisation, matching the checkpoint's own extractor
        w = (w - w.mean()) / (w.std() + 1e-7)

        t0 = start / SR
        evs = [(a - t0, b - t0) for a, b in e.get("events", [])]
        cats = e.get("cats", [])
        y = DS.rasterise(evs, cats, self.T, mode=self.raster)
        lw = np.ones(self.T, np.float32)                      # B8 long-ref coverage weight
        if self.long_cov_w != 1.0:
            hop_s = self.L / SR / self.T
            for a_, b_ in evs:
                if b_ - a_ >= self.long_cov_min:
                    lw[max(0, int(a_ / hop_s)):min(self.T, int(np.ceil(b_ / hop_s)))] = self.long_cov_w
        if self.shoulder_w != 1.0:                         # L3 shoulder loss
            hop_s = self.L / SR / self.T; occ = np.zeros(self.T, bool)
            for a_, b_ in evs:
                occ[max(0, int(a_ / hop_s)):min(self.T, int(np.ceil(b_ / hop_s)))] = True
            for a_, b_ in evs:
                if b_ - a_ >= self.shoulder_max:
                    continue
                tw = max(0.2 * (b_ - a_), 0.05) + 0.04
                for lo, hi in ((a_ - tw, a_), (b_, b_ + tw)):
                    i0, i1 = max(0, int(lo / hop_s)), min(self.T, int(np.ceil(hi / hop_s)))
                    for k_ in range(i0, i1):
                        if not occ[k_]:
                            lw[k_] = max(lw[k_], self.shoulder_w)
        bd, bv = DS.boundary_targets(DS._merge(evs) if self.bd_union else evs, self.T, tau_valid=self.edge_tau)
        if self.cat_bd:
            # PER-CATEGORY edges, appended as channels 2.. (on_k, off_k).
            # 27% of natural events overlap another and 96% of those overlaps
            # are cross-category; one "nearest edge of any event" target is
            # ambiguous exactly there, so each category gets its own.
            cb, cv = [], []
            for k in range(len(DS.CATS)):
                ek = [ev for ev, ct in zip(evs, cats) if DS.CAT2IDX.get(ct) == k]
                d_, v_ = DS.boundary_targets(ek, self.T)
                cb.append(d_); cv.append(v_)
            bd = np.concatenate([bd] + cb, axis=1)
            bv = np.concatenate([bv] + cv, axis=1)
        if self.lane2:
            # B6a lane B: presence -> y[:, 8] (same raster as ch0), distance -> bd/bv after the category columns
            lb = DS.lane_b_events(evs)
            y = np.concatenate([y, DS.rasterise(lb, [], self.T, mode=self.raster)[:, :1]], axis=1)
            ld, lv = DS.boundary_targets(lb, self.T)
            bd = np.concatenate([bd, ld], axis=1)
            bv = np.concatenate([bv, lv], axis=1)
        if self.box:
            # instance-box targets appended LAST: bd[..., -6:], bv[..., -6:]
            if self.box_kind == "dense":
                # dense intervals (densedec / model.dense_box_loss). Silver (no_boundary) and Bronze (weak)
                # edges are not trusted for instance regression: their targets are masked.
                bt, bm = DS.dense_targets(evs, cats, self.T)
                if e.get("no_boundary") or e.get("weak"):
                    bm = np.zeros_like(bm)
            else:
                bt, bm = DS.box_targets(evs, self.T)
            bd = np.concatenate([bd, bt], axis=1)
            bv = np.concatenate([bv, bm], axis=1)
        if self.collar:
            # collar windows appended LAST (after any box targets)
            cw = DS.collar_targets(evs, self.T)
            bd = np.concatenate([bd, cw], axis=1)
            bv = np.concatenate([bv, cw], axis=1)
        if e.get("no_boundary") and self.mask_nb:
            # --no-boundary-stores historically down-weighted only the frame
            # loss; Silver's timestamps still trained the boundary head. Truly
            # masking them (top_A) cut natural F1 0.371 -> 0.308 at epoch 2,
            # so Silver edges HELP and masking is opt-in.
            bv = np.zeros_like(bv)
        # impulse targets at the exact onset / offset frames, for the OOL
        if self.ool_collar:
            # collar ids instead of exact-frame impulses (--ool-collar; see model.collar_ool_loss)
            ion, ioff = DS.collar_window_ids_cls(evs, cats, self.T)
            if not self.red_classes:
                ion, ioff = ion[:, :1], ioff[:, :1]
        else:
            ion, ioff = (DS.impulse_targets_cls(evs, cats, self.T) if self.red_classes
                         else DS.impulse_targets(evs, self.T))
        if self.lane2:
            lon, loff = DS.impulse_targets(DS._merge(lb), self.T)
            ion, ioff = np.concatenate([ion, lon], axis=1), np.concatenate([ioff, loff], axis=1)
        clip_t = weak_clip_target(cats) if e.get("weak") else y.max(axis=0)
        if clip_t.shape[0] < y.shape[1]:                      # B6a: weak clips carry no lane-B evidence
            clip_t = np.pad(clip_t, (0, y.shape[1] - clip_t.shape[0]))
        wt = self.clip_weights.get(cid, 1.0)
        wc = wt if self.clip_loss_weights is None else self.clip_loss_weights.get(cid, wt)
        sn = str(e.get("_store") or "")
        tier = source_head(sn, self.src_heads)
        return (torch.from_numpy(w.astype(np.float32)), torch.from_numpy(y),
                torch.from_numpy(clip_t), torch.tensor([wt, wc], dtype=torch.float32),
                torch.from_numpy(bd), torch.from_numpy(bv),
                torch.from_numpy(aux), torch.from_numpy(ion),
                torch.from_numpy(ioff), torch.tensor(tier, dtype=torch.long), torch.from_numpy(lw))


class UnlabelledCrops(Dataset):
    """Fixed-length crops from audio with NO labels at all.

    The mean teacher needs only the waveform: it asks the student to agree
    with an EMA copy of itself on the same clip under different perturbation.
    That is why this works on Bronze, which has been unusable all project for
    want of timestamps, and on the test audio, which has no labels by
    definition (transductive use of the provided audio only, disclosed).
    """

    def __init__(self, arrs, index, ids, crop_s=5.0, seed=0, ast=None, n_frames=None):
        self.arrs = arrs
        self.index = index
        self.ids = [c for c in ids if c in index]
        self.L = int(crop_s * SR)
        self.rng = np.random.default_rng(seed)
        self.ast = ast            # {store: (frames, clips, hop, dim)} or None
        self.T = n_frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        e = self.index[self.ids[i]]
        n = e["n"]
        start = 0 if n <= self.L else int(self.rng.integers(0, n - self.L + 1))
        w = A.get_wav(self.arrs[e.get("_store")], e, start, self.L)
        if len(w) < self.L:
            w = np.pad(w, (0, self.L - len(w)))
        w_std = float(w.std() + 1e-7)
        w = (w - w.mean()) / (w.std() + 1e-7)
        aux = np.zeros((0,), np.float32)
        if self.ast == "scale":
            aux = np.array([w_std], np.float32)
        elif self.ast is not None:
            ar, ix, hop, dim = self.ast[e.get("_store")]
            ae = ix.get(e.get("_cid"))
            if ae is None:
                raise KeyError(f"no aux features for unlabelled clip {e.get('_cid')}")
            h = np.asarray(ar[ae["off"]: ae["off"] + ae["n"]], dtype=np.float32)
            a0 = int(start / SR / hop)
            a1 = a0 + int(np.ceil(self.L / SR / hop)) + 1
            aux = AF.upsample(h[a0:a1], self.T, hop_in=hop).astype(np.float32)
        return torch.from_numpy(w.astype(np.float32)), torch.from_numpy(aux)


@torch.no_grad()
def ema_update(teacher, student, decay):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(decay).add_(sp.detach(), alpha=1 - decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def mt_rampup(step, total):
    """exp[-5(1 - t/T)^2] - the standard Laine/Tarvainen ramp.

    The unsupervised term starts at zero and grows. Schmid/Shao both warn
    that letting it dominate before the teacher is reliable is the main way
    to get mean teacher wrong; here the teacher starts from an already-
    trained checkpoint, which is the 'stage 1' their ablation says is
    mandatory (skipping it took 0.562 down to 0.483, worse than not
    fine-tuning at all).
    """
    t = min(max(step / max(total, 1), 0.0), 1.0)
    return float(np.exp(-5.0 * (1.0 - t) ** 2))


class FineTuneSED(nn.Module):
    """WavLM fine-tuned, optionally fused with a frozen AudioSet stream.

    EXP-33, with a matched-resolution control: at an identical 101 ms grid
    AST is worse than WavLM overall (AP 0.4496 vs 0.5250) and BETTER on events
    of >=1.5 s (0.3799 vs 0.3511), and CONCATENATING them beats either alone
    by +0.036 overall and +0.052 on long events. The two encoders hold
    different information - WavLM places edges precisely, AST knows what a
    sustained source is, because AudioSet has 527 classes of them and WavLM's
    pre-training had none.

    That matters because EXP-32 showed the model reports AUDIBILITY where the
    reference marks PRESENCE, and long events carry 80.6% of the Dice loss. A
    coarse stream is a better presence detector precisely because it pools
    over 100 ms and survives a masking that flattens a 20 ms frame.

    The AST stream is FROZEN and pre-cached, so it costs nothing per step and
    a failed experiment costs extraction time rather than a training run.
    """

    def __init__(self, n_layers=2, rnn_dim=256, dropout=0.3, input_dropout=0.1,
                 boundary_head=True, freeze_conv=True, aux_dim=0,
                 red_head=False, layer_sum=False, init_layer=None,
                 grad_ckpt=False, cat_boundary=0,
                 enc_name="microsoft/wavlm-base-plus", box_head=False,
                 conv_heads=1, collar_head=False, no_wavlm=False, conformer=0,
                 aux_mode=None, red_ch=1, box_kind="peaks", lane2=False):
        super().__init__()
        self.enc = Encoder(n_layers, freeze_conv, name=enc_name,
                           layer_sum=layer_sum, init_layer=init_layer,
                           grad_ckpt=grad_ckpt)
        # no_wavlm: the head sees ONLY the second stream; the encoder is kept
        # for its frame arithmetic (frames_for) but never run or trained
        self.no_wavlm = no_wavlm
        if no_wavlm:
            assert aux_dim, "--no-wavlm needs a second stream (--use-ast --aux ...)"
            for p in self.enc.parameters():
                p.requires_grad = False
        # aux_mode="beats_ft": the second stream is computed INSIDE the model by
        # a trainable BEATs encoder from raw-amplitude audio; the aux input then
        # carries only each crop's std so raw = normalised_wav * std
        self.aux_mode = aux_mode
        if aux_mode in AF.IN_MODEL:
            import beats_enc as BE
            m = AF.IN_MODEL[aux_mode]
            self.aux_enc = BE.MultiEncoder(m) if isinstance(m, tuple) else BE.make_encoder(m)
            aux_dim = self.aux_enc.out_dim
        self.aux_dim = 0 if no_wavlm else aux_dim
        in_dim = aux_dim if no_wavlm else self.enc.out_dim
        self.head = M.SEDHead(in_dim=in_dim, aux_dim=self.aux_dim, rnn_dim=rnn_dim, conformer=conformer,
                              dropout=dropout, input_dropout=input_dropout,
                              boundary_head=boundary_head,
                              red_head=red_head, red_ch=red_ch, cat_boundary=cat_boundary,
                              box_head=box_head, box_kind=box_kind, conv_heads=conv_heads,
                              collar_head=collar_head, lane2=lane2)

    def beats_features(self, wav, scale):
        """Raw-amplitude audio for BEATs from the normalised crop and its std."""
        return self.aux_enc(wav * scale.view(-1, 1).to(wav.dtype))

    def forward(self, wav, mask=None, aux=None, tier=None):
        if self.no_wavlm:
            h = aux.float()
            if mask is None:
                mask = torch.ones(h.shape[0], h.shape[1], device=h.device)
            return self.head(h, mask, tier=tier)
        h = self.enc(wav)
        if self.aux_mode in AF.IN_MODEL:
            f40 = self.beats_features(wav, aux)                   # (B, T40, 1215)
            j = torch.clamp((torch.arange(h.shape[1], device=h.device) * 0.5).long(),
                            max=f40.shape[1] - 1)                  # 20 ms -> 40 ms index
            aux = f40[:, j]
        if self.aux_dim:
            if aux is None:
                aux = torch.zeros(h.shape[0], h.shape[1], self.aux_dim,
                                  device=h.device, dtype=h.dtype)
            a = aux.to(h.dtype)
            t = min(h.shape[1], a.shape[1])
            h = torch.cat([h[:, :t], a[:, :t]], dim=-1)
        if mask is None:
            mask = torch.ones(h.shape[0], h.shape[1], device=h.device)
        return self.head(h, mask, tier=tier)


def weak_clip_target(cats):
    """Clip-level target of a weak (Bronze) clip: ch0 = 1 (every tagged clip holds an event) + its categories."""
    t = np.zeros(DS.N_OUT, np.float32); t[0] = 1.0
    for k in cats:
        if k in DS.CAT2IDX:
            t[1 + DS.CAT2IDX[k]] = 1.0
    return t


def build_clip_weights(idx, ids, silver_w, clip_w_fix=False, keep_deg_frames=False, weak_w=0.0):
    """(frame_w, clip_w) per clip. frame_w is 0 for degenerate clips (one event
    or the union covering >= 95%: no usable frame contrast). Without the fix the
    same weight also zeroed the clip and boundary losses, silently discarding
    ~8% of gold/valtune (audit 2026-10-01); with clip_w_fix those losses keep
    the tier weight (TF.clip_weight). Silver no_boundary clips are scaled by
    silver_w in both (their boundary valid mask is zeroed in the dataset)."""
    fw, cw = {}, {}
    for c in ids:
        e = idx[c]
        d = TF.clip_diagnosis(e.get("events", []),
                              e.get("dur_measured") or e.get("dur", 0))
        w = TF.frame_weight(d, e.get("tier", "gold"), {"gold": 1.0})
        wc = TF.clip_weight(d, e.get("tier", "gold"), {"gold": 1.0})
        if keep_deg_frames:
            # whole-clip events are a valid all-positive frame target; with weight 0
            # the model never learns whole-clip presence [MEASURED 2026-10-01: such
            # events hold 30% of natural ref frames and are only 39% covered]
            w = wc
        if e.get("no_boundary"):
            w *= silver_w
            wc *= silver_w
        if e.get("weak"):                    # Bronze: clip-level tags only (round 2, R2_04)
            w, wc = 0.0, weak_w
        fw[c], cw[c] = w, (wc if clip_w_fix else w)    # without the fix: old single weight
    return fw, cw


def parse_store_mult(items):
    """["gold_audio=2", ...] -> {"gold_audio": 2}"""
    return {k: int(v) for k, v in (s.split("=") for s in (items or []))}


def oversample(ids, idx, mult):
    """Repeat each clip of store s mult[s] times (training list only). Synthetic
    and Silver stores contain NO whole-clip events [MEASURED 2026-10-01], so the
    natural stores are the only source of the test-time label distribution."""
    out = []
    for c in ids:
        out.extend([c] * int(mult.get(idx[c].get("_store"), 1)))
    return out


def encoder_params(net):
    """Encoder tensors that are trainable by design (WavLM layers + in-model
    BEATs); the permanently frozen conv front-end is excluded."""
    ps = [p for n_, p in net.enc.named_parameters() if p.requires_grad and n_ != "layer_w"]
    if getattr(net, "aux_mode", None) in AF.IN_MODEL:
        ps += [p for p in net.aux_enc.parameters() if p.requires_grad]
    return ps


def set_trainable(params, flag):
    for p in params:
        p.requires_grad_(flag)


def _beats_depth(name):
    """LLRD depth of a BEATs tensor: layer i -> i+1, stem (below layer 0) -> 0,
    everything above the stack (final norm, SED heads) -> 13."""
    import re
    m = re.search(r"(?:encoder\.layers|blocks)\.(\d+)\.", name)   # BEATs | ATST-F
    if m:
        return int(m.group(1)) + 1
    if any(k in name for k in ("patch_embedding", "post_extract_proj", "pos_conv",
                               "beats.layer_norm", "atst.patch_embed", "atst.pos_embed",
                               "atst.mask_embed", "time_pos_embed", "freq_pos_embed",
                               "init_bn", "patch_embed.proj")):          # last four: Dasheng stem
        return 0
    return 13


def param_groups(net, lr_enc, lr_head, beats_top=None, llrd=None, n_layers=12):
    """Optimizer groups. Default: [encoder @ lr_enc, head @ lr_head] (old recipe).
    With llrd: in-model BEATs gets layer-wise decay, layer i at
    beats_top * llrd**(n_layers-1-i), the stem one step below layer 0, the
    top norm/heads at beats_top; WavLM stays at lr_enc."""
    # the layer mixture is a handful of logits: it trains at the head's rate
    mix_p = [p for n_, p in net.enc.named_parameters() if n_ == "layer_w" and p.requires_grad]
    wav_p = [p for n_, p in net.enc.named_parameters() if p.requires_grad and n_ != "layer_w"]
    head_p = list(net.head.parameters()) + mix_p
    if getattr(net, "aux_mode", None) not in AF.IN_MODEL or not llrd:
        enc_p = wav_p + ([p for p in net.aux_enc.parameters() if p.requires_grad]
                         if getattr(net, "aux_mode", None) in AF.IN_MODEL else [])
        return [{"params": enc_p, "lr": lr_enc}, {"params": head_p, "lr": lr_head}]
    by_depth = {}
    for n_, p in net.aux_enc.named_parameters():
        if p.requires_grad:
            by_depth.setdefault(_beats_depth(n_), []).append(p)
    groups = [{"params": wav_p, "lr": lr_enc}]
    for dpt in sorted(by_depth):
        lr = beats_top if dpt == 13 else beats_top * llrd ** (n_layers - dpt)
        groups.append({"params": by_depth[dpt], "lr": lr})
    groups.append({"params": head_p, "lr": lr_head})
    return groups


@torch.no_grad()
def clip_posteriors(net, arr, index, ids, crop_s, device, batch=8, ast=None,
                    all_channels=False, snap_last=False, with_box=False, taper_s=0.0):
    """Whole-clip posteriors by overlapping equal-length crops.

    all_channels=True returns (T, C) per clip: column 0 is exactly the default
    decoded track, columns 1.. the per-category heads (for per-class decoding).
    taper_s > 0: each crop's frames are weighted by a linear ramp over the first/last taper_s seconds
    (one-sided context near a crop edge gets less say where another crop covers the frame; frames covered
    by one crop only are unchanged, since the weighted mean renormalises). 0 = plain average (shipped).
    crop_s <= 0: one crop per clip = the WHOLE clip (no windows, no padding; batch is 1 per clip).
    """
    whole = crop_s <= 0
    L = int(crop_s * SR) if not whole else None
    net.eval()
    Tw = net.enc.frames_for(L) if not whole else None
    post, dist, clipp, boxes = {}, {}, {}, {}
    for cid in ids:
        e = index[cid]
        n = e["n"]
        if whole:
            L = max(n, SR); Tw = net.enc.frames_for(L)
        starts = [0] if n <= L else list(range(0, n - L + 1, L // 2))
        if n > L and starts[-1] + L < n:
            # optional snap onto the 20 ms frame grid (an off-grid start shifts
            # this crop's posteriors by up to +-10 ms vs its neighbours); any
            # frame no crop reaches is filled below either way.
            # snap_last=False keeps the old n-L start: on DECIDE the snapped
            # version measured -0.0015 (CI [-0.0050, +0.0023], P(better)=0.23),
            # i.e. not better, so the shipped behaviour stays the default.
            last = (n - L) // HOP * HOP if snap_last else n - L
            if last != starts[-1]:
                starts.append(last)
        crops, auxs = [], []
        for s in starts:
            w = A.get_wav(arr, e, s, L)
            if len(w) < L:
                w = np.pad(w, (0, L - len(w)))
            crops.append((w - w.mean()) / (w.std() + 1e-7))
            if ast == "scale":
                auxs.append(np.array([w.std() + 1e-7], np.float32))
            elif ast is not None:
                aa, ix, hop, dim = ast
                ae = ix.get(cid)
                if ae is None:
                    auxs.append(np.zeros((Tw, dim), np.float32))
                else:
                    h = np.asarray(aa[ae["off"]: ae["off"] + ae["n"]],
                                   dtype=np.float32)
                    a0 = int(s / SR / hop)
                    a1 = a0 + int(np.ceil(L / SR / hop)) + 1
                    auxs.append(AF.upsample(h[a0:a1], Tw, hop_in=hop))
        nT = net.enc.frames_for(n) if n >= L else Tw
        acc = None if all_channels else np.zeros(nT, np.float32)   # (nT, C) sized on first batch
        cnt = np.zeros(nT, np.float32)
        dac = np.zeros((nT, 2), np.float32)
        bac = np.zeros((nT, 6), np.float32) if with_box else None
        cmax = 0.0
        for i in range(0, len(crops), batch):
            xb = torch.from_numpy(
                np.stack(crops[i:i + batch]).astype(np.float32)).to(device)
            ab = (torch.from_numpy(np.stack(auxs[i:i + batch])).to(device)
                  if auxs else None)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):   # CPU bf16 is emulated and ~10x slower
                res = net(xb, aux=ab)
            if len(res) > 3 and res[3] is not None:
                # Decode the LATCHED presence, not the frame posterior. The
                # frame head still exists as an auxiliary target, but the
                # whole point of RED is that presence persists through frames
                # where the source is inaudible, so that is what the readout
                # must see.
                pres, _, _ = M.red_presence(*res[3])
                p = pres[:, :, 0].float().cpu().numpy()
            else:
                p = torch.sigmoid(res[0][:, :, 0].float()).cpu().numpy()
            if all_channels:
                if len(res) > 3 and res[3] is not None and res[3][0].shape[-1] > 1:
                    p = pres.float().cpu().numpy()          # per-class RED: latched track per channel
                else:
                    p = np.concatenate([p[:, :, None], torch.sigmoid(
                        res[0][:, :, 1:].float()).cpu().numpy()], axis=2)
            dd = (res[2].float().cpu().numpy()
                  if len(res) > 2 and res[2] is not None else None)
            bx = None
            if with_box and len(res) > 5 and res[5] is not None:   # box head: heatmaps -> probabilities
                bx = res[5].float()
                if getattr(net.head, "box_kind", "peaks") == "dense":   # dense: distances in seconds
                    bx = torch.nn.functional.softplus(bx).cpu().numpy()
                else:
                    bx = torch.cat([torch.sigmoid(bx[..., :2]), bx[..., 2:]], -1).cpu().numpy()
            cmax = max(cmax, float(torch.sigmoid(res[1][:, 0].float()).max()))
            if acc is None:
                acc = np.zeros((nT, p.shape[2]), np.float32)
            if taper_s > 0:
                k = max(1, int(round(taper_s / FS)))
                ramp = np.minimum(1.0, np.minimum(np.arange(Tw) + 0.5, Tw - np.arange(Tw) - 0.5) / k).astype(np.float32)
            else:
                ramp = np.ones(Tw, np.float32)
            for j, s in enumerate(starts[i:i + batch]):
                a = int(round(s / SR / FS))
                take = min(Tw, nT - a)
                if take <= 0:
                    continue
                wgt = ramp[:take]
                acc[a:a + take] += p[j, :take] * (wgt[:, None] if p.ndim == 3 else wgt)
                cnt[a:a + take] += wgt
                if dd is not None:
                    dac[a:a + take] += dd[j, :take] * wgt[:, None]
                if bx is not None:
                    bac[a:a + take] += bx[j, :take] * wgt[:, None]
        # frames no crop reached (tail < 1 hop) copy the nearest covered frame
        # instead of reading as 0
        fill = np.maximum.accumulate(np.where(cnt > 0, np.arange(nT), 0))
        den = np.where(cnt > 0, cnt, 1).astype(np.float32)
        post[cid] = (acc / (den[:, None] if all_channels else den))[fill]
        if dd is not None:
            dist[cid] = (dac / den[:, None])[fill]
        clipp[cid] = cmax
        if with_box:
            boxes[cid] = (bac / den[:, None])[fill]
    if with_box:
        return post, dist, clipp, boxes
    return post, dist, clipp


def _pad_inputs(arr, index, ids, pad, ast):
    """Copy of the clips with `pad` seconds of silence on both sides (and the
    precomputed aux stream padded by the same time), as an in-memory store."""
    ps = int(round(pad * SR))
    chunks, idx2, off = [], {}, 0
    for cid in ids:
        e = index[cid]
        w = np.asarray(arr[e["off"]: e["off"] + e["n"]])
        w = np.concatenate([np.zeros(ps, w.dtype), w, np.zeros(ps, w.dtype)])
        chunks.append(w); idx2[cid] = {"off": off, "n": len(w)}; off += len(w)
    arr2 = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    if ast is None or isinstance(ast, str):
        return arr2, idx2, ast
    aa, ix, hop, dim = ast
    pa = int(round(pad / hop))
    achunks, aix2, aoff = [], {}, 0
    for cid in ids:
        ae = ix.get(cid)
        if ae is None:
            continue
        h = np.asarray(aa[ae["off"]: ae["off"] + ae["n"]], dtype=np.float32)
        h = np.concatenate([np.zeros((pa, h.shape[1]), np.float32), h, np.zeros((pa, h.shape[1]), np.float32)])
        achunks.append(h); aix2[cid] = {"off": aoff, "n": len(h)}; aoff += len(h)
    aa2 = np.concatenate(achunks) if achunks else np.zeros((0, dim), np.float32)
    return arr2, idx2, (aa2, aix2, hop, dim)


def tta_posteriors(net, arr, index, ids, crop_s, device, batch=8, ast=None,
                   pads=(0.0, 0.64, 1.28), **kw):
    """Test-time augmentation: run clip_posteriors on clips zero-padded by p
    seconds on both sides (gives the clip start/end real context and shifts
    the crop grid), trim the p seconds back off, and average over pads.
    Pads must be multiples of 40 ms so both 20 ms frames and 40 ms aux stay
    on grid. Output lengths equal the unpadded clip_posteriors' lengths."""
    base, bdist, bclip = clip_posteriors(net, arr, index, ids, crop_s, device, batch, ast, **kw)
    acc = {c: [base[c]] for c in ids}
    dacc = {c: [bdist[c]] for c in ids if c in bdist}
    for pad in pads:
        if pad <= 0:
            continue
        assert abs(pad / 0.04 - round(pad / 0.04)) < 1e-6, pad
        a2, i2, ast2 = _pad_inputs(arr, index, ids, pad, ast)
        p2, d2, _ = clip_posteriors(net, a2, i2, ids, crop_s, device, batch, ast2, **kw)
        k = int(round(pad / FS))
        for c in ids:
            n = len(base[c])
            q = p2[c][k:k + n]
            if len(q) < n:
                q = np.concatenate([q, np.repeat(q[-1:], n - len(q), axis=0)])
            acc[c].append(q)
            if c in d2 and c in dacc:
                dq = d2[c][k:k + n]
                if len(dq) < n:
                    dq = np.concatenate([dq, np.repeat(dq[-1:], n - len(dq), axis=0)])
                dacc[c].append(dq)
    post = {c: np.mean(acc[c], axis=0) for c in ids}
    dist = {c: np.mean(dacc[c], axis=0) for c in dacc}
    return post, dist, bclip


import faulthandler as _fh, signal as _sig
if hasattr(_sig, "SIGUSR1"):        # `kill -USR1 <pid>` dumps every thread's stack to stderr (the log) - for hangs
    _fh.register(_sig.SIGUSR1, all_threads=True)


def _lane_split(fn, p, t, mask, **kw):
    """B6a: with a 9th lane-B channel, fn (which averages over channels) runs on channels 0..7 exactly as without
    --lane2, and lane B is added at one channel's weight, so ch0..7 keep their objective. No-op for 8-channel models."""
    n = DS.N_OUT
    if p.shape[-1] <= n:
        return fn(p, t, mask, **kw)
    cut = lambda x, s: x[..., s] if (x is not None and x.dim() == p.dim()) else x  # noqa: E731  per-channel masks only
    return (fn(p[..., :n], t[..., :n], cut(mask, slice(0, n)), **kw)
            + fn(p[..., n:], t[..., n:], cut(mask, slice(n, None)), **kw) / n)


def validate_objective_flags(a):
    """Fail at startup on objective flags that would otherwise be silently ignored."""
    if getattr(a, "lane2", False):
        if not getattr(a, "red_classes", False):
            raise SystemExit("--lane2 needs --red-classes (lane B is RED channel 8)")
        if getattr(a, "ool_collar", False):
            raise SystemExit("--lane2 does not support --ool-collar (collar ids have no lane-B channel)")
    if getattr(a, "dice_red", False):
        if not (getattr(a, "red", False) or getattr(a, "red_classes", False)):
            raise SystemExit("--dice-red needs an active RED head (--red or --red-classes)")
        if not getattr(a, "dice_w", 0) > 0:
            raise SystemExit("--dice-red needs --dice-w > 0")
    if getattr(a, "ool_collar", False) and not (getattr(a, "red", False)
                                                or getattr(a, "red_classes", False)):
        raise SystemExit("--ool-collar shapes the RED onset/offset loss; needs --red or --red-classes")
    foreign = [st for st in (getattr(a, "train_stores", None) or [])
               if is_foreign_store(st)]
    if foreign and not getattr(a, "src_heads", False):
        raise SystemExit(f"{foreign} carry a foreign label convention (pseudo-labels / external corpora): "
                         "train them only with --src-heads so they never reach the Gold head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stores", nargs="+", default=["gold_audio"],
                    help="one or more audio stores; synth_audio carries exact "
                         "boundaries and is the point of the exercise")
    ap.add_argument("--val-store", default="val_audio")
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crop", type=float, default=5.0)
    ap.add_argument("--lr-enc", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--long-cov-w", type=float, default=1.0, help="B8: loss weight on frames inside refs >= --long-cov-min s")
    ap.add_argument("--long-cov-min", type=float, default=2.0)
    ap.add_argument("--shoulder-w", type=float, default=1.0, help="L3: weight on negative frames just outside short refs")
    ap.add_argument("--shoulder-max", type=float, default=0.5)
    ap.add_argument("--dice-red", action="store_true",
                    help="apply --dice-w to the RED latched union presence (the deployed output) instead of the frame head")
    ap.add_argument("--dice-w", type=float, default=0.0,
                    help="weight of a per-clip soft Dice loss on the any-noise channel (arXiv 2107.10471)")
    ap.add_argument("--clip-loss-w", type=float, default=0.3)
    ap.add_argument("--boundary-w", type=float, default=0.3)
    ap.add_argument("--no-boundary-stores", nargs="*", default=["silver_audio"],
                    help="stores excluded from the BOUNDARY loss but kept for "
                         "the frame and clip losses; Silver's timestamps are "
                         "too coarse to teach edges but its audio is real")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="score the DECIDE half every N epochs, not just at "
                         "the end. Turns a long run from a blind bet into a "
                         "monitored one - if the honest number stops moving, "
                         "kill it and keep the hours")
    ap.add_argument("--mt-stores", nargs="*", default=[],
                    help="audio stores used as UNLABELLED data for the mean "
                         "teacher; their labels are never read")
    ap.add_argument("--mt-w", type=float, default=70.0,
                    help="max weight on the consistency loss. DCASE's default "
                         "is 2; ATST-SED ramps to 70 and reports that this is "
                         "the whole trick (0.492 -> 0.562 PSDS1)")
    ap.add_argument("--mt-decay", type=float, default=0.999)
    ap.add_argument("--dur-sample", action="store_true", help="B11: repeat each training clip ceil(duration/crop) times per epoch")
    ap.add_argument("--store-mult", nargs="*", default=None,
                    help="oversample training stores, e.g. gold_audio=2 valtune_audio=3")
    ap.add_argument("--keep-deg-frames", action="store_true",
                    help="degenerate clips (whole-clip events) keep FRAME-loss weight too")
    ap.add_argument("--clip-w-fix", action="store_true",
                    help="degenerate clips (one event >= 95%%) keep clip/boundary "
                         "loss weight (frame weight stays 0)")
    ap.add_argument("--llrd", type=float, default=None,
                    help="layer-wise LR decay for the in-model BEATs stack")
    ap.add_argument("--lr-beats-top", type=float, default=1e-4,
                    help="LR of the top BEATs layer/heads under --llrd")
    ap.add_argument("--warm-head-epochs", type=int, default=0,
                    help="train only the head for the first N epochs (encoders frozen)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a trained checkpoint. Mandatory for "
                         "mean teacher - the teacher must already be reliable "
                         "before the unsupervised term is allowed to dominate")
    ap.add_argument("--layer-sum", action="store_true",
                    help="learned softmax mixture over ALL encoder hidden "
                         "states instead of the last one")
    ap.add_argument("--init-layer", type=int, default=None,
                    help="put ~97%% of the initial mixture on this hidden "
                         "state (4 reproduces a depth-4 warm start)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--mask-nb-boundary", action="store_true",
                    help="really drop --no-boundary-stores from the boundary "
                         "loss (measured harmful: top_A ep2 natural F1 -0.063)")
    ap.add_argument("--enc-name", default="microsoft/wavlm-base-plus",
                    help="HF encoder id; microsoft/wavlm-large has 24 layers")
    ap.add_argument("--collar-head", action="store_true",
                    help="onset/offset peak heads trained with the collar-"
                         "aware loss (Kalda & Alumae 2022)")
    ap.add_argument("--collar-w", type=float, default=0.5)
    ap.add_argument("--dev-stores", nargs="*", default=None,
                    help="restrict the checkpoint-selection dev split to these stores (e.g. gold_audio "
                         "valtune_audio); their other would-be dev clips go back to training")
    ap.add_argument("--src-heads", action="store_true",
                    help="one frame/boundary/RED head per label source (gold, silver, synth, Vaani pseudo, "
                         "external strong corpora) on a shared encoder; inference uses the gold head")
    ap.add_argument("--conv-heads", action="store_true",
                    help="separate frame/boundary heads for Gold (decoded), "
                         "Silver and synthetic label conventions")
    ap.add_argument("--box-head", action="store_true",
                    help="instance box head: onset/offset heatmaps + "
                         "durations (nested events become separate boxes)")
    ap.add_argument("--box-w", type=float, default=1.0)
    ap.add_argument("--box-kind", choices=["peaks", "dense"], default="peaks",
                    help="box head flavour: CenterNet edge peaks (v7) or dense interval regression with DIoU in 3 lanes "
                         "(any / human_non_speech / other), decoded by densedec")
    ap.add_argument("--cat-boundary", action="store_true",
                    help="per-category onset/offset distance head")
    ap.add_argument("--catb-w", type=float, default=0.3)
    ap.add_argument("--p-catbal", type=float, default=0.0,
                    help="prob. a training crop is centred on an event of a uniformly chosen category")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save <tag>_ep<N>.pt every N epochs (default: follow --eval-every); <tag>_last.pt always")
    ap.add_argument("--red-tier-w", action="store_true",
                    help="apply per-clip tier weights (silver_w etc.) to the RED presence/onset/offset losses too")
    ap.add_argument("--weak-w", type=float, default=0.0,
                    help="R2_04: clip-loss weight of weak (Bronze, e['weak']) clips; their frame/RED/Dice weight is 0")
    ap.add_argument("--bd-union", action="store_true",
                    help="R2: train the ch0 boundary head on merged events (nested breaths no longer create inner edge targets)")
    ap.add_argument("--red-pres-loss", choices=["focal", "bce"], default="focal",
                    help="RED presence loss: focal gamma=2 (legacy) or bce = plain BCE + --pos-weight (Schmid et al.)")
    ap.add_argument("--raster", choices=["touch", "centre"], default="touch",
                    help="v11 Arm L: presence target convention; centre = [round(on), round(off)), unbiased")
    ap.add_argument("--edge-tau", action="store_true",
                    help="v11 Arm L: boundary-head loss in each event's own tolerance tau = max(0.2L, 50 ms)")
    ap.add_argument("--lane2", action="store_true",
                    help="B6a: 9th class-free lane-B channel (greedy interval colouring, gap 0.1 s) with its own RED latch "
                         "and boundary target; ch0 unchanged; posteriors become (T, 9). Needs --red-classes")
    ap.add_argument("--red-classes", action="store_true",
                    help="per-class RED: one latched start/end track per channel (implies --red)")
    ap.add_argument("--red", action="store_true",
                    help="Recurrent Event Detection head + onset-offset loss")
    ap.add_argument("--ool-w", type=float, default=100.0,
                    help="onset-offset loss weight; Schmid et al. fix it at 100")
    ap.add_argument("--ool-collar", action="store_true",
                    help="RED onset/offset loss rewards an edge anywhere inside the scorer's own "
                         "collar max(0.2*dur, 0.05 s) (disjoint per-edge windows) instead of only "
                         "the exact frame; docs/controller/ROOT_CAUSE_NATURAL_2026_10_02.md")
    ap.add_argument("--use-ast", action="store_true",
                    help="fuse the frozen AudioSet stream (EXP-33: +0.052 AP "
                         "on long events over WavLM alone)")
    ap.add_argument("--conformer", type=int, default=0,
                    help="number of Conformer blocks before the BiGRU head")
    ap.add_argument("--no-wavlm", action="store_true",
                    help="detector runs on the second stream alone (needs --use-ast)")
    ap.add_argument("--aux", choices=sorted(AF.AUX), default="ast",
                    help="which frozen second-stream encoder --use-ast fuses")
    ap.add_argument("--p-center", type=float, default=0.5,
                    help="probability a training crop is centred on a real "
                         "event rather than placed uniformly at random")
    ap.add_argument("--silver-w", type=float, default=0.4,
                    help="frame-loss weight for the no-boundary stores")
    ap.add_argument("--offset-w", type=float, default=1.0,
                    help="extra weight on the OFFSET channel of the boundary "
                         "loss; EXP-28 shows offsets, not onsets, cap natural F1")
    ap.add_argument("--no-boundary-head", action="store_true")
    ap.add_argument("--unfreeze-conv", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    validate_objective_flags(a)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT.mkdir(parents=True, exist_ok=True)

    net = FineTuneSED(a.enc_layers, boundary_head=not a.no_boundary_head,
                      freeze_conv=not a.unfreeze_conv,
                      aux_dim=AF.AUX[a.aux][1] if a.use_ast else 0,
                      aux_mode=a.aux if a.aux in AF.IN_MODEL else None,
                      no_wavlm=a.no_wavlm, conformer=a.conformer,
                      red_head=a.red or a.red_classes, layer_sum=a.layer_sum,
                      red_ch=DS.N_OUT if a.red_classes else 1,
                      init_layer=a.init_layer, grad_ckpt=a.grad_ckpt,
                      cat_boundary=len(DS.CATS) if a.cat_boundary else 0,
                      enc_name=a.enc_name, box_head=a.box_head, box_kind=a.box_kind,
                      conv_heads=n_conv_heads(a),
                      collar_head=a.collar_head, lane2=a.lane2).to(dev)
    if a.init_from:
        blob0 = torch.load(CKPT / f"{a.init_from}.pt", map_location=dev,
                           weights_only=False)
        # strict=False on purpose: a checkpoint trained without the RED head
        # has no weights for it, and we WANT the encoder and BiGRU warm and
        # the new head fresh. Report what matched so a silent mismatch cannot
        # pass as a warm start.
        missing, unexpected = net.load_state_dict(
            widen_state(blob0["state"], net), strict=False)
        print(f"  warm-started from {a.init_from}: "
              f"{len(blob0['state'])-len(unexpected)} tensors loaded, "
              f"{len(missing)} left fresh "
              f"({', '.join(m.split('.')[-2] for m in missing[:4]) or 'none'})",
              flush=True)

    if a.conv_heads or a.src_heads:
        # extra heads start as exact copies of the (warm) Gold heads (frame, boundary AND RED),
        # so the run begins at the parent model's behaviour
        init_source_heads(net)
        print(f"  {n_conv_heads(a)} source heads (gold decoded; silver, synth"
              f"{', pseudo, external' if a.src_heads else ''}), initialised from the gold head", flush=True)

    teacher, ul_loader = None, None
    if a.mt_stores:
        import copy as _copy
        teacher = _copy.deepcopy(net).to(dev)
        for q in teacher.parameters():
            q.requires_grad_(False)
        teacher.eval()
        u_arrs, u_idx = {}, {}
        for sn in a.mt_stores:
            ua, um = A.load_audio_store(AUD, sn)
            u_arrs[sn] = ua
            for cid, e in um["clips"].items():
                u_idx[f"{sn}:{cid}"] = {**e, "_store": sn, "_cid": cid}
        print(f"  mean teacher: {len(u_idx):,} UNLABELLED clips from "
              f"{', '.join(a.mt_stores)} (labels never read)", flush=True)
        ul_loader = DataLoader(
            UnlabelledCrops(u_arrs, u_idx, list(u_idx), a.crop, a.seed,
                            ast=(("scale" if a.aux in AF.IN_MODEL else
                                  load_ast_stores(a.mt_stores, AUD.parent / AF.AUX[a.aux][0]))
                                 if a.use_ast else None),
                            n_frames=net.enc.frames_for(int(a.crop * SR))),
            batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    Tw = net.enc.frames_for(int(a.crop * SR))
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"=== finetune [{a.tag}] enc_layers={a.enc_layers} crop={a.crop}s "
          f"-> {Tw} frames | trainable {n_tr/1e6:.1f}M ===", flush=True)

    # Several stores are presented as one by prefixing clip ids. Keeping the
    # arrays separate matters: each is its own memmap, and a synthetic clip and
    # a Gold clip can legitimately share an id.
    arrs, idx, ast_stores = {}, {}, {}
    for sname in a.train_stores:
        ar, me = A.load_audio_store(AUD, sname)
        arrs[sname] = ar
        n_before = len(idx)
        nb = any(k in sname for k in a.no_boundary_stores)
        for cid, e in me["clips"].items():
            idx[f"{sname}:{cid}"] = {**e, "_store": sname, "no_boundary": nb,
                                     "_cid": cid}
        print(f"  {sname}: {len(idx)-n_before:,} clips", flush=True)
    if a.use_ast:
        ast_stores = ("scale" if a.aux in AF.IN_MODEL
                      else load_ast_stores(a.train_stores, AUD.parent / AF.AUX[a.aux][0]))
        if ast_stores == "scale":
            print("  second stream: BEATs computed in-model (fine-tuned)", flush=True)
        else:
            print(f"  AST stream: {sum(len(v[1]) for v in ast_stores.values()):,} "
                  f"clips with AudioSet features", flush=True)
    ids = list(idx)[:a.limit] if a.limit else list(idx)

    # speaker-disjoint dev split, same rule as train.py
    groups = {}
    for c in ids:
        e = idx[c]
        # synthetic clips inherit their host's grouping key, so a host and its
        # synthetic offspring can never straddle the train/dev boundary
        base = e.get("host") or c.split(":", 1)[1]
        k = (f"{e.get('state')}_{e.get('district')}_{e.get('lang')}"
             if "district" in e else SP.speaker_key(base))
        groups.setdefault(k, []).append(c)
    gk = sorted(groups)
    np.random.default_rng(a.seed).shuffle(gk)
    n_dev = max(1, int(0.1 * len(gk)))
    dev_ids = [c for g in gk[:n_dev] for c in groups[g]]
    tr_ids = [c for g in gk[n_dev:] for c in groups[g]]
    dev_ids, tr_ids = restrict_dev(dev_ids, tr_ids, a.dev_stores)
    if a.store_mult:
        tr_ids = oversample(tr_ids, idx, parse_store_mult(a.store_mult))
    if a.dur_sample:
        import math as _m
        tr_ids = [c for c in tr_ids for _ in range(max(1, int(_m.ceil(idx[c]["n"] / SR / a.crop))))]
        print(f"  --dur-sample: {len(tr_ids):,} training items (clips repeated ceil(duration/crop) times)", flush=True)
        print(f"  --store-mult {a.store_mult}: {len(tr_ids):,} training items after oversampling", flush=True)

    cw, cwc = build_clip_weights(idx, ids, a.silver_w, a.clip_w_fix, a.keep_deg_frames, weak_w=a.weak_w)
    n_nb = sum(1 for c in ids if idx[c].get("no_boundary") and cw[c] > 0)
    if a.clip_w_fix:
        print(f"  --clip-w-fix: {sum(1 for c in ids if cw[c] == 0 and cwc[c] > 0):,} "
              f"degenerate clips keep clip/boundary weight", flush=True)
    if n_nb:
        print(f"  {n_nb:,} clips contribute frame/clip supervision only "
              f"(weight {a.silver_w}), excluded from the boundary loss")
    print(f"  {sum(1 for v in cw.values() if v > 0):,}/{len(cw):,} clips carry "
          f"usable boundary information")
    print(f"  train {len(tr_ids):,} | dev {len(dev_ids):,} clips "
          f"(speaker-disjoint)", flush=True)

    mk = lambda i, t: CropDataset(  # noqa: E731
        arrs, idx, i, a.crop, t, a.seed, cw, Tw, clip_loss_weights=cwc,
        p_center=a.p_center if t else 0.0,
        ast=(ast_stores or None) if a.use_ast else None,
        cat_bd=a.cat_boundary, mask_nb=a.mask_nb_boundary, box=a.box_head, box_kind=a.box_kind,
        collar=a.collar_head, red_classes=a.red_classes, long_cov_w=a.long_cov_w, long_cov_min=a.long_cov_min, shoulder_w=a.shoulder_w, shoulder_max=a.shoulder_max,
        p_catbal=a.p_catbal if t else 0.0, ool_collar=a.ool_collar, src_heads=a.src_heads,
        raster=a.raster, edge_tau=a.edge_tau, bd_union=a.bd_union, lane2=a.lane2)
    tl = DataLoader(mk(tr_ids, True), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=True)
    dl = DataLoader(mk(dev_ids, False), batch_size=a.batch, shuffle=False,
                    num_workers=a.workers)

    groups = param_groups(net, a.lr_enc, a.lr_head, beats_top=a.lr_beats_top, llrd=a.llrd)
    enc_train = encoder_params(net)          # toggled by --warm-head-epochs
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    steps = a.epochs * max(1, len(tl) // a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(          # one max_lr PER group
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
    if a.llrd:
        print("  LLRD groups: " + " ".join(f"{g['lr']:.1e}x{len(g['params'])}" for g in groups),
              flush=True)

    refs, _ = D.load_validation()
    tune_ids, decide_ids = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    varr, vmeta = A.load_audio_store(AUD, a.val_store)
    vidx = vmeta["clips"]
    tune_ids = [c for c in tune_ids if c in vidx]
    decide_ids = [c for c in decide_ids if c in vidx]

    from sklearn.metrics import average_precision_score
    v_ast_probe = None
    if a.use_ast:   # no silent fallback: a model trained with AST must be scored with it
        _vs = AF.AUX[a.aux][2] or a.val_store
        v_ast_probe = ("scale" if a.aux in AF.IN_MODEL
                       else load_ast_stores([_vs], AUD.parent / AF.AUX[a.aux][0])[_vs])
    best, best_state, hist = -1.0, None, []
    gstep = 0
    ul_iter = iter(ul_loader) if ul_loader is not None else None
    ramp_steps = max(1, len(tl) * min(3, a.epochs))      # ramp over ~3 epochs
    for ep in range(1, a.epochs + 1):
        net.train()
        if a.warm_head_epochs:
            set_trainable(enc_train, ep > a.warm_head_epochs)
            if ep in (1, a.warm_head_epochs + 1):
                print(f"  epoch {ep}: encoders {'TRAINING' if ep > a.warm_head_epochs else 'FROZEN (warm head)'}",
                      flush=True)
        t0, tot, nb = time.time(), 0.0, 0
        comp, ncomp = {}, 0                         # sampled loss components (every 50 batches)
        opt.zero_grad(set_to_none=True)
        for i, (w, y, c, sw, bd, bv, ax, ion, ioff, tr, lw) in enumerate(tl):
            w, y, c, sw, bd, bv, ax, ion, ioff, tr, lw = (
                t.to(dev, non_blocking=True)
                for t in (w, y, c, sw, bd, bv, ax, ion, ioff, tr, lw))
            swf, sw = sw[:, 0], sw[:, 1]     # frame-loss weight | clip/boundary-loss weight
            m = torch.ones(w.shape[0], Tw, device=dev)
            mw = m * lw[:, :Tw]              # B8: frame mask carrying the long-ref coverage weight
            with torch.autocast(dev, dtype=torch.bfloat16):
                res = net(w, m, aux=ax if a.use_ast else None,
                          tier=tr if (a.conv_heads or a.src_heads) else None)
                loss = M.masked_bce(res[0], y, mw, pos_weight=a.pos_weight,
                                    sample_weight=swf)
                if a.dice_w and not a.dice_red:
                    loss = loss + a.dice_w * M.soft_dice_loss(res[0][..., 0], y[..., 0], m,
                                                              sample_weight=swf)
                loss = loss + a.clip_loss_w * _lane_split(lambda p_, t_, _m, **k_: M.clip_bce(p_, t_, **k_),
                                                          res[1], c, None, sample_weight=sw)
                if len(res) > 2 and res[2] is not None:
                    loss = loss + a.boundary_w * M.boundary_loss(
                        res[2], bd[..., :2], bv[..., :2], m, sample_weight=sw,
                        offset_w=a.offset_w, tau_unit=a.edge_tau)
                ncol = 2 * DS.COLLAR_K if a.collar_head else 0
                if ncol and len(res) > 6 and res[6] is not None:
                    loss = loss + a.collar_w * M.collar_loss(
                        res[6], bd[..., -ncol:], m, sample_weight=sw)
                    bd, bv = bd[..., :-ncol], bv[..., :-ncol]
                nbx = DS.BOX_CH if a.box_head else 0
                if len(res) > 4 and res[4] is not None:
                    loss = loss + a.catb_w * M.boundary_loss(
                        res[4], bd[..., 2:bd.shape[-1] - nbx],
                        bv[..., 2:bv.shape[-1] - nbx], m, sample_weight=sw)
                if len(res) > 5 and res[5] is not None:
                    bl = M.dense_box_loss if a.box_kind == "dense" else M.box_loss
                    loss = loss + a.box_w * bl(
                        res[5], bd[..., -nbx:], bv[..., -nbx:], m,
                        sample_weight=sw)
                if len(res) > 3 and res[3] is not None:
                    # RED + ONSET-OFFSET LOSS (Schmid et al. arXiv:2601.04178).
                    # Presence is derived from the start/end logits by a latch,
                    # and supervised through that recurrence; the OOL puts a
                    # focal loss on the exact onset and offset frames. Their
                    # ablation attributes the entire gain to the OOL term
                    # (CRNN collar-F1 30.4 -> 39.7 of a final 40.6), with
                    # lambda fixed at 100 across every experiment.
                    pres, p_on, p_off = M.red_presence(*res[3])
                    l_dice = None
                    if a.dice_w and a.dice_red:            # Dice on the deployed union presence
                        l_dice = M.soft_dice_prob(pres[..., 0], y[..., 0], m, sample_weight=sw)
                        loss = loss + a.dice_w * l_dice
                    # CLIP weight (tier weight; kept for whole-clip clips under --clip-w-fix), not the
                    # frame weight swf, which is 0 for whole-clip clips without --keep-deg-frames and
                    # would silently drop their sustained-presence RED supervision (professor, 2026-10-01).
                    # --weak-w: RED follows the FRAME weight, so weak (Bronze) clips never train RED edges
                    rw = swf if a.weak_w > 0 else (sw if a.red_tier_w else None)
                    if a.red_pres_loss == "bce":   # bts_R2: paper-style presence loss (plain BCE + pos_weight)
                        l_pres = _lane_split(M.focal_bce, pres, y[..., :pres.shape[-1]], m, sample_weight=rw,
                                             gamma=0.0, pos_weight=a.pos_weight)
                    else:
                        l_pres = _lane_split(M.focal_bce, pres, y[..., :pres.shape[-1]], mw, sample_weight=rw)
                    if a.ool_collar:   # an edge anywhere inside the scorer's collar counts
                        l_on = M.collar_ool_loss(p_on, ion, M.onset_censor_mask(y, ion, m),
                                                 sample_weight=rw)
                        l_off = M.collar_ool_loss(p_off, ioff, m, sample_weight=rw)
                    else:
                        l_on = _lane_split(M.focal_bce, p_on, ion, M.onset_censor_mask(y, ion, m), sample_weight=rw)
                        l_off = _lane_split(M.focal_bce, p_off, ioff, m, sample_weight=rw)
                    loss = loss + l_pres + a.ool_w * (l_on + l_off)
                    if i % 50 == 0:                       # raw (unweighted) components, sampled
                        ncomp += 1
                        for k_, v_ in (("red_pres", l_pres), ("red_on", l_on), ("red_off", l_off),
                                       ("dice", l_dice), ("total", loss)):
                            if v_ is not None:
                                comp[k_] = comp.get(k_, 0.0) + float(v_.detach())
            # ---- MEAN TEACHER on unlabelled real audio ----------------
            # The student and an EMA copy of itself see the SAME clip with
            # different noise, and must agree. No labels are involved, which
            # is why this can use Bronze (no timestamps) and the test audio.
            # ATST-SED's ablation is emphatic that the WEIGHT is the trick:
            # DCASE's default is 2, they ramp to 70, and "without supervised
            # losses" still scores 0.538 of their 0.562 - almost all of the
            # gain comes from the unlabelled data.
            if teacher is not None:
                try:
                    uw, uax = next(ul_iter)
                except (StopIteration, TypeError):
                    ul_iter = iter(ul_loader)
                    uw, uax = next(ul_iter)
                uw = uw.to(dev, non_blocking=True)
                uax = uax.to(dev, non_blocking=True) if a.use_ast else None
                um = torch.ones(uw.shape[0], Tw, device=dev)
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                    t_out = teacher(uw + 0.01 * torch.randn_like(uw), um, aux=uax)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    s_out = net(uw, um, aux=uax)
                w_mt = a.mt_w * mt_rampup(gstep, ramp_steps)
                cons = torch.nn.functional.mse_loss(
                    torch.sigmoid(s_out[0].float()),
                    torch.sigmoid(t_out[0].float()))
                if len(s_out) > 3 and s_out[3] is not None:
                    sp, _, _ = M.red_presence(*s_out[3])
                    tp, _, _ = M.red_presence(*t_out[3])
                    cons = cons + torch.nn.functional.mse_loss(sp, tp.detach())
                loss = loss + w_mt * cons
            gstep += 1

            (loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                if teacher is not None:
                    ema_update(teacher, net, a.mt_decay)
                if sched.last_epoch < steps - 1:
                    sched.step()
                opt.zero_grad(set_to_none=True)
            tot += float(loss.detach())
            nb += 1
            if not np.isfinite(tot):
                raise RuntimeError(f"loss became non-finite at epoch {ep}")

        net.eval()
        ys, ps, berr = [], [], []
        with torch.no_grad():
            for w, y, c, sw, bd, bv, ax, ion, ioff, _tr, _lw in dl:
                w, ax = w.to(dev), ax.to(dev)
                m = torch.ones(w.shape[0], Tw, device=dev)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    res = net(w, m, aux=ax if a.use_ast else None)
                fr = res[0]
                ps.append(torch.sigmoid(fr[:, :, 0].float()).cpu().numpy().ravel())
                ys.append(y[:, :, 0].numpy().ravel())
                if len(res) > 2:
                    # Mean |error| of the distance head, over frames that are
                    # actually near an edge. dev_AP is BLIND to this head - it
                    # only sees channel 0 - so selecting on dev_AP alone can
                    # discard the epoch where the boundary estimate is best.
                    d = res[2].float().cpu()
                    bd, bv = bd[..., :2], bv[..., :2]
                    v = bv.bool()
                    if v.any():
                        berr.append(float((d - bd).abs()[v].mean()))
        yv, pv = np.concatenate(ys), np.concatenate(ps)
        ap_ = float(average_precision_score(yv, pv)) if yv.max() > 0 else 0.0
        mae = float(np.mean(berr)) if berr else float("nan")
        hist.append({"ap": ap_, "bmae": mae})
        # Selection score: frame AP, minus a penalty for a coarse boundary head.
        # The scale is set by the tolerance floor - a head whose mean error is a
        # full 0.05 s costs as much as 0.05 AP, which is roughly the gap between
        # a good and a bad epoch. Falls back to plain AP with no boundary head.
        sel = ap_ - (0.0 if np.isnan(mae) else min(mae, 0.25))
        star = ""
        if sel > best:
            best, star = sel, "  <- best"
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        print(f"  ep {ep:>2}/{a.epochs}  loss {tot/max(nb,1):.4f}  "
              f"dev_AP {ap_:.4f}  b_mae {mae:.4f}s  sel {sel:.4f}  "
              f"{time.time()-t0:.0f}s{star}", flush=True)
        if ncomp:
            print("     components (raw, mean of sampled batches): " +
                  "  ".join(f"{k} {v / ncomp:.4f}" for k, v in comp.items()), flush=True)

        # PERIODIC HONEST CHECK. dev_AP is drawn from the training stores and
        # has misled us before (fin_X had the best dev_AP of any model and was
        # the worst). The decide half is the only number that has ever
        # predicted the leaderboard, so on a long run we pay ~2 minutes every
        # few epochs to watch it, and can stop the moment it flattens.
        se = a.save_every or a.eval_every                 # save cadence, independent of evaluation
        if se and ep % se == 0 and ep < a.epochs:
            try:
                torch.save({"state": net.state_dict(), "epoch": ep,
                            "cfg": {**vars(a), "finetune": True},
                            "readout": {}, "dev_ap": best, "dev_hist": hist},
                           CKPT / f"{a.tag}_ep{ep}.pt")
            except Exception as ex:
                print(f"     >> periodic save failed: {type(ex).__name__}", flush=True)
        if a.eval_every and ep % a.eval_every == 0 and ep < a.epochs:
            try:
                pv_, dv_, _ = clip_posteriors(net, varr, vidx, decide_ids,
                                              a.crop, dev, ast=v_ast_probe)
                rc_ = {"kind": "global", "thr": 0.6, "med": 7,
                       "min_dur": 0.10, "merge_gap": 0.06}
                pr_ = T.decode_all(pv_, rc_, dv_, decide_ids)
                sc_ = S.score({c: refs[c] for c in decide_ids}, pr_)
                print(f"     >> DECIDE @ep{ep}: {sc_['combined']:.4f}  "
                      f"(F1 {sc_['event_f1']:.4f} Dice "
                      f"{sc_['segment_dice']:.4f})  [fixed readout, "
                      f"trend only]", flush=True)
                net.train()
            except Exception as ex:
                print(f"     >> periodic eval failed: {type(ex).__name__}",
                      flush=True)

    # the RAW last epoch, before the proxy-best (dev_AP - b_mae) state is restored:
    # for RED models that proxy reads the frame head, not the decoded output, so the
    # true best epoch must be chosen later with the deployed decoder (professor 2026-10-01).
    torch.save({"state": net.state_dict(), "epoch": a.epochs,
                "cfg": {**vars(a), "finetune": True}, "readout": {}, "dev_ap": best,
                "dev_hist": hist}, CKPT / f"{a.tag}_last.pt")
    if best_state is not None:
        net.load_state_dict(best_state)
        # hist holds dicts now (ap + boundary MAE), so the best epoch must be
        # found with the same selection score the loop used. np.argmax over a
        # list of dicts RAISES - and it raised after a 63-minute run had
        # finished training but before the checkpoint was written. Hence also
        # the save-before-evaluate below.
        sels = [h["ap"] - (0.0 if np.isnan(h["bmae"]) else min(h["bmae"], 0.25))
                for h in hist]
        i = int(np.argmax(sels))
        print(f"\n  restored best: sel {sels[i]:.4f} (dev_AP "
              f"{hist[i]['ap']:.4f}, b_mae {hist[i]['bmae']:.4f}s) at epoch "
              f"{i+1} of {a.epochs}", flush=True)

    # SAVE THE WEIGHTS BEFORE EVALUATING. The readout sweep and the ALL-clip
    # scoring take minutes and touch a lot of code; anything that throws there
    # used to destroy a finished training run. Re-saved with scores afterwards.
    out = CKPT / f"{a.tag}.pt"
    torch.save({"state": net.state_dict(), "cfg": {**vars(a), "finetune": True},
                "readout": {}, "dev_ap": best, "dev_hist": hist}, out)
    print(f"  weights saved to {out} before evaluation", flush=True)

    print("\n=== evaluating on the official validation set ===")
    # A model trained on validation is being scored against clips it has seen.
    # That is fine for the final models - they are validated by proxy through
    # nat_val, which trained on the tune half only - but a number printed
    # without a warning is a number someone will eventually believe.
    if any("valall" in x or "valtune" in x for x in a.train_stores):
        which = ("ALL of validation" if any("valall" in x for x in a.train_stores)
                 else "the validation TUNE half")
        print(f"  !! CONTAMINATED: this model trained on {which}. TUNE is")
        print("     meaningless here, and DECIDE too if the store was valall.")
        print("     Comparable only against models trained the same way.",
              flush=True)
    v_ast = None
    if a.use_ast:
        _vs = AF.AUX[a.aux][2] or a.val_store
        v_ast = ("scale" if a.aux in AF.IN_MODEL
                 else load_ast_stores([_vs], AUD.parent / AF.AUX[a.aux][0])[_vs])
    post, dist, _ = clip_posteriors(net, varr, vidx, tune_ids + decide_ids,
                                    a.crop, dev, ast=v_ast)
    grid = dict(__import__("decode").DEFAULT_GRID)
    grid["clip_gate"] = [0.0]
    cfg, tune_score = T.choose_readout(post, refs, tune_ids, dist or None, grid)
    tuned = S.score({c: refs[c] for c in tune_ids},
                    T.decode_all(post, cfg, dist, tune_ids))
    held = S.score({c: refs[c] for c in decide_ids},
                   T.decode_all(post, cfg, dist, decide_ids))
    allsc = S.score({c: refs[c] for c in post},
                    T.decode_all(post, cfg, dist, list(post)))
    print()
    print("=" * 70)
    print(f"TUNE   : F1 {tuned['event_f1']:.4f} Dice {tuned['segment_dice']:.4f}"
          f" combined {tuned['combined']:.4f}")
    print(f"DECIDE : F1 {held['event_f1']:.4f} Dice {held['segment_dice']:.4f}"
          f" combined {held['combined']:.4f}   <-- the honest number")
    print(f"ALL    : F1 {allsc['event_f1']:.4f} Dice "
          f"{allsc['segment_dice']:.4f} combined {allsc['combined']:.4f}"
          f"   <-- compare to the incumbent's 1.1190")
    print(f"readout: {cfg}")
    print("=" * 70)

    out = CKPT / f"{a.tag}.pt"
    torch.save({"state": net.state_dict(), "cfg": {**vars(a), "finetune": True},
                "readout": cfg, "dev_ap": best, "dev_hist": hist,
                "tune": tuned, "decide": held, "all": allsc}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
