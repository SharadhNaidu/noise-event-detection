"""Train the detection head on cached features, then evaluate against the real metric.

Protocol (see docs/specs and EXP-03):
  * train/dev split inside the training features is SPEAKER-DISJOINT
  * the official validation set is split into TUNE and DECIDE halves, also
    speaker-disjoint. Readout parameters are swept on TUNE only; the reported
    number - the one that decides whether a submission is worth spending - comes
    from DECIDE, which no tuning has touched.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import data as D          # noqa: E402
import dataset as DS      # noqa: E402
import decode as DEC      # noqa: E402
import features as F      # noqa: E402
import model as M         # noqa: E402
import readout as R      # noqa: E402
import scorer as S        # noqa: E402
import splits as SP       # noqa: E402
import tierfilter as TF   # noqa: E402

FEAT = (P.ROOT / "features")
CKPT = (P.ROOT / "ckpt")
WIN = 250          # 5 s at 20 ms


def build_loaders(stores, weights, win, batch, dev_frac=0.1, seed=42, workers=0,
                  tier_filter=True):
    """stores: list of feature-store names to train on (e.g. ['gold'])."""
    arrs, idx = {}, {}
    for name in stores:
        a, meta = F.load_store(FEAT, name)
        arrs[name] = a
        for cid, e in meta["clips"].items():
            idx[f"{name}:{cid}"] = {**e, "_store": name}

    # speaker-disjoint dev split. Parquet-sourced ids are synthetic (sNNNrNNNNN)
    # and carry no speaker, so fall back to grouping by district+language there.
    def key(cid):
        e = idx[cid]
        if "district" in e:
            return f"{e.get('state')}_{e.get('district')}_{e.get('lang')}"
        return SP.speaker_key(cid.split(":", 1)[1])

    groups = {}
    for cid in idx:
        groups.setdefault(key(cid), []).append(cid)
    gk = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(gk)
    n_dev = max(1, int(len(gk) * dev_frac))
    dev_ids = [c for g in gk[:n_dev] for c in groups[g]]
    tr_ids = [c for g in gk[n_dev:] for c in groups[g]]

    class Multi:
        """Presents several stores as one array-like via (store, offset)."""
        def __init__(self, arrs, idx):
            self.arrs, self.idx = arrs, idx
        def __getitem__(self, s):
            raise RuntimeError("index through WindowDataset only")

    # simplest correct approach: one WindowDataset per store, concatenated
    from torch.utils.data import ConcatDataset
    tr_sets, dev_sets = [], []
    for name in stores:
        a, meta = F.load_store(FEAT, name)
        ids_tr = [c.split(":", 1)[1] for c in tr_ids if c.startswith(name + ":")]
        ids_dev = [c.split(":", 1)[1] for c in dev_ids if c.startswith(name + ":")]
        cw = None
        if tier_filter:
            cw = {}
            for cid, e in meta["clips"].items():
                d = TF.clip_diagnosis(e.get("events", []),
                                      e.get("dur_measured") or e.get("dur", 0))
                cw[cid] = TF.frame_weight(d, e.get("tier", name), weights)
            kept = sum(1 for v in cw.values() if v > 0)
            print(f"    {name}: {kept:,}/{len(cw):,} clips usable for the frame "
                  f"loss ({100*kept/max(len(cw),1):.1f}%)")
        if ids_tr:
            tr_sets.append(DS.WindowDataset(a, meta["clips"], ids_tr, win=win,
                                            weights=weights, train=True, seed=seed,
                                            clip_weights=cw))
        if ids_dev:
            dev_sets.append(DS.WindowDataset(a, meta["clips"], ids_dev, win=win,
                                             weights=weights, train=False,
                                             clip_weights=cw))
    tr = ConcatDataset(tr_sets)
    dv = ConcatDataset(dev_sets)
    print(f"  train windows {len(tr):,} | dev windows {len(dv):,} "
          f"({len(tr_ids):,} / {len(dev_ids):,} clips, speaker-disjoint)")
    return (DataLoader(tr, batch_size=batch, shuffle=True, collate_fn=DS.collate,
                       num_workers=workers, drop_last=True),
            DataLoader(dv, batch_size=batch, shuffle=False, collate_fn=DS.collate,
                       num_workers=workers))


@torch.no_grad()
def posteriors_for(net, store_name, clip_ids=None, win=WIN, batch=64, device="cuda"):
    """Full-clip posteriors by windowing with overlap-average. Returns
    {clip_id: (T,) any-noise probability} plus {clip_id: clip prob}."""
    arr, meta = F.load_store(FEAT, store_name)
    clips = meta["clips"]
    ids = clip_ids if clip_ids is not None else list(clips)
    net.eval()
    out, clipp, dist = {}, {}, {}
    STRIDE = win // 2
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        for cid in chunk:
            e = clips[cid]
            n = e["n"]
            x = np.asarray(arr[e["off"]: e["off"] + n], dtype=np.float32)
            starts = [0] if n <= win else list(range(0, n - win + 1, STRIDE))
            if n > win and starts[-1] + win < n:
                starts.append(n - win)
            acc = np.zeros(n, dtype=np.float32)
            cnt = np.zeros(n, dtype=np.float32)
            xb, sl = [], []
            for s in starts:
                w = np.zeros((win, x.shape[1]), dtype=np.float32)
                take = min(win, n - s)
                w[:take] = x[s:s + take]
                xb.append(w)
                sl.append((s, take))
            xb = torch.from_numpy(np.stack(xb)).to(device)
            mask = torch.zeros(xb.shape[0], win, device=device)
            for j, (s, take) in enumerate(sl):
                mask[j, :take] = 1
            res = net(xb, mask)
            fr, cp = res[0], res[1]
            dd = res[2].float().cpu().numpy() if len(res) > 2 else None
            fr = torch.sigmoid(fr[:, :, 0].float()).cpu().numpy()
            dacc = np.zeros((n, 2), dtype=np.float32) if dd is not None else None
            for j, (s, take) in enumerate(sl):
                acc[s:s + take] += fr[j, :take]
                cnt[s:s + take] += 1
                if dd is not None:
                    dacc[s:s + take] += dd[j, :take]
            out[cid] = acc / np.maximum(cnt, 1)
            if dd is not None:
                dist[cid] = dacc / np.maximum(cnt, 1)[:, None]
            clipp[cid] = float(torch.sigmoid(cp[:, 0]).max().item())
    return (out, clipp, dist) if dist else (out, clipp)


def choose_readout(post, refs, tune_ids, dist=None, grid=None):
    """Sweep EVERY readout family on TUNE and return the best configuration.

    This is not a nicety. The same `sunny_gs_filtered` posteriors scored 0.9130
    with train.py's swept global threshold and 0.8371 with the fixed adaptive
    parameters that had been fitted to a different checkpoint. Readout settings
    do NOT transfer between models, so each checkpoint must carry its own or a
    better model ships ~0.07 below its true score.
    """
    tr = {c: refs[c] for c in tune_ids}
    best = (-1.0, None)

    # family 1: global threshold + median filter (decode.sweep)
    cfg, top = DEC.sweep({c: post[c] for c in tune_ids}, tr, S, grid=grid,
                         verbose=False)
    best = max(best, (top[0][0]["combined"], {"kind": "global", **cfg}),
               key=lambda z: z[0])

    # family 2: per-clip adaptive threshold
    for rel in (0.35, 0.45, 0.5, 0.6):
        for floor in (0.10, 0.15, 0.25, 0.35):
            for md in (0.08, 0.10, 0.14):
                pr = {c: R.adaptive_threshold(post[c], rel=rel, floor=floor,
                                              min_dur=md, merge_gap=0.06)
                      for c in tune_ids}
                sc = S.score(tr, pr)["combined"]
                if sc > best[0]:
                    best = (sc, {"kind": "adaptive", "rel": rel, "floor": floor,
                                 "min_dur": md, "merge_gap": 0.06})

    # family 3: asymmetric thresholds. EXP-28: the posterior's fall is 36%
    # slower than its rise on natural clips, so one threshold cannot serve both
    # edges well.
    for ton in (0.35, 0.5, 0.6, 0.7):
        for toff in (0.35, 0.5, 0.6, 0.7, 0.8):
            for md in (0.08, 0.12):
                pr = {c: R.asymmetric(post[c], thr_on=ton, thr_off=toff,
                                      min_dur=md, merge_gap=0.06)
                      for c in tune_ids}
                sc = S.score(tr, pr)["combined"]
                if sc > best[0]:
                    best = (sc, {"kind": "asym", "thr_on": ton, "thr_off": toff,
                                 "min_dur": md, "merge_gap": 0.06})

    # family 4: whichever of the above, then boundary refinement
    if dist:
        base = dict(best[1])
        for radius in (0.10, 0.15, 0.20, 0.30):
            pr = {c: R.refine(apply_readout(post[c], base), dist[c],
                              radius=radius) for c in tune_ids}
            sc = S.score(tr, pr)["combined"]
            if sc > best[0]:
                best = (sc, {**base, "refine": radius})
    return best[1], best[0]


def apply_readout(p, cfg):
    if cfg.get("kind") == "asym":
        return R.asymmetric(p, thr_on=cfg["thr_on"], thr_off=cfg["thr_off"],
                            min_dur=cfg["min_dur"], merge_gap=cfg["merge_gap"])
    if cfg.get("kind") == "adaptive":
        return R.adaptive_threshold(p, rel=cfg["rel"], floor=cfg["floor"],
                                    min_dur=cfg["min_dur"],
                                    merge_gap=cfg["merge_gap"])
    return DEC.prob_to_events(p, thr=cfg["thr"], med=cfg["med"],
                              min_dur=cfg["min_dur"],
                              merge_gap=cfg["merge_gap"])


def decode_all(post, cfg, dist=None, ids=None):
    ids = ids if ids is not None else list(post)
    out = {}
    for c in ids:
        ev = apply_readout(post[c], cfg)
        # split BEFORE refine: splitting introduces new boundaries that the
        # refine step should then be allowed to place properly.
        if cfg.get("split") and dist and c in dist:
            ev = R.split_on_crossings(ev, dist[c])
        if cfg.get("refine") and dist:
            ev = R.refine(ev, dist[c], radius=cfg["refine"])
        if cfg.get("dilate"):
            ev = R.bias_correct(ev, -cfg["dilate"] / 2, cfg["dilate"] / 2)
        out[c] = ev
    return out


def evaluate(net, tune_ids, decide_ids, refs, grid=None, device="cuda",
             val_store="val"):
    """Sweep every readout on TUNE, report the untouched DECIDE score."""
    res = posteriors_for(net, val_store, tune_ids + decide_ids, device=device)
    post, dist = res[0], (res[2] if len(res) > 2 else None)
    cfg, tune_score = choose_readout(post, refs, tune_ids, dist, grid)

    tuned = S.score({c: refs[c] for c in tune_ids},
                    decode_all(post, cfg, dist, tune_ids))
    held = S.score({c: refs[c] for c in decide_ids},
                   decode_all(post, cfg, dist, decide_ids))
    return cfg, tuned, held, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", default=["gold"])
    # The validation store MUST come from the same encoder layer as the training
    # stores. A model trained on layer 1 and evaluated on layer-10 features
    # produces well-formed, validator-passing garbage - the same silent-failure
    # class as the clean-twin clip_id collision. F.assert_same_layer enforces it.
    ap.add_argument("--val-store", default="val")
    ap.add_argument("--weights", default='{"gold":1.0,"silver":0.4,"bronze":0.0}')
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--rnn-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--input-dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip-loss-w", type=float, default=0.3)
    ap.add_argument("--boundary-head", action="store_true",
                    help="add the onset/offset distance-regression head")
    ap.add_argument("--boundary-w", type=float, default=1.0)
    ap.add_argument("--tag", default="sunny")
    ap.add_argument("--no-tier-filter", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    CKPT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    weights = json.loads(a.weights)

    layer = F.assert_same_layer(FEAT, list(a.stores) + [a.val_store])
    print(f"=== train [{a.tag}] stores={a.stores} weights={weights} "
          f"layer={layer} ===")
    tr, dv = build_loaders(a.stores, weights, WIN, a.batch, seed=a.seed,
                           tier_filter=not a.no_tier_filter)

    net = M.SEDHead(rnn_dim=a.rnn_dim, dropout=a.dropout,
                    input_dropout=a.input_dropout,
                    boundary_head=a.boundary_head).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  head params {n_par/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(tr), pct_start=0.2)

    refs, meta = D.load_validation()
    tune_ids, decide_ids = SP.split_by_speaker(list(refs), frac=0.5, seed=42)
    print(f"  validation: tune {len(tune_ids)} | decide {len(decide_ids)}")

    best = -1.0
    best_state = None
    hist = []
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, m, c, w, bd, bv, _, _ in tr:
            x, y, m, c, w, bd, bv = (
                t.to(dev, non_blocking=True) for t in (x, y, m, c, w, bd, bv))
            with torch.autocast(dev, dtype=torch.bfloat16):
                res = net(x, m)
                fr, cp = res[0], res[1]
                loss = M.masked_bce(fr, y, m, pos_weight=a.pos_weight,
                                    sample_weight=w)
                loss = loss + a.clip_loss_w * M.clip_bce(cp, c, sample_weight=w)
                if len(res) > 2:
                    loss = loss + a.boundary_w * M.boundary_loss(
                        res[2], bd, bv, m, sample_weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            if not np.isfinite(tot):
                raise RuntimeError(f"loss became non-finite at epoch {ep}")
            nb += 1

        # Dev check: frame-level average precision on the held-out
        # SPEAKER-DISJOINT split. Cheap (no decode, no sweep) but monotonically
        # related to what we care about, so it is a sound early-stopping signal.
        # EXP-10 showed the final epoch is far from the best one - 2 epochs beat
        # 30 - so selecting on this is the single most important fix.
        net.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, m, c, w, bd, bv, _, _ in dv:
                x, m = x.to(dev), m.to(dev)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    fr = net(x, m)[0]
                sel = m.bool().cpu().numpy().ravel()
                ps.append(torch.sigmoid(fr[:, :, 0].float()).cpu().numpy().ravel()[sel])
                ys.append(y[:, :, 0].numpy().ravel()[sel])
        yv, pv = np.concatenate(ys), np.concatenate(ps)
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(yv, pv)) if yv.max() > 0 else 0.0
        hist.append(ap)
        star = ""
        if ap > best:
            best = ap
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            star = "  <- best"
        print(f"  ep {ep:>2}/{a.epochs}  loss {tot/max(nb,1):.4f}  "
              f"dev_AP {ap:.4f}  {time.time()-t0:.0f}s{star}", flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\n  restored best checkpoint: dev_AP {best:.4f} "
              f"(epoch {int(np.argmax(hist))+1} of {a.epochs})")

    print("\n=== evaluating on the official validation set ===")
    grid = dict(DEC.DEFAULT_GRID)
    grid["clip_gate"] = [0.0]          # EXP-03: only 2/2501 clips are empty
    cfg, tuned, held, post = evaluate(net, tune_ids, decide_ids, refs, grid, dev,
                                      val_store=a.val_store)

    print()
    print("=" * 70)
    print(f"TUNE half   : F1 {tuned['event_f1']:.4f}  Dice {tuned['segment_dice']:.4f}"
          f"  combined {tuned['combined']:.4f}")
    print(f"DECIDE half : F1 {held['event_f1']:.4f}  Dice {held['segment_dice']:.4f}"
          f"  combined {held['combined']:.4f}   <-- the honest number")
    print(f"readout     : {cfg}")
    print(f"baseline to beat: 1.1835   rank 10: 1.5576   rank 1: 1.6100")
    print("=" * 70)

    out = CKPT / f"{a.tag}.pt"
    torch.save({"state": net.state_dict(), "cfg": {**vars(a), "layer": layer},
                "readout": cfg,
                "dev_ap": best, "dev_hist": hist,
                "tune": tuned, "decide": held}, out)
    np.savez_compressed(CKPT / f"{a.tag}_valpost.npz",
                        **{c: p.astype(np.float16) for c, p in post.items()})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
