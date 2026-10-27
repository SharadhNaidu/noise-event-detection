import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import event_proposals as EP
import splits as SP


class ProposalDataset(Dataset):
    def __init__(self, cache_root, audio_root, stores, seed=13):
        self.features = {}
        self.items = {}
        self.provenance = {}
        self.silver_loc_weight = .5
        _, meta = D.load_validation()
        _, decide = SP.split_by_speaker(list(meta))
        forbidden = {SP.speaker_key(c) for c in decide}
        for store in stores:
            if store in ['val_audio', 'valall_audio', 'test_audio', 'bronze_audio']:
                raise RuntimeError(f'Forbidden training store: {store}')
            source = cache_root/f'{store}.json'
            index = json.loads(source.read_text())
            if not index['complete'] or not index['provenance']['may_train']:
                raise RuntimeError('Unfinished or nontraining feature cache')
            audio_path = audio_root/f'{store}.json'
            if hashlib.sha256(audio_path.read_bytes()).hexdigest() != index['provenance']['audio_index_sha256']:
                raise RuntimeError('Audio metadata changed after export')
            audio = json.loads(audio_path.read_text())['clips']
            dim = index['provenance']['dimension']
            self.features[store] = np.memmap(cache_root/f'{store}.f16', dtype=np.float16, mode='r', shape=(index['total_frames'], dim))
            self.provenance[store] = index['provenance']
            for cid, clip in index['clips'].items():
                entry = audio[cid]
                host = entry.get('host') or cid
                if SP.speaker_key(host) in forbidden:
                    raise RuntimeError('DECIDE source group in head training data')
                key = f'{store}:{cid}'
                self.items[key] = dict(entry, feature=clip, _store=store, _cid=cid)
        cfgs = [v['backbone_cfg'] for v in self.provenance.values()]
        if not all(c == cfgs[0] for c in cfgs) or not all(v['checkpoint_sha256'] == next(iter(self.provenance.values()))['checkpoint_sha256'] for v in self.provenance.values()):
            raise RuntimeError('Mixed acoustic checkpoints')
        groups = {}
        for cid, e in self.items.items():
            base = e.get('host') or e['_cid']
            group = f'{e.get("state")}_{e.get("district")}_{e.get("lang")}' if 'district' in e else SP.speaker_key(base)
            groups.setdefault(group, []).append(cid)
        keys = sorted(groups)
        np.random.default_rng(seed).shuffle(keys)
        ndev = max(1, int(.1*len(keys)))
        self.train_ids = [c for k in keys[ndev:] for c in groups[k]]
        self.dev_ids = [c for k in keys[:ndev] for c in groups[k]]
        self.ids = self.train_ids
        self.target_stats = Counter()
        for e in self.items.values():
            _, _, _, stats = EP.interval_targets(e.get('events', []), e.get('cats', []), e['feature']['n'], e['n']/16000)
            self.target_stats.update(stats)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        e = self.items[self.ids[i]]
        f = e['feature']
        x = self.features[e['_store']][f['off']:f['off']+f['n']].astype(np.float32)
        target, weights, _, _ = EP.interval_targets(e.get('events', []), e.get('cats', []), f['n'], e['n']/16000)
        if e['_store'] == 'silver_long':
            weights *= self.silver_loc_weight
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(weights)


def collate(rows):
    return tuple(pad_sequence([r[i] for r in rows], batch_first=True) for i in range(3))+(torch.tensor([len(r[0]) for r in rows]),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--audio', default=str(P.ROOT / "audio"))
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['probability', 'acoustic'], required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--collar-weight', type=float, default=0.)
    ap.add_argument('--silver-loc-weight', type=float, default=.5)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    if not 0 <= args.silver_loc_weight <= 1:
        raise ValueError('Silver localization weight must be between zero and one')
    if os.name == 'nt':
        ownership = json.loads((Path(__file__).resolve().parents[1]/'docs/controller/COMPUTE_OWNERSHIP.json').read_text())
        if ownership['laptop_reserved_for_track2'] or args.threads > ownership['laptop_cpu_thread_limit']:
            raise RuntimeError('Laptop reservation or CPU limit prohibits training')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    stores = ['gold_audio', 'synth_aug', 'synth_mask_t', 'valtune_audio', 'silver_long']
    data = ProposalDataset(Path(args.features), Path(args.audio), stores, args.seed)
    data.silver_loc_weight = args.silver_loc_weight
    validation = ProposalDataset.__new__(ProposalDataset)
    validation.__dict__ = dict(data.__dict__)
    validation.ids = data.dev_ids[:args.limit] if args.limit else data.dev_ids
    if args.limit:
        data.ids = data.train_ids[:args.limit]
    train = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate)
    dev = DataLoader(validation, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)
    net = EP.EventProposalNet(acoustic_dim=512 if args.mode == 'acoustic' else 0).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr*.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(args=vars(args), training_clips=len(data), dev_clips=len(validation), target_stats=dict(data.target_stats),
                    train_id_hash=hashlib.sha256('\n'.join(data.ids).encode()).hexdigest(), dev_id_hash=hashlib.sha256('\n'.join(validation.ids).encode()).hexdigest(),
                    feature_provenance=data.provenance, source={p: hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest() for p in ['train_epn.py', 'event_proposals.py']},
                    parameter_count=sum(p.numel() for p in net.parameters()),
                    adaptations='Full-clip sequence after frozen V5 crop stitching; exact per-event mean IoU; union plus category lanes; ambiguous ownership masked. Acoustic arm additionally reads the existing 512-dimensional shared acoustic state. No quality head, new data or Dice term.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ['training_clips', 'dev_clips', 'target_stats', 'parameter_count']}), flush=True)
    best = float('inf')
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        net.train()
        total = weight_sum = 0.
        for i, (x, target, weights, lengths) in enumerate(train, 1):
            x, target, weights = x.cuda(), target.cuda(), weights.cuda()
            opt.zero_grad(set_to_none=True)
            pred, _ = net(x, lengths)
            loss = EP.proposal_loss(pred, target, weights, args.collar_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite interval loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
            count = float(weights.sum())
            total += float(loss.detach())*count
            weight_sum += count
            if i % 100 == 0:
                print(f'epoch {epoch} batch {i}/{len(train)} loss={total/max(weight_sum,1):.5f} elapsed={time.monotonic()-started:.0f}s', flush=True)
        net.eval()
        dev_total = dev_weight = 0.
        with torch.no_grad():
            for x, target, weights, lengths in dev:
                x, target, weights = x.cuda(), target.cuda(), weights.cuda()
                pred, _ = net(x, lengths)
                loss = EP.proposal_loss(pred, target, weights, args.collar_weight)
                count = float(weights.sum())
                dev_total += float(loss)*count
                dev_weight += count
        score = dev_total/max(dev_weight, 1.)
        history.append(dict(epoch=epoch, train_loss=total/max(weight_sum, 1.), dev_loss=score, elapsed=time.monotonic()-started))
        checkpoint = dict(state=net.state_dict(), mode=args.mode, acoustic_dim=net.acoustic_dim, cfg=vars(args), manifest=manifest, history=history, epoch=epoch)
        torch.save(checkpoint, out/f'ep{epoch}.pt')
        if score < best:
            best = score
            torch.save(checkpoint, out/'best.pt')
        (out/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
        sched.step()
    print('EPN_TRAINING_DONE', flush=True)


if __name__ == '__main__':
    main()
