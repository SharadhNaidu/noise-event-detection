"""Train only event-quality confidence, keeping the measured geometry frozen."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import event_proposals as EP
from proposal_quality import quality_targets, quality_loss
from train_epn import ProposalDataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QualityDataset(ProposalDataset):
    def __getitem__(self, i):
        e = self.items[self.ids[i]]
        f = e['feature']
        x = self.features[e['_store']][f['off']:f['off']+f['n']].astype(np.float32)
        target, weights, active, _ = EP.interval_targets(e.get('events', []), e.get('cats', []), f['n'], e['n']/16000)
        return tuple(torch.from_numpy(v) for v in [x, target, weights, active])


def collate(rows):
    return tuple(pad_sequence([r[i] for r in rows], batch_first=True) for i in range(4))+(torch.tensor([len(r[0]) for r in rows]),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--evaluation', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(13)
    np.random.seed(13)
    random.seed(13)
    measured = json.loads(Path(args.evaluation).read_text())
    if measured['mode'] != 'acoustic' or digest(measured['checkpoint']) != measured['checkpoint_sha256']:
        raise RuntimeError('Expected the exact measured acoustic geometry checkpoint')
    blob = torch.load(measured['checkpoint'], map_location='cpu', weights_only=False)
    if blob['cfg']['collar_weight'] != 0:
        raise RuntimeError('Use the original acoustic arm selected by TUNE, not the rejected collar arm')
    net = EP.EventProposalNet(acoustic_dim=512, quality=True).cuda()
    missing, unexpected = net.load_state_dict(blob['state'], strict=False)
    if set(missing) != {'quality.weight', 'quality.bias'} or unexpected:
        raise RuntimeError('Unexpected source checkpoint structure')
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith('quality.'))
    data = QualityDataset(Path(args.features), Path(args.audio), ['gold_audio', 'synth_aug', 'synth_mask_t', 'valtune_audio', 'silver_long'], 13)
    if data.provenance != blob['manifest']['feature_provenance']:
        raise RuntimeError('Frozen feature provenance changed')
    train_hash = hashlib.sha256('\n'.join(data.train_ids).encode()).hexdigest()
    dev_hash = hashlib.sha256('\n'.join(data.dev_ids).encode()).hexdigest()
    if train_hash != blob['manifest']['train_id_hash'] or dev_hash != blob['manifest']['dev_id_hash']:
        raise RuntimeError('Source train/development groups changed')
    # Maintain original split, but never learn precise quality from Silver timestamps.
    data.ids = [c for c in data.train_ids if data.items[c]['_store'] != 'silver_long']
    dev = copy.copy(data)
    dev.ids = [c for c in data.dev_ids if data.items[c]['_store'] != 'silver_long']
    if args.limit:
        data.ids, dev.ids = data.ids[:args.limit], dev.ids[:args.limit]
    train = DataLoader(data, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate)
    validation = DataLoader(dev, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    opt = torch.optim.AdamW(net.quality.parameters(), lr=.001, weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=.0001)
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    manifest = dict(source_geometry_checkpoint_sha256=measured['checkpoint_sha256'], source_evaluation_sha256=digest(args.evaluation),
                    optimization=dict(seed=13, epochs=args.epochs, batch=16, lr=.001, weight_decay=.001, cosine_min_lr=.0001, net_mode='eval', positive_background_mass='equal'),
                    feature_provenance=data.provenance, training_clips=len(data), dev_clips=len(dev),
                    full_original_train_id_hash=train_hash, full_original_dev_id_hash=dev_hash,
                    train_id_hash=hashlib.sha256('\n'.join(data.ids).encode()).hexdigest(), dev_id_hash=hashlib.sha256('\n'.join(dev.ids).encode()).hexdigest(),
                    parameter_count=sum(p.numel() for p in net.parameters()), trainable_parameter_count=sum(p.numel() for p in net.parameters() if p.requires_grad),
                    source={p: digest(Path(__file__).parent/p) for p in ['event_proposals.py', 'train_epn_quality.py', 'proposal_quality.py']},
                    adaptations='Only quality Linear(128,8) learns. Frozen EPN geometry/GRU/acoustic projection and frozen V5. Binary target: both category/union interval edges inside official collars, real background zero; active censored/ambiguous and padding masked. Equal event-balanced active/background mass. Silver precise confidence targets excluded. Direct learned score used identically for ranking and acceptance; no presence/quality multiplication or region-score averaging.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    best, history, started = float('inf'), [], time.monotonic()
    for epoch in range(1, args.epochs+1):
        net.eval()  # Frozen geometry must retain the exact deterministic inference path.
        for x, target, weights, active, lengths in train:
            x, target, weights, active, lengths = (v.cuda() for v in [x, target, weights, active, lengths])
            distances, logits = net(x, lengths)
            labels, pw, nw = quality_targets(distances, target, weights, active, lengths)
            loss = quality_loss(logits, labels, pw, nw)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite quality loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total = 0.
        with torch.no_grad():
            for x, target, weights, active, lengths in validation:
                x, target, weights, active, lengths = (v.cuda() for v in [x, target, weights, active, lengths])
                distances, logits = net(x, lengths)
                labels, pw, nw = quality_targets(distances, target, weights, active, lengths)
                total += float(quality_loss(logits, labels, pw, nw))*len(x)
        score = total/len(dev)
        for name, value in blob['state'].items():
            if not torch.equal(value, net.state_dict()[name].cpu()):
                raise RuntimeError('Frozen proposal geometry changed')
        history.append(dict(epoch=epoch, dev_loss=score, elapsed=time.monotonic()-started))
        checkpoint = dict(state=net.state_dict(), mode='acoustic', acoustic_dim=512, quality_enabled=True,
                          cfg=vars(args), manifest=manifest, history=history, epoch=epoch)
        torch.save(checkpoint, out/f'ep{epoch}.pt')
        if score < best:
            best = score
            torch.save(checkpoint, out/'best.pt')
        print(json.dumps(history[-1]), flush=True)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        sched.step()
    print('EPN_QUALITY_TRAINING_DONE', flush=True)


if __name__ == '__main__':
    main()
