"""Joint presence/interval adaptation with a matched optimization control."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import dataset as DS
import event_proposals as EP
from joint_interval import JointIntervalNet, presence_loss
from train_epn import ProposalDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'docs/controller'))
from epn_recipe import decode_sequence


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class JointDataset(ProposalDataset):
    def __getitem__(self, i):
        x, target, weights = super().__getitem__(i)
        e = self.items[self.ids[i]]
        # Same any-part-of-frame occupancy convention as the existing V5 task.
        labels = DS.rasterise(e.get('events', []), e.get('cats', []), len(x))
        return x, target, weights, torch.from_numpy(labels)


def collate(rows):
    return tuple(pad_sequence([row[k] for row in rows], batch_first=True) for k in range(4))+(torch.tensor([len(row[0]) for row in rows]),)


def check_initial_parity(net, blob, data, out):
    original = EP.EventProposalNet(acoustic_dim=512).cuda().eval()
    original.load_state_dict(blob['state'])
    net.eval()
    index = json.loads((data.cache_root/'valtune_audio.json').read_text())
    features = data.features['valtune_audio']
    inspected, max_distance, max_probability, mismatches = 0, 0., 0., 0
    with torch.no_grad():
        for cid, f in sorted(index['clips'].items()):
            e = data.items['valtune_audio:'+cid]
            x = features[f['off']:f['off']+f['n']].astype(np.float32)
            tx, lengths = torch.from_numpy(x)[None].cuda(), torch.tensor([len(x)])
            od, _ = original(tx, lengths)
            nd, logits = net(tx, lengths)
            max_distance = max(max_distance, float((od-nd).abs().max()))
            p = torch.sigmoid(logits[0]).cpu().numpy()
            max_probability = max(max_probability, float(np.abs(p-x[:, :8]).max()))
            d = nd[0].cpu().numpy()
            for kind in ['union', 'categories', 'union_human_nested']:
                for threshold in [.3, .5, .7]:
                    spec = dict(kind=kind, threshold=threshold)
                    a = decode_sequence(x[:, :8], d, e['n']/16000, spec)
                    b = decode_sequence(p, d, e['n']/16000, spec)
                    mismatches += a != b
            inspected += 1
    result = dict(clips=inspected, configurations=9, max_distance_difference=max_distance,
                  max_probability_difference=max_probability, decoded_mismatches=mismatches,
                  scope='Permitted TUNE zero-step parity only; numerical clamping may alter probabilities, no labels fitted.')
    out.write_text(json.dumps(result, indent=2))
    del original
    torch.cuda.empty_cache()
    if max_distance != 0 or mismatches:
        raise RuntimeError('Zero-step decoder or geometry differs; inspect parity before training')
    print('JOINT_INITIAL_PARITY_PASSED '+json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--init-checkpoint', required=True)
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--objective', choices=['joint', 'iou'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--verify-initial-parity', action='store_true')
    args = ap.parse_args()
    if os.name == 'nt':
        ownership = json.loads((ROOT/'docs/controller/COMPUTE_OWNERSHIP.json').read_text())
        if ownership['laptop_reserved_for_track2'] or args.threads > ownership['laptop_cpu_thread_limit']:
            raise RuntimeError('Laptop reservation or CPU limit prohibits training')
    if not torch.cuda.is_available():
        raise RuntimeError('Architecture heads must train on the assigned GPU')
    torch.set_num_threads(args.threads)
    torch.manual_seed(13)
    np.random.seed(13)
    random.seed(13)
    if digest(args.init_checkpoint) != args.expected_sha256:
        raise RuntimeError('Warm-start checkpoint SHA256 mismatch')
    blob = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    if blob['mode'] != 'acoustic' or blob['cfg']['collar_weight'] != 0 or blob.get('quality_enabled'):
        raise RuntimeError('Expected original measured acoustic interval geometry')
    net = JointIntervalNet(acoustic_dim=512).cuda()
    missing, unexpected = net.load_state_dict(blob['state'], strict=False)
    if set(missing) != {'presence_residual.weight', 'presence_residual.bias'} or unexpected:
        raise RuntimeError('Unexpected warm-start state structure')
    for name, value in net.named_parameters():
        value.requires_grad_(args.objective == 'joint' or not name.startswith('presence_residual.'))
    data = JointDataset(Path(args.features), Path(args.audio), ['gold_audio','synth_aug','synth_mask_t','valtune_audio','silver_long'], 13)
    data.cache_root = Path(args.features)
    if data.provenance != blob['manifest']['feature_provenance']:
        raise RuntimeError('Warm-start backbone feature provenance differs')
    for key, ids in [('train_id_hash', data.train_ids), ('dev_id_hash', data.dev_ids)]:
        if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != blob['manifest'][key]:
            raise RuntimeError('Original group split differs')
    # Match the measured original arm for both continuation objectives.
    data.silver_loc_weight = .5
    dev = copy.copy(data)
    dev.ids = data.dev_ids
    if args.limit:
        data.ids, dev.ids = data.train_ids[:args.limit], data.dev_ids[:args.limit]
    train = DataLoader(data, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate)
    validation = DataLoader(dev, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.verify_initial_parity:
        check_initial_parity(net, blob, data, out/'initial_parity.json')
    manifest = dict(args=vars(args), arm=args.objective, training_clips=len(data), dev_clips=len(dev),
                    train_id_hash=hashlib.sha256('\n'.join(data.ids).encode()).hexdigest(),
                    dev_id_hash=hashlib.sha256('\n'.join(dev.ids).encode()).hexdigest(), feature_provenance=data.provenance,
                    source_geometry_checkpoint_sha256=args.expected_sha256,
                    optimization=dict(seed=13, epochs=args.epochs, batch=16, lr=.001, weight_decay=.001, cosine_min_lr=.0001,
                                      presence_coefficient=1 if args.objective=='joint' else 0, silver_loc_weight=.5, collar_weight=0),
                    parameter_count=sum(p.numel() for p in net.parameters()), trainable_parameter_count=sum(p.numel() for p in net.parameters() if p.requires_grad),
                    source={p:digest(ROOT/'src'/p) for p in ['joint_interval.py','train_joint_interval.py','event_proposals.py','train_epn.py']},
                    adaptations='Head-only jointly learned residual category/union presence and positive interval distances from the existing frozen V5 post-fusion state. Zero residual warm start; unweighted per-clip masked BCE plus event-balanced IoU. Matching IoU-only continuation control freezes the zero residual. Original nine-config presence decoder; no quality decoder. Not joint encoder fine-tuning or a RED reproduction.',
                    backend=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key:manifest[key] for key in ['arm','training_clips','dev_clips','parameter_count','trainable_parameter_count','backend']}), flush=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.001, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=.0001)
    best, history, start = float('inf'), [], time.monotonic()
    for epoch in range(1,args.epochs+1):
        net.train()
        totals = np.zeros(3)
        count = 0
        for i, (x, target, weights, labels, lengths) in enumerate(train,1):
            x,target,weights,labels=(a.cuda() for a in [x,target,weights,labels])
            d,z = net(x,lengths)
            loc = EP.proposal_loss(d,target,weights)
            activity = presence_loss(z,labels,lengths)
            loss = loc+activity if args.objective=='joint' else loc
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite joint loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            opt.step()
            totals += np.array([float(loss.detach()),float(loc.detach()),float(activity.detach())])*len(x)
            count += len(x)
            if i % 200 == 0:
                print(f'epoch {epoch} batch {i}/{len(train)} loss={totals[0]/count:.6f} elapsed={time.monotonic()-start:.0f}s',flush=True)
        net.eval()
        val = np.zeros(3)
        with torch.no_grad():
            for x,target,weights,labels,lengths in validation:
                x,target,weights,labels=(a.cuda() for a in [x,target,weights,labels])
                d,z=net(x,lengths)
                loc=EP.proposal_loss(d,target,weights)
                activity=presence_loss(z,labels,lengths)
                loss=loc+activity if args.objective=='joint' else loc
                val+=np.array([float(loss),float(loc),float(activity)])*len(x)
        val/=len(dev)
        if args.objective=='iou' and any(torch.count_nonzero(p).item() for p in net.presence_residual.parameters()):
            raise RuntimeError('Frozen continuation control residual changed')
        history.append(dict(epoch=epoch,train_loss=(totals/count).tolist(),dev_loss=float(val[0]),dev_components=val.tolist(),elapsed=time.monotonic()-start))
        ckpt=dict(state=net.state_dict(),mode='acoustic',acoustic_dim=512,presence_residual_enabled=True,cfg=vars(args),manifest=manifest,history=history,epoch=epoch)
        torch.save(ckpt,out/f'ep{epoch}.pt')
        if val[0]<best:
            best=val[0]
            torch.save(ckpt,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(history[-1]),flush=True)
        scheduler.step()
    print('JOINT_INTERVAL_TRAINING_DONE',flush=True)


if __name__=='__main__':
    main()
