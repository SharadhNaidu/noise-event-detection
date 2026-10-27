"""Controlled pre-fusion information comparison; both arms use the joint loss."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from aligned_interval import AlignedIntervalNet
import event_proposals as EP
from joint_interval import presence_loss
from train_joint_interval import ROOT, JointDataset, collate, decode_sequence, digest

EXPECTED = '28a2a23502e8df1b6340dd41c84c1e608912b78ba746131beb0f9fd14fb7ac4e'
STORES = ['gold_audio', 'synth_aug', 'synth_mask_t', 'valtune_audio', 'silver_long']


class AlignedDataset(JointDataset):
    """Join the frozen 768-column skip by the exact registered coarse layout."""
    def __init__(self, cache_root, skip_root, audio_root, stores=STORES, seed=13):
        super().__init__(cache_root, audio_root, stores, seed)
        self.cache_root, self.skip_root = Path(cache_root), Path(skip_root)
        self.skip_features, self.skip_provenance, self.skip_integrity = {}, {}, {}
        for store in stores:
            coarse_path, skip_path = self.cache_root/f'{store}.json', self.skip_root/f'{store}.json'
            coarse, fine = json.loads(coarse_path.read_text()), json.loads(skip_path.read_text())
            cp, fp = coarse['provenance'], fine['provenance']
            if not fine['complete'] or fp['dimension'] != 768 or not fp['may_train']:
                raise RuntimeError(f'Incomplete or nontraining pre-fusion cache: {store}')
            if cp['dimension'] != 536 or fp['frame_seconds'] != .02 or cp['frame_seconds'] != .02:
                raise RuntimeError(f'Incorrect stream dimensions or timestamp grid: {store}')
            if fp['checkpoint_sha256'] != cp['checkpoint_sha256'] or fp['backbone_cfg'] != cp['backbone_cfg']:
                raise RuntimeError(f'Pre-fusion and coarse backbone differ: {store}')
            if fp['audio_index_sha256'] != cp['audio_index_sha256'] or fp['audio_index_sha256'] != digest(Path(audio_root)/f'{store}.json'):
                raise RuntimeError(f'Pre-fusion audio metadata differs: {store}')
            if fp.get('reference_index_sha256') != digest(coarse_path) or 'export_localization_skip.py' not in fp.get('source', {}):
                raise RuntimeError(f'Unregistered or legacy skip export: {store}')
            if self.skip_provenance and fp['source'] != next(iter(self.skip_provenance.values()))['source']:
                raise RuntimeError(f'Mixed pre-fusion exporter sources: {store}')
            if fine['clips'] != coarse['clips'] or fine['total_frames'] != coarse['total_frames']:
                raise RuntimeError(f'Pre-fusion clip IDs, offsets or frame counts differ: {store}')
            if (self.cache_root/f'{store}.f16').stat().st_size != coarse['total_frames']*536*2:
                raise RuntimeError(f'Coarse payload size mismatch: {store}')
            payload = self.skip_root/f'{store}.f16'
            if payload.stat().st_size != fine['total_frames']*768*2:
                raise RuntimeError(f'Pre-fusion payload size mismatch: {store}')
            if not fine.get('payload_sha256') or digest(payload) != fine['payload_sha256']:
                raise RuntimeError(f'Pre-fusion payload SHA256 mismatch: {store}')
            self.skip_features[store] = np.memmap(payload, dtype=np.float16, mode='r', shape=(fine['total_frames'], 768))
            self.skip_provenance[store] = fp
            self.skip_integrity[store] = dict(payload_sha256=fine['payload_sha256'], index_sha256=digest(skip_path))
            print(f'ALIGNED_CACHE_VERIFIED {store}', flush=True)

    def __getitem__(self, i):
        x, target, weights, labels = super().__getitem__(i)
        entry = self.items[self.ids[i]]
        f = entry['feature']
        fine = torch.from_numpy(self.skip_features[entry['_store']][f['off']:f['off']+f['n']].astype(np.float32))
        if fine.shape != (len(x), 768) or not torch.isfinite(fine).all():
            raise RuntimeError('Nonfinite or incorrectly joined localization skip')
        return torch.cat([x, fine], dim=-1), target, weights, labels


def check_initial_parity(net, blob, data, out):
    """Report widened-GRU numerical differences; block changed detections."""
    original = EP.EventProposalNet(acoustic_dim=512).cuda().eval()
    original.load_state_dict(blob['state'])
    net.eval()
    index = json.loads((data.cache_root/'valtune_audio.json').read_text())
    coarse, fine = data.features['valtune_audio'], data.skip_features['valtune_audio']
    maximum_distance = maximum_probability = maximum_edge = 0.
    raw_changes = serialized_changes = detection_changes = inspected = 0
    with torch.no_grad():
        for cid, f in sorted(index['clips'].items()):
            entry = data.items['valtune_audio:'+cid]
            a, b = f['off'], f['off']+f['n']
            x = coarse[a:b].astype(np.float32)
            joined = np.concatenate([x, fine[a:b].astype(np.float32)], axis=-1)
            if not np.isfinite(joined).all():
                raise RuntimeError('Nonfinite features during initial parity')
            lengths = torch.tensor([len(x)])
            od, _ = original(torch.from_numpy(x)[None].cuda(), lengths)
            nd, logits = net(torch.from_numpy(joined)[None].cuda(), lengths)
            if not all(torch.isfinite(value).all() for value in [od, nd, logits]):
                raise RuntimeError('Nonfinite outputs during initial parity')
            maximum_distance = max(maximum_distance, float((od-nd).abs().max()))
            p, old_d, new_d = torch.sigmoid(logits[0]).cpu().numpy(), od[0].cpu().numpy(), nd[0].cpu().numpy()
            maximum_probability = max(maximum_probability, float(np.abs(p-x[:, :8]).max()))
            for kind in ['union', 'categories', 'union_human_nested']:
                for threshold in [.3, .5, .7]:
                    spec = dict(kind=kind, threshold=threshold)
                    original_events = decode_sequence(x[:, :8], old_d, entry['n']/16000, spec)
                    aligned_events = decode_sequence(p, new_d, entry['n']/16000, spec)
                    raw_changes += original_events != aligned_events
                    serialized_changes += [tuple(round(v, 3) for v in e) for e in original_events] != [tuple(round(v, 3) for v in e) for e in aligned_events]
                    if len(original_events) != len(aligned_events):
                        detection_changes += 1
                    elif original_events:
                        edge_error = float(np.abs(np.asarray(original_events)-np.asarray(aligned_events)).max())
                        maximum_edge = max(maximum_edge, edge_error)
                        detection_changes += edge_error > 1e-5
            inspected += 1
    result = dict(clips=inspected, configurations=9, max_distance_difference=maximum_distance,
                  max_probability_difference=maximum_probability, max_endpoint_difference=maximum_edge,
                  raw_decoder_changes=raw_changes, serialized_decoder_changes=serialized_changes,
                  detection_changes_beyond_tolerance=detection_changes, tolerance_seconds=1e-5,
                  scope='TUNE inference only, no label fitting; widened GRU and probability clamping can change numerical reductions. Serialized differences are reported, not hidden as exact baseline parity.')
    out.write_text(json.dumps(result, indent=2))
    del original
    torch.cuda.empty_cache()
    if maximum_distance > 1e-5 or maximum_probability > 2e-6 or detection_changes:
        raise RuntimeError('Initial detection/geometry differs beyond numerical tolerance; inspect before training')
    print('ALIGNED_INITIAL_PARITY_PASSED '+json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--skip-features', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--init-checkpoint', required=True)
    ap.add_argument('--expected-sha256', default=EXPECTED)
    ap.add_argument('--skip-mode', choices=['real', 'zero'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.threads <= 4 or args.epochs < 1 or (args.limit is not None and args.limit < 1):
        raise RuntimeError('Positive budget and at most four CPU threads are required')
    if os.name == 'nt':
        ownership = json.loads((ROOT/'docs/controller/COMPUTE_OWNERSHIP.json').read_text())
        if ownership['laptop_reserved_for_track2'] or args.threads > ownership['laptop_cpu_thread_limit']:
            raise RuntimeError('Laptop reservation or CPU limit prohibits training')
    if not torch.cuda.is_available():
        raise RuntimeError('Aligned architecture heads require the assigned GPU')
    if args.expected_sha256 != EXPECTED or digest(args.init_checkpoint) != EXPECTED:
        raise RuntimeError('Use the exact original measured acoustic checkpoint')
    torch.set_num_threads(args.threads)
    torch.manual_seed(13)
    np.random.seed(13)
    random.seed(13)
    blob = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    if blob['mode'] != 'acoustic' or blob['cfg']['collar_weight'] != 0 or blob.get('quality_enabled') or blob.get('presence_residual_enabled'):
        raise RuntimeError('Expected original measured acoustic interval geometry')
    net = AlignedIntervalNet(acoustic_dim=512, skip_mode=args.skip_mode).cuda()
    warm_start = net.warm_start(blob['state'])
    data = AlignedDataset(Path(args.features), Path(args.skip_features), Path(args.audio))
    if data.provenance != blob['manifest']['feature_provenance']:
        raise RuntimeError('Warm-start coarse feature provenance differs')
    for key, ids in [('train_id_hash', data.train_ids), ('dev_id_hash', data.dev_ids)]:
        if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != blob['manifest'][key]:
            raise RuntimeError('Original group split differs')
    data.silver_loc_weight = .5
    dev = copy.copy(data)
    dev.ids = data.dev_ids
    if args.limit:
        data.ids, dev.ids = data.train_ids[:args.limit], data.dev_ids[:args.limit]
    train = DataLoader(data, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate)
    validation = DataLoader(dev, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    check_initial_parity(net, blob, data, out/'initial_parity.json')
    manifest = dict(args=vars(args), arm='aligned_'+args.skip_mode, training_clips=len(data), dev_clips=len(dev),
                    train_id_hash=hashlib.sha256('\n'.join(data.ids).encode()).hexdigest(), dev_id_hash=hashlib.sha256('\n'.join(dev.ids).encode()).hexdigest(),
                    feature_provenance=data.provenance, skip_feature_provenance=data.skip_provenance, skip_cache_integrity=data.skip_integrity,
                    source_geometry_checkpoint_sha256=EXPECTED, alignment_skip_enabled=True, presence_residual_enabled=True,
                    skip_mode=args.skip_mode, fine_dimension=768, warm_start=warm_start,
                    optimization=dict(seed=13, epochs=args.epochs, batch=16, lr=.001, weight_decay=.001, cosine_min_lr=.0001,
                                      presence_coefficient=1, silver_loc_weight=.5, collar_weight=0),
                    parameter_count=sum(p.numel() for p in net.parameters()), trainable_parameter_count=sum(p.numel() for p in net.parameters() if p.requires_grad),
                    source={p:digest(ROOT/'src'/p) for p in ['aligned_interval.py','joint_interval.py','event_proposals.py','train_aligned_interval.py','train_joint_interval.py','train_epn.py','dataset.py']},
                    adaptations='Same joint presence BCE plus event-balanced IoU in real/zero arms. Both read identical verified 536+768 frozen V5 cache and preserve stores, original group split, labels, seed and budget. Only skip information is enabled or zeroed. Frozen encoders; no BEAT pooling change or established timing gain.',
                    backend=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key:manifest[key] for key in ['arm','training_clips','dev_clips','parameter_count','backend']}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=.0001)
    best, history, started = float('inf'), [], time.monotonic()
    for epoch in range(1, args.epochs+1):
        net.train()
        totals, count = np.zeros(3), 0
        for i, (x, target, weights, labels, lengths) in enumerate(train, 1):
            x, target, weights, labels = (value.cuda() for value in [x, target, weights, labels])
            distances, logits = net(x, lengths)
            loc, activity = EP.proposal_loss(distances, target, weights), presence_loss(logits, labels, lengths)
            loss = loc+activity
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite aligned interval loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
            totals += np.array([float(loss.detach()), float(loc.detach()), float(activity.detach())])*len(x)
            count += len(x)
            if i % 200 == 0:
                print(f'epoch {epoch} batch {i}/{len(train)} loss={totals[0]/count:.6f} elapsed={time.monotonic()-started:.0f}s', flush=True)
        net.eval()
        val = np.zeros(3)
        with torch.no_grad():
            for x, target, weights, labels, lengths in validation:
                x, target, weights, labels = (value.cuda() for value in [x, target, weights, labels])
                distances, logits = net(x, lengths)
                loc, activity = EP.proposal_loss(distances, target, weights), presence_loss(logits, labels, lengths)
                val += np.array([float(loc+activity), float(loc), float(activity)])*len(x)
        val /= len(dev)
        if not np.isfinite(val).all():
            raise RuntimeError('Nonfinite aligned development loss')
        history.append(dict(epoch=epoch, train_loss=(totals/count).tolist(), dev_loss=float(val[0]), dev_components=val.tolist(), elapsed=time.monotonic()-started))
        ckpt = dict(state=net.state_dict(), mode='acoustic', acoustic_dim=512, fine_dimension=768,
                    alignment_skip_enabled=True, presence_residual_enabled=True, skip_mode=args.skip_mode,
                    cfg=vars(args), manifest=manifest, history=history, epoch=epoch)
        torch.save(ckpt, out/f'ep{epoch}.pt')
        if val[0] < best:
            best = val[0]
            torch.save(ckpt, out/'best.pt')
        (out/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
        scheduler.step()
    print('ALIGNED_INTERVAL_TRAINING_DONE', flush=True)


if __name__ == '__main__':
    main()
