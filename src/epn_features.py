import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audiostore as A
import finetune as FT
import gate as G
import model as M


@torch.no_grad()
def red_sequence(net, arr, entry, crop_s, batch=2, device='cuda', with_acoustic=False):
    length = int(crop_s*A.SR)
    n = entry['n']
    tw = net.enc.frames_for(length)
    starts = [0] if n <= length else list(range(0, n-length+1, length//2))
    if n > length and starts[-1]+length < n:
        starts.append(n-length)
    n_frames = net.enc.frames_for(n) if n >= length else tw
    dim = 24+512 if with_acoustic else 24
    acc = np.zeros((n_frames, dim), np.float32)
    count = np.zeros(n_frames, np.float32)
    for i in range(0, len(starts), batch):
        waves, scales = [], []
        for start in starts[i:i+batch]:
            wave = A.get_wav(arr, entry, start, length)
            wave = np.pad(wave, (0, max(0, length-len(wave))))
            scale = wave.std()+1e-7
            waves.append((wave-wave.mean())/scale)
            scales.append([scale])
        w = torch.from_numpy(np.stack(waves).astype(np.float32)).to(device)
        scale = torch.tensor(scales, dtype=torch.float32, device=device)
        hidden = []
        handle = net.head.rnn.register_forward_hook(lambda module, inputs, output: hidden.append(output[0])) if with_acoustic else None
        try:
            with torch.autocast(device, dtype=torch.bfloat16):
                res = net(w, aux=scale)
        finally:
            if handle is not None:
                handle.remove()
        if len(res) < 4 or res[3] is None or res[3][0].shape[-1] != 8:
            raise RuntimeError('EPN feature export requires eight RED tracks')
        outputs = torch.cat(M.red_presence(*res[3]), dim=-1)
        if with_acoustic:
            outputs = torch.cat([outputs, hidden[0].float()], dim=-1)
        probabilities = outputs.float().cpu().numpy()
        for j, start in enumerate(starts[i:i+batch]):
            a = int(round(start/A.SR/FT.FS))
            take = min(tw, n_frames-a)
            if take > 0:
                acc[a:a+take] += probabilities[j, :take]
                count[a:a+take] += 1
    fill = np.maximum.accumulate(np.where(count>0, np.arange(n_frames), 0))
    return (acc/np.maximum(count, 1)[:, None])[fill]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stores', nargs='+', required=True)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--with-acoustic', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(6)
    net, blob = G.load(args.tag, 'cuda')
    net.eval()
    if blob['cfg'].get('aux') != 'beats_ft':
        raise RuntimeError('This export is specific to the V5 trainable-BEATs path')
    if 'test_audio' in blob['cfg']['train_stores']:
        raise RuntimeError('Test audio must never appear in backbone training stores')
    allowed = set(blob['cfg']['train_stores'])|{'val_audio', 'test_audio'}
    if set(args.stores)-allowed:
        raise RuntimeError('Unregistered export store')
    checkpoint = FT.CKPT/f'{args.tag}.pt'
    stamp = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                 backbone_cfg=blob['cfg'], source={p: digest(Path(__file__).parent/p) for p in ['finetune.py', 'model.py', 'epn_features.py']},
                 channels='presence[8], onset[8], offset[8]'+(', BiGRU acoustic state[512]' if args.with_acoustic else ''), dimension=536 if args.with_acoustic else 24, frame_seconds=FT.FS,
                 acoustic_path='Existing V5 crop normalization and probability stitching; no acoustic model updates')
    if args.smoke:
        arr, meta = A.load_audio_store(FT.AUD, 'val_audio')
        ids = sorted(meta['clips'])
        chosen = [min(ids, key=lambda c: meta['clips'][c]['n']), max(ids, key=lambda c: meta['clips'][c]['n']), ids[len(ids)//2]]
        p, _, _ = FT.clip_posteriors(net, arr, meta['clips'], chosen, blob['cfg']['crop'], 'cuda', batch=args.batch,
                                    ast='scale', all_channels=True)
        error = {c: float(np.max(abs(p[c]-red_sequence(net, arr, meta['clips'][c], blob['cfg']['crop'], args.batch, with_acoustic=args.with_acoustic)[:, :8]))) for c in chosen}
        if max(error.values()) > 1e-6:
            raise RuntimeError(f'Acoustic path mismatch: {error}')
        print(json.dumps(dict(smoke_passed=True, max_probability_difference=error)), flush=True)
        return
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for store in args.stores:
        arr, meta = A.load_audio_store(FT.AUD, store)
        idx_path = dest/f'{store}.json'
        data_path = dest/f'{store}.f16'
        partial = dest/f'{store}.f16.partial'
        store_stamp = dict(stamp, audio_index_sha256=digest(FT.AUD/f'{store}.json'), store=store,
                           may_train=store in blob['cfg']['train_stores'])
        if idx_path.exists():
            index = json.loads(idx_path.read_text())
            if index['provenance'] != store_stamp:
                raise RuntimeError('Existing feature cache provenance mismatch')
            if index.get('complete'):
                print(f'{store}: already complete', flush=True)
                continue
        else:
            clips, total = {}, 0
            for c, entry in sorted(meta['clips'].items()):
                n = max(1, net.enc.frames_for(entry['n']))
                clips[c] = dict(off=total, n=n)
                total += n
            index = dict(provenance=store_stamp, clips=clips, total_frames=total,
                         completed_clips=[], complete=False)
            idx_path.write_text(json.dumps(index))
        done = set(index['completed_clips'])
        if done and not partial.exists():
            raise RuntimeError('Partial cache is missing')
        data = np.memmap(partial, dtype=np.float16, mode='r+' if partial.exists() else 'w+', shape=(index['total_frames'], stamp['dimension']))
        for i, (c, entry) in enumerate(sorted(meta['clips'].items()), 1):
            if c in done:
                continue
            seq = red_sequence(net, arr, entry, blob['cfg']['crop'], args.batch, with_acoustic=args.with_acoustic)
            target = index['clips'][c]
            data[target['off']:target['off']+target['n']] = seq[:target['n']]
            index['completed_clips'].append(c)
            if i % 250 == 0:
                data.flush()
                temp_index = idx_path.with_suffix('.json.partial')
                temp_index.write_text(json.dumps(index))
                os.replace(temp_index, idx_path)
                print(f'{store}: {i}/{len(meta["clips"])} elapsed={time.monotonic()-started:.0f}s', flush=True)
        data.flush()
        del data
        os.replace(partial, data_path)
        index['complete'] = True
        index['completed_clips'] = []
        temp_index = idx_path.with_suffix('.json.partial')
        temp_index.write_text(json.dumps(index))
        os.replace(temp_index, idx_path)
        print(f'{store}: COMPLETE frames={index["total_frames"]} elapsed={time.monotonic()-started:.0f}s', flush=True)
    print('EPN_FEATURE_EXPORT_DONE', flush=True)


if __name__ == '__main__':
    main()
