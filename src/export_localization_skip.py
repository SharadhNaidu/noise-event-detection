"""Export the measured V5 pre-fusion WavLM stream; no model adaptation."""
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import audiostore as A
import finetune as FT
import gate as G

EXPECTED = '832eece040fea8e1a6cf07c57ae022eefaaf3aa342437fc66d44be9babe31d86'
STORES = ['val_audio', 'gold_audio', 'synth_aug', 'synth_mask_t', 'valtune_audio', 'silver_long']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


@torch.no_grad()
def sequence(enc, arr, entry, crop_s, batch, full=None):
    """Identical crop geometry/normalization/averaging to epn_features.red_sequence."""
    length, n = int(crop_s*A.SR), entry['n']
    tw = enc.frames_for(length)
    starts = [0] if n <= length else list(range(0, n-length+1, length//2))
    if n > length and starts[-1]+length < n:
        starts.append(n-length)
    frames = enc.frames_for(n) if n >= length else tw
    acc, count = np.zeros((frames, 768), np.float32), np.zeros(frames, np.float32)
    maximum_error = 0.
    for i in range(0, len(starts), batch):
        waves, scales = [], []
        for start in starts[i:i+batch]:
            wave = A.get_wav(arr, entry, start, length)
            wave = np.pad(wave, (0, max(0, length-len(wave))))
            scale = wave.std()+1e-7
            waves.append((wave-wave.mean())/scale)
            scales.append([scale])
        w = torch.from_numpy(np.stack(waves).astype(np.float32)).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = enc(w)
            if full is not None:
                captured = []
                handle = full.enc.register_forward_hook(lambda module, inputs, out: captured.append(out))
                try:
                    full(w, aux=torch.tensor(scales, dtype=torch.float32, device='cuda'))
                finally:
                    handle.remove()
                maximum_error = max(maximum_error, float((outputs.float()-captured[0].float()).abs().max()))
        outputs = outputs.float().cpu().numpy()
        if outputs.shape != (len(waves), tw, 768) or not np.isfinite(outputs).all():
            raise RuntimeError('Invalid encoder output dimensions or nonfinite values')
        for j, start in enumerate(starts[i:i+batch]):
            a = int(round(start/A.SR/FT.FS))
            take = min(tw, frames-a)
            if take > 0:
                acc[a:a+take] += outputs[j, :take]
                count[a:a+take] += 1
    fill = np.maximum.accumulate(np.where(count > 0, np.arange(frames), 0))
    return (acc/np.maximum(count, 1)[:, None])[fill], maximum_error


def reference(root, audio, store, enc, cfg, checkpoint_hash):
    path = root/f'{store}.json'
    index = json.loads(path.read_text())
    provenance = index['provenance']
    if not index['complete'] or provenance['dimension'] != 536 or provenance['checkpoint_sha256'] != checkpoint_hash:
        raise RuntimeError('Reference must be the complete measured V5 acoustic cache')
    if provenance['backbone_cfg'] != cfg or provenance['audio_index_sha256'] != digest(audio/f'{store}.json'):
        raise RuntimeError('Reference backbone or source audio metadata changed')
    if (root/f'{store}.f16').stat().st_size != index['total_frames']*536*2:
        raise RuntimeError('Reference feature payload size mismatch')
    arr, meta = A.load_audio_store(audio, store)
    if set(meta['clips']) != set(index['clips']):
        raise RuntimeError('Reference and audio clip IDs differ')
    offset = 0
    for cid, entry in sorted(meta['clips'].items()):
        f = index['clips'][cid]
        if f != dict(off=offset, n=max(1, enc.frames_for(entry['n']))):
            raise RuntimeError('Reference frame counts/offsets disagree with encoder geometry')
        offset += f['n']
    if offset != index['total_frames']:
        raise RuntimeError('Reference total frames mismatch')
    return arr, meta, index, digest(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=str(P.ROOT / "ckpt/codex_v5_S.pt"))
    ap.add_argument('--reference-root', required=True)
    ap.add_argument('--audio', default=str(P.ROOT / "audio"))
    ap.add_argument('--out', required=True)
    ap.add_argument('--stores', nargs='+', default=STORES)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if not set(args.stores) <= set(STORES) or len(set(args.stores)) != len(args.stores):
        raise RuntimeError('Only the registered training stores and validation inference are permitted')
    if args.limit is not None and (not args.smoke or args.limit < 1):
        raise RuntimeError('Limit is permitted only for smoke; full cache geometry is preserved')
    if not 1 <= args.threads <= 4 or args.batch < 1 or not torch.cuda.is_available():
        raise RuntimeError('CUDA and at most four CPU threads are required')
    if os.name == 'nt':
        ownership = json.loads((Path(__file__).resolve().parents[1]/'docs/controller/COMPUTE_OWNERSHIP.json').read_text())
        if ownership['laptop_reserved_for_track2'] or args.threads > ownership['laptop_cpu_thread_limit']:
            raise RuntimeError('Laptop reservation or CPU limit prohibits export')
    torch.set_num_threads(args.threads)
    checkpoint, root, audio, dest = map(Path, [args.checkpoint, args.reference_root, args.audio, args.out])
    if dest.resolve() == root.resolve() or digest(checkpoint) != EXPECTED:
        raise RuntimeError('Preserve reference files and use the exact measured V5 checkpoint')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = blob['cfg']
    if cfg.get('no_wavlm') or cfg.get('aux') != 'beats_ft' or 'test_audio' in cfg['train_stores'] or set(args.stores)-set(cfg['train_stores'])-{'val_audio'}:
        raise RuntimeError('Expected the measured two-stream V5 configuration')
    enc = FT.Encoder(cfg['enc_layers'], freeze_conv=not cfg.get('unfreeze_conv', False),
                     name=cfg.get('enc_name', 'microsoft/wavlm-base-plus'), layer_sum=cfg.get('layer_sum', False),
                     init_layer=cfg.get('init_layer')).cuda().eval()
    enc.load_state_dict({k[4:]: v for k, v in blob['state'].items() if k.startswith('enc.')}, strict=True)
    enc.requires_grad_(False)
    del blob
    sources = {p: digest(Path(__file__).parent/p) for p in ['export_localization_skip.py', 'epn_features.py', 'finetune.py', 'model.py', 'beats_enc.py']}
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for store in args.stores:
        arr, meta, ref, reference_hash = reference(root, audio, store, enc, cfg, EXPECTED)
        stamp = dict(checkpoint_sha256=EXPECTED, backbone_cfg=cfg, source=sources, reference_index_sha256=reference_hash,
                     audio_index_sha256=digest(audio/f'{store}.json'), dimension=768, frame_seconds=FT.FS,
                     torch_version=torch.__version__, autocast_dtype='bfloat16', output_dtype='float16', batch=args.batch,
                     store=store, may_train=bool(ref['provenance']['may_train']) and store != 'val_audio',
                     acoustic_path='Frozen V5 pre-fusion WavLM; identical crop padding/normalization/20ms stitching; no architecture trained')
        if args.smoke:
            FT.CKPT = checkpoint.parent
            full, full_blob = G.load(checkpoint.stem, 'cuda')
            del full_blob
            full.eval().requires_grad_(False)
            ids = sorted(meta['clips'], key=lambda c: meta['clips'][c]['n'])
            chosen = list(dict.fromkeys([ids[0], ids[-1], ids[len(ids)//2]]))[:args.limit or 3]
            errors = {c: sequence(enc, arr, meta['clips'][c], cfg['crop'], args.batch, full)[1] for c in chosen}
            if max(errors.values()) > 1e-6:
                raise RuntimeError(f'Original pipeline encoder mismatch: {errors}')
            smoke_path = dest/f'{store}_smoke.json'
            if smoke_path.exists():
                raise RuntimeError('Existing smoke report is preserved; choose a new output folder')
            atomic_json(smoke_path, dict(provenance=stamp, max_absolute_error=errors, smoke_passed=True))
            del full
            print(json.dumps(dict(store=store, smoke_passed=True, max_absolute_error=errors)), flush=True)
            continue
        index_path, payload, partial = dest/f'{store}.json', dest/f'{store}.f16', dest/f'{store}.f16.partial'
        size = ref['total_frames']*768*2
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index['provenance'] != stamp or index['clips'] != ref['clips'] or index['total_frames'] != ref['total_frames']:
                raise RuntimeError('Existing output provenance/layout mismatch; preserve it')
            if index['complete']:
                if payload.stat().st_size != size or digest(payload) != index['payload_sha256']:
                    raise RuntimeError('Completed output payload changed')
                continue
        else:
            if payload.exists() or partial.exists():
                raise RuntimeError('Unregistered existing output payload; preserve it')
            index = dict(provenance=stamp, clips=ref['clips'], total_frames=ref['total_frames'], completed_clips=[], complete=False)
            atomic_json(index_path, index)
        done = set(index['completed_clips'])
        if not done <= set(ref['clips']) or len(done) != len(index['completed_clips']):
            raise RuntimeError('Invalid completion ledger; preserve partial output')
        if not partial.exists():
            if payload.exists() or done:
                raise RuntimeError('Interrupted output needs explicit review; preserve its files')
            with partial.open('xb') as f:
                f.truncate(size)
        if partial.stat().st_size != size:
            raise RuntimeError('Partial output size mismatch; never truncate it')
        data = np.memmap(partial, dtype=np.float16, mode='r+', shape=(ref['total_frames'], 768))
        for i, (cid, entry) in enumerate(sorted(meta['clips'].items()), 1):
            if cid in done:
                continue
            seq, _ = sequence(enc, arr, entry, cfg['crop'], args.batch)
            target = ref['clips'][cid]
            exported = seq[:target['n']].astype(np.float16)
            if exported.shape != (target['n'], 768) or not np.isfinite(exported).all():
                raise RuntimeError('Joined feature shape or float16 finiteness failure')
            data[target['off']:target['off']+target['n']] = exported
            index['completed_clips'].append(cid)
            if i % 25 == 0:
                data.flush()
                atomic_json(index_path, index)
                print(f'{store}: {i}/{len(meta["clips"])} elapsed={time.monotonic()-started:.0f}s', flush=True)
        data.flush()
        del data
        if payload.exists():
            raise RuntimeError('Output payload already exists; never replace it')
        partial.rename(payload)
        index.update(complete=True, completed_clips=[], payload_sha256=digest(payload))
        atomic_json(index_path, index)
        print(f'{store}: COMPLETE elapsed={time.monotonic()-started:.0f}s', flush=True)
    print('LOCALIZATION_SKIP_SMOKE_DONE' if args.smoke else 'LOCALIZATION_SKIP_EXPORT_DONE', flush=True)


if __name__ == '__main__':
    main()
