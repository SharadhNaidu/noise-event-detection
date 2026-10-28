#!/bin/bash
# Recovered verbatim from the old A100 server (2026-10-10); paths made repo-relative for release.
# generic member chain: TAG AUX SEED [EXTRA flags...] -> smoke, load+infer check, 6 ep, val posteriors.
# No --src-heads (extra heads are never decoded at inference). Smoke -> load+infer check -> 6-epoch run -> val posteriors.
TAG=$1; AUXN=$2; SEED=$3; shift 3; EXTRA="$*"
cd "$(dirname "$0")/../src"; export INDOML_ROOT=${INDOML_ROOT:-../workdir} HF_HUB_DISABLE_XET=1 NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export TAG; PY=${PY:-python}; LOGS=$INDOML_ROOT/logs; mkdir -p $LOGS; L=$LOGS/${TAG}.log
B="--use-ast --aux $AUXN --no-boundary-stores silver_long --silver-w 0.5 --p-center 0.0 --crop 5.0 --boundary-w 0.3 --lr-enc 5e-5 --lr-head 1e-3 --clip-w-fix --keep-deg-frames --llrd 0.8 --lr-beats-top 1e-4 --warm-head-epochs 1 --train-stores gold_audio synth_aug synth_mask_t valtune_audio silver_long"
F="$B --enc-layers 4 --red-classes $EXTRA"
echo "== $TAG start $(date)" >> $L
timeout 900 $PY -u finetune.py --tag ${TAG}_smoke $F --eval-every 0 --limit 120 --epochs 2 --batch ${BATCH:-6} --accum 1 --workers 2 --seed $SEED > $LOGS/${TAG}_smoke.log 2>&1
if ! grep -q "ep  2/2" $LOGS/${TAG}_smoke.log || grep -q Traceback $LOGS/${TAG}_smoke.log; then echo "== SMOKE FAILED $(date)" >> $L; tail -15 $LOGS/${TAG}_smoke.log >> $L; exit 1; fi
$PY - >> $L 2>&1 <<'PYX' || { echo "== LOAD FAILED $(date)" >> $L; exit 1; }
import os, gate as G, finetune as FT, audiostore as A
arr, m = A.load_audio_store(FT.AUD, "val_audio"); idx = m["clips"]; ids = list(idx)[:4]
net, blob = G.load(os.environ["TAG"] + "_smoke", "cuda"); assert net.head.red.weight.shape[0] == 16, net.head.red.weight.shape
p, d, _ = FT.clip_posteriors(net, arr, idx, ids, 5.0, "cuda", ast=G.aux_for(blob["cfg"]), all_channels=True)
assert all(p[c].shape[1] == 8 for c in ids); print("LOAD+INFER OK", p[ids[0]].shape)
PYX
rm -f $INDOML_ROOT/ckpt/${TAG}_smoke*.pt; echo "== smoke+load OK, training $(date)" >> $L
$PY -u finetune.py --tag $TAG $F --eval-every 2 --save-every 2 --epochs ${EPOCHS:-6} --batch ${BATCH:-6} --accum ${ACCUM:-3} --workers 4 --seed $SEED > $LOGS/${TAG}_train.log 2>&1
echo "== $TAG trained $(date)" >> $L; grep -E "^DECIDE :" $LOGS/${TAG}_train.log >> $L
$PY - >> $L 2>&1 <<'PYX'
import os, audiostore as A, finetune as FT, submit_ens as SE
arr, me = A.load_audio_store(FT.AUD, "val_audio"); idx = me["clips"]
post, dist = SE.member_posteriors(os.environ["TAG"], "val_audio", list(idx), "cuda", arr, idx); print(os.environ["TAG"], "val posteriors", len(post))
PYX
echo ${TAG}_DONE >> $L
