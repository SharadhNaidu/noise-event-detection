# How each champion member was trained

Every member was trained with `src/finetune.py`, from pretrained WavLM-base+ (plus BEATs where noted).

**Ground truth for the arguments:** `configs/members_cfg.json`, the full argparse `cfg` stored inside each checkpoint (hon_A, bts_C, bft_S, bts_R, v5_S, v12_ep4). `python train/cfg_to_cmd.py` turns it into exact, verified launch lines. The lines below are the ones recovered from session logs; where they are truncated, trust the cfg.

In this repo every path comes from `src/paths.py` (`INDOML_ROOT`, default `./workdir`); see the README for building the audio stores. During the competition the server layout was `/root/work/v9/src` (code), `INDOML_ROOT=/root/work/indoml` (stores) and `/root/work/.venv2` (venv); only historical.

Always cap threads (`OMP_NUM_THREADS=12`). Uncapped, torch spawned 138 threads and ran 4.6x slower.

Train stores, unless noted: `gold_audio synth_aug synth_mask_t valtune_audio silver_long`. Silver (`--no-boundary-stores silver_long`) has weight 0.5 on every loss term. Its boundary targets still train the boundary head, because that is masked only with `--mask-nb-boundary`, which no member used.

| member | launched | command |
|---|---|---|
| hon_A | laptop, 09-27 | `finetune.py --tag hon_A ...` WavLM only, 8 ep, seed 42. The recovered line is truncated; the full args are in the cfg. |
| bts_C | laptop, 09-30 | `finetune.py --tag bts_C --use-ast --aux beats --conformer 2 --eval-every 2 --train-stores gold_audio synth_aug synth_mask_t valtune_audio silver_long --no-boundary-stores silver_long --silver-w 0.5 --p-center 0.0 --enc-layers 4 --epochs 8 --batch 6 --accum 3 --crop 5.0 --boundary-w 0.3 --lr-enc 5e-5 --lr-head 1e-3 --workers 0 --seed 42` |
| bft_S | A100, 09-30 | `finetune.py --tag bft_S --use-ast --aux beats_ft --eval-every 2 --train-stores gold_audio synth_aug synth_mask_t valtune_audio silver_long --no-boundary-stores silver_long --silver-w 0.5 --p-center 0.0 --enc-layers 4 --epochs 8 --batch 6 --accum 3 --crop 5.0 --boundary-w 0.3 --lr-enc 5e-5 --lr-head 1e-3 --workers 4 --seed 42` |
| bts_R | laptop, 10-03 | `finetune.py --tag bts_R $BTS --eval-every 2 --save-every 2 --epochs 8 --batch 6 --accum 3`. `$BTS` = the bts_C flags + RED presence classes (`--red-classes`); see the cfg. |
| v5_S | A100, 10-01 | `finetune.py --tag v5_S $F5 --eval-every 2 --epochs 6 --batch 6 --accum 3 --workers 4 --seed 13`. `$F5` = the `member_go_b.sh` base flags with `--enc-layers 4 --red-classes` (clip-w-fix, keep-deg-frames, LLRD 0.8, lr-beats-top 1e-4, warm-head 1). |
| v12 | A100, 10-07 | `bash train/member_go_b.sh v12 beats_ft 61 --enc-layers 12 --layer-sum --init-layer 4 --grad-ckpt` (launched with the earlier `member_go.sh`, same flags) |
| v12s62 | A100, 10-10 (`q12b.sh`) | `BATCH=6 ACCUM=3 bash train/member_go_b.sh v12s62 beats_ft 62 --enc-layers 12 --layer-sum --init-layer 4 --grad-ckpt --train-stores gold_audio synth_aug synth_mask_t valtune_audio silver_long` |
| v5all | A100, 10-07 | `bash train/member_go_b.sh v5all beats_ft 13 --train-stores gold_audio synth_aug synth_mask_t valall_audio silver_long` (launched with the earlier `member_go_all.sh`, same flags). This is the v5_S recipe with `valall_audio` (all validation) replacing `valtune_audio`. |

Notes on the launchers:
- **`member_go.sh` / `member_go_all.sh`:** earlier copies of `train/member_go_b.sh`, not shipped. The `_b` version only adds the `BATCH`/`ACCUM`/`EPOCHS` env overrides, whose defaults (6 / 3 / 6) equal the values those runs used, so the `_b` lines above are equivalent. All of them build the flags as `B + --enc-layers 4 --red-classes + EXTRA`.
- **How `EXTRA` works:** argparse keeps the last value, so the `--enc-layers 12` and `--train-stores ...` in `EXTRA` override the base values.

**Test posteriors:** `test_posteriors.sh TAG` (`submit_ens.member_posteriors` on `test_audio`) wrote `post8_test_audio_<TAG>.pkl`. That is the format in `../posteriors/`.

**Lost checkpoints:** the final `v12.pt`, `v12s62.pt` and `v5all.pt` were lost with the old A100 notebook (replaced 2026-10-10). Retrain them with the commands above. Seeds are fixed, but bit-exact equality with the old checkpoints is not guaranteed.
