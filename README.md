# Noise Event Detection: Datathon@IndoML 2026, Track 1

Frame-level sound-event detection of noise events in Indian-language speech (Vaani). The code takes a set of `.wav` clips and returns, for each clip, a list of `(onset, offset)` noise events in seconds.

**Final leaderboard score: 1.5926 (rank 6).** File `submissions/sub_1016_tta3.zip`, inner `predictions.jsonl` md5 `628e807db067dcc10825b11d5e6f1650`.

| route | Event F1 | Segment Dice |
|---|---|---|
| natural clips (3,437) | 0.4936 | 0.7178 |
| synthetic clips (2,080) | 0.9929 | 0.9845 |

## Contents
1. [Quick start (inference)](#quick-start-inference)
2. [Method](#method)
3. [Repository layout](#repository-layout)
4. [Environment and hardware](#environment-and-hardware)
5. [Checkpoints and assets](#checkpoints-and-assets)
6. [Reproducing the leaderboard file](#reproducing-the-leaderboard-file)
7. [Training](#training)
8. [Tests](#tests)
9. [Score history](#score-history)

## Quick start (inference)

```bash
git clone https://github.com/SharadhNaidu/noise-event-detection && cd noise-event-detection
python3.10 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate  (Ubuntu: apt install python3.10-venv)
pip install -r requirements_inference.txt
bash scripts/run_inference.sh /path/to/wavs predictions.zip   # or a .zip of wavs, e.g. input_data.zip
```

`run_inference.sh` downloads and md5-checks the checkpoints and the BEATs weights on the first run, then calls `inference/infer.py`. The same steps by hand:

```bash
python scripts/download_assets.py --what checkpoints        # -> checkpoints/  (GitHub release v1.0, ~2.3 GB)
python scripts/download_assets.py --what beats              # -> ext/PretrainedSED/resources/BEATs_strong_1.pt
python inference/infer.py --audio /path/to/wavs --out predictions.zip [--device cuda|cpu] [--batch 8]
```

**Input:** a folder (searched recursively) or a `.zip` of `.wav` files. The clip id is the file name without `.wav`, and ids must be unique. Any sample rate and channel count is accepted: audio is down-mixed to mono and resampled to 16 kHz.

**Output:** a zip holding one `predictions.jsonl`, the official submission format, with one line per input clip:
```json
{"clip_id": "IISc_VaaniProject_..._0332", "events": [{"onset": 0.42, "offset": 1.86}, {"onset": 3.1, "offset": 3.52}]}
```
Times are in seconds, rounded to 3 decimals. Every input clip gets a line. The decoder has the empty-clip fallback used on the leaderboard: a clip with nothing above threshold gets its most likely span, so the leaderboard file has no empty clips. Only inputs too short to form a frame, or zero-length/header-only files, get `"events": []`. These inputs were tested end to end: mono and stereo, 16, 22.05 and 48 kHz, 8-bit PCM, 16-bit PCM and float wavs, clips from 1 sample to 40 s, digital silence, and a header-only file.

**Other flags:** `--work DIR` sets the scratch folder for the audio store, BEATs features and linked checkpoints (default `workdir/infer`). The large per-run audio and feature files are deleted at the end unless you pass `--keep-work`. `--limit N` runs the first N clips (a smoke test).

**Run time** for the 5,517 test clips (about 10.8 h of audio) on an RTX 4070 Laptop (8 GB) is about 75 min, mostly the three TTA members. A GPU with 8 GB or more is recommended. CPU works but is slow.

**GPU support:** the pinned `torch 2.4.1+cu118` wheels ship kernels for compute capability 3.7–9.0, i.e. Kepler (K80) to Hopper (A100, H100, RTX 20/30/40, T4, V100). They need an NVIDIA driver ≥ 450 (CUDA 11.x). RTX 50xx and other Blackwell GPUs need a newer torch build, which we have not tested. On such machines, `--device cpu` runs every step on CPU, including BEATs feature extraction. If you run short of GPU memory, lower `--batch` (default 8).

**Messages you can ignore:** transformers prints that `pos_conv_embed ... weight_g/weight_v` were "not used" and `parametrizations.weight.original0/1` were "newly initialized". This comes from how newer torch names weight-norm parameters when the pretrained hub weights load. Every checkpoint stores those tensors and is loaded with `strict=True` right afterwards, so nothing stays at its random initial value.

## Method

Every member is a frame model at 20 ms that outputs 8 tracks: "any noise" plus 7 noise categories. Members are fine-tuned from `microsoft/wavlm-base-plus` (4 or 12 transformer layers). Some also use BEATs AudioSet-strong features, either frozen ("bts") or fine-tuned ("bft"). They are trained with `src/finetune.py` on 5 s crops.

1. **Routing.** The test set mixes real ("natural") recordings and synthetic mixtures. Natural clips peak at int16 full scale (≥ 32765), synthetic clips at 0.99 (`submit_ens.natural_by_peak`).
2. **Ensemble.** Each route averages the member posteriors with equal weight (`configs/members.json`):
   - natural: `hon_A (TTA), bts_C, bft_S (TTA), bts_R, v12, v5_S (TTA), v12s62`
   - synthetic: `hon_A, bts_C, bft_S, bts_R, v5all, v5all`
   - **TTA** means the clip is also run zero-padded by 0.64 s and by 1.28 s on both sides. This shifts the 5 s crop grid and gives the clip edges context. The pads are trimmed off and the three outputs averaged (`finetune.tta_posteriors`).
3. **Decoding** (`src/postpass.py`; constants in `configs/postpass.json`):
   - **Readouts.** Each route uses a frozen hysteresis readout (`configs/readouts/md02_natural.json`, `ens5_v5w2_ov.json`), then an empty-clip fallback.
   - **Synthetic level filter.** Synthetic events quieter than −16 dB relative to the clip are dropped. The labels mark only *inserted* noise, not breaths already present in the clean speech.
   - **Background extension (`bgext`).** An event of at least 1 s whose dominant category is a long background class (0, 3, 5) is extended while the any-noise posterior stays ≥ 0.30. Two such long events less than 0.5 s apart are joined when the gap itself stays ≥ 0.30.
   - **Habit full-span.** Some annotators label a persistent background as one event covering the whole clip. A small model (`src/habit.py`: HistGradientBoosting on clip-pooled BEATs tags, recording state and duration) predicts this from the clip. For natural clips with p ≥ 0.5 whose events cover less than 90 % of the clip, the events become one full-span event.
   - **Type-C inner events.** A category run (score ≥ 0.65 for at least 0.3 s, gaps under 0.12 s bridged) is added as its own nested event when it sits inside a live event at least 2× its length whose dominant category is different. The habit full-span event is never a host.
   - **Synthetic edge shift.** Synthetic onsets and offsets move by +4/−4 ms and then +4/−2 ms, clamped to the clip.

## Repository layout

```
inference/infer.py           wav folder/zip -> predictions.zip (live models; entry point for evaluation)
scripts/run_inference.sh     one-command inference (downloads assets, then infer.py)
scripts/download_assets.py   fetch + md5-verify checkpoints / posteriors / BEATs weights
scripts/build_audio_store.py wav folder/zip -> 16 kHz int16 store (the format every src/ module reads)
scripts/tta_posteriors.py    per-member posteriors (plain or TTA) on a store -> post8_<store>_<tag>[_tta].pkl
reproduce/rebuild_lb.py      cached test posteriors -> the leaderboard file (md5-checked)
reproduce/habit_test_shipped.json  the 124 natural test clips the shipped habit model turned full-span
src/                         training + model + decoding library (finetune.py, model.py, gate.py, postpass.py, habit.py, ...)
src/paths.py                 $INDOML_ROOT (default ./workdir): audio stores, features, checkpoints; src/data.py reads $INDOML_DATA (default ./data)
configs/members.json         ensemble lists, TTA members, lost-checkpoint substitutes
configs/members_cfg.json     full training argparse config of every shipped checkpoint
configs/postpass.json        all decode constants;  configs/readouts/*.json  the two frozen readouts
train/                       training launchers, cfg_to_cmd.py (exact launch lines), train_habit.py, TRAINING.md
ext/PretrainedSED/           vendored BEATs code from fschmid56/PretrainedSED (MIT; weights downloaded separately)
tests/                       pytest suite
submissions/sub_1016_tta3.zip  the file scored 1.5926 on the leaderboard
```

## Environment and hardware

| | |
|---|---|
| Python | 3.10 (3.10.21 tested) |
| PyTorch | 2.4.1 + CUDA 11.8 wheels (`--extra-index-url https://download.pytorch.org/whl/cu118`) |
| Key packages | transformers 4.44.2, numpy 2.2.6, scikit-learn 1.7.2 (the habit-model pickle needs this exact version), scipy 1.15.3, librosa 0.10.2.post1, soundfile 0.12.1, einops 0.8.1, huggingface_hub 0.34.4. Pinned in `requirements_*.txt` |
| Training GPUs | **NVIDIA A100** cloud notebook (bft_S, v5_S, v12, v12s62, v5all; exact memory variant not recorded) and **NVIDIA GeForce RTX 4070 Laptop, 8 GB** (hon_A, bts_C, bts_R). Later work ran on an **A100 MIG 3g.20gb slice (20 GB)** with 15 CPU cores and 16 GB RAM, driver 470 (CUDA 11.4 runtime; cu118 wheels run on it, cu121+ do not) |
| Training CUDA | 11.8 wheels (driver 470 / CUDA 11.4 on the server, driver 561.09 on the laptop) |
| Inference (tested) | Windows 11, Intel i9-14900HX, 32 GB RAM, RTX 4070 Laptop 8 GB, driver 561.09, CUDA 11.8 wheels |
| Precision | bf16 autocast on CUDA, fp32 on CPU |
| Threads | set `OMP_NUM_THREADS=12 MKL_NUM_THREADS=12` on many-core hosts. Uncapped torch ran 4.6× slower on the 256-core server |
| System dependencies | `libsndfile` (bundled in the `soundfile` wheels on Windows/macOS; `apt install libsndfile1` on minimal Linux), `python3.10-venv` on Ubuntu, `bash` for the `.sh` launchers (they call `python` from the active venv; override with `PYTHON=/path/to/python`), internet access on the first run (asset download and the `wavlm-base-plus` config/weights from the HF hub). No ffmpeg is needed: inputs are wav |

### Pretrained dependencies

| model | source | version |
|---|---|---|
| WavLM Base+ | [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus) (HF hub, fetched by transformers; only its config matters, every weight is overwritten by our checkpoints) | tested with revision `4c66d4806a428f2e922ccfa1a962776e232d487b` |
| BEATs (AudioSet-strong fine-tuned, `BEATs_strong_1.pt`) | [PretrainedSED release v0.0.1](https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1), MIT | md5 `0b3d1253749058df5551beeb0a8f5e87` |

The model code for BEATs is vendored in `ext/PretrainedSED/` (MIT, licence included).

## Checkpoints and assets

The checkpoints and cached posteriors are too large for git. They are attached to the GitHub release **[v1.0](https://github.com/SharadhNaidu/noise-event-detection/releases/tag/v1.0)** of this repo. `scripts/download_assets.py` downloads them into the right folders and checks each md5 against `md5.txt` (also a release asset).

| file | role | size |
|---|---|---|
| `checkpoints/hon_A.pt` | WavLM-base+ (4 layers), 8 epochs, seed 42 | 162 MB |
| `checkpoints/bts_C.pt` | + frozen BEATs-strong features, 2-block conformer head | 173 MB |
| `checkpoints/bft_S.pt` | + fine-tuned BEATs-strong | 533 MB |
| `checkpoints/bts_R.pt` | bts_C recipe + per-class presence (RED) head | 173 MB |
| `checkpoints/v5_S.pt` | + fine-tuned BEATs, LLRD 0.8, 6 epochs, seed 13 | 533 MB |
| `checkpoints/v12_ep4.pt` | 12-layer WavLM (layer sum) + fine-tuned BEATs, epoch 4 of the v12 run | 760 MB |
| `checkpoints/habit_model.pkl` | labelling-habit model (scikit-learn 1.7.2) | 1.5 MB |
| `posteriors/post8_test_audio_<tag>.pkl` | cached test posteriors of all 8 members (+ `_tta` for hon_A, bft_S, v5_S) | 11 × 81 MB |

### Lost checkpoints (please read)

The final weights of three ensemble members, **`v12`, `v12s62` and `v5all`**, were lost when the cloud A100 notebook was replaced on 2026-10-10. Their **test posteriors survive** and are published, so the leaderboard file can still be rebuilt from them (next section). The **live** `inference/infer.py` substitutes them and prints a banner saying so:

| lost member | used in | live substitute |
|---|---|---|
| `v12` (12-layer, seed 61, final epoch) | natural | `v12_ep4` (same run, epoch 4) |
| `v12s62` (12-layer, seed 62) | natural | `v12_ep4` |
| `v5all` (v5_S recipe trained on all validation data) | synthetic (×2) | `v5_S` |

All three can be retrained with the commands in [Training](#training). Seeds are fixed, but bit-exact equality with the lost weights is not guaranteed.

## Reproducing the leaderboard file

This rebuilds `sub_1016_tta3.zip` from the cached posteriors without running any network:

```bash
pip install -r requirements_inference.txt
python scripts/download_assets.py --what posteriors
python scripts/build_audio_store.py /path/to/input_data.zip test_audio      # organisers' test zip -> workdir/audio/
python reproduce/rebuild_lb.py                                             # --stage tta3 (default) or typec_syn86
```

Expected output:
```
inner md5 1315f264bb9ed3864db26530ad50d1f9 | expected for tta3 from the published posteriors: 1315f264... -> EXACT
vs uploaded LB file sub_1016_tta3.zip: clips differing 383/5517, events 13530 vs 13538, event F1 (0.2 s collar) 0.9996, segment Dice 0.9989
```

**Why tta3 is not byte-identical to the uploaded file.** The uploaded file was built from TTA posterior caches that were lost with the old machine. The published `*_tta` caches were recomputed on an RTX 4070 from the same checkpoints and code. bf16 kernels differ slightly between GPUs (mean |Δ| ≈ 4e-3 per frame), so a few edges move by one frame. Against the uploaded file the rebuild agrees at event F1 0.9996 and Dice 0.9989.

**Exact check.** `--stage typec_syn86` runs the same decode chain on the non-TTA caches, which are the originals. It reproduces the competition-time file `sub_1016_typec_syn86` **byte for byte** (md5 `f820fdfd4a94aa0c0fa203fb1b7f1023`). That shows the decode chain itself is exact. tta3 is that file with the TTA posteriors swapped in for hon_A, bft_S and v5_S.

**Habit decisions.** By default the rebuild uses the 124 decisions of the shipped habit model (`reproduce/habit_test_shipped.json`). `--habit-model checkpoints/habit_model.pkl` recomputes them with the retrained model instead, which agrees with the shipped decisions on 98.8 % of natural clips.

## Training

Paths come from two variables: `INDOML_ROOT` (`src/paths.py`, default `./workdir`) holds `audio/` stores and `ckpt/` outputs, and `INDOML_DATA` (`src/data.py`, default `./data`) holds the organisers' `validation.zip` and `input_data.zip`. Install with `pip install -r requirements_train.txt`.

**1. Data.** The training data is the [Vaani Noise Event Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) (CC-BY-4.0). It is gated: accept its terms on Hugging Face and `export HF_TOKEN=...`. Download its parquet shards to `$INDOML_ROOT/raw/data/`, e.g. `huggingface-cli download ARTPARK-IISc/Vaani-Noise-Event-Dataset --repo-type dataset --local-dir ${INDOML_ROOT:-workdir}/raw`. Put `validation.zip` and `input_data.zip` (competition files) in `$INDOML_DATA`.

**2. Audio stores** (run from `src/`):

| store | clips | command |
|---|---|---|
| `gold_audio` | 11,111 | `python extract_audio.py hf --name gold_audio --tier verified_timestamps --workers 14` |
| `val_audio` | 2,501 | `python extract_audio.py val --name val_audio --workers 14` |
| `test_audio` | 5,517 | `python extract_audio.py test --name test_audio` |
| `valtune_audio` | 1,257 | `python make_valtune.py --name valtune_audio --half tune` (speaker half of validation, `split_by_speaker(0.5, seed=42)`) |
| `valall_audio` | 2,501 | `python make_valtune.py --name valall_audio --half all` (only for `v5all`) |
| `synth_aug` | 9,982 | `python synth.py --name synth_aug --n 10000` (seed 0, SNR −5…15 dB, 1–4 inserted events, TUNE-half noise bank) |
| `synth_mask_t` | 9,000 | `python -c "import synth; synth.build_masked('synth_mask_t')"` (seed 5, SNR −12…2 dB, 2–8 s long events masked by speech, TUNE-half bank) |
| `silver_long` | 12,014 | the `unverified_timestamps` tier: `python extract_audio.py hf --name silver_audio --tier unverified_timestamps`, then keep clips whose longest event is ≥ 1.5 s |

The `gold_audio` and `val_audio` commands are the verbatim launch lines. The other rows follow the scripts' defaults and our notes. The original launch lines for `synth_aug` (`--n`) and `silver_long` (an extra filter removed 216 of 12,230 candidate clips) were not recorded exactly.

**3. Members.** Each checkpoint stores its full argparse config, which is also kept in `configs/members_cfg.json`. `train/cfg_to_cmd.py` prints the exact `finetune.py` command for every member and checks that it re-parses to the stored config:

```bash
python train/cfg_to_cmd.py            # all members; run the printed lines from src/
```
For example (hon_A):
```bash
cd src && python finetune.py --train-stores gold_audio synth_aug synth_mask_t valtune_audio silver_long --tag hon_A \
  --enc-layers 4 --batch 6 --accum 3 --no-boundary-stores silver_long --p-center 0.0 --silver-w 0.5 --workers 0
```
`train/TRAINING.md` records where and when each member was launched, along with the server launchers (`member_go_b.sh`) used for `v12`, `v12s62` and `v5all`. For `v12s62` use `--seed 62`, and for `v5all` the v5_S line with `valall_audio` in place of `valtune_audio`. Checkpoints are written to `$INDOML_ROOT/ckpt/<tag>.pt`.

**4. Habit model.**
```bash
python train/train_habit.py --out checkpoints/habit_model.pkl
```
This trains on gold plus the TUNE half of validation and reports AUC on the held-out DECIDE half. Expected: gold hold-out AUC ≈ 0.79, DECIDE AUC ≈ 0.73.

**5. Test posteriors** for the rebuild. Training writes checkpoints to `$INDOML_ROOT/ckpt/`, which is where this script reads them. To use the downloaded ones instead, copy them there first:
```bash
mkdir -p workdir/ckpt && cp checkpoints/*.pt workdir/ckpt/
python scripts/tta_posteriors.py hon_A test_audio [--tta] --out posteriors/     # needs workdir/audio/test_audio + BEATs features for bts_*
```

## Tests

```bash
pytest            # decode, habit, inference I/O, asset download, training-config round trip; needs no data or weights
pytest -m data    # model/feature tests; need scripts/download_assets.py (+ checkpoints in $INDOML_ROOT/ckpt) and the audio stores
```

## Score history

| LB | what changed |
|---|---|
| 1.5831 | 7-member natural ensemble and 6-member synthetic ensemble, frozen readouts, synthetic level filter |
| 1.5852 | background extension (`bgext`) + synthetic edge shift +4/−4 ms |
| 1.5902 | habit full-span model |
| 1.5907 | type-C inner events in habit clips |
| **1.5926** | second synthetic shift (+4/−2 ms) + TTA (0 / 0.64 / 1.28 s pads) on hon_A, bft_S, v5_S. **Final.** |
