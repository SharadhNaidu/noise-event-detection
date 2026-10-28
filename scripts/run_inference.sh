#!/usr/bin/env bash
# Phase 2 evaluation entry point:  bash scripts/run_inference.sh <wav folder or zip> [out.zip] [extra infer.py flags]
# Downloads/verifies the checkpoints on first use, then writes the submission zip (predictions.jsonl inside).
set -euo pipefail
cd "$(dirname "$0")/.."
AUDIO=${1:?usage: run_inference.sh <wav folder or zip> [out.zip]}; OUT=${2:-predictions.zip}; shift $(( $# > 1 ? 2 : 1 ))
python scripts/download_assets.py --what checkpoints
python scripts/download_assets.py --what beats
python inference/infer.py --audio "$AUDIO" --out "$OUT" "$@"
