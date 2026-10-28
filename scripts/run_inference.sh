#!/usr/bin/env bash
# Phase 2 evaluation entry point:  bash scripts/run_inference.sh <wav folder or zip> [out.zip] [extra infer.py flags]
# Downloads/verifies the checkpoints on first use, then writes the submission zip (predictions.jsonl inside).
# Paths are taken relative to the directory you call it from. PYTHON overrides the interpreter (default: python).
set -euo pipefail
abspath() { case "$1" in /*|[A-Za-z]:[/\\]*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
AUDIO=$(abspath "${1:?usage: run_inference.sh <wav folder or zip> [out.zip] [extra infer.py flags]}")
OUT=$(abspath "${2:-predictions.zip}"); shift $(( $# > 1 ? 2 : 1 ))
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
"$PY" scripts/download_assets.py --what checkpoints
"$PY" scripts/download_assets.py --what beats
"$PY" inference/infer.py --audio "$AUDIO" --out "$OUT" "$@"
