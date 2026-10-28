#!/bin/bash
# How every post8_test_audio_<TAG>.pkl was made (the testpost/post function of the server queue scripts).
# usage: bash test_posteriors.sh TAG     (checkpoint indoml/ckpt/<TAG>.pt -> indoml/ckpt/post8_test_audio_<TAG>.pkl)
cd "$(dirname "$0")/../src"
export INDOML_ROOT=${INDOML_ROOT:-../workdir} HF_HUB_DISABLE_XET=1 NVIDIA_TF32_OVERRIDE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 TAG=$1
${PY:-python} - <<"PYX"
import os, audiostore as A, finetune as FT, submit_ens as SE
arr, me = A.load_audio_store(FT.AUD, "test_audio"); idx = me["clips"]
post, dist = SE.member_posteriors(os.environ["TAG"], "test_audio", list(idx), "cuda", arr, idx); print(os.environ["TAG"], "TEST posteriors", len(post))
PYX
