#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Reproducible CPU/GPU preflight for the CWAM RoboCasa365 Cosmos3-Nano recipe.
# COSMOS_SOURCE should point to the immutable source snapshot being qualified.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${VALIDATION_SAMPLES_PER_SPLIT:=3}"
: "${RUN_CPU_GATE:=1}"
: "${RUN_GPU_GATE:=0}"
: "${RUN_PRODUCTION_BATCH_GATE:=0}"
: "${VALIDATION_LOG_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/logs/cosmos3_robocasa365_validation}"

[[ "$RUN_GPU_GATE" =~ ^[01]$ ]] ||
    { echo "ERROR: RUN_GPU_GATE must be 0 or 1" >&2; exit 1; }
[[ "$RUN_CPU_GATE" =~ ^[01]$ ]] ||
    { echo "ERROR: RUN_CPU_GATE must be 0 or 1" >&2; exit 1; }
[[ "$RUN_PRODUCTION_BATCH_GATE" =~ ^[01]$ ]] ||
    { echo "ERROR: RUN_PRODUCTION_BATCH_GATE must be 0 or 1" >&2; exit 1; }
if [[ "$RUN_CPU_GATE" == 0 && "$RUN_GPU_GATE" == 0 && "$RUN_PRODUCTION_BATCH_GATE" == 0 ]]; then
    echo "ERROR: at least one validation gate must be 1" >&2
    exit 1
fi
[[ "$VALIDATION_SAMPLES_PER_SPLIT" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: VALIDATION_SAMPLES_PER_SPLIT must be positive" >&2; exit 1; }

test -d "$COSMOS_SOURCE"
case "$COSMOS_SOURCE" in
    /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots/*) ;;
    *)
        echo "ERROR: COSMOS_SOURCE must be a personal immutable run snapshot: $COSMOS_SOURCE" >&2
        exit 1
        ;;
esac
CWAM_WRITABLE_ENTRY="$(find "$COSMOS_SOURCE" -xdev -perm /222 -print -quit)"
if [[ -n "$CWAM_WRITABLE_ENTRY" ]]; then
    echo "ERROR: immutable source contains a write-enabled entry: $CWAM_WRITABLE_ENTRY" >&2
    exit 1
fi
test -x "$COSMOS_ENV/bin/python"
test -f "$COSMOS_ENV/.cwam_env_ready.json"
test -f "$ROBOCASA365_ROOT/_SUCCESS.json"
test -d "$ROBOCASA365_SOURCE_ROOT/videos"
test -f "$BASE_CHECKPOINT_PATH/model/.metadata"
test -f "$WAN_VAE_PATH"
test -f "$QWEN3_VL_TOKENIZER_PATH/tokenizer_config.json"

mkdir -p "$VALIDATION_LOG_ROOT"
VALIDATION_RUN_ID="${VALIDATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
VALIDATION_LOG="$VALIDATION_LOG_ROOT/${VALIDATION_RUN_ID}.log"
exec > >(tee -a "$VALIDATION_LOG") 2>&1

echo "VALIDATION_LOG=$VALIDATION_LOG"
echo "COSMOS_SOURCE=$COSMOS_SOURCE"
date -Is

export PATH="$COSMOS_ENV/bin:$PATH"
export PYTHONPATH="$COSMOS_SOURCE"
export COSMOS_ENV
export ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT
export BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export COSMOS_INTERNAL=0
export COSMOS_TRAINING=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# CPU nodes do not receive the NVIDIA container runtime's CUDA library
# injection. Point Transformer Engine at the pinned PyPI CUDA libraries so
# CPU-only collection can import the training modules without a GPU.
COSMOS_NVIDIA_ROOT="$COSMOS_ENV/lib/python3.13/site-packages/nvidia"
if [[ -d "$COSMOS_NVIDIA_ROOT" ]]; then
    export NVRTC_HOME="${NVRTC_HOME:-$COSMOS_NVIDIA_ROOT/cuda_nvrtc}"
    export CURAND_HOME="${CURAND_HOME:-$COSMOS_NVIDIA_ROOT/curand}"
    COSMOS_NVIDIA_LIBRARY_PATH=""
    while IFS= read -r library_dir; do
        COSMOS_NVIDIA_LIBRARY_PATH="${COSMOS_NVIDIA_LIBRARY_PATH:+$COSMOS_NVIDIA_LIBRARY_PATH:}$library_dir"
    done < <(find "$COSMOS_NVIDIA_ROOT" -type d -name lib -print)
    export LD_LIBRARY_PATH="${COSMOS_NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
source "$COSMOS_SOURCE/examples/h_cluster/configure_torchcodec_runtime.sh"
# The marker and test contracts use Python assertions; never inherit -O.
unset PYTHONOPTIMIZE
if [[ "$RUN_GPU_GATE" == 1 ]]; then
    export COSMOS_DEVICE=cuda
else
    export COSMOS_DEVICE=cpu
fi
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

cd "$COSMOS_SOURCE"
"$COSMOS_ENV/bin/python" examples/h_cluster/validate_cu128_train_env_marker.py \
    --environment "$COSMOS_ENV"

if [[ "$RUN_CPU_GATE" == 1 ]]; then
    "$COSMOS_ENV/bin/python" -m pytest -q -o addopts= --noconftest -p no:cacheprovider \
        examples/h_cluster/validate_cu128_train_env_marker_test.py \
        cosmos_framework/utils/distributed_test.py \
        cosmos_framework/data/generator/action/transforms_test.py \
        cosmos_framework/data/generator/action/datasets/action_sft_dataset_test.py \
        cosmos_framework/data/generator/action/datasets/robocasa365_segment_dataset_test.py \
        cosmos_framework/data/generator/joint_dataloader_test.py \
        cosmos_framework/callbacks/action_gradient_evidence_test.py \
        cosmos_framework/callbacks/production_batch_memory_gate_test.py \
        cosmos_framework/callbacks/wandb_log_eval_test.py \
        cosmos_framework/checkpoint/dcp_test.py \
        cosmos_framework/model/generator/omni_mot_validation_test.py

    "$COSMOS_ENV/bin/python" examples/validate_action_policy_robocasa365_nano.py \
        --dataset-root "$ROBOCASA365_ROOT" \
        --source-root "$ROBOCASA365_SOURCE_ROOT" \
        --checkpoint "$BASE_CHECKPOINT_PATH" \
        --wan-vae "$WAN_VAE_PATH" \
        --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
        --samples-per-split "$VALIDATION_SAMPLES_PER_SPLIT"
fi

if [[ "$RUN_PRODUCTION_BATCH_GATE" == 1 ]]; then
    PRODUCTION_BATCH_EVIDENCE="$VALIDATION_LOG_ROOT/${VALIDATION_RUN_ID}.production-batch.json"
    "$COSMOS_ENV/bin/python" examples/validate_action_policy_robocasa365_nano_batch.py \
        --dataset-root "$ROBOCASA365_ROOT" \
        --source-root "$ROBOCASA365_SOURCE_ROOT" \
        --checkpoint "$BASE_CHECKPOINT_PATH" \
        --wan-vae "$WAN_VAE_PATH" \
        --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
        --output-json "$PRODUCTION_BATCH_EVIDENCE"
    test -s "$PRODUCTION_BATCH_EVIDENCE"
    echo "PRODUCTION_BATCH_EVIDENCE=$PRODUCTION_BATCH_EVIDENCE"
fi

if [[ "$RUN_GPU_GATE" == 1 ]]; then
    "$COSMOS_ENV/bin/python" examples/validate_action_policy_robocasa365_nano_gpu.py \
        --dataset-root "$ROBOCASA365_ROOT" \
        --source-root "$ROBOCASA365_SOURCE_ROOT" \
        --wan-vae "$WAN_VAE_PATH" \
        --split val
fi

echo "VALIDATION_PASSED cpu_gate=$RUN_CPU_GATE gpu_gate=$RUN_GPU_GATE production_batch_gate=$RUN_PRODUCTION_BATCH_GATE"
date -Is
