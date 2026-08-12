#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launcher for the CWAM RoboCasa365 Cosmos3-Nano recipe.
#
# Required:
#   ROBOCASA365_ROOT         converted CWAM view with _SUCCESS.json
#   ROBOCASA365_SOURCE_ROOT  original RoboCasa365 tree referenced by the view
#   QWEN3_VL_TOKENIZER_PATH  local Qwen3-VL-8B-Instruct processor directory
#
# Optional:
#   BASE_CHECKPOINT_PATH     default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH             default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   OUTPUT_ROOT              default: outputs/train
#   RUN_NAME                 unique run name; defaults to a UTC timestamp
#   NPROC_PER_NODE           local GPU count; default: 8
# Strong-v1, B2, and B3 only:
#   ROBOCASA365_CONTROL_CACHE_ROOT        immutable state/action cache
#   ROBOCASA365_STRONG_LATENT_CACHE_ROOT  triple-view 480 Wan latent cache
#   ROBOCASA365_STRONG_LATENT_CACHE_MODE  must be required for formal training
#   CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER must be 1 with that cache
#   RESUME_EXISTING_RUN      0 for a new run (default), 1 for functional
#                            model/optim/scheduler/trainer resume
#   ALLOW_DATA_STREAM_REPLAY must be 1 with RESUME_EXISTING_RUN=1, acknowledging
#                            that the legacy packed loader restarts its data stream
#   EXTRA_TAIL_OVERRIDES     space-separated Hydra overrides

: "${TOML_FILE:=examples/toml/sft_config/action_policy_robocasa365_nano.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${NPROC_PER_NODE:=8}"
EXPERIMENT_SLUG="$(basename "$TOML_FILE" .toml)"
: "${RUN_NAME:=${EXPERIMENT_SLUG}_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${RESUME_EXISTING_RUN:=0}"
: "${ALLOW_DATA_STREAM_REPLAY:=0}"
if [[ -z "${MASTER_PORT:-}" ]]; then
    CWAM_PORT_HASH="$(printf '%s' "$RUN_NAME" | cksum)"
    CWAM_PORT_HASH="${CWAM_PORT_HASH%% *}"
    # Keep automatic ports inside H-cluster's recommended 20000-29999 range.
    MASTER_PORT="$((20000 + CWAM_PORT_HASH % 10000))"
fi

export ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-}"
export ROBOCASA365_SOURCE_ROOT="${ROBOCASA365_SOURCE_ROOT:-}"
export QWEN3_VL_TOKENIZER_PATH="${QWEN3_VL_TOKENIZER_PATH:-}"
export ROBOCASA365_CONTROL_CACHE_ROOT="${ROBOCASA365_CONTROL_CACHE_ROOT:-}"
export ROBOCASA365_STRONG_LATENT_CACHE_ROOT="${ROBOCASA365_STRONG_LATENT_CACHE_ROOT:-}"
export ROBOCASA365_STRONG_LATENT_CACHE_MODE="${ROBOCASA365_STRONG_LATENT_CACHE_MODE:-disabled}"
export CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER="${CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER:-0}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export COSMOS_DEVICE=cuda
export COSMOS_INTERNAL=0
export COSMOS_TRAINING=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export MASTER_PORT
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: NPROC_PER_NODE must be a positive integer (got: $NPROC_PER_NODE)" >&2; exit 1; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: RUN_NAME may contain only letters, digits, dot, underscore, and hyphen" >&2; exit 1; }
[[ "$RESUME_EXISTING_RUN" =~ ^[01]$ ]] ||
    { echo "ERROR: RESUME_EXISTING_RUN must be 0 or 1" >&2; exit 1; }
[[ "$ALLOW_DATA_STREAM_REPLAY" =~ ^[01]$ ]] ||
    { echo "ERROR: ALLOW_DATA_STREAM_REPLAY must be 0 or 1" >&2; exit 1; }
if [[ "$RESUME_EXISTING_RUN" == 1 && "$ALLOW_DATA_STREAM_REPLAY" != 1 ]]; then
    echo "ERROR: packed-loader state is not checkpointed; functional resume restarts the data stream." >&2
    echo "Set ALLOW_DATA_STREAM_REPLAY=1 only for a documented smoke/restart attempt." >&2
    exit 1
fi

LAUNCHER_WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LAUNCHER_WORKDIR/outputs/train}"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$LAUNCHER_WORKDIR/$OUTPUT_ROOT"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
[[ "$IMAGINAIRE_OUTPUT_ROOT" = /* ]] ||
    IMAGINAIRE_OUTPUT_ROOT="$LAUNCHER_WORKDIR/$IMAGINAIRE_OUTPUT_ROOT"
export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT" ||
    { echo "ERROR: cannot create output root: $IMAGINAIRE_OUTPUT_ROOT" >&2; exit 1; }
[[ -w "$IMAGINAIRE_OUTPUT_ROOT" ]] ||
    { echo "ERROR: output root is not writable: $IMAGINAIRE_OUTPUT_ROOT" >&2; exit 1; }

RUN_OUTPUT_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft/$RUN_NAME"
if [[ "$RESUME_EXISTING_RUN" == 0 && (-e "$RUN_OUTPUT_DIR" || -L "$RUN_OUTPUT_DIR") ]]; then
    echo "ERROR: refusing implicit resume into existing run directory: $RUN_OUTPUT_DIR" >&2
    exit 1
fi
if [[ "$RESUME_EXISTING_RUN" == 1 && ! -f "$RUN_OUTPUT_DIR/checkpoints/latest_checkpoint.txt" ]]; then
    echo "ERROR: explicit resume requested but latest_checkpoint.txt is absent under: $RUN_OUTPUT_DIR" >&2
    exit 1
fi
if [[ "$RESUME_EXISTING_RUN" == 1 ]]; then
    echo "WARNING: resuming model/optim/scheduler/trainer with a restarted data stream." >&2
fi
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
: "${LOG_FILENAME:=${RUN_NAME}.launch-$(date -u +%Y%m%dT%H%M%SZ).log}"

EXTRA_DATASET_CHECK='
[[ -f "$ROBOCASA365_ROOT/_SUCCESS.json" ]] ||
    { echo "ERROR: ROBOCASA365_ROOT must contain _SUCCESS.json (got: $ROBOCASA365_ROOT)" >&2; exit 1; }
[[ -f "$ROBOCASA365_ROOT/dataset_manifest.json" ]] ||
    { echo "ERROR: ROBOCASA365_ROOT must contain dataset_manifest.json" >&2; exit 1; }
[[ -f "$ROBOCASA365_ROOT/splits/train.parquet" ]] ||
    { echo "ERROR: ROBOCASA365_ROOT must contain splits/train.parquet" >&2; exit 1; }
[[ -f "$ROBOCASA365_ROOT/splits/val.parquet" ]] ||
    { echo "ERROR: ROBOCASA365_ROOT must contain splits/val.parquet" >&2; exit 1; }
[[ -f "$ROBOCASA365_ROOT/splits/test.parquet" ]] ||
    { echo "ERROR: ROBOCASA365_ROOT must contain splits/test.parquet" >&2; exit 1; }
[[ -f "$ROBOCASA365_ROOT/stats/robocasa365_native_frame_wise_relative_rot6d.json" ]] ||
    { echo "ERROR: ROBOCASA365_ROOT is missing action normalization statistics" >&2; exit 1; }
[[ -d "$ROBOCASA365_SOURCE_ROOT/videos" ]] ||
    { echo "ERROR: ROBOCASA365_SOURCE_ROOT must contain the source videos tree" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] ||
    { echo "ERROR: BASE_CHECKPOINT_PATH is not a Cosmos DCP (missing model/.metadata)" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]] ||
    { echo "ERROR: BASE_CHECKPOINT_PATH is missing checkpoint.json" >&2; exit 1; }
[[ "$(find "$BASE_CHECKPOINT_PATH/model" -maxdepth 1 -type f -name "*.distcp" | wc -l)" -eq 6 ]] ||
    { echo "ERROR: BASE_CHECKPOINT_PATH must contain the audited six Nano DCP shards" >&2; exit 1; }
[[ -d "$QWEN3_VL_TOKENIZER_PATH" ]] ||
    { echo "ERROR: QWEN3_VL_TOKENIZER_PATH must be a local directory" >&2; exit 1; }
[[ -f "$QWEN3_VL_TOKENIZER_PATH/tokenizer_config.json" ]] ||
    { echo "ERROR: QWEN3_VL_TOKENIZER_PATH is missing tokenizer_config.json" >&2; exit 1; }
[[ -f "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json" ]] ||
    { echo "ERROR: QWEN3_VL_TOKENIZER_PATH is missing tokenizer.json" >&2; exit 1; }
[[ -f "$QWEN3_VL_TOKENIZER_PATH/config.json" ]] ||
    { echo "ERROR: QWEN3_VL_TOKENIZER_PATH is missing config.json" >&2; exit 1; }
case "$(basename "$TOML_FILE")" in
action_policy_robocasa365_strong_nano.toml|\
action_policy_robocasa365_b2_nano.toml)
    [[ "$NPROC_PER_NODE" -eq 8 ]] ||
        { echo "ERROR: Strong-v1/B2 formal training requires exactly 8 ranks" >&2; exit 1; }
    ;;
action_policy_robocasa365_b3_dselect_nano.toml)
    [[ "$NPROC_PER_NODE" -eq 4 || "$NPROC_PER_NODE" -eq 8 ]] ||
        { echo "ERROR: B3 formal training requires exactly 4 or 8 ranks" >&2; exit 1; }
    ;;
esac
case "$(basename "$TOML_FILE")" in
action_policy_robocasa365_strong_nano.toml|\
action_policy_robocasa365_b2_nano.toml|\
action_policy_robocasa365_b3_dselect_nano.toml)
    [[ "$ROBOCASA365_STRONG_LATENT_CACHE_MODE" == required ]] ||
        { echo "ERROR: Strong-v1/B2/B3 requires ROBOCASA365_STRONG_LATENT_CACHE_MODE=required" >&2; exit 1; }
    [[ "$CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER" == 1 ]] ||
        { echo "ERROR: Strong-v1/B2/B3 requires CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER=1" >&2; exit 1; }
    [[ -f "$ROBOCASA365_CONTROL_CACHE_ROOT/manifest.json" && -f "$ROBOCASA365_CONTROL_CACHE_ROOT/_SUCCESS.json" ]] ||
        { echo "ERROR: Strong-v1/B2/B3 control cache is incomplete" >&2; exit 1; }
    [[ -f "$ROBOCASA365_STRONG_LATENT_CACHE_ROOT/manifest.json" && -f "$ROBOCASA365_STRONG_LATENT_CACHE_ROOT/_SUCCESS.json" ]] ||
        { echo "ERROR: Strong-v1/B2/B3 triple-view 480 latent cache is incomplete" >&2; exit 1; }
    ;;
esac
'

TAIL_OVERRIDES=(
    "job.name=${RUN_NAME}"
    "model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE}"
    ${EXTRA_TAIL_OVERRIDES:-}
)
if [[ "$RESUME_EXISTING_RUN" == 1 ]]; then
    TAIL_OVERRIDES+=("checkpoint.strict_resume=true")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
