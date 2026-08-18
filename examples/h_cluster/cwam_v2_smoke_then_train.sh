#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable CWAM-v2 source snapshot.}"
: "${FORMAL_RUN_NAME:=CWAM-COSMOS3-V2-20260818-R1}"
: "${WORLD_SIZE:=8}"

[[ "$WORLD_SIZE" == 8 ]] || {
    echo "ERROR: CWAM-v2 formal training requires exactly 8 GPUs" >&2
    exit 1
}

export EXPERIMENT_LABEL=CWAM_V2
export WORLD_SIZE
export RECIPE_NAME=action_policy_robocasa365_cwam_v2_nano
export EXPECTED_HOST="${EXPECTED_HOST:-gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"
export CONTROL_ROOT="${CONTROL_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-v2-20260818/control}"
export ROBOCASA365_CONTROL_CACHE_ROOT="${ROBOCASA365_CONTROL_CACHE_ROOT:-/nvme/cwam-c1/inputs/control-cache}"
export ROBOCASA365_STRONG_LATENT_CACHE_ROOT="${ROBOCASA365_STRONG_LATENT_CACHE_ROOT:-/nvme/cwam-c1/inputs/latent-cache}"
export RUN_ROOT="${RUN_ROOT:-/nvme/cwam-v2/runs/$FORMAL_RUN_NAME}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-/dev/shm/cwam-v2-rclone/rclone.conf}"
export OSS_ROOT="${OSS_ROOT:-eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/V2/$FORMAL_RUN_NAME}"
export TOML_FILE="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_robocasa365_cwam_v2_nano.toml"
export NCCL_IB_DISABLE=1

exec bash "$SOURCE_DIR/cwam_b2_smoke_then_train.sh"
