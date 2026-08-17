#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable C1 source snapshot.}"
: "${FORMAL_RUN_NAME:=CWAM-COSMOS3-C1-20260817-R1}"

export EXPERIMENT_LABEL=C1
export WORLD_SIZE=6
export RECIPE_NAME=action_policy_robocasa365_causal_state_nano
export EXPECTED_HOST="${EXPECTED_HOST:-gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn}"
export CONTROL_ROOT="${CONTROL_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-c1-20260817/control}"
export ROBOCASA365_CONTROL_CACHE_ROOT="${ROBOCASA365_CONTROL_CACHE_ROOT:-/nvme/cwam-c1/inputs/control-cache}"
export ROBOCASA365_STRONG_LATENT_CACHE_ROOT="${ROBOCASA365_STRONG_LATENT_CACHE_ROOT:-/nvme/cwam-c1/inputs/latent-cache}"
export RUN_ROOT="${RUN_ROOT:-/nvme/cwam-c1/runs/$FORMAL_RUN_NAME}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-/dev/shm/cwam-c1-rclone/rclone.conf}"
export OSS_ROOT="${OSS_ROOT:-eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1/$FORMAL_RUN_NAME}"
export TOML_FILE="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_robocasa365_causal_state_nano.toml"
export NCCL_IB_DISABLE=1

exec bash "$SOURCE_DIR/cwam_b2_smoke_then_train.sh"
