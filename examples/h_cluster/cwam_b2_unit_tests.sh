#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail

: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${SOURCE_ROOT:=$PERSONAL_ROOT/code/cwam-cosmos3-b2-20260811/source}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"

PYTHON_BIN="$COSMOS_ENV/bin/python"
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: B2 Python is not executable: $PYTHON_BIN" >&2
    exit 1
}

cd "$SOURCE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

"$PYTHON_BIN" -m pytest -q -o addopts= --confcutdir="$SOURCE_ROOT/cosmos_framework" \
    examples/h_cluster/validate_cwam_b2_training_log_test.py \
    cosmos_framework/model/generator/mot/action_relevance_test.py \
    cosmos_framework/model/generator/mot/strong_action_expert_test.py \
    cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa365_s1_nano_test.py \
    cosmos_framework/model/generator/omni_mot_denoise_test.py
