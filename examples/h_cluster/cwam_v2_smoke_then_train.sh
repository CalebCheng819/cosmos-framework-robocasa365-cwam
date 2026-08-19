#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable CWAM-v2 source snapshot.}"
: "${FORMAL_RUN_NAME:=CWAM-COSMOS3-V2-20260818-R1}"
: "${WORLD_SIZE:=8}"
: "${C1_R13_SERVING_CHECKPOINT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/staging/cwam-serving-checkpoints/C1-R13-iter_000020000-serving-ema-bf16-metadata-v1}"

[[ "$WORLD_SIZE" == 8 ]] || {
    echo "ERROR: CWAM-v2 formal training requires exactly 8 GPUs" >&2
    exit 1
}

export EXPERIMENT_LABEL=CWAM_V2
export WORLD_SIZE
export RECIPE_NAME=action_policy_robocasa365_cwam_v2_nano
export EXPECTED_HOST="${EXPECTED_HOST:-gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"
export CONTROL_ROOT="${CONTROL_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-v2-20260818/control}"
export C1_INPUT_ROOT="${C1_INPUT_ROOT:-/nvme/cwam-v2/inputs/shared}"
export ROBOCASA365_CONTROL_CACHE_ROOT="${ROBOCASA365_CONTROL_CACHE_ROOT:-$C1_INPUT_ROOT/control-cache}"
export ROBOCASA365_STRONG_LATENT_CACHE_ROOT="${ROBOCASA365_STRONG_LATENT_CACHE_ROOT:-$C1_INPUT_ROOT/latent-cache}"
export RUN_ROOT="${RUN_ROOT:-/nvme/cwam-v2/runs/$FORMAL_RUN_NAME}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-/dev/shm/cwam-v2-rclone/rclone.conf}"
export OSS_ROOT="${OSS_ROOT:-eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/V2/$FORMAL_RUN_NAME}"
export TOML_FILE="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_robocasa365_cwam_v2_nano.toml"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/nvme/cwam-v2/inputs/$FORMAL_RUN_NAME/c1-r13-ema-warmstart}"
export BASE_CHECKPOINT_EXPECTED_SHARDS=8
export NCCL_IB_DISABLE=1

[[ -f "$C1_R13_SERVING_CHECKPOINT/model/.metadata" ]] || {
    echo "ERROR: C1 R13 serving checkpoint metadata is absent" >&2
    exit 1
}
[[ "$(find "$C1_R13_SERVING_CHECKPOINT/model" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 8 ]] || {
    echo "ERROR: C1 R13 serving checkpoint must contain exactly 8 DCP shards" >&2
    exit 1
}

mkdir -p "$BASE_CHECKPOINT_PATH"
if [[ -e "$BASE_CHECKPOINT_PATH/model" || -L "$BASE_CHECKPOINT_PATH/model" ]]; then
    [[ "$(readlink -f "$BASE_CHECKPOINT_PATH/model")" == "$(readlink -f "$C1_R13_SERVING_CHECKPOINT/model")" ]] || {
        echo "ERROR: refusing to reuse a warm-start model link with a different target" >&2
        exit 1
    }
else
    ln -s "$C1_R13_SERVING_CHECKPOINT/model" "$BASE_CHECKPOINT_PATH/model"
fi
if [[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]]; then
    python3 - "$BASE_CHECKPOINT_PATH/checkpoint.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    if json.load(handle) != {}:
        raise SystemExit("warm-start checkpoint.json must contain an empty object")
PY
else
    printf '{}\n' > "$BASE_CHECKPOINT_PATH/checkpoint.json.tmp"
    mv "$BASE_CHECKPOINT_PATH/checkpoint.json.tmp" "$BASE_CHECKPOINT_PATH/checkpoint.json"
fi

echo "CWAM_V2_WARMSTART_READY source=$C1_R13_SERVING_CHECKPOINT target=$BASE_CHECKPOINT_PATH shards=8 initialization=c1-r13-ema optimizer=fresh"

exec bash "$SOURCE_DIR/cwam_b2_smoke_then_train.sh"
