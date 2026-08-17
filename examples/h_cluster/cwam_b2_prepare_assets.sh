#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Prepare the immutable B2 inputs on the allocated worker NVMe. The rclone
# credential is handed to the worker as an encrypted CMS envelope and remains
# only in /dev/shm after decryption.

set -Eeuo pipefail
umask 077

: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${CONTROL_ROOT:=$PERSONAL_ROOT/experiments/cwam-b2-20260811/control}"
: "${SOURCE_ROOT:=$PERSONAL_ROOT/code/cwam-cosmos3-b2-20260811/source}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${B2_INPUT_ROOT:=/nvme/cwam-b2/inputs}"
: "${LATENT_REMOTE:=eailab-hdd2:eailab/chengjuntao/caches/robocasa365-wan22-triple480-fullpad-bf16/CWAM-E35STRONG-FULLPAD-WAN-CACHE-R3-8H200-20260804}"

PYTHON_BIN="$COSMOS_ENV/bin/python"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
OPENSSL_BIN="/usr/bin/openssl"
SECRET_ROOT="/dev/shm/cwam-b2-rclone"
HANDOFF_ROOT="$CONTROL_ROOT/rclone-handoff"
CONTROL_CACHE_ROOT="$B2_INPUT_ROOT/control-cache"
LATENT_CACHE_ROOT="$B2_INPUT_ROOT/latent-cache"
LATENT_STAGE_ROOT="$B2_INPUT_ROOT/.latent-cache.partial"

for executable in "$PYTHON_BIN" "$RCLONE_BIN" "$OPENSSL_BIN"; do
    [[ -x "$executable" ]] || {
        echo "ERROR: required executable is unavailable: $executable" >&2
        exit 1
    }
done
[[ -f "$ROBOCASA365_ROOT/_SUCCESS.json" && -f "$ROBOCASA365_ROOT/dataset_manifest.json" ]] || {
    echo "ERROR: converted RoboCasa365 input is incomplete" >&2
    exit 1
}
[[ -d "$ROBOCASA365_SOURCE_ROOT/videos" && -f "$ROBOCASA365_SOURCE_ROOT/meta/info.json" ]] || {
    echo "ERROR: source RoboCasa365 input is incomplete" >&2
    exit 1
}
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: Wan VAE is missing" >&2; exit 1; }
for tokenizer_file in config.json tokenizer.json tokenizer_config.json; do
    [[ -f "$QWEN3_VL_TOKENIZER_PATH/$tokenizer_file" ]] || {
        echo "ERROR: tokenizer file is missing: $tokenizer_file" >&2
        exit 1
    }
done
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" && -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] || {
    echo "ERROR: Cosmos3-Nano DCP metadata is incomplete" >&2
    exit 1
}
mapfile -t dcp_shards < <(find "$BASE_CHECKPOINT_PATH/model" -maxdepth 1 -type f -name '*.distcp' -print)
[[ "${#dcp_shards[@]}" -eq 6 ]] || {
    echo "ERROR: Cosmos3-Nano DCP must contain exactly six distcp shards" >&2
    exit 1
}

mkdir -p "$B2_INPUT_ROOT" "$HANDOFF_ROOT" "$CONTROL_ROOT/results"
mkdir -p "$SECRET_ROOT"

PRIVATE_KEY="$SECRET_ROOT/recipient.key"
CERTIFICATE="$HANDOFF_ROOT/recipient.pem"
ENCRYPTED_CONFIG="$HANDOFF_ROOT/rclone-config.cms.der"
RCLONE_CONFIG="$SECRET_ROOT/rclone.conf"

if [[ ! -s "$RCLONE_CONFIG" ]]; then
    if [[ ! -s "$PRIVATE_KEY" || ! -s "$CERTIFICATE" ]]; then
        rm -f "$PRIVATE_KEY" "$CERTIFICATE" "$ENCRYPTED_CONFIG" "$HANDOFF_ROOT/request.ready"
        "$OPENSSL_BIN" req -x509 -newkey rsa:3072 -nodes -days 1 \
            -subj '/CN=cwam-b2-assets' \
            -keyout "$PRIVATE_KEY" \
            -out "$CERTIFICATE" >/dev/null 2>&1
        chmod 0400 "$PRIVATE_KEY"
        chmod 0444 "$CERTIFICATE"
        date -Is > "$HANDOFF_ROOT/request.ready"
    fi
    echo "RCLONE_HANDOFF_WAITING certificate=$CERTIFICATE"
    for _ in $(seq 1 240); do
        [[ -s "$ENCRYPTED_CONFIG" ]] && break
        sleep 5
    done
    [[ -s "$ENCRYPTED_CONFIG" ]] || {
        echo "ERROR: timed out waiting for the encrypted rclone configuration" >&2
        exit 1
    }
    "$OPENSSL_BIN" cms -decrypt -binary -inform DER \
        -in "$ENCRYPTED_CONFIG" \
        -recip "$CERTIFICATE" \
        -inkey "$PRIVATE_KEY" \
        -out "$RCLONE_CONFIG"
    chmod 0400 "$RCLONE_CONFIG"
    rm -f "$PRIVATE_KEY"
fi

RCLONE_CONFIG="$RCLONE_CONFIG" "$RCLONE_BIN" listremotes | grep -Fx 'eailab-hdd2:' >/dev/null
echo "RCLONE_HANDOFF_READY remote=eailab-hdd2"

control_pid=""
if [[ ! -s "$CONTROL_CACHE_ROOT/manifest.json" || ! -s "$CONTROL_CACHE_ROOT/_SUCCESS.json" ]]; then
    [[ ! -e "$CONTROL_CACHE_ROOT" ]] || {
        echo "ERROR: incomplete control cache already exists: $CONTROL_CACHE_ROOT" >&2
        exit 1
    }
    (
        cd "$SOURCE_ROOT"
        export PYTHONNOUSERSITE=1
        export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        "$PYTHON_BIN" examples/h_cluster/build_robocasa365_control_cache.py \
            --dataset-root "$ROBOCASA365_ROOT" \
            --source-root "$ROBOCASA365_SOURCE_ROOT" \
            --output-root "$CONTROL_CACHE_ROOT"
    ) > >(tee "$CONTROL_ROOT/results/control-cache-build.log") 2>&1 &
    control_pid="$!"
else
    echo "CONTROL_CACHE_REUSED root=$CONTROL_CACHE_ROOT"
fi

if [[ ! -s "$LATENT_CACHE_ROOT/manifest.json" || ! -s "$LATENT_CACHE_ROOT/_SUCCESS.json" ]]; then
    [[ ! -e "$LATENT_CACHE_ROOT" ]] || {
        echo "ERROR: incomplete final latent cache exists: $LATENT_CACHE_ROOT" >&2
        exit 1
    }
    mkdir -p "$LATENT_STAGE_ROOT"
    echo "LATENT_CACHE_STAGE_STARTED remote=$LATENT_REMOTE destination=$LATENT_STAGE_ROOT"
    RCLONE_CONFIG="$RCLONE_CONFIG" "$RCLONE_BIN" copy "$LATENT_REMOTE" "$LATENT_STAGE_ROOT" \
        --exclude '/_SUCCESS.json' \
        --transfers 32 \
        --checkers 64 \
        --multi-thread-cutoff 128M \
        --multi-thread-streams 8 \
        --buffer-size 32M \
        --retries 5 \
        --low-level-retries 20 \
        --stats 30s \
        --stats-one-line
    RCLONE_CONFIG="$RCLONE_CONFIG" "$RCLONE_BIN" copyto \
        "$LATENT_REMOTE/_SUCCESS.json" "$LATENT_STAGE_ROOT/_SUCCESS.json"
    [[ -s "$LATENT_STAGE_ROOT/manifest.json" && -s "$LATENT_STAGE_ROOT/_SUCCESS.json" ]] || {
        echo "ERROR: staged latent cache is missing terminal metadata" >&2
        exit 1
    }
    mv "$LATENT_STAGE_ROOT" "$LATENT_CACHE_ROOT"
else
    echo "LATENT_CACHE_REUSED root=$LATENT_CACHE_ROOT"
fi

if [[ -n "$control_pid" ]]; then
    wait "$control_pid"
fi
[[ -s "$CONTROL_CACHE_ROOT/manifest.json" && -s "$CONTROL_CACHE_ROOT/_SUCCESS.json" ]] || {
    echo "ERROR: control cache did not reach terminal completeness" >&2
    exit 1
}

export CONTROL_CACHE_ROOT LATENT_CACHE_ROOT LATENT_REMOTE BASE_CHECKPOINT_PATH
"$PYTHON_BIN" - <<'PY' > "$CONTROL_ROOT/results/assets-ready.json"
import datetime
import json
import os
from pathlib import Path


def inventory(root: Path) -> dict[str, object]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "root": str(root),
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "manifest_present": (root / "manifest.json").is_file(),
        "success_present": (root / "_SUCCESS.json").is_file(),
    }


payload = {
    "status": "ready",
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": os.uname().nodename,
    "control_cache": inventory(Path(os.environ["CONTROL_CACHE_ROOT"])),
    "latent_cache": inventory(Path(os.environ["LATENT_CACHE_ROOT"])),
    "latent_remote": os.environ["LATENT_REMOTE"],
    "base_checkpoint": inventory(Path(os.environ["BASE_CHECKPOINT_PATH"])),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

df -hT /nvme
cat "$CONTROL_ROOT/results/assets-ready.json"
echo "B2_ASSETS_READY control=$CONTROL_CACHE_ROOT latent=$LATENT_CACHE_ROOT"
