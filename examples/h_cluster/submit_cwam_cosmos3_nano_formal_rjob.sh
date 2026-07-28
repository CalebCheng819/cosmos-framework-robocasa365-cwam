#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Submit one normal-priority, single-node 6/8-H200 production-probe + formal
# run. Execute this script from an authenticated H-cluster workspace.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_PERSONAL_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly PINNED_FORMAL_OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal"
readonly PINNED_FORMAL_MIN_FREE_BYTES="5497558138880"
readonly PINNED_FORMAL_ESTIMATED_REQUIRED_BYTES="4398046511104"
readonly PINNED_RJOB_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_RJOB_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly SCHEDULER_RECEIPT_ATTEMPTS=86400
readonly SCHEDULER_RECEIPT_RETRY_SECONDS=1
readonly SCHEDULER_RECEIPT_TIMEOUT_SECONDS=90000

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the clean immutable formal snapshot.}"
: "${FORMAL_EXPERIMENT_ID:?Set FORMAL_EXPERIMENT_ID.}"
: "${FORMAL_RUN_NAME:=$FORMAL_EXPERIMENT_ID}"
: "${FORMAL_MANIFEST_PATH:?Set FORMAL_MANIFEST_PATH.}"
: "${FORMAL_MANIFEST_SHA256:?Set FORMAL_MANIFEST_SHA256.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_WORKFLOW_SNAPSHOT_SHA256:?Set EXPECTED_WORKFLOW_SNAPSHOT_SHA256.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE to 6 or 8.}"
: "${RJOB_NAME:?Set a unique H-cluster rjob show name.}"
: "${RJOB_PREDICT_ONLY:=0}"
: "${RJOB_IMAGE:=$PINNED_CONTAINER_IMAGE}"
: "${RJOB_QUOTA_GROUP:=$PINNED_RJOB_QUOTA_GROUP}"
: "${RJOB_EXCLUDED_NODE_TAG:=$PINNED_RJOB_EXCLUDED_NODE_TAG}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${FORMAL_OUTPUT_ROOT:=$PINNED_FORMAL_OUTPUT_ROOT}"
: "${FORMAL_MIN_FREE_BYTES:=$PINNED_FORMAL_MIN_FREE_BYTES}"
: "${FORMAL_ESTIMATED_REQUIRED_BYTES:=$PINNED_FORMAL_ESTIMATED_REQUIRED_BYTES}"
SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
SECURE_PATH_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_secure_paths.py"
SECURE_TEE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_secure_tee.py"

[[ "$NPROC_PER_NODE" == 6 || "$NPROC_PER_NODE" == 8 ]] ||
    { echo "ERROR: NPROC_PER_NODE must be exactly 6 or 8" >&2; exit 1; }
if [[ "$NPROC_PER_NODE" == 8 ]]; then
    PROTOCOL_EXPERIMENT_FAMILY="CWAM-E35"
else
    PROTOCOL_EXPERIMENT_FAMILY="CWAM-E35A"
fi
[[ "$FORMAL_EXPERIMENT_ID" == "$PROTOCOL_EXPERIMENT_FAMILY"-* ]] ||
    {
        echo "ERROR: world_size=$NPROC_PER_NODE requires a $PROTOCOL_EXPERIMENT_FAMILY experiment ID" >&2
        exit 1
    }
[[ "$FORMAL_RUN_NAME" == "$FORMAL_EXPERIMENT_ID" ]] ||
    { echo "ERROR: FORMAL_RUN_NAME must equal FORMAL_EXPERIMENT_ID" >&2; exit 1; }
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] ||
    { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    { echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2; exit 1; }
[[ "$FORMAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: FORMAL_RUN_NAME contains unsupported characters" >&2; exit 1; }
[[ "$FORMAL_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: FORMAL_EXPERIMENT_ID contains unsupported characters" >&2; exit 1; }
[[ "$FORMAL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: FORMAL_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase Git SHA" >&2; exit 1; }
[[ "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_WORKFLOW_SNAPSHOT_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_SNAPSHOT_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$RJOB_IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] ||
    { echo "ERROR: RJOB_IMAGE must be repository@sha256:<64 lowercase hex>" >&2; exit 1; }
[[ "$RJOB_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
    { echo "ERROR: RJOB_IMAGE does not match the audited image digest" >&2; exit 1; }
[[ "$FORMAL_MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: FORMAL_MIN_FREE_BYTES must be a positive integer" >&2; exit 1; }
[[ "$FORMAL_ESTIMATED_REQUIRED_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: FORMAL_ESTIMATED_REQUIRED_BYTES must be a positive integer" >&2; exit 1; }
[[ "$FORMAL_OUTPUT_ROOT" == "$PINNED_FORMAL_OUTPUT_ROOT" ]] ||
    { echo "ERROR: FORMAL_OUTPUT_ROOT does not match the audited GPFS output root" >&2; exit 1; }
[[ "$FORMAL_MIN_FREE_BYTES" == "$PINNED_FORMAL_MIN_FREE_BYTES" ]] ||
    { echo "ERROR: FORMAL_MIN_FREE_BYTES does not match the audited 5-TiB gate" >&2; exit 1; }
[[ "$FORMAL_ESTIMATED_REQUIRED_BYTES" == "$PINNED_FORMAL_ESTIMATED_REQUIRED_BYTES" ]] ||
    { echo "ERROR: FORMAL_ESTIMATED_REQUIRED_BYTES does not match the audited 4-TiB estimate" >&2; exit 1; }
[[ "$RJOB_QUOTA_GROUP" == "$PINNED_RJOB_QUOTA_GROUP" ]] ||
    { echo "ERROR: RJOB_QUOTA_GROUP does not match the audited quota group" >&2; exit 1; }
[[ "$RJOB_EXCLUDED_NODE_TAG" == "$PINNED_RJOB_EXCLUDED_NODE_TAG" ]] ||
    { echo "ERROR: RJOB_EXCLUDED_NODE_TAG does not match the audited exclusion" >&2; exit 1; }

test -d "$COSMOS_SOURCE"
test -s "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
test -s "$COSMOS_SOURCE/examples/h_cluster/capture_cwam_formal_rjob_receipt.py"
test -s "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"
test -s "$SECURE_PATH_HELPER"
test -s "$SECURE_TEE_HELPER"
test -s "$FORMAL_MANIFEST_PATH"
SUBMIT_SCRIPT_REAL="$(realpath -e -- "${BASH_SOURCE[0]}")"
EXPECTED_SUBMIT_SCRIPT="$COSMOS_SOURCE/examples/h_cluster/submit_cwam_cosmos3_nano_formal_rjob.sh"
[[ "$SUBMIT_SCRIPT_REAL" == "$(realpath -e -- "$EXPECTED_SUBMIT_SCRIPT")" ]] ||
    {
        echo "ERROR: formal submit wrapper must execute from the verified COSMOS_SOURCE snapshot" >&2
        exit 1
    }
ACTUAL_MANIFEST_SHA256="$(sha256sum "$FORMAL_MANIFEST_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_MANIFEST_SHA256" != "$FORMAL_MANIFEST_SHA256" ]]; then
    echo "ERROR: formal manifest checksum mismatch" >&2
    exit 1
fi
command -v python3 >/dev/null
command -v rjob >/dev/null
command -v timeout >/dev/null
python3 "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
python3 "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py" verify-formal-contract \
    --snapshot-root "$COSMOS_SOURCE" \
    --snapshot-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    --formal-manifest "$FORMAL_MANIFEST_PATH" \
    --formal-manifest-sha256 "$FORMAL_MANIFEST_SHA256" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --expected-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-workflow-snapshot-sha256 "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    --nproc-per-node "$NPROC_PER_NODE" \
    --expected-container-image "$RJOB_IMAGE" \
    --dataset-root "$ROBOCASA365_ROOT" \
    --source-root "$ROBOCASA365_SOURCE_ROOT" \
    --base-checkpoint "$BASE_CHECKPOINT_PATH" \
    --wan-vae "$WAN_VAE_PATH" \
    --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
    --environment "$COSMOS_ENV"

if [[ "$NPROC_PER_NODE" == 8 ]]; then
    RJOB_CPU=128
    RJOB_MEMORY_MIB=1500000
else
    RJOB_CPU=96
    RJOB_MEMORY_MIB=1125000
fi

SUBMISSION_RESERVATION_ROOT="$FORMAL_OUTPUT_ROOT/formal_submission_reservations"
SUBMISSION_RESERVATION_DIR="$SUBMISSION_RESERVATION_ROOT/$FORMAL_EXPERIMENT_ID"
SUBMISSION_PENDING_PATH="$SUBMISSION_RESERVATION_DIR/submission-pending.json"
SUBMISSION_RESERVATION_PATH="$SUBMISSION_RESERVATION_DIR/submission.json"
SUBMISSION_UNKNOWN_PATH="$SUBMISSION_RESERVATION_DIR/submission-unknown.json"
SUBMISSION_LOG_PATH="$SUBMISSION_RESERVATION_DIR/rjob-submit.log"
SCHEDULER_EXPECTED_CONTRACT_PATH="$SUBMISSION_RESERVATION_DIR/rjob-expected-contract.json"
SCHEDULER_CAPTURE_LOG_PATH="$SUBMISSION_RESERVATION_DIR/rjob-scheduler-capture.log"
SCHEDULER_RECEIPT_PATH="$SUBMISSION_RESERVATION_DIR/rjob-scheduler-receipt.json"
SCHEDULER_TEXT_RECEIPT_PATH="$SUBMISSION_RESERVATION_DIR/rjob-get.log"

python3 "$SECURE_PATH_HELPER" validate-root \
    --root "$PINNED_PERSONAL_GPFS_ROOT" >/dev/null
python3 "$SECURE_PATH_HELPER" ensure-dir \
    --root "$PINNED_PERSONAL_GPFS_ROOT" \
    --path "$FORMAL_OUTPUT_ROOT" \
    --create-parents \
    --directory-mode 0755 >/dev/null

if [[ "$RJOB_PREDICT_ONLY" == 0 ]]; then
    python3 "$SECURE_PATH_HELPER" ensure-dir \
        --root "$FORMAL_OUTPUT_ROOT" \
        --path "$SUBMISSION_RESERVATION_ROOT" \
        --create-parents \
        --directory-mode 0755 >/dev/null
    if ! python3 "$SECURE_PATH_HELPER" create-dir-exclusive \
        --root "$FORMAL_OUTPUT_ROOT" \
        --path "$SUBMISSION_RESERVATION_DIR" \
        --directory-mode 0755 >/dev/null; then
        echo "ERROR: formal run already has a submit-side reservation: $SUBMISSION_RESERVATION_DIR" >&2
        exit 1
    fi
    python3 - \
        "$SUBMISSION_PENDING_PATH" \
        "$FORMAL_EXPERIMENT_ID" \
        "$FORMAL_RUN_NAME" \
        "$RJOB_NAME" \
        "$NPROC_PER_NODE" \
        "$RJOB_IMAGE" \
        "$FORMAL_MANIFEST_SHA256" \
        "$EXPECTED_SOURCE_COMMIT" \
        "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
        "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
        "$FORMAL_OUTPUT_ROOT" \
        "$FORMAL_MIN_FREE_BYTES" \
        "$FORMAL_ESTIMATED_REQUIRED_BYTES" \
        "$RJOB_QUOTA_GROUP" \
        "$RJOB_CPU" \
        "$RJOB_MEMORY_MIB" \
        "$RJOB_EXCLUDED_NODE_TAG" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "status": "submitting",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": sys.argv[2],
    "run_name": sys.argv[3],
    "rjob_name": sys.argv[4],
    "world_size": int(sys.argv[5]),
    "container_image": sys.argv[6],
    "formal_manifest_sha256": sys.argv[7],
    "source_commit": sys.argv[8],
    "workflow_snapshot_sha256": sys.argv[9],
    "source_manifest_sha256": sys.argv[10],
    "formal_output_root": sys.argv[11],
    "minimum_free_bytes": int(sys.argv[12]),
    "estimated_required_bytes": int(sys.argv[13]),
    "quota_group": sys.argv[14],
    "cpu": int(sys.argv[15]),
    "memory_mib": int(sys.argv[16]),
    "negative_tag": sys.argv[17],
    "task_type": "normal",
    "private_machine": "group",
    "priority": 9,
    "restart_policy": "never",
    "backoff_limit": 1,
}
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o444,
)
with os.fdopen(descriptor, "w") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
fi

printf -v Q_COSMOS_SOURCE '%q' "$COSMOS_SOURCE"
printf -v Q_FORMAL_EXPERIMENT_ID '%q' "$FORMAL_EXPERIMENT_ID"
printf -v Q_FORMAL_RUN_NAME '%q' "$FORMAL_RUN_NAME"
printf -v Q_FORMAL_MANIFEST_PATH '%q' "$FORMAL_MANIFEST_PATH"
printf -v Q_FORMAL_MANIFEST_SHA256 '%q' "$FORMAL_MANIFEST_SHA256"
printf -v Q_EXPECTED_SOURCE_COMMIT '%q' "$EXPECTED_SOURCE_COMMIT"
printf -v Q_EXPECTED_WORKFLOW_SNAPSHOT_SHA256 '%q' "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256"
printf -v Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256 '%q' "$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
printf -v Q_NPROC_PER_NODE '%q' "$NPROC_PER_NODE"
printf -v Q_RJOB_NAME '%q' "$RJOB_NAME"
printf -v Q_RJOB_IMAGE '%q' "$RJOB_IMAGE"
printf -v Q_COSMOS_ENV '%q' "$COSMOS_ENV"
printf -v Q_ROBOCASA365_ROOT '%q' "$ROBOCASA365_ROOT"
printf -v Q_ROBOCASA365_SOURCE_ROOT '%q' "$ROBOCASA365_SOURCE_ROOT"
printf -v Q_BASE_CHECKPOINT_PATH '%q' "$BASE_CHECKPOINT_PATH"
printf -v Q_WAN_VAE_PATH '%q' "$WAN_VAE_PATH"
printf -v Q_QWEN3_VL_TOKENIZER_PATH '%q' "$QWEN3_VL_TOKENIZER_PATH"
printf -v Q_FORMAL_OUTPUT_ROOT '%q' "$FORMAL_OUTPUT_ROOT"
printf -v Q_FORMAL_MIN_FREE_BYTES '%q' "$FORMAL_MIN_FREE_BYTES"
printf -v Q_FORMAL_ESTIMATED_REQUIRED_BYTES '%q' "$FORMAL_ESTIMATED_REQUIRED_BYTES"
printf -v Q_SUBMISSION_RESERVATION_PATH '%q' "$SUBMISSION_RESERVATION_PATH"
printf -v Q_RJOB_QUOTA_GROUP '%q' "$RJOB_QUOTA_GROUP"
printf -v Q_RJOB_EXCLUDED_NODE_TAG '%q' "$RJOB_EXCLUDED_NODE_TAG"
printf -v Q_WRAPPER '%q' "$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh"
FORMAL_COMMAND="$(
    printf '%s\n' \
        'set -Eeuo pipefail' \
        "export COSMOS_SOURCE=$Q_COSMOS_SOURCE" \
        "export FORMAL_EXPERIMENT_ID=$Q_FORMAL_EXPERIMENT_ID" \
        "export FORMAL_RUN_NAME=$Q_FORMAL_RUN_NAME" \
        "export FORMAL_MANIFEST_PATH=$Q_FORMAL_MANIFEST_PATH" \
        "export FORMAL_MANIFEST_SHA256=$Q_FORMAL_MANIFEST_SHA256" \
        "export EXPECTED_SOURCE_COMMIT=$Q_EXPECTED_SOURCE_COMMIT" \
        "export EXPECTED_WORKFLOW_SNAPSHOT_SHA256=$Q_EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
        "export EXPECTED_SNAPSHOT_MANIFEST_SHA256=$Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
        "export EXPECTED_CONTAINER_IMAGE=$Q_RJOB_IMAGE" \
        "export NPROC_PER_NODE=$Q_NPROC_PER_NODE" \
        "export RJOB_NAME=$Q_RJOB_NAME" \
        "export COSMOS_ENV=$Q_COSMOS_ENV" \
        "export ROBOCASA365_ROOT=$Q_ROBOCASA365_ROOT" \
        "export ROBOCASA365_SOURCE_ROOT=$Q_ROBOCASA365_SOURCE_ROOT" \
        "export BASE_CHECKPOINT_PATH=$Q_BASE_CHECKPOINT_PATH" \
        "export WAN_VAE_PATH=$Q_WAN_VAE_PATH" \
        "export QWEN3_VL_TOKENIZER_PATH=$Q_QWEN3_VL_TOKENIZER_PATH" \
        "export FORMAL_OUTPUT_ROOT=$Q_FORMAL_OUTPUT_ROOT" \
        "export FORMAL_MIN_FREE_BYTES=$Q_FORMAL_MIN_FREE_BYTES" \
        "export FORMAL_ESTIMATED_REQUIRED_BYTES=$Q_FORMAL_ESTIMATED_REQUIRED_BYTES" \
        "export SUBMISSION_RESERVATION_PATH=$Q_SUBMISSION_RESERVATION_PATH" \
        "export RJOB_QUOTA_GROUP=$Q_RJOB_QUOTA_GROUP" \
        "export RJOB_EXCLUDED_NODE_TAG=$Q_RJOB_EXCLUDED_NODE_TAG" \
        'export RUN_FORMAL_AFTER_PROBE=1' \
        "bash $Q_WRAPPER"
)"
FORMAL_COMMAND_SHA256="$(printf %s "$FORMAL_COMMAND" | sha256sum | awk '{print $1}')"

SUBMIT_ARGS=(
    submit
    --name "$RJOB_NAME"
    --group "$RJOB_QUOTA_GROUP"
    --charged-group "$RJOB_QUOTA_GROUP"
    --private-machine group
    --priority 9
    --task-type normal
    --restart-policy never
    --termination-grace-period-seconds 60
    --backoff_limit 1
    --image "$RJOB_IMAGE"
    --cpu "$RJOB_CPU"
    --gpu "$NPROC_PER_NODE"
    --memory "$RJOB_MEMORY_MIB"
    --negative-tags "$RJOB_EXCLUDED_NODE_TAG"
    --mount
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    SUBMIT_ARGS+=(--predict-only true)
fi

echo "RJOB_NAME=$RJOB_NAME"
echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "RJOB_CPU=$RJOB_CPU"
echo "RJOB_MEMORY_MIB=$RJOB_MEMORY_MIB"
echo "RJOB_IMAGE=$RJOB_IMAGE"
echo "COSMOS_SOURCE=$COSMOS_SOURCE"
echo "FORMAL_MANIFEST_SHA256=$FORMAL_MANIFEST_SHA256"
echo "EXPECTED_SNAPSHOT_MANIFEST_SHA256=$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
echo "SUBMISSION_RESERVATION_PATH=$SUBMISSION_RESERVATION_PATH"
echo "SCHEDULER_RECEIPT_WAIT=${SCHEDULER_RECEIPT_ATTEMPTS}x${SCHEDULER_RECEIPT_RETRY_SECONDS}s"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    rjob "${SUBMIT_ARGS[@]}" -- bash -lc "$FORMAL_COMMAND"
    echo "FORMAL_SUBMISSION_PREDICT_ONLY reservation_created=false"
    exit 0
fi

mark_submission_unknown() {
    local reason="$1"
    local rjob_status="$2"
    local tee_status="$3"
    python3 - \
        "$SUBMISSION_PENDING_PATH" \
        "$SUBMISSION_UNKNOWN_PATH" \
        "$SUBMISSION_LOG_PATH" \
        "$reason" \
        "$rjob_status" \
        "$tee_status" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

pending = Path(sys.argv[1])
target = Path(sys.argv[2])
if pending.is_symlink() or not pending.is_file() or pending.stat().st_mode & 0o777 != 0o444:
    raise SystemExit(f"pending submission record is missing, mutable, or unsafe: {pending}")
payload = json.loads(pending.read_text())
if payload.get("status") == "submitting":
    submission_log = Path(sys.argv[3])
    payload.update(
        {
            "status": "submission_unknown",
            "submission_unknown_at": datetime.now(timezone.utc).isoformat(),
            "submission_unknown_reason": sys.argv[4],
            "rjob_exit_code": int(sys.argv[5]),
            "tee_exit_code": int(sys.argv[6]),
            "submission_log": str(submission_log),
            "submission_log_sha256": (
                hashlib.sha256(submission_log.read_bytes()).hexdigest()
                if submission_log.is_file() and not submission_log.is_symlink()
                else None
            ),
        }
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
PY
}

set +e
timeout --foreground --kill-after=10s 180s \
    rjob "${SUBMIT_ARGS[@]}" -- bash -lc "$FORMAL_COMMAND" 2>&1 |
    python3 "$SECURE_TEE_HELPER" \
        --root "$FORMAL_OUTPUT_ROOT" \
        --path "$SUBMISSION_LOG_PATH" \
        --file-mode 0444
SUBMIT_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
RJOB_SUBMIT_STATUS="${SUBMIT_PIPE_STATUS[0]}"
TEE_SUBMIT_STATUS="${SUBMIT_PIPE_STATUS[1]}"
if [[ "$RJOB_SUBMIT_STATUS" -ne 0 || "$TEE_SUBMIT_STATUS" -ne 0 ]]; then
    mark_submission_unknown "rjob_or_tee_returned_nonzero_after_submission_attempt" \
        "$RJOB_SUBMIT_STATUS" "$TEE_SUBMIT_STATUS"
    echo "ERROR: ambiguous rjob submission; durable reservation retained at $SUBMISSION_RESERVATION_PATH" >&2
    exit 1
fi

set +e
SCHEDULER_RJOB_NAME="$(
    python3 - "$SUBMISSION_LOG_PATH" "$RJOB_NAME" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
matches = re.findall(r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)", text)
if len(matches) != 1 or not matches[0].startswith(f"{sys.argv[2]}-"):
    raise SystemExit(f"could not bind one actual scheduler rjob name: {matches!r}")
print(matches[0])
PY
)"
SCHEDULER_NAME_STATUS=$?
set -e
if [[ "$SCHEDULER_NAME_STATUS" -ne 0 || -z "$SCHEDULER_RJOB_NAME" ]]; then
    mark_submission_unknown "accepted_cli_result_without_parseable_scheduler_name" 0 0
    echo "ERROR: rjob returned success but its actual scheduler name was not provable" >&2
    exit 1
fi

python3 - \
    "$SCHEDULER_EXPECTED_CONTRACT_PATH" \
    "$SCHEDULER_RJOB_NAME" \
    "$RJOB_NAME" \
    "$RJOB_QUOTA_GROUP" \
    "$RJOB_IMAGE" \
    "$RJOB_CPU" \
    "$RJOB_MEMORY_MIB" \
    "$NPROC_PER_NODE" \
    "$RJOB_EXCLUDED_NODE_TAG" \
    "$COSMOS_SOURCE" \
    "$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh" \
    "$FORMAL_EXPERIMENT_ID" \
    "$FORMAL_COMMAND_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "scheduler_rjob_name": sys.argv[2],
    "showname": sys.argv[3],
    "quota_group": sys.argv[4],
    "task_type": "normal",
    "priority": 9,
    "backoff_limit": 1,
    "task_name": "t0",
    "replicas": 1,
    "image": sys.argv[5],
    "cpu": int(sys.argv[6]),
    "memory_mib": int(sys.argv[7]),
    "gpu": int(sys.argv[8]),
    "negative_tags": [sys.argv[9]],
    "private_machine": "group",
    "restart_policy": "Never",
    "termination_grace_period_seconds": 60,
    "mounts": [
        "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
        "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
    ],
    "snapshot_root": sys.argv[10],
    "wrapper_path": sys.argv[11],
    "experiment_id": sys.argv[12],
    "command_sha256": sys.argv[13],
}
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

set +e
timeout --foreground --kill-after=10s "${SCHEDULER_RECEIPT_TIMEOUT_SECONDS}s" \
    python3 "$COSMOS_SOURCE/examples/h_cluster/capture_cwam_formal_rjob_receipt.py" \
        --scheduler-rjob-name "$SCHEDULER_RJOB_NAME" \
        --expected-json "$SCHEDULER_EXPECTED_CONTRACT_PATH" \
        --output "$SCHEDULER_RECEIPT_PATH" \
        --attempts "$SCHEDULER_RECEIPT_ATTEMPTS" \
        --retry-seconds "$SCHEDULER_RECEIPT_RETRY_SECONDS" 2>&1 |
    python3 "$SECURE_TEE_HELPER" \
        --root "$FORMAL_OUTPUT_ROOT" \
        --path "$SCHEDULER_CAPTURE_LOG_PATH" \
        --file-mode 0444
SCHEDULER_CAPTURE_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
if [[ "${SCHEDULER_CAPTURE_PIPE_STATUS[0]}" -ne 0 ||
    "${SCHEDULER_CAPTURE_PIPE_STATUS[1]}" -ne 0 ]]; then
    mark_submission_unknown "accepted_cli_result_without_validated_scheduler_spec" 0 0
    echo "ERROR: scheduler spec/replica validation failed; durable reservation retained" >&2
    exit 1
fi

SCHEDULER_TEXT_RECEIPT_OK=0
for _ in {1..10}; do
    set +e
    SCHEDULER_TEXT_RECEIPT_CONTENT="$(
        timeout --foreground --kill-after=5s 20s \
            rjob get "$SCHEDULER_RJOB_NAME" 2>&1
    )"
    RJOB_GET_STATUS=$?
    set -e
    if [[ "$RJOB_GET_STATUS" -eq 0 ]] &&
        grep -Fq "rjob $SCHEDULER_RJOB_NAME (showname=$RJOB_NAME):" \
            <<<"$SCHEDULER_TEXT_RECEIPT_CONTENT" &&
        python3 "$SECURE_PATH_HELPER" write-file-exclusive \
            --root "$FORMAL_OUTPUT_ROOT" \
            --path "$SCHEDULER_TEXT_RECEIPT_PATH" \
            --file-mode 0444 \
            --text "$SCHEDULER_TEXT_RECEIPT_CONTENT" >/dev/null; then
        SCHEDULER_TEXT_RECEIPT_OK=1
        break
    fi
    sleep 1
done
if [[ "$SCHEDULER_TEXT_RECEIPT_OK" != 1 ]]; then
    mark_submission_unknown "accepted_cli_result_without_scheduler_readback" 0 0
    echo "ERROR: scheduler readback failed; durable reservation retained" >&2
    exit 1
fi

python3 - \
    "$SUBMISSION_PENDING_PATH" \
    "$SUBMISSION_RESERVATION_PATH" \
    "$SCHEDULER_RJOB_NAME" \
    "$SUBMISSION_LOG_PATH" \
    "$SCHEDULER_EXPECTED_CONTRACT_PATH" \
    "$SCHEDULER_CAPTURE_LOG_PATH" \
    "$SCHEDULER_RECEIPT_PATH" \
    "$SCHEDULER_TEXT_RECEIPT_PATH" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

pending = Path(sys.argv[1])
target = Path(sys.argv[2])
if pending.is_symlink() or not pending.is_file() or pending.stat().st_mode & 0o777 != 0o444:
    raise SystemExit(f"pending submission record is missing, mutable, or unsafe: {pending}")
payload = json.loads(pending.read_text())
if payload.get("status") != "submitting":
    raise SystemExit(f"unexpected reservation state after accepted submission: {payload!r}")
submission_log = Path(sys.argv[4])
scheduler_expected_contract = Path(sys.argv[5])
scheduler_capture_log = Path(sys.argv[6])
scheduler_receipt = Path(sys.argv[7])
scheduler_text_receipt = Path(sys.argv[8])
payload.update(
    {
        "status": "accepted",
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "scheduler_rjob_name": sys.argv[3],
        "submission_log": str(submission_log),
        "submission_log_sha256": hashlib.sha256(submission_log.read_bytes()).hexdigest(),
        "scheduler_expected_contract": str(scheduler_expected_contract),
        "scheduler_expected_contract_sha256": hashlib.sha256(
            scheduler_expected_contract.read_bytes()
        ).hexdigest(),
        "scheduler_capture_log": str(scheduler_capture_log),
        "scheduler_capture_log_sha256": hashlib.sha256(
            scheduler_capture_log.read_bytes()
        ).hexdigest(),
        "scheduler_receipt": str(scheduler_receipt),
        "scheduler_receipt_sha256": hashlib.sha256(scheduler_receipt.read_bytes()).hexdigest(),
        "scheduler_text_receipt": str(scheduler_text_receipt),
        "scheduler_text_receipt_sha256": hashlib.sha256(
            scheduler_text_receipt.read_bytes()
        ).hexdigest(),
    }
)
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o444,
)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
PY
echo "FORMAL_SUBMISSION_ACCEPTED scheduler_rjob=$SCHEDULER_RJOB_NAME reservation=$SUBMISSION_RESERVATION_PATH"
