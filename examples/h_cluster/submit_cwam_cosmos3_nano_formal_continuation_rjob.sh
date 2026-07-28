#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Submit one normal-priority, single-node, eight-H200 formal continuation.
# Selection is read only from a checksum-pinned passed throughput matrix.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly PINNED_FORMAL_OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal"
readonly SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
readonly NPROC_REQUIRED=8
readonly RJOB_MEMORY_MIB=1500000

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable formal snapshot.}"
: "${FORMAL_EXPERIMENT_ID:?Set FORMAL_EXPERIMENT_ID.}"
: "${FORMAL_RUN_NAME:=$FORMAL_EXPERIMENT_ID}"
: "${FORMAL_MANIFEST_PATH:?Set FORMAL_MANIFEST_PATH.}"
: "${FORMAL_MANIFEST_SHA256:?Set FORMAL_MANIFEST_SHA256.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_WORKFLOW_SNAPSHOT_SHA256:?Set EXPECTED_WORKFLOW_SNAPSHOT_SHA256.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${GPFS_LATENT_CACHE_ROOT:?Set GPFS_LATENT_CACHE_ROOT.}"
: "${EXPECTED_CACHE_SUCCESS_SHA256:?Set EXPECTED_CACHE_SUCCESS_SHA256.}"
: "${EXPECTED_CACHE_MANIFEST_SHA256:?Set EXPECTED_CACHE_MANIFEST_SHA256.}"
: "${THROUGHPUT_MATRIX_SUMMARY:?Set THROUGHPUT_MATRIX_SUMMARY.}"
: "${EXPECTED_THROUGHPUT_MATRIX_SHA256:?Set EXPECTED_THROUGHPUT_MATRIX_SHA256.}"
: "${R6_ITER1000_CHECKPOINT_PATH:?Set R6_ITER1000_CHECKPOINT_PATH.}"
: "${EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256:?Set EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256.}"
: "${RJOB_NAME:?Set a unique lowercase H-cluster rjob show name.}"
: "${RJOB_PREDICT_ONLY:=1}"
: "${RJOB_IMAGE:=$PINNED_CONTAINER_IMAGE}"
: "${RJOB_QUOTA_GROUP:=$PINNED_QUOTA_GROUP}"
: "${RJOB_EXCLUDED_NODE_TAG:=$PINNED_EXCLUDED_NODE_TAG}"
: "${FORMAL_OUTPUT_ROOT:=$PINNED_FORMAL_OUTPUT_ROOT}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"

[[ "$FORMAL_RUN_NAME" == "$FORMAL_EXPERIMENT_ID" ]] ||
    { echo "ERROR: FORMAL_RUN_NAME must equal FORMAL_EXPERIMENT_ID" >&2; exit 1; }
[[ "$FORMAL_EXPERIMENT_ID" =~ ^CWAM-E35-[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: formal continuation must use the CWAM-E35 experiment family" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    { echo "ERROR: invalid RJOB_NAME" >&2; exit 1; }
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] ||
    { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: invalid EXPECTED_SOURCE_COMMIT" >&2; exit 1; }
for digest in \
    "$FORMAL_MANIFEST_SHA256" \
    "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$EXPECTED_CACHE_SUCCESS_SHA256" \
    "$EXPECTED_CACHE_MANIFEST_SHA256" \
    "$EXPECTED_THROUGHPUT_MATRIX_SHA256" \
    "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
        { echo "ERROR: expected identities must be lowercase SHA-256 digests" >&2; exit 1; }
done
[[ "$RJOB_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
    { echo "ERROR: RJOB_IMAGE differs from the audited digest" >&2; exit 1; }
[[ "$RJOB_QUOTA_GROUP" == "$PINNED_QUOTA_GROUP" ]] ||
    { echo "ERROR: RJOB_QUOTA_GROUP differs from the audited quota group" >&2; exit 1; }
[[ "$RJOB_EXCLUDED_NODE_TAG" == "$PINNED_EXCLUDED_NODE_TAG" ]] ||
    { echo "ERROR: RJOB_EXCLUDED_NODE_TAG differs from the audited exclusion" >&2; exit 1; }
[[ "$FORMAL_OUTPUT_ROOT" == "$PINNED_FORMAL_OUTPUT_ROOT" ]] ||
    { echo "ERROR: FORMAL_OUTPUT_ROOT differs from the audited root" >&2; exit 1; }

SUBMITTER="$COSMOS_SOURCE/examples/h_cluster/submit_cwam_cosmos3_nano_formal_continuation_rjob.sh"
RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_nano_formal_continuation.sh"
CONTINUATION_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_cosmos3_formal_continuation.py"
TRAINING_EVIDENCE="$COSMOS_SOURCE/examples/h_cluster/cwam_formal_training_evidence.py"
PROVENANCE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"
for path in \
    "$SUBMITTER" "$RUNNER" "$CONTINUATION_HELPER" "$TRAINING_EVIDENCE" \
    "$PROVENANCE_HELPER" "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" "$FORMAL_MANIFEST_PATH"; do
    test -s "$path"
done
[[ "$(realpath -e -- "${BASH_SOURCE[0]}")" == "$(realpath -e -- "$SUBMITTER")" ]] ||
    { echo "ERROR: submitter must execute from the verified COSMOS_SOURCE snapshot" >&2; exit 1; }
command -v python3 >/dev/null
command -v rjob >/dev/null
command -v sha256sum >/dev/null
python3 "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null

EXPECTED_RUNNER_SHA256="$(sha256sum "$RUNNER" | awk '{print $1}')"
EXPECTED_CONTINUATION_HELPER_SHA256="$(sha256sum "$CONTINUATION_HELPER" | awk '{print $1}')"
EXPECTED_TRAINING_EVIDENCE_SHA256="$(sha256sum "$TRAINING_EVIDENCE" | awk '{print $1}')"
INPUT_CONTRACT_DIR="$(mktemp -d)"
INPUT_CONTRACT="$INPUT_CONTRACT_DIR/continuation-inputs.json"
trap 'rm -f -- "$INPUT_CONTRACT"; rmdir -- "$INPUT_CONTRACT_DIR" 2>/dev/null || true' EXIT
python3 "$CONTINUATION_HELPER" validate-inputs \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --run-name "$FORMAL_RUN_NAME" \
    --world-size "$NPROC_REQUIRED" \
    --cache-root "$GPFS_LATENT_CACHE_ROOT" \
    --cache-success-sha256 "$EXPECTED_CACHE_SUCCESS_SHA256" \
    --cache-manifest-sha256 "$EXPECTED_CACHE_MANIFEST_SHA256" \
    --matrix-summary "$THROUGHPUT_MATRIX_SUMMARY" \
    --matrix-summary-sha256 "$EXPECTED_THROUGHPUT_MATRIX_SHA256" \
    --resume-checkpoint "$R6_ITER1000_CHECKPOINT_PATH" \
    --resume-checkpoint-fingerprint-sha256 "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256" \
    --output "$INPUT_CONTRACT" >/dev/null
read -r SELECTED_WORKERS SELECTED_STORAGE SELECTED_AC BEST_CELL_ID < <(
    python3 -c \
        'import json,sys; p=json.load(open(sys.argv[1]))["throughput"]; print(p["workers_per_rank"], p["storage"], p["activation_checkpointing"], p["best_observed_cell_id"])' \
        "$INPUT_CONTRACT"
)
if [[ "$SELECTED_WORKERS" == 12 || "$SELECTED_WORKERS" == 16 ]]; then
    RJOB_CPU=160
else
    RJOB_CPU=128
fi
python3 "$PROVENANCE_HELPER" verify-formal-contract \
    --snapshot-root "$COSMOS_SOURCE" \
    --snapshot-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    --formal-manifest "$FORMAL_MANIFEST_PATH" \
    --formal-manifest-sha256 "$FORMAL_MANIFEST_SHA256" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --expected-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-workflow-snapshot-sha256 "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    --nproc-per-node "$NPROC_REQUIRED" \
    --expected-container-image "$RJOB_IMAGE" \
    --dataset-root "$ROBOCASA365_ROOT" \
    --source-root "$ROBOCASA365_SOURCE_ROOT" \
    --base-checkpoint "$BASE_CHECKPOINT_PATH" \
    --wan-vae "$WAN_VAE_PATH" \
    --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
    --environment "$COSMOS_ENV" \
    --expected-rjob-cpu "$RJOB_CPU" \
    --expected-formal-wrapper run_cwam_cosmos3_nano_formal_continuation.sh >/dev/null

mkdir -p "$FORMAL_OUTPUT_ROOT"
FORMAL_OUTPUT_ROOT="$(realpath -e -- "$FORMAL_OUTPUT_ROOT")"
RESERVATION_ROOT="$FORMAL_OUTPUT_ROOT/formal_continuation_submission_reservations"
RESERVATION_DIR="$RESERVATION_ROOT/$FORMAL_EXPERIMENT_ID"
PENDING_PATH="$RESERVATION_DIR/submission-pending.json"
ACCEPTED_PATH="$RESERVATION_DIR/submission.json"
SUBMISSION_LOG="$RESERVATION_DIR/rjob-submit.log"
SCHEDULER_READBACK="$RESERVATION_DIR/rjob-get.log"

printf -v Q_COSMOS_SOURCE %q "$COSMOS_SOURCE"
printf -v Q_FORMAL_EXPERIMENT_ID %q "$FORMAL_EXPERIMENT_ID"
printf -v Q_FORMAL_RUN_NAME %q "$FORMAL_RUN_NAME"
printf -v Q_FORMAL_MANIFEST_PATH %q "$FORMAL_MANIFEST_PATH"
printf -v Q_FORMAL_MANIFEST_SHA256 %q "$FORMAL_MANIFEST_SHA256"
printf -v Q_EXPECTED_SOURCE_COMMIT %q "$EXPECTED_SOURCE_COMMIT"
printf -v Q_EXPECTED_WORKFLOW_SNAPSHOT_SHA256 %q "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256"
printf -v Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256 %q "$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
printf -v Q_RJOB_IMAGE %q "$RJOB_IMAGE"
printf -v Q_EXPECTED_RUNNER_SHA256 %q "$EXPECTED_RUNNER_SHA256"
printf -v Q_EXPECTED_CONTINUATION_HELPER_SHA256 %q "$EXPECTED_CONTINUATION_HELPER_SHA256"
printf -v Q_EXPECTED_TRAINING_EVIDENCE_SHA256 %q "$EXPECTED_TRAINING_EVIDENCE_SHA256"
printf -v Q_RJOB_NAME %q "$RJOB_NAME"
printf -v Q_RJOB_CPU %q "$RJOB_CPU"
printf -v Q_SELECTED_WORKERS %q "$SELECTED_WORKERS"
printf -v Q_SELECTED_STORAGE %q "$SELECTED_STORAGE"
printf -v Q_SELECTED_AC %q "$SELECTED_AC"
printf -v Q_CACHE_ROOT %q "$GPFS_LATENT_CACHE_ROOT"
printf -v Q_CACHE_SUCCESS_SHA256 %q "$EXPECTED_CACHE_SUCCESS_SHA256"
printf -v Q_CACHE_MANIFEST_SHA256 %q "$EXPECTED_CACHE_MANIFEST_SHA256"
printf -v Q_MATRIX_SUMMARY %q "$THROUGHPUT_MATRIX_SUMMARY"
printf -v Q_MATRIX_SHA256 %q "$EXPECTED_THROUGHPUT_MATRIX_SHA256"
printf -v Q_R6_CHECKPOINT %q "$R6_ITER1000_CHECKPOINT_PATH"
printf -v Q_R6_FINGERPRINT %q "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256"
printf -v Q_COSMOS_ENV %q "$COSMOS_ENV"
printf -v Q_ROBOCASA365_ROOT %q "$ROBOCASA365_ROOT"
printf -v Q_ROBOCASA365_SOURCE_ROOT %q "$ROBOCASA365_SOURCE_ROOT"
printf -v Q_BASE_CHECKPOINT %q "$BASE_CHECKPOINT_PATH"
printf -v Q_WAN_VAE %q "$WAN_VAE_PATH"
printf -v Q_QWEN %q "$QWEN3_VL_TOKENIZER_PATH"
printf -v Q_RUNNER %q "$RUNNER"

FORMAL_COMMAND="$(
    printf '%s\n' \
        "cd $Q_COSMOS_SOURCE" \
        "export COSMOS_SOURCE=$Q_COSMOS_SOURCE" \
        "export FORMAL_EXPERIMENT_ID=$Q_FORMAL_EXPERIMENT_ID" \
        "export FORMAL_RUN_NAME=$Q_FORMAL_RUN_NAME" \
        "export FORMAL_MANIFEST_PATH=$Q_FORMAL_MANIFEST_PATH" \
        "export FORMAL_MANIFEST_SHA256=$Q_FORMAL_MANIFEST_SHA256" \
        "export EXPECTED_SOURCE_COMMIT=$Q_EXPECTED_SOURCE_COMMIT" \
        "export EXPECTED_WORKFLOW_SNAPSHOT_SHA256=$Q_EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
        "export EXPECTED_SNAPSHOT_MANIFEST_SHA256=$Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
        "export EXPECTED_CONTAINER_IMAGE=$Q_RJOB_IMAGE" \
        "export EXPECTED_RUNNER_SHA256=$Q_EXPECTED_RUNNER_SHA256" \
        "export EXPECTED_CONTINUATION_HELPER_SHA256=$Q_EXPECTED_CONTINUATION_HELPER_SHA256" \
        "export EXPECTED_TRAINING_EVIDENCE_SHA256=$Q_EXPECTED_TRAINING_EVIDENCE_SHA256" \
        "export RJOB_NAME=$Q_RJOB_NAME" \
        "export EXPECTED_RJOB_CPU=$Q_RJOB_CPU" \
        "export EXPECTED_SELECTED_WORKERS=$Q_SELECTED_WORKERS" \
        "export EXPECTED_SELECTED_STORAGE=$Q_SELECTED_STORAGE" \
        "export EXPECTED_SELECTED_ACTIVATION_CHECKPOINTING=$Q_SELECTED_AC" \
        "export GPFS_LATENT_CACHE_ROOT=$Q_CACHE_ROOT" \
        "export EXPECTED_CACHE_SUCCESS_SHA256=$Q_CACHE_SUCCESS_SHA256" \
        "export EXPECTED_CACHE_MANIFEST_SHA256=$Q_CACHE_MANIFEST_SHA256" \
        "export THROUGHPUT_MATRIX_SUMMARY=$Q_MATRIX_SUMMARY" \
        "export EXPECTED_THROUGHPUT_MATRIX_SHA256=$Q_MATRIX_SHA256" \
        "export R6_ITER1000_CHECKPOINT_PATH=$Q_R6_CHECKPOINT" \
        "export EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256=$Q_R6_FINGERPRINT" \
        "export COSMOS_ENV=$Q_COSMOS_ENV" \
        "export ROBOCASA365_ROOT=$Q_ROBOCASA365_ROOT" \
        "export ROBOCASA365_SOURCE_ROOT=$Q_ROBOCASA365_SOURCE_ROOT" \
        "export BASE_CHECKPOINT_PATH=$Q_BASE_CHECKPOINT" \
        "export WAN_VAE_PATH=$Q_WAN_VAE" \
        "export QWEN3_VL_TOKENIZER_PATH=$Q_QWEN" \
        "export NPROC_PER_NODE=8" \
        "bash $Q_RUNNER"
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
    --replicas 1
    --restart-policy never
    --termination-grace-period-seconds 60
    --backoff_limit 1
    --image "$RJOB_IMAGE"
    --cpu "$RJOB_CPU"
    --gpu "$NPROC_REQUIRED"
    --memory "$RJOB_MEMORY_MIB"
    --negative-tags "$RJOB_EXCLUDED_NODE_TAG"
    --mount
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
)
if [[ "$SELECTED_STORAGE" == nvme ]]; then
    SUBMIT_ARGS+=(--store-host-nvme)
fi
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    SUBMIT_ARGS+=(--predict-only true)
fi

echo "RJOB_NAME=$RJOB_NAME"
echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "BEST_OBSERVED_CELL_ID=$BEST_CELL_ID"
echo "SELECTED_WORKERS_PER_RANK=$SELECTED_WORKERS"
echo "SELECTED_STORAGE=$SELECTED_STORAGE"
echo "SELECTED_ACTIVATION_CHECKPOINTING=$SELECTED_AC"
echo "RJOB_CPU=$RJOB_CPU"
echo "RJOB_GPU=$NPROC_REQUIRED"
echo "RJOB_MEMORY_MIB=$RJOB_MEMORY_MIB"
echo "STORE_HOST_NVME=$([[ "$SELECTED_STORAGE" == nvme ]] && echo 1 || echo 0)"
echo "FORMAL_COMMAND_SHA256=$FORMAL_COMMAND_SHA256"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    rjob "${SUBMIT_ARGS[@]}" -- \
        /usr/bin/env -u BASH_ENV -u ENV /usr/bin/bash --noprofile --norc -c "$FORMAL_COMMAND"
    echo "FORMAL_CONTINUATION_SUBMISSION_PREDICT_ONLY reservation_created=false"
    exit 0
fi

mkdir -p "$RESERVATION_ROOT"
if ! mkdir "$RESERVATION_DIR"; then
    echo "ERROR: duplicate continuation submission reservation: $RESERVATION_DIR" >&2
    exit 1
fi
python3 - \
    "$PENDING_PATH" "$FORMAL_EXPERIMENT_ID" "$RJOB_NAME" "$RJOB_CPU" \
    "$SELECTED_WORKERS" "$SELECTED_STORAGE" "$SELECTED_AC" "$BEST_CELL_ID" \
    "$EXPECTED_CACHE_SUCCESS_SHA256" "$EXPECTED_CACHE_MANIFEST_SHA256" \
    "$EXPECTED_THROUGHPUT_MATRIX_SHA256" "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256" \
    "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$RJOB_IMAGE" "$FORMAL_COMMAND_SHA256" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": 1,
    "status": "submitting",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": sys.argv[2],
    "rjob_show_name": sys.argv[3],
    "cpu": int(sys.argv[4]),
    "gpu": 8,
    "memory_mib": 1_500_000,
    "workers_per_rank": int(sys.argv[5]),
    "storage": sys.argv[6],
    "activation_checkpointing": sys.argv[7],
    "best_observed_cell_id": sys.argv[8],
    "cache_success_sha256": sys.argv[9],
    "cache_manifest_sha256": sys.argv[10],
    "throughput_matrix_sha256": sys.argv[11],
    "r6_checkpoint_fingerprint_sha256": sys.argv[12],
    "source_commit": sys.argv[13],
    "snapshot_manifest_sha256": sys.argv[14],
    "container_image": sys.argv[15],
    "command_sha256": sys.argv[16],
    "store_host_nvme": sys.argv[6] == "nvme",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

set +e
rjob "${SUBMIT_ARGS[@]}" -- \
    /usr/bin/env -u BASH_ENV -u ENV /usr/bin/bash --noprofile --norc -c "$FORMAL_COMMAND" \
    2>&1 | tee "$SUBMISSION_LOG"
SUBMIT_STATUS=("${PIPESTATUS[@]}")
set -e
if [[ "${SUBMIT_STATUS[0]}" -ne 0 || "${SUBMIT_STATUS[1]}" -ne 0 ]]; then
    echo "ERROR: rjob submission failed or became ambiguous; reservation retained" >&2
    exit 1
fi
SCHEDULER_RJOB_NAME="$(
    python3 - "$SUBMISSION_LOG" "$RJOB_NAME" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
matches = re.findall(r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)", text)
if len(matches) != 1 or not matches[0].startswith(sys.argv[2] + "-"):
    raise SystemExit(f"cannot bind one scheduler rjob name: {matches!r}")
print(matches[0])
PY
)"
READBACK_OK=0
for _ in {1..10}; do
    set +e
    READBACK="$(rjob get "$SCHEDULER_RJOB_NAME" 2>&1)"
    READBACK_STATUS=$?
    set -e
    if [[ "$READBACK_STATUS" -eq 0 ]]; then
        printf '%s\n' "$READBACK" >"$SCHEDULER_READBACK"
        READBACK_OK=1
        break
    fi
    sleep 1
done
[[ "$READBACK_OK" == 1 ]] ||
    { echo "ERROR: accepted rjob has no scheduler readback" >&2; exit 1; }
python3 - \
    "$PENDING_PATH" "$ACCEPTED_PATH" "$SCHEDULER_RJOB_NAME" \
    "$SUBMISSION_LOG" "$SCHEDULER_READBACK" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

pending, target = Path(sys.argv[1]), Path(sys.argv[2])
payload = json.loads(pending.read_text())
payload.update({
    "status": "accepted",
    "accepted_at": datetime.now(timezone.utc).isoformat(),
    "scheduler_rjob_name": sys.argv[3],
    "submission_log": sys.argv[4],
    "submission_log_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "scheduler_readback": sys.argv[5],
    "scheduler_readback_sha256": hashlib.sha256(Path(sys.argv[5]).read_bytes()).hexdigest(),
})
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
chmod 0444 "$PENDING_PATH" "$ACCEPTED_PATH" "$SUBMISSION_LOG" "$SCHEDULER_READBACK"
echo "FORMAL_CONTINUATION_SUBMISSION_ACCEPTED scheduler_rjob=$SCHEDULER_RJOB_NAME reservation=$ACCEPTED_PATH"
