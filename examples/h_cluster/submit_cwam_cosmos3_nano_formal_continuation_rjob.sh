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
readonly SUBMISSION_TIMEOUT_SECONDS=180
readonly SCHEDULER_CAPTURE_TIMEOUT_SECONDS=180
readonly SCHEDULER_CAPTURE_ATTEMPTS=90
readonly SCHEDULER_CAPTURE_RETRY_SECONDS=1
readonly PERSONAL_GPFS_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS_MOUNT="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable formal snapshot.}"
: "${FORMAL_EXPERIMENT_ID:?Set FORMAL_EXPERIMENT_ID.}"
: "${FORMAL_EXPERIMENT_MODE:?Set FORMAL_EXPERIMENT_MODE to exploratory or confirmatory.}"
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
: "${EXPECTED_BENCHMARK_SOURCE_ROOT:?Set the immutable throughput benchmark snapshot root.}"
: "${EXPECTED_BENCHMARK_SOURCE_COMMIT:?Set the throughput benchmark source commit.}"
: "${EXPECTED_BENCHMARK_SNAPSHOT_MANIFEST_SHA256:?Set the throughput benchmark snapshot digest.}"
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
: "${EXPECTED_WAN_VAE_SHA256:=20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"

[[ "$FORMAL_RUN_NAME" == "$FORMAL_EXPERIMENT_ID" ]] ||
    { echo "ERROR: FORMAL_RUN_NAME must equal FORMAL_EXPERIMENT_ID" >&2; exit 1; }
[[ "$FORMAL_EXPERIMENT_ID" =~ ^CWAM-E35-[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: formal continuation must use the CWAM-E35 experiment family" >&2; exit 1; }
[[ "$FORMAL_EXPERIMENT_MODE" == exploratory ||
   "$FORMAL_EXPERIMENT_MODE" == confirmatory ]] ||
    { echo "ERROR: FORMAL_EXPERIMENT_MODE must be exploratory or confirmatory" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    { echo "ERROR: invalid RJOB_NAME" >&2; exit 1; }
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] ||
    { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: invalid EXPECTED_SOURCE_COMMIT" >&2; exit 1; }
[[ "$EXPECTED_BENCHMARK_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: invalid EXPECTED_BENCHMARK_SOURCE_COMMIT" >&2; exit 1; }
for digest in \
    "$FORMAL_MANIFEST_SHA256" \
    "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$EXPECTED_CACHE_SUCCESS_SHA256" \
    "$EXPECTED_CACHE_MANIFEST_SHA256" \
    "$EXPECTED_THROUGHPUT_MATRIX_SHA256" \
    "$EXPECTED_BENCHMARK_SNAPSHOT_MANIFEST_SHA256" \
    "$EXPECTED_WAN_VAE_SHA256" \
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
SCHEDULER_CAPTURE_HELPER="$COSMOS_SOURCE/examples/h_cluster/capture_cwam_formal_rjob_receipt.py"
PYTHON="$COSMOS_ENV/bin/python"
SYSTEM_PYTHON="$(command -v python3)"
for path in \
    "$SUBMITTER" "$RUNNER" "$CONTINUATION_HELPER" "$TRAINING_EVIDENCE" \
    "$PROVENANCE_HELPER" "$SCHEDULER_CAPTURE_HELPER" \
    "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" "$FORMAL_MANIFEST_PATH"; do
    test -s "$path"
done
[[ "$(realpath -e -- "${BASH_SOURCE[0]}")" == "$(realpath -e -- "$SUBMITTER")" ]] ||
    { echo "ERROR: submitter must execute from the verified COSMOS_SOURCE snapshot" >&2; exit 1; }
test -x "$PYTHON"
test -x "$SYSTEM_PYTHON"
command -v rjob >/dev/null
command -v sha256sum >/dev/null
command -v timeout >/dev/null
command -v tee >/dev/null
"$PYTHON" "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null

EXPECTED_RUNNER_SHA256="$(sha256sum "$RUNNER" | awk '{print $1}')"
EXPECTED_CONTINUATION_HELPER_SHA256="$(sha256sum "$CONTINUATION_HELPER" | awk '{print $1}')"
EXPECTED_TRAINING_EVIDENCE_SHA256="$(sha256sum "$TRAINING_EVIDENCE" | awk '{print $1}')"
INPUT_CONTRACT_DIR="$(mktemp -d)"
INPUT_CONTRACT="$INPUT_CONTRACT_DIR/continuation-inputs.json"
trap 'rm -f -- "$INPUT_CONTRACT"; rmdir -- "$INPUT_CONTRACT_DIR" 2>/dev/null || true' EXIT
"$PYTHON" "$CONTINUATION_HELPER" validate-inputs \
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
    --expected-benchmark-source-root "$EXPECTED_BENCHMARK_SOURCE_ROOT" \
    --expected-benchmark-source-commit "$EXPECTED_BENCHMARK_SOURCE_COMMIT" \
    --expected-benchmark-snapshot-manifest-sha256 "$EXPECTED_BENCHMARK_SNAPSHOT_MANIFEST_SHA256" \
    --wan-vae-path "$WAN_VAE_PATH" \
    --expected-wan-vae-sha256 "$EXPECTED_WAN_VAE_SHA256" \
    --require-torch-dcp-metadata \
    --output "$INPUT_CONTRACT" >/dev/null
read -r SELECTED_WORKERS SELECTED_STORAGE SELECTED_AC BEST_CELL_ID < <(
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1]))["throughput"]; print(p["workers_per_rank"], p["storage"], p["activation_checkpointing"], p["best_observed_cell_id"])' \
        "$INPUT_CONTRACT"
)
if [[ "$SELECTED_WORKERS" == 12 || "$SELECTED_WORKERS" == 16 ]]; then
    RJOB_CPU=160
else
    RJOB_CPU=128
fi
"$PYTHON" "$PROVENANCE_HELPER" verify-formal-contract \
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
    --expected-formal-wrapper run_cwam_cosmos3_nano_formal_continuation.sh \
    --expected-experiment-mode "$FORMAL_EXPERIMENT_MODE" >/dev/null
rm -f -- "$INPUT_CONTRACT"
rmdir -- "$INPUT_CONTRACT_DIR"
trap - EXIT

mkdir -p "$FORMAL_OUTPUT_ROOT"
FORMAL_OUTPUT_ROOT="$(realpath -e -- "$FORMAL_OUTPUT_ROOT")"
RESERVATION_ROOT="$FORMAL_OUTPUT_ROOT/formal_continuation_submission_reservations"
RESERVATION_DIR="$RESERVATION_ROOT/$FORMAL_EXPERIMENT_ID"
PENDING_PATH="$RESERVATION_DIR/submission-pending.json"
ACCEPTED_PATH="$RESERVATION_DIR/submission.json"
SUBMISSION_UNKNOWN="$RESERVATION_DIR/submission-unknown.json"
SUBMISSION_LOG="$RESERVATION_DIR/rjob-submit.log"
SCHEDULER_EXPECTED_CONTRACT="$RESERVATION_DIR/rjob-expected-contract.json"
SCHEDULER_CAPTURE_LOG="$RESERVATION_DIR/rjob-scheduler-capture.log"
SCHEDULER_STRUCTURED_RECEIPT="$RESERVATION_DIR/rjob-scheduler-receipt.json"
SCHEDULER_READBACK="$RESERVATION_DIR/rjob-get.log"
SUBMISSION_NONCE="$("$PYTHON" -I -c 'import secrets; print(secrets.token_hex(32))')"
[[ "$SUBMISSION_NONCE" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: failed to generate submission nonce" >&2; exit 1; }
FORMAL_SUBMISSION_RESERVATION_PATH="$ACCEPTED_PATH"

printf -v Q_COSMOS_SOURCE %q "$COSMOS_SOURCE"
printf -v Q_FORMAL_EXPERIMENT_ID %q "$FORMAL_EXPERIMENT_ID"
printf -v Q_FORMAL_EXPERIMENT_MODE %q "$FORMAL_EXPERIMENT_MODE"
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
printf -v Q_SUBMISSION_NONCE %q "$SUBMISSION_NONCE"
printf -v Q_SUBMISSION_RESERVATION %q "$FORMAL_SUBMISSION_RESERVATION_PATH"
printf -v Q_RJOB_CPU %q "$RJOB_CPU"
printf -v Q_SELECTED_WORKERS %q "$SELECTED_WORKERS"
printf -v Q_SELECTED_STORAGE %q "$SELECTED_STORAGE"
printf -v Q_SELECTED_AC %q "$SELECTED_AC"
printf -v Q_CACHE_ROOT %q "$GPFS_LATENT_CACHE_ROOT"
printf -v Q_CACHE_SUCCESS_SHA256 %q "$EXPECTED_CACHE_SUCCESS_SHA256"
printf -v Q_CACHE_MANIFEST_SHA256 %q "$EXPECTED_CACHE_MANIFEST_SHA256"
printf -v Q_MATRIX_SUMMARY %q "$THROUGHPUT_MATRIX_SUMMARY"
printf -v Q_MATRIX_SHA256 %q "$EXPECTED_THROUGHPUT_MATRIX_SHA256"
printf -v Q_BENCHMARK_SOURCE_ROOT %q "$EXPECTED_BENCHMARK_SOURCE_ROOT"
printf -v Q_BENCHMARK_SOURCE_COMMIT %q "$EXPECTED_BENCHMARK_SOURCE_COMMIT"
printf -v Q_BENCHMARK_SNAPSHOT_SHA256 %q "$EXPECTED_BENCHMARK_SNAPSHOT_MANIFEST_SHA256"
printf -v Q_R6_CHECKPOINT %q "$R6_ITER1000_CHECKPOINT_PATH"
printf -v Q_R6_FINGERPRINT %q "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256"
printf -v Q_COSMOS_ENV %q "$COSMOS_ENV"
printf -v Q_ROBOCASA365_ROOT %q "$ROBOCASA365_ROOT"
printf -v Q_ROBOCASA365_SOURCE_ROOT %q "$ROBOCASA365_SOURCE_ROOT"
printf -v Q_BASE_CHECKPOINT %q "$BASE_CHECKPOINT_PATH"
printf -v Q_WAN_VAE %q "$WAN_VAE_PATH"
printf -v Q_WAN_VAE_SHA256 %q "$EXPECTED_WAN_VAE_SHA256"
printf -v Q_QWEN %q "$QWEN3_VL_TOKENIZER_PATH"
printf -v Q_RUNNER %q "$RUNNER"

FORMAL_COMMAND="$(
    printf '%s\n' \
        "set -Eeuo pipefail" \
        "unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE PYTHONPATH PYTHONUSERBASE PYTHONBREAKPOINT LD_PRELOAD LD_AUDIT" \
        "cd $Q_COSMOS_SOURCE" \
        "export COSMOS_SOURCE=$Q_COSMOS_SOURCE" \
        "export FORMAL_EXPERIMENT_ID=$Q_FORMAL_EXPERIMENT_ID" \
        "export FORMAL_EXPERIMENT_MODE=$Q_FORMAL_EXPERIMENT_MODE" \
        "export FORMAL_RUN_NAME=$Q_FORMAL_RUN_NAME" \
        "export FORMAL_SUBMISSION_NONCE=$Q_SUBMISSION_NONCE" \
        "export FORMAL_SUBMISSION_RESERVATION_PATH=$Q_SUBMISSION_RESERVATION" \
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
        "export EXPECTED_BENCHMARK_SOURCE_ROOT=$Q_BENCHMARK_SOURCE_ROOT" \
        "export EXPECTED_BENCHMARK_SOURCE_COMMIT=$Q_BENCHMARK_SOURCE_COMMIT" \
        "export EXPECTED_BENCHMARK_SNAPSHOT_MANIFEST_SHA256=$Q_BENCHMARK_SNAPSHOT_SHA256" \
        "export R6_ITER1000_CHECKPOINT_PATH=$Q_R6_CHECKPOINT" \
        "export EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256=$Q_R6_FINGERPRINT" \
        "export COSMOS_ENV=$Q_COSMOS_ENV" \
        "export ROBOCASA365_ROOT=$Q_ROBOCASA365_ROOT" \
        "export ROBOCASA365_SOURCE_ROOT=$Q_ROBOCASA365_SOURCE_ROOT" \
        "export BASE_CHECKPOINT_PATH=$Q_BASE_CHECKPOINT" \
        "export WAN_VAE_PATH=$Q_WAN_VAE" \
        "export EXPECTED_WAN_VAE_SHA256=$Q_WAN_VAE_SHA256" \
        "export QWEN3_VL_TOKENIZER_PATH=$Q_QWEN" \
        "export NPROC_PER_NODE=8" \
        "exec bash $Q_RUNNER"
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
    "$PERSONAL_GPFS_MOUNT"
    "$PUBLIC_GPFS_MOUNT"
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
echo "SUBMISSION_NONCE_SHA256=$(printf %s "$SUBMISSION_NONCE" | sha256sum | awk '{print $1}')"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    rjob "${SUBMIT_ARGS[@]}" -- bash -lc "$FORMAL_COMMAND"
    echo "FORMAL_CONTINUATION_SUBMISSION_PREDICT_ONLY reservation_created=false"
    exit 0
fi

mkdir -p "$RESERVATION_ROOT"
RESERVATION_ROOT="$(realpath -e -- "$RESERVATION_ROOT")"
[[ "$RESERVATION_ROOT" == "$FORMAL_OUTPUT_ROOT/formal_continuation_submission_reservations" ]] ||
    { echo "ERROR: non-canonical continuation reservation root" >&2; exit 1; }
if ! mkdir --mode=0700 -- "$RESERVATION_DIR"; then
    echo "ERROR: duplicate continuation submission reservation: $RESERVATION_DIR" >&2
    exit 1
fi
"$PYTHON" - \
    "$PENDING_PATH" "$FORMAL_EXPERIMENT_ID" "$RJOB_NAME" "$RJOB_CPU" \
    "$SELECTED_WORKERS" "$SELECTED_STORAGE" "$SELECTED_AC" "$BEST_CELL_ID" \
    "$EXPECTED_CACHE_SUCCESS_SHA256" "$EXPECTED_CACHE_MANIFEST_SHA256" \
    "$EXPECTED_THROUGHPUT_MATRIX_SHA256" "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256" \
    "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$RJOB_IMAGE" "$FORMAL_COMMAND_SHA256" "$SUBMISSION_NONCE" \
    "$FORMAL_SUBMISSION_RESERVATION_PATH" "$PERSONAL_GPFS_MOUNT" \
    "$PUBLIC_GPFS_MOUNT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": 2,
    "kind": "cwam_cosmos3_formal_continuation_submission_reservation",
    "status": "submitting",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": sys.argv[2],
    "run_name": sys.argv[2],
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
    "submission_nonce": sys.argv[17],
    "submission_reservation_path": sys.argv[18],
    "mounts": [sys.argv[19], sys.argv[20]],
}
target = Path(sys.argv[1])
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

mark_submission_unknown() {
    local reason="$1"
    local scheduler_name="${2:-}"
    local artifact
    trap - ERR EXIT INT TERM
    if ! "$PYTHON" - \
        "$PENDING_PATH" "$ACCEPTED_PATH" "$SUBMISSION_UNKNOWN" \
        "$reason" "$scheduler_name" \
        "$SUBMISSION_LOG" "$SCHEDULER_EXPECTED_CONTRACT" \
        "$SCHEDULER_CAPTURE_LOG" "$SCHEDULER_STRUCTURED_RECEIPT" \
        "$SCHEDULER_READBACK" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

pending_path = Path(sys.argv[1])
accepted_path = Path(sys.argv[2])
target = Path(sys.argv[3])
if pending_path.is_symlink() or not pending_path.is_file():
    raise SystemExit("submission pending receipt is missing or unsafe")
if accepted_path.is_symlink() or (
    accepted_path.exists() and not accepted_path.is_file()
):
    raise SystemExit("accepted submission target exists and is unsafe")
if target.is_symlink() or (target.exists() and not target.is_file()):
    raise SystemExit("submission-unknown target already exists and is unsafe")
if accepted_path.exists():
    accepted_path.unlink()
if target.exists():
    existing = json.loads(target.read_text())
    if (
        existing.get("kind")
        != "cwam_cosmos3_formal_continuation_submission_unknown"
        or existing.get("status") != "submission_unknown"
    ):
        raise SystemExit("existing submission-unknown marker has invalid semantics")
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    raise SystemExit(0)
payload = {
    "schema_version": 1,
    "kind": "cwam_cosmos3_formal_continuation_submission_unknown",
    "status": "submission_unknown",
    "submission_unknown_at": datetime.now(timezone.utc).isoformat(),
    "submission_unknown_reason": sys.argv[4],
    "scheduler_rjob_name": sys.argv[5] or None,
    "submission_reservation": {
        "path": str(pending_path),
        "sha256": hashlib.sha256(pending_path.read_bytes()).hexdigest(),
    },
    "available_artifacts": {},
}
for literal in sys.argv[6:]:
    path = Path(literal)
    if path.is_file() and not path.is_symlink():
        payload["available_artifacts"][str(path)] = {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
    then
        echo "CRITICAL: failed to persist submission-unknown marker; reservation remains unsealed: $RESERVATION_DIR" >&2
        return 125
    fi
    for artifact in \
        "$PENDING_PATH" "$SUBMISSION_LOG" "$SCHEDULER_EXPECTED_CONTRACT" \
        "$SCHEDULER_CAPTURE_LOG" "$SCHEDULER_STRUCTURED_RECEIPT" \
        "$SCHEDULER_READBACK" "$SUBMISSION_UNKNOWN"; do
        if [[ -e "$artifact" || -L "$artifact" ]]; then
            if [[ ! -f "$artifact" || -L "$artifact" ]] ||
                ! chmod 0444 "$artifact"; then
                echo "CRITICAL: cannot make submission artifact immutable: $artifact" >&2
                return 125
            fi
        fi
    done
    if ! "$PYTHON" - "$SUBMISSION_UNKNOWN" "$PENDING_PATH" "$ACCEPTED_PATH" <<'PY'
import json
import stat
import sys
from pathlib import Path

unknown = Path(sys.argv[1])
pending = Path(sys.argv[2])
accepted = Path(sys.argv[3])
try:
    accepted.lstat()
except FileNotFoundError:
    pass
else:
    raise SystemExit("accepted and submission-unknown receipts must be mutually exclusive")
for path in (unknown, pending):
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit(f"submission artifact is not regular immutable mode 0444: {path}")
payload = json.loads(unknown.read_text())
if (
    payload.get("schema_version") != 1
    or payload.get("kind")
    != "cwam_cosmos3_formal_continuation_submission_unknown"
    or payload.get("status") != "submission_unknown"
):
    raise SystemExit("submission-unknown marker failed semantic verification")
PY
    then
        echo "CRITICAL: submission-unknown marker verification failed; reservation remains unsealed" >&2
        return 125
    fi
    if ! chmod 0555 "$RESERVATION_DIR" ||
        [[ "$(stat -c %a "$RESERVATION_DIR")" != 555 ]]; then
        echo "CRITICAL: failed to seal submission-unknown reservation directory" >&2
        return 125
    fi
}

SUBMISSION_ATTEMPTED=0
POST_ACCEPT_COMPLETE=0
submission_attempt_exit() {
    local status="$1"
    trap - ERR EXIT INT TERM
    if [[ "$SUBMISSION_ATTEMPTED" == 1 &&
        "$POST_ACCEPT_COMPLETE" != 1 &&
        ! -e "$SUBMISSION_UNKNOWN" ]]; then
        if ! mark_submission_unknown \
            "unexpected_post_accept_exit" \
            "${SCHEDULER_RJOB_NAME:-}"; then
            exit 125
        fi
    fi
    exit "$status"
}
submission_attempt_failure() {
    local status="$1"
    trap - ERR EXIT INT TERM
    if [[ "$SUBMISSION_ATTEMPTED" == 1 &&
        "$POST_ACCEPT_COMPLETE" != 1 &&
        ! -e "$SUBMISSION_UNKNOWN" ]]; then
        if ! mark_submission_unknown \
            "unexpected_post_accept_failure" \
            "${SCHEDULER_RJOB_NAME:-}"; then
            exit 125
        fi
    fi
    exit "$status"
}
submission_attempt_signal() {
    local signal_name="$1"
    local status="$2"
    trap - ERR EXIT INT TERM
    if [[ "$SUBMISSION_ATTEMPTED" == 1 &&
        "$POST_ACCEPT_COMPLETE" != 1 &&
        ! -e "$SUBMISSION_UNKNOWN" ]]; then
        if ! mark_submission_unknown \
            "submission_attempt_interrupted_${signal_name,,}" \
            "${SCHEDULER_RJOB_NAME:-}"; then
            exit 125
        fi
    fi
    exit "$status"
}
trap 'submission_attempt_exit "$?"' EXIT
trap 'submission_attempt_failure "$?"' ERR
trap 'submission_attempt_signal INT 130' INT
trap 'submission_attempt_signal TERM 143' TERM
SUBMISSION_ATTEMPTED=1

trap - ERR
set +e
timeout --foreground --kill-after=10s "${SUBMISSION_TIMEOUT_SECONDS}s" \
    rjob "${SUBMIT_ARGS[@]}" -- bash -lc "$FORMAL_COMMAND" \
    2>&1 | tee "$SUBMISSION_LOG"
SUBMIT_STATUS=("${PIPESTATUS[@]}")
set -e
trap 'submission_attempt_failure "$?"' ERR
if [[ "${SUBMIT_STATUS[0]}" -ne 0 || "${SUBMIT_STATUS[1]}" -ne 0 ]]; then
    if ! mark_submission_unknown "rjob_submit_failed_or_ambiguous"; then
        exit 125
    fi
    echo "ERROR: rjob submission failed or became ambiguous; submission is unknown" >&2
    exit 1
fi

chmod 0444 "$SUBMISSION_LOG"

trap - ERR
set +e
SCHEDULER_RJOB_NAME="$(
    "$PYTHON" - "$SUBMISSION_LOG" "$RJOB_NAME" <<'PY'
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
PARSE_STATUS=$?
set -e
trap 'submission_attempt_failure "$?"' ERR
if [[ "$PARSE_STATUS" -ne 0 || -z "$SCHEDULER_RJOB_NAME" ]]; then
    if ! mark_submission_unknown \
        "accepted_cli_without_unique_scheduler_name"; then
        exit 125
    fi
    echo "ERROR: cannot bind one scheduler RJob name" >&2
    exit 1
fi

"$PYTHON" - \
    "$SCHEDULER_EXPECTED_CONTRACT" "$SCHEDULER_RJOB_NAME" "$RJOB_NAME" \
    "$RJOB_QUOTA_GROUP" "$RJOB_IMAGE" "$RJOB_CPU" "$RJOB_EXCLUDED_NODE_TAG" \
    "$PERSONAL_GPFS_MOUNT" "$PUBLIC_GPFS_MOUNT" "$COSMOS_SOURCE" "$RUNNER" \
    "$FORMAL_EXPERIMENT_ID" "$FORMAL_COMMAND_SHA256" "$SUBMISSION_NONCE" \
    "$FORMAL_SUBMISSION_RESERVATION_PATH" "$SELECTED_STORAGE" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema_version": 2,
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
    "memory_mib": 1_500_000,
    "gpu": 8,
    "negative_tags": [sys.argv[7]],
    "private_machine": "group",
    "restart_policy": "Never",
    "termination_grace_period_seconds": 60,
    "mounts": [sys.argv[8], sys.argv[9]],
    "snapshot_root": sys.argv[10],
    "wrapper_path": sys.argv[11],
    "experiment_id": sys.argv[12],
    "command_sha256": sys.argv[13],
    "submission_nonce": sys.argv[14],
    "submission_reservation_path": sys.argv[15],
    "store_host_nvme": sys.argv[16] == "nvme",
}
target = Path(sys.argv[1])
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

trap - ERR
set +e
timeout --foreground --kill-after=10s "${SCHEDULER_CAPTURE_TIMEOUT_SECONDS}s" \
    "$SYSTEM_PYTHON" "$SCHEDULER_CAPTURE_HELPER" \
        --scheduler-rjob-name "$SCHEDULER_RJOB_NAME" \
        --expected-json "$SCHEDULER_EXPECTED_CONTRACT" \
        --output "$SCHEDULER_STRUCTURED_RECEIPT" \
        --attempts "$SCHEDULER_CAPTURE_ATTEMPTS" \
        --retry-seconds "$SCHEDULER_CAPTURE_RETRY_SECONDS" \
        >"$SCHEDULER_CAPTURE_LOG" 2>&1
CAPTURE_STATUS=$?
set -e
trap 'submission_attempt_failure "$?"' ERR
chmod 0444 "$SCHEDULER_EXPECTED_CONTRACT" "$SCHEDULER_CAPTURE_LOG"
if [[ "$CAPTURE_STATUS" -ne 0 || ! -s "$SCHEDULER_STRUCTURED_RECEIPT" ]]; then
    if ! mark_submission_unknown \
        "accepted_cli_without_validated_scheduler_crd_and_replica" \
        "$SCHEDULER_RJOB_NAME"; then
        exit 125
    fi
    echo "ERROR: scheduler CRD/Replica validation failed" >&2
    exit 1
fi

READBACK_OK=0
READBACK=
for _ in {1..10}; do
    trap - ERR
    set +e
    READBACK="$(
        timeout --foreground --kill-after=5s 20s \
            rjob get "$SCHEDULER_RJOB_NAME" 2>&1
    )"
    READBACK_STATUS=$?
    set -e
    trap 'submission_attempt_failure "$?"' ERR
    if [[ "$READBACK_STATUS" -eq 0 &&
        "$READBACK" == *"rjob $SCHEDULER_RJOB_NAME (showname=$RJOB_NAME):"* ]]; then
        printf '%s\n' "$READBACK" >"$SCHEDULER_READBACK"
        READBACK_OK=1
        break
    fi
    sleep 1
done
if [[ "$READBACK_OK" != 1 ]]; then
    printf '%s\n' "$READBACK" >"$SCHEDULER_READBACK"
    if ! mark_submission_unknown \
        "validated_scheduler_without_exact_text_readback" \
        "$SCHEDULER_RJOB_NAME"; then
        exit 125
    fi
    echo "ERROR: accepted RJob has no exact text readback" >&2
    exit 1
fi
chmod 0444 "$SCHEDULER_READBACK"

"$PYTHON" - \
    "$PENDING_PATH" "$ACCEPTED_PATH" "$SCHEDULER_RJOB_NAME" \
    "$SUBMISSION_LOG" "$SCHEDULER_EXPECTED_CONTRACT" \
    "$SCHEDULER_CAPTURE_LOG" "$SCHEDULER_STRUCTURED_RECEIPT" \
    "$SCHEDULER_READBACK" "$SUBMISSION_UNKNOWN" <<'PY'
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

pending, target = Path(sys.argv[1]), Path(sys.argv[2])
unknown = Path(sys.argv[9])
try:
    unknown.lstat()
except FileNotFoundError:
    pass
else:
    raise SystemExit("cannot publish accepted receipt beside submission-unknown")
payload = json.loads(pending.read_text())
structured_path = Path(sys.argv[7])
structured = json.loads(structured_path.read_text())
expected = json.loads(Path(sys.argv[5]).read_text())
if structured.get("kind") != "cwam_formal_rjob_scheduler_receipt":
    raise SystemExit("structured scheduler receipt kind mismatch")
if structured.get("validation") != {
    "status": "passed",
    "expected_contract_schema_version": 2,
}:
    raise SystemExit("structured scheduler receipt validation mismatch")
metadata = structured.get("metadata", {})
if metadata.get("name") != sys.argv[3] or metadata.get("showname") != payload["rjob_show_name"]:
    raise SystemExit("structured scheduler identity mismatch")
replicas = structured.get("replicas")
if not isinstance(replicas, list) or len(replicas) != 1:
    raise SystemExit("structured scheduler receipt must contain exactly one Replica")
task = structured.get("spec", {}).get("task", {})
container = task.get("container", {})
host_nvme = {
    "enabled": payload["store_host_nvme"],
    "annotation": "kubebrain.brainpp.cn/writeread-hostpath-mounts",
    "value": (
        "/nvme:/mnt/nvme1n1,/cfs:/mnt/cfs"
        if payload["store_host_nvme"]
        else None
    ),
    "container_target": "/nvme" if payload["store_host_nvme"] else None,
    "host_source": "/mnt/nvme1n1" if payload["store_host_nvme"] else None,
}
if task.get("store_host_nvme") != host_nvme:
    raise SystemExit("structured scheduler host-NVMe contract mismatch")
if replicas[0].get("worker_spec", {}).get("store_host_nvme") != host_nvme:
    raise SystemExit("expanded Replica host-NVMe contract mismatch")
if task.get("submission_nonce") != payload["submission_nonce"]:
    raise SystemExit("structured scheduler submission nonce mismatch")
if task.get("submission_reservation_path") != str(target):
    raise SystemExit("structured scheduler reservation path mismatch")
if task.get("mounts") != payload["mounts"]:
    raise SystemExit("structured scheduler mounts mismatch")
if container.get("image") != payload["container_image"]:
    raise SystemExit("structured scheduler image mismatch")
if container.get("resources") != {
    "requests_equal_core_limits": True,
    "cpu": payload["cpu"],
    "memory_mib": payload["memory_mib"],
    "gpu": payload["gpu"],
    "platform_injected_limit_keys": [],
}:
    raise SystemExit("structured scheduler resources mismatch")
command = container.get("command")
if (
    not isinstance(command, list)
    or len(command) != 3
    or command[:2] != ["bash", "-lc"]
    or hashlib.sha256(command[2].encode()).hexdigest() != payload["command_sha256"]
):
    raise SystemExit("structured scheduler command digest mismatch")
if expected.get("command_sha256") != payload["command_sha256"]:
    raise SystemExit("expected scheduler command digest mismatch")
if container.get("command_sha256") != payload["command_sha256"]:
    raise SystemExit("structured scheduler command SHA-256 field mismatch")


def descriptor(literal):
    path = Path(literal)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


payload.update({
    "status": "accepted",
    "accepted_at": datetime.now(timezone.utc).isoformat(),
    "scheduler_rjob_name": sys.argv[3],
    "scheduler_rjob_uid": metadata.get("uid"),
    "scheduler_replica_name": replicas[0].get("replica_name"),
    "scheduler_pod_name": replicas[0].get("pod_name"),
    "artifacts": {
        "submission_log": descriptor(sys.argv[4]),
        "scheduler_expected_contract": descriptor(sys.argv[5]),
        "scheduler_capture_log": descriptor(sys.argv[6]),
        "scheduler_structured_receipt": descriptor(sys.argv[7]),
        "scheduler_text_readback": descriptor(sys.argv[8]),
    },
})
temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
descriptor_fd = os.open(
    temporary,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
published = False
try:
    with os.fdopen(descriptor_fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        unknown.lstat()
    except FileNotFoundError:
        pass
    else:
        raise SystemExit("submission-unknown appeared before accepted publication")
    os.link(temporary, target, follow_symlinks=False)
    published = True
    directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except Exception:
    if published:
        target.unlink(missing_ok=True)
    raise
finally:
    temporary.unlink(missing_ok=True)
PY
if [[ -e "$SUBMISSION_UNKNOWN" || -L "$SUBMISSION_UNKNOWN" ]]; then
    if ! mark_submission_unknown \
        "accepted_receipt_conflicted_with_submission_unknown" \
        "$SCHEDULER_RJOB_NAME"; then
        exit 125
    fi
    echo "ERROR: accepted and submission-unknown receipts became ambiguous" >&2
    exit 1
fi
chmod 0444 \
    "$PENDING_PATH" "$ACCEPTED_PATH" "$SUBMISSION_LOG" \
    "$SCHEDULER_EXPECTED_CONTRACT" "$SCHEDULER_CAPTURE_LOG" \
    "$SCHEDULER_STRUCTURED_RECEIPT" "$SCHEDULER_READBACK"
chmod 0555 "$RESERVATION_DIR"
[[ ! -e "$SUBMISSION_UNKNOWN" && ! -L "$SUBMISSION_UNKNOWN" ]] ||
    { echo "ERROR: sealed accepted receipt conflicts with submission-unknown" >&2; exit 1; }
POST_ACCEPT_COMPLETE=1
trap - ERR EXIT INT TERM
echo "FORMAL_CONTINUATION_SUBMISSION_ACCEPTED scheduler_rjob=$SCHEDULER_RJOB_NAME reservation=$ACCEPTED_PATH"
