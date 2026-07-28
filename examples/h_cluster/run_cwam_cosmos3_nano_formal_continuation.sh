#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run one formal eight-H200 continuation from the exact R6 iter-1000 state.
# The throughput matrix selects loader workers, cache storage, and activation
# checkpointing.  The benchmark-only fixed-index trace is explicitly removed.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_FORMAL_OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal"
readonly PINNED_FORMAL_LOG_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/logs/cwam-cosmos3-formal"
readonly SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
readonly NPROC_REQUIRED=8

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable formal snapshot.}"
: "${FORMAL_EXPERIMENT_ID:?Set FORMAL_EXPERIMENT_ID.}"
: "${FORMAL_RUN_NAME:=$FORMAL_EXPERIMENT_ID}"
: "${FORMAL_MANIFEST_PATH:?Set FORMAL_MANIFEST_PATH.}"
: "${FORMAL_MANIFEST_SHA256:?Set FORMAL_MANIFEST_SHA256.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_WORKFLOW_SNAPSHOT_SHA256:?Set EXPECTED_WORKFLOW_SNAPSHOT_SHA256.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${EXPECTED_CONTAINER_IMAGE:=$PINNED_CONTAINER_IMAGE}"
: "${EXPECTED_RUNNER_SHA256:?Set EXPECTED_RUNNER_SHA256.}"
: "${EXPECTED_CONTINUATION_HELPER_SHA256:?Set EXPECTED_CONTINUATION_HELPER_SHA256.}"
: "${EXPECTED_TRAINING_EVIDENCE_SHA256:?Set EXPECTED_TRAINING_EVIDENCE_SHA256.}"
: "${RJOB_NAME:?Set RJOB_NAME.}"
: "${EXPECTED_RJOB_CPU:?Set EXPECTED_RJOB_CPU from the selected throughput cell.}"
: "${EXPECTED_SELECTED_WORKERS:?Set EXPECTED_SELECTED_WORKERS.}"
: "${EXPECTED_SELECTED_STORAGE:?Set EXPECTED_SELECTED_STORAGE.}"
: "${EXPECTED_SELECTED_ACTIVATION_CHECKPOINTING:?Set EXPECTED_SELECTED_ACTIVATION_CHECKPOINTING.}"
: "${GPFS_LATENT_CACHE_ROOT:?Set GPFS_LATENT_CACHE_ROOT.}"
: "${EXPECTED_CACHE_SUCCESS_SHA256:?Set EXPECTED_CACHE_SUCCESS_SHA256.}"
: "${EXPECTED_CACHE_MANIFEST_SHA256:?Set EXPECTED_CACHE_MANIFEST_SHA256.}"
: "${THROUGHPUT_MATRIX_SUMMARY:?Set THROUGHPUT_MATRIX_SUMMARY.}"
: "${EXPECTED_THROUGHPUT_MATRIX_SHA256:?Set EXPECTED_THROUGHPUT_MATRIX_SHA256.}"
: "${R6_ITER1000_CHECKPOINT_PATH:?Set R6_ITER1000_CHECKPOINT_PATH.}"
: "${EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256:?Set EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256.}"
: "${NPROC_PER_NODE:=8}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${FORMAL_OUTPUT_ROOT:=$PINNED_FORMAL_OUTPUT_ROOT}"
: "${FORMAL_LOG_ROOT:=$PINNED_FORMAL_LOG_ROOT}"
: "${NVME_MOUNT_ROOT:=/nvme}"

[[ "$NPROC_PER_NODE" == "$NPROC_REQUIRED" ]] ||
    { echo "ERROR: formal continuation requires exactly eight ranks" >&2; exit 1; }
[[ "$FORMAL_RUN_NAME" == "$FORMAL_EXPERIMENT_ID" ]] ||
    { echo "ERROR: FORMAL_RUN_NAME must equal FORMAL_EXPERIMENT_ID" >&2; exit 1; }
[[ "$FORMAL_EXPERIMENT_ID" =~ ^CWAM-E35-[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: formal continuation must use the CWAM-E35 experiment family" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    { echo "ERROR: invalid RJOB_NAME" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: invalid EXPECTED_SOURCE_COMMIT" >&2; exit 1; }
for digest in \
    "$FORMAL_MANIFEST_SHA256" \
    "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$EXPECTED_RUNNER_SHA256" \
    "$EXPECTED_CONTINUATION_HELPER_SHA256" \
    "$EXPECTED_TRAINING_EVIDENCE_SHA256" \
    "$EXPECTED_CACHE_SUCCESS_SHA256" \
    "$EXPECTED_CACHE_MANIFEST_SHA256" \
    "$EXPECTED_THROUGHPUT_MATRIX_SHA256" \
    "$EXPECTED_R6_CHECKPOINT_FINGERPRINT_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
        { echo "ERROR: all expected identities must be lowercase SHA-256 digests" >&2; exit 1; }
done
[[ "$EXPECTED_CONTAINER_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
    { echo "ERROR: container image differs from the audited digest" >&2; exit 1; }
[[ "$FORMAL_OUTPUT_ROOT" == "$PINNED_FORMAL_OUTPUT_ROOT" ]] ||
    { echo "ERROR: FORMAL_OUTPUT_ROOT differs from the audited root" >&2; exit 1; }
[[ "$FORMAL_LOG_ROOT" == "$PINNED_FORMAL_LOG_ROOT" ]] ||
    { echo "ERROR: FORMAL_LOG_ROOT differs from the audited root" >&2; exit 1; }
[[ "$EXPECTED_RJOB_CPU" == 128 || "$EXPECTED_RJOB_CPU" == 160 ]] ||
    { echo "ERROR: EXPECTED_RJOB_CPU must be 128 or 160" >&2; exit 1; }
[[ "$EXPECTED_SELECTED_WORKERS" == 4 || "$EXPECTED_SELECTED_WORKERS" == 8 ||
   "$EXPECTED_SELECTED_WORKERS" == 12 || "$EXPECTED_SELECTED_WORKERS" == 16 ]] ||
    { echo "ERROR: invalid selected worker count" >&2; exit 1; }
if [[ "$EXPECTED_SELECTED_WORKERS" == 12 || "$EXPECTED_SELECTED_WORKERS" == 16 ]]; then
    [[ "$EXPECTED_RJOB_CPU" == 160 ]] ||
        { echo "ERROR: workers 12/16 require the full 160-CPU request" >&2; exit 1; }
else
    [[ "$EXPECTED_RJOB_CPU" == 128 ]] ||
        { echo "ERROR: workers 4/8 use the audited 128-CPU request" >&2; exit 1; }
fi
[[ "$EXPECTED_SELECTED_STORAGE" == gpfs || "$EXPECTED_SELECTED_STORAGE" == nvme ]] ||
    { echo "ERROR: selected storage must be gpfs or nvme" >&2; exit 1; }
[[ "$EXPECTED_SELECTED_ACTIVATION_CHECKPOINTING" == selective ||
   "$EXPECTED_SELECTED_ACTIVATION_CHECKPOINTING" == none ]] ||
    { echo "ERROR: invalid selected activation-checkpointing mode" >&2; exit 1; }

RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_nano_formal_continuation.sh"
CONTINUATION_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_cosmos3_formal_continuation.py"
THROUGHPUT_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_cosmos3_latent_cache_throughput.py"
TRAINING_EVIDENCE="$COSMOS_SOURCE/examples/h_cluster/cwam_formal_training_evidence.py"
PROVENANCE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"
SECURE_TEE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_secure_tee.py"
LAUNCHER="$COSMOS_SOURCE/examples/launch_sft_action_policy_robocasa365_nano.sh"
PYTHON="$COSMOS_ENV/bin/python"
TORCHRUN="$COSMOS_ENV/bin/torchrun"
for path in \
    "$RUNNER" "$CONTINUATION_HELPER" "$THROUGHPUT_HELPER" "$TRAINING_EVIDENCE" \
    "$PROVENANCE_HELPER" "$SECURE_TEE_HELPER" "$LAUNCHER" \
    "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" "$FORMAL_MANIFEST_PATH"; do
    test -s "$path"
done
test -x "$PYTHON"
test -x "$TORCHRUN"
[[ "$(realpath -e -- "${BASH_SOURCE[0]}")" == "$(realpath -e -- "$RUNNER")" ]] ||
    { echo "ERROR: continuation runner must execute from COSMOS_SOURCE" >&2; exit 1; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$EXPECTED_RUNNER_SHA256" ]] ||
    { echo "ERROR: continuation runner SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256sum "$CONTINUATION_HELPER" | awk '{print $1}')" == "$EXPECTED_CONTINUATION_HELPER_SHA256" ]] ||
    { echo "ERROR: continuation helper SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256sum "$TRAINING_EVIDENCE" | awk '{print $1}')" == "$EXPECTED_TRAINING_EVIDENCE_SHA256" ]] ||
    { echo "ERROR: training-evidence helper SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256sum "$FORMAL_MANIFEST_PATH" | awk '{print $1}')" == "$FORMAL_MANIFEST_SHA256" ]] ||
    { echo "ERROR: formal manifest SHA-256 mismatch" >&2; exit 1; }

"$PYTHON" "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null
"$PYTHON" "$PROVENANCE_HELPER" verify-formal-contract \
    --snapshot-root "$COSMOS_SOURCE" \
    --snapshot-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    --formal-manifest "$FORMAL_MANIFEST_PATH" \
    --formal-manifest-sha256 "$FORMAL_MANIFEST_SHA256" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --expected-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-workflow-snapshot-sha256 "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    --nproc-per-node "$NPROC_REQUIRED" \
    --expected-container-image "$EXPECTED_CONTAINER_IMAGE" \
    --dataset-root "$ROBOCASA365_ROOT" \
    --source-root "$ROBOCASA365_SOURCE_ROOT" \
    --base-checkpoint "$BASE_CHECKPOINT_PATH" \
    --wan-vae "$WAN_VAE_PATH" \
    --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
    --environment "$COSMOS_ENV" \
    --expected-rjob-cpu "$EXPECTED_RJOB_CPU" \
    --expected-formal-wrapper run_cwam_cosmos3_nano_formal_continuation.sh >/dev/null
if [[ -e "$COSMOS_SOURCE/.git" ]]; then
    [[ "$(git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$EXPECTED_SOURCE_COMMIT" ]] ||
        { echo "ERROR: COSMOS_SOURCE commit mismatch" >&2; exit 1; }
fi

export PATH="$COSMOS_ENV/bin:$PATH"
export PYTHONPATH="$COSMOS_SOURCE"
export COSMOS_DEVICE=cuda COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NPROC_PER_NODE
source "$COSMOS_SOURCE/examples/h_cluster/configure_torchcodec_runtime.sh"
"$PYTHON" "$COSMOS_SOURCE/examples/h_cluster/validate_cu128_train_env_marker.py" \
    --environment "$COSMOS_ENV"

mkdir -p "$FORMAL_OUTPUT_ROOT" "$FORMAL_LOG_ROOT"
FORMAL_OUTPUT_ROOT="$(realpath -e -- "$FORMAL_OUTPUT_ROOT")"
FORMAL_LOG_ROOT="$(realpath -e -- "$FORMAL_LOG_ROOT")"
EVIDENCE_PARENT="$FORMAL_OUTPUT_ROOT/formal_continuation_evidence"
EVIDENCE_DIR="$EVIDENCE_PARENT/$FORMAL_RUN_NAME"
mkdir -p "$EVIDENCE_PARENT"
if ! mkdir "$EVIDENCE_DIR"; then
    echo "ERROR: continuation evidence directory already exists: $EVIDENCE_DIR" >&2
    exit 1
fi
CONTRACT_PATH="$EVIDENCE_DIR/continuation-inputs.json"
FORMAL_LOG="$FORMAL_LOG_ROOT/${FORMAL_RUN_NAME}.continuation.log"
SUMMARY_PATH="$EVIDENCE_DIR/summary.json"
RUN_DIR="$FORMAL_OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft/$FORMAL_RUN_NAME"
CURRENT_PHASE=input_validation
FINAL_STATUS=failed

record_failure() {
    local exit_code=$?
    trap - EXIT
    if [[ "$FINAL_STATUS" != passed && -d "$EVIDENCE_DIR" ]]; then
        "$PYTHON" - "$SUMMARY_PATH" "$FORMAL_EXPERIMENT_ID" "$CURRENT_PHASE" "$exit_code" <<'PY' || true
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "failed",
    "mode": "continuation",
    "experiment_id": sys.argv[2],
    "failed_phase": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n")
PY
    fi
    exit "$exit_code"
}
trap record_failure EXIT

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
    --output "$CONTRACT_PATH" >/dev/null
CONTRACT_SHA256="$(sha256sum "$CONTRACT_PATH" | awk '{print $1}')"
read -r SELECTED_WORKERS SELECTED_STORAGE SELECTED_AC < <(
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1]))["throughput"]; print(p["workers_per_rank"], p["storage"], p["activation_checkpointing"])' \
        "$CONTRACT_PATH"
)
[[ "$SELECTED_WORKERS" == "$EXPECTED_SELECTED_WORKERS" &&
   "$SELECTED_STORAGE" == "$EXPECTED_SELECTED_STORAGE" &&
   "$SELECTED_AC" == "$EXPECTED_SELECTED_ACTIVATION_CHECKPOINTING" ]] ||
    { echo "ERROR: runtime throughput selection differs from the submit-side contract" >&2; exit 1; }

CURRENT_PHASE=node_validation
"$PYTHON" "$THROUGHPUT_HELPER" capture-node \
    --output "$EVIDENCE_DIR/rjob-node-eight-h200.json" \
    --expected-gpus "$NPROC_REQUIRED" \
    --minimum-cpus "$EXPECTED_RJOB_CPU" \
    --requested-cpus "$EXPECTED_RJOB_CPU" \
    --rjob-show-name "$RJOB_NAME" >/dev/null

if [[ "$SELECTED_STORAGE" == nvme ]]; then
    CURRENT_PHASE=nvme_stage
    [[ "$NVME_MOUNT_ROOT" == /nvme ]] ||
        { echo "ERROR: H host-NVMe must be mounted exactly at /nvme" >&2; exit 1; }
    [[ "$(findmnt -n -T /nvme -o TARGET)" == /nvme ]] ||
        { echo "ERROR: /nvme is not a distinct mount target" >&2; exit 1; }
    [[ "$(findmnt -n -T /nvme -o SOURCE)" == /dev/nvme1n1 ]] ||
        { echo "ERROR: /nvme is not backed by /dev/nvme1n1" >&2; exit 1; }
    [[ "$(findmnt -n -T /nvme -o FSTYPE)" == xfs ]] ||
        { echo "ERROR: /nvme is not xfs" >&2; exit 1; }
    NVME_STAGE_PARENT="/nvme/.cwam-cosmos3-formal-continuation/$FORMAL_RUN_NAME"
    NVME_CACHE_ROOT="$NVME_STAGE_PARENT/cache"
    mkdir -p "$NVME_STAGE_PARENT"
    [[ ! -e "$NVME_CACHE_ROOT" && ! -L "$NVME_CACHE_ROOT" ]] ||
        { echo "ERROR: run-specific NVMe cache stage already exists" >&2; exit 1; }
    STAGE_PORT_HASH="$(printf '%s' "$FORMAL_RUN_NAME-nvme-stage" | cksum)"
    STAGE_PORT_HASH="${STAGE_PORT_HASH%% *}"
    STAGE_MASTER_PORT="$((20000 + STAGE_PORT_HASH % 10000))"
    "$TORCHRUN" \
        --standalone \
        --nproc_per_node="$NPROC_REQUIRED" \
        --master_port="$STAGE_MASTER_PORT" \
        "$THROUGHPUT_HELPER" stage-nvme \
        --source "$GPFS_LATENT_CACHE_ROOT" \
        --destination "$NVME_CACHE_ROOT" \
        --evidence-dir "$EVIDENCE_DIR/nvme-stage"
    RUNTIME_LATENT_CACHE_ROOT="$NVME_CACHE_ROOT"
else
    CURRENT_PHASE=gpfs_cache_validation
    "$PYTHON" "$THROUGHPUT_HELPER" validate-cache \
        --root "$GPFS_LATENT_CACHE_ROOT" \
        --output "$EVIDENCE_DIR/gpfs-latent-cache-validation.json" >/dev/null
    RUNTIME_LATENT_CACHE_ROOT="$GPFS_LATENT_CACHE_ROOT"
fi

CURRENT_PHASE=run_prepare
"$PYTHON" "$CONTINUATION_HELPER" prepare-run \
    --output-root "$FORMAL_OUTPUT_ROOT" \
    --run-name "$FORMAL_RUN_NAME" \
    --contract "$CONTRACT_PATH" \
    --contract-sha256 "$CONTRACT_SHA256" >"$EVIDENCE_DIR/run-prepare.json"

# The fixed trace belongs only to the throughput benchmark.  Formal training
# must regain its normal iterable-shuffle stream, with replay explicitly
# acknowledged because the R6 DCP has no dataloader cursor.
unset ROBOCASA365_FIXED_INDEX_TRACE_PATH
export ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT
export BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export ROBOCASA365_LATENT_CACHE_ROOT="$RUNTIME_LATENT_CACHE_ROOT"
export ROBOCASA365_LATENT_CACHE_MODE=required
export CWAM_METADATA_ONLY_VIDEO_TOKENIZER=1
export ROBOCASA365_TRAIN_NUM_WORKERS="$SELECTED_WORKERS"
export ROBOCASA365_VAL_NUM_WORKERS=2
export CWAM_ACTIVATION_CHECKPOINTING_MODE="$SELECTED_AC"
export OUTPUT_ROOT="$FORMAL_OUTPUT_ROOT"
export IMAGINAIRE_OUTPUT_ROOT="$FORMAL_OUTPUT_ROOT"
export RUN_NAME="$FORMAL_RUN_NAME"
export RESUME_EXISTING_RUN=1
export ALLOW_DATA_STREAM_REPLAY=1
export LOG_FILENAME="${FORMAL_RUN_NAME}.continuation.torchrun.log"
export EXTRA_TAIL_OVERRIDES="
optimizer.lr=5.0e-5
scheduler.warm_up_steps=[500]
scheduler.cycle_lengths=[16000]
trainer.max_iter=10000
trainer.max_val_iter=8
trainer.validation_iter=500
trainer.logging_iter=50
checkpoint.save_iter=500
checkpoint.dcp_async_mode_enabled=false
checkpoint.load_training_state=true
checkpoint.keys_to_skip_loading=[]
checkpoint.keys_not_to_resume=[dataloader]
checkpoint.strict_resume=true
trainer.callbacks.action_gradient_evidence.every_n=50
trainer.callbacks.production_batch_memory_gate.required=true
"

CURRENT_PHASE=formal_training
set +e
bash "$LAUNCHER" 2>&1 |
    "$PYTHON" "$SECURE_TEE_HELPER" \
        --root "$FORMAL_LOG_ROOT" \
        --path "$FORMAL_LOG" \
        --file-mode 0444
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
[[ "${PIPE_STATUS[0]}" -eq 0 ]] ||
    { echo "ERROR: continuation training failed with exit ${PIPE_STATUS[0]}" >&2; exit "${PIPE_STATUS[0]}"; }
[[ "${PIPE_STATUS[1]}" -eq 0 ]] ||
    { echo "ERROR: continuation log capture failed with exit ${PIPE_STATUS[1]}" >&2; exit "${PIPE_STATUS[1]}"; }

CURRENT_PHASE=formal_evidence
"$PYTHON" "$TRAINING_EVIDENCE" verify-formal \
    --mode continuation \
    --run-dir "$RUN_DIR" \
    --log "$FORMAL_LOG" \
    --world-size "$NPROC_REQUIRED" \
    --formal-run-name "$FORMAL_RUN_NAME" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --continuation-contract "$CONTRACT_PATH" \
    --continuation-contract-sha256 "$CONTRACT_SHA256" \
    --output-json "$SUMMARY_PATH"

FINAL_STATUS=passed
CURRENT_PHASE=complete
echo "CWAM_COSMOS3_FORMAL_CONTINUATION_COMPLETED summary=$SUMMARY_PATH control_success_claim=false"
