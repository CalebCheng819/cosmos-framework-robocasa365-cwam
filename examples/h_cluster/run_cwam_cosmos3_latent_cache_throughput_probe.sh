#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run the CWAM/Cosmos3-Nano Wan-latent-cache throughput probe on one H-cluster
# node with exactly eight visible H200 GPUs.
#
# Default adaptive6 protocol:
#   1. GPFS + selective AC at 4/8/12/16 workers per rank;
#   2. node-local NVMe + selective AC at the best passed worker count;
#   3. no AC at the best passed storage/worker pair.
# PROBE_MATRIX_MODE=full16 retains the complete 4x2x2 factorial protocol.
# Every cell discards 10 warm-up optimizer steps and measures the following 50.
#
# Every cell resumes the exact R6 iter-1000 model/optimizer/scheduler/trainer
# state; only the packed data stream is explicitly restarted.
# Probe checkpoints are never treated as formal artifacts and are removed after
# the per-cell metrics JSON is durable.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly NPROC_REQUIRED=8
readonly WARMUP_STEPS=10
readonly MEASURE_STEPS=50
readonly TOTAL_STEPS=60
readonly NVME_MIN_FREE_BYTES_REQUIRED=128849018880  # 120 GiB
readonly NVME_COPY_HEADROOM_BYTES=8589934592        # 8 GiB
readonly MINIMUM_EFFECTIVE_CPUS=160
readonly REQUESTED_CPUS=192
readonly OFFICIAL_WAN22_VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
readonly -a WORKER_MATRIX=(4 8 12 16)
readonly -a STORAGE_MATRIX=(gpfs nvme)
readonly -a ACTIVATION_CHECKPOINTING_MATRIX=(selective none)

RUN_ROOT=
CURRENT_PHASE=preflight
record_runner_exit() {
    local status=$?
    trap - EXIT
    if [[ -n "$RUN_ROOT" && -d "$RUN_ROOT" ]]; then
        python3 -c '
import json, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "completed" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "last_phase": sys.argv[3],
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
' "$RUN_ROOT/run-exit.json" "$status" "$CURRENT_PHASE" || true
    fi
    exit "$status"
}
trap record_runner_exit EXIT

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable Cosmos source snapshot.}"
: "${THROUGHPUT_EXPERIMENT_ID:?Set THROUGHPUT_EXPERIMENT_ID.}"
: "${THROUGHPUT_RUN_ID:?Set a unique THROUGHPUT_RUN_ID.}"
: "${THROUGHPUT_OUTPUT_ROOT:?Set the GPFS throughput output root.}"
: "${GPFS_LATENT_CACHE_ROOT:?Set GPFS_LATENT_CACHE_ROOT to a complete cache root.}"
: "${ROBOCASA365_FIXED_INDEX_TRACE_PATH:?Set ROBOCASA365_FIXED_INDEX_TRACE_PATH to the audited .npy trace.}"
: "${NVME_MOUNT_ROOT:=/nvme}"
: "${RJOB_NAME:?Set RJOB_NAME to the submit-side show name.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${EXPECTED_RUNNER_SHA256:?Set EXPECTED_RUNNER_SHA256.}"
: "${EXPECTED_HELPER_SHA256:?Set EXPECTED_HELPER_SHA256.}"
: "${NPROC_PER_NODE:=8}"
: "${PROBE_MATRIX_MODE:=adaptive6}"
: "${PROBE_RESUME_MODE:=r6_iter1000_model_optimizer}"
: "${R6_ITER1000_CHECKPOINT_PATH:?Set R6_ITER1000_CHECKPOINT_PATH to the stable R6 iter-1000 DCP.}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${ROBOCASA365_VAL_NUM_WORKERS:=2}"
: "${NVME_MIN_FREE_BYTES:=$NVME_MIN_FREE_BYTES_REQUIRED}"
: "${GPU_SAMPLE_INTERVAL_SECONDS:=2}"
: "${NO_AC_CELL_TIMEOUT_SECONDS:=3600}"

[[ "$NPROC_PER_NODE" == "$NPROC_REQUIRED" ]] ||
    { echo "ERROR: NPROC_PER_NODE must be exactly 8" >&2; exit 1; }
[[ "$WARMUP_STEPS" == 10 && "$MEASURE_STEPS" == 50 && "$TOTAL_STEPS" == 60 ]] ||
    { echo "ERROR: the probe timing contract is fixed at warmup=10 + measure=50" >&2; exit 1; }
[[ "$PROBE_MATRIX_MODE" == adaptive6 || "$PROBE_MATRIX_MODE" == full16 ]] ||
    { echo "ERROR: PROBE_MATRIX_MODE must be adaptive6 or full16" >&2; exit 1; }
[[ "$PROBE_RESUME_MODE" == r6_iter1000_model_optimizer ]] ||
    { echo "ERROR: every comparison cell must resume R6 iter1000 model/optimizer state" >&2; exit 1; }
[[ "$THROUGHPUT_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: THROUGHPUT_EXPERIMENT_ID contains unsupported characters" >&2; exit 1; }
[[ "$THROUGHPUT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: THROUGHPUT_RUN_ID contains unsupported characters" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase Git SHA" >&2; exit 1; }
[[ "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_SNAPSHOT_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$EXPECTED_RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_RUNNER_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$EXPECTED_HELPER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_HELPER_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$NVME_MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ && "$NVME_MIN_FREE_BYTES" -ge "$NVME_MIN_FREE_BYTES_REQUIRED" ]] ||
    { echo "ERROR: NVME_MIN_FREE_BYTES must be at least 120 GiB" >&2; exit 1; }
[[ "$GPU_SAMPLE_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: GPU_SAMPLE_INTERVAL_SECONDS must be a positive integer" >&2; exit 1; }
[[ "$NO_AC_CELL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ &&
   "$NO_AC_CELL_TIMEOUT_SECONDS" -ge 600 ]] ||
    { echo "ERROR: NO_AC_CELL_TIMEOUT_SECONDS must be an integer >= 600" >&2; exit 1; }
[[ "$ROBOCASA365_VAL_NUM_WORKERS" =~ ^[0-9]+$ ]] ||
    { echo "ERROR: ROBOCASA365_VAL_NUM_WORKERS must be a non-negative integer" >&2; exit 1; }
[[ "$NVME_MOUNT_ROOT" == /nvme ]] ||
    { echo "ERROR: H --store-host-nvme contract requires NVME_MOUNT_ROOT=/nvme" >&2; exit 1; }

RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_latent_cache_throughput_probe.sh"
HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_cosmos3_latent_cache_throughput.py"
LAUNCHER="$COSMOS_SOURCE/examples/launch_sft_action_policy_robocasa365_nano.sh"
PROVENANCE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"
SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
PYTHON="$COSMOS_ENV/bin/python"
TORCHRUN="$COSMOS_ENV/bin/torchrun"
[[ "$(realpath -e -- "${BASH_SOURCE[0]}")" == "$(realpath -e -- "$RUNNER")" ]] ||
    { echo "ERROR: runner must execute from COSMOS_SOURCE" >&2; exit 1; }
test -s "$HELPER"
test -s "$LAUNCHER"
test -s "$PROVENANCE_HELPER"
test -s "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
test -x "$PYTHON"
test -x "$TORCHRUN"
export PATH="$COSMOS_ENV/bin:$PATH"
[[ "$(realpath -e -- "$(command -v python)")" == "$(realpath -e -- "$PYTHON")" ]] ||
    { echo "ERROR: PATH python does not resolve to COSMOS_ENV" >&2; exit 1; }
[[ "$(realpath -e -- "$(command -v torchrun)")" == "$(realpath -e -- "$TORCHRUN")" ]] ||
    { echo "ERROR: PATH torchrun does not resolve to COSMOS_ENV" >&2; exit 1; }
command -v findmnt >/dev/null
command -v lsblk >/dev/null
command -v nvidia-smi >/dev/null
command -v sha256sum >/dev/null
command -v df >/dev/null
command -v timeout >/dev/null

ACTUAL_RUNNER_SHA256="$(sha256sum "$RUNNER" | awk '{print $1}')"
ACTUAL_HELPER_SHA256="$(sha256sum "$HELPER" | awk '{print $1}')"
[[ "$ACTUAL_RUNNER_SHA256" == "$EXPECTED_RUNNER_SHA256" ]] ||
    { echo "ERROR: runner SHA-256 differs from the submit-side contract" >&2; exit 1; }
[[ "$ACTUAL_HELPER_SHA256" == "$EXPECTED_HELPER_SHA256" ]] ||
    { echo "ERROR: helper SHA-256 differs from the submit-side contract" >&2; exit 1; }
"$PYTHON" "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null

if [[ -e "$COSMOS_SOURCE/.git" ]]; then
    ACTUAL_SOURCE_COMMIT="$(
        git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse HEAD
    )"
    [[ "$ACTUAL_SOURCE_COMMIT" == "$EXPECTED_SOURCE_COMMIT" ]] ||
        { echo "ERROR: COSMOS_SOURCE commit mismatch" >&2; exit 1; }
fi

test -d "$ROBOCASA365_ROOT"
test -d "$ROBOCASA365_SOURCE_ROOT/videos"
test -f "$BASE_CHECKPOINT_PATH/model/.metadata"
test -f "$BASE_CHECKPOINT_PATH/checkpoint.json"
test -f "$WAN_VAE_PATH"
test -f "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json"
test -d "$GPFS_LATENT_CACHE_ROOT"
ACTUAL_WAN_VAE_SHA256="$(sha256sum "$WAN_VAE_PATH" | awk '{print $1}')"
[[ "$ACTUAL_WAN_VAE_SHA256" == "$OFFICIAL_WAN22_VAE_SHA256" ]] ||
    { echo "ERROR: WAN_VAE_PATH does not match the official audited Wan2.2 VAE digest" >&2; exit 1; }

mkdir -p "$THROUGHPUT_OUTPUT_ROOT"
THROUGHPUT_OUTPUT_ROOT="$(realpath -e -- "$THROUGHPUT_OUTPUT_ROOT")"
RUN_ROOT="$THROUGHPUT_OUTPUT_ROOT/$THROUGHPUT_RUN_ID"
if ! mkdir "$RUN_ROOT"; then
    echo "ERROR: duplicate or ambiguous probe run; output already exists: $RUN_ROOT" >&2
    exit 1
fi
EVIDENCE_DIR="$RUN_ROOT/evidence"
CELLS_DIR="$RUN_ROOT/cells"
mkdir "$EVIDENCE_DIR" "$CELLS_DIR"

CURRENT_PHASE=cache_validation
CACHE_VALIDATION_JSON="$EVIDENCE_DIR/gpfs-latent-cache-validation.json"
"$PYTHON" "$HELPER" validate-cache \
    --root "$GPFS_LATENT_CACHE_ROOT" \
    --output "$CACHE_VALIDATION_JSON" >/dev/null
read -r CACHE_MANIFEST_SHA256 CACHE_TOTAL_BYTES CACHE_WAN_VAE_SHA256 < <(
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["manifest_sha256"], p["total_bytes"], p["wan_vae_sha256"])' \
        "$CACHE_VALIDATION_JSON"
)
[[ "$CACHE_WAN_VAE_SHA256" == "$ACTUAL_WAN_VAE_SHA256" ]] ||
    { echo "ERROR: cache and runtime Wan VAE digests differ" >&2; exit 1; }
CURRENT_PHASE=fixed_index_trace_prepare
TRACE_VALIDATION_JSON="$EVIDENCE_DIR/fixed-index-trace-validation.json"
"$PYTHON" "$HELPER" prepare-fixed-trace \
    --path "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" \
    --dataset-root "$ROBOCASA365_ROOT" \
    --source-root "$ROBOCASA365_SOURCE_ROOT" \
    --output "$TRACE_VALIDATION_JSON" >/dev/null
read -r FIXED_TRACE_SHA256 FIXED_TRACE_MANIFEST_SHA256 < <(
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["array_sha256"], p["manifest_sha256"])' \
        "$TRACE_VALIDATION_JSON"
)
ROBOCASA365_FIXED_INDEX_TRACE_PATH="$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' \
        "$TRACE_VALIDATION_JSON"
)"

NODE_EVIDENCE_JSON="$EVIDENCE_DIR/rjob-node-eight-h200.json"
"$PYTHON" "$HELPER" capture-node \
    --output "$NODE_EVIDENCE_JSON" \
    --expected-gpus "$NPROC_REQUIRED" \
    --minimum-cpus "$MINIMUM_EFFECTIVE_CPUS" \
    --requested-cpus "$REQUESTED_CPUS" \
    --rjob-show-name "$RJOB_NAME" >/dev/null

NVME_MOUNT_ROOT="$(realpath -e -- "$NVME_MOUNT_ROOT")"
CURRENT_PHASE=nvme_validation_and_stage
[[ -d "$NVME_MOUNT_ROOT" && -w "$NVME_MOUNT_ROOT" ]] ||
    { echo "ERROR: NVME_MOUNT_ROOT must be a writable directory" >&2; exit 1; }
NVME_MOUNT_TARGET="$(findmnt -n -T "$NVME_MOUNT_ROOT" -o TARGET)"
NVME_MOUNT_SOURCE="$(findmnt -n -T "$NVME_MOUNT_ROOT" -o SOURCE)"
NVME_MOUNT_FSTYPE="$(findmnt -n -T "$NVME_MOUNT_ROOT" -o FSTYPE)"
[[ -n "$NVME_MOUNT_TARGET" && -n "$NVME_MOUNT_SOURCE" && -n "$NVME_MOUNT_FSTYPE" ]] ||
    { echo "ERROR: findmnt could not resolve the NVMe contract" >&2; exit 1; }
NVME_MOUNT_TARGET="$(realpath -e -- "$NVME_MOUNT_TARGET")"
[[ "$NVME_MOUNT_TARGET" == /nvme ]] ||
    { echo "ERROR: H host-NVMe mount target must resolve exactly to /nvme" >&2; exit 1; }
[[ "$NVME_MOUNT_SOURCE" == /dev/nvme1n1 ]] ||
    { echo "ERROR: H host-NVMe mount source must be /dev/nvme1n1" >&2; exit 1; }
[[ "$NVME_MOUNT_FSTYPE" == xfs ]] ||
    { echo "ERROR: H host-NVMe filesystem must be xfs" >&2; exit 1; }
set +e
NVME_BLOCK_EVIDENCE="$(lsblk -sno NAME,TYPE,TRAN "$NVME_MOUNT_SOURCE" 2>&1)"
LSBLK_STATUS=$?
set -e
[[ "$LSBLK_STATUS" -eq 0 ]] ||
    { echo "ERROR: cannot trace NVMe mount source through lsblk: $NVME_BLOCK_EVIDENCE" >&2; exit 1; }
grep -Eq '(^|[[:space:]])nvme([[:space:]]|$)' <<<"$NVME_BLOCK_EVIDENCE" ||
    { echo "ERROR: mount source has no NVMe block-device evidence" >&2; exit 1; }
NVME_FREE_BYTES="$(df -B1 --output=avail "$NVME_MOUNT_ROOT" | tail -n 1 | tr -d '[:space:]')"
[[ "$NVME_FREE_BYTES" =~ ^[0-9]+$ ]] ||
    { echo "ERROR: cannot determine free bytes for NVME_MOUNT_ROOT" >&2; exit 1; }
NVME_REQUIRED_FOR_CACHE="$((CACHE_TOTAL_BYTES + NVME_COPY_HEADROOM_BYTES))"
if ((NVME_REQUIRED_FOR_CACHE < NVME_MIN_FREE_BYTES)); then
    NVME_REQUIRED_FOR_CACHE="$NVME_MIN_FREE_BYTES"
fi
((NVME_FREE_BYTES >= NVME_REQUIRED_FOR_CACHE)) ||
    {
        echo "ERROR: NVMe free capacity is insufficient: free=$NVME_FREE_BYTES required=$NVME_REQUIRED_FOR_CACHE" >&2
        exit 1
    }
"$PYTHON" -c '
import json, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "passed",
    "nvme_mount_root": sys.argv[2],
    "mount_target": sys.argv[3],
    "mount_source": sys.argv[4],
    "filesystem_type": sys.argv[5],
    "free_bytes": int(sys.argv[6]),
    "required_bytes": int(sys.argv[7]),
    "minimum_free_bytes": int(sys.argv[8]),
    "cache_total_bytes": int(sys.argv[9]),
    "lsblk_ancestry": sys.argv[10],
}
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
' \
    "$EVIDENCE_DIR/nvme-mount-and-capacity.json" \
    "$NVME_MOUNT_ROOT" \
    "$NVME_MOUNT_TARGET" \
    "$NVME_MOUNT_SOURCE" \
    "$NVME_MOUNT_FSTYPE" \
    "$NVME_FREE_BYTES" \
    "$NVME_REQUIRED_FOR_CACHE" \
    "$NVME_MIN_FREE_BYTES" \
    "$CACHE_TOTAL_BYTES" \
    "$NVME_BLOCK_EVIDENCE"

NVME_STAGE_PARENT="$NVME_MOUNT_ROOT/.cwam-cosmos3-latent-cache/$THROUGHPUT_RUN_ID"
mkdir -p "$NVME_STAGE_PARENT"
NVME_LATENT_CACHE_ROOT="$NVME_STAGE_PARENT/cache"
[[ ! -e "$NVME_LATENT_CACHE_ROOT" && ! -L "$NVME_LATENT_CACHE_ROOT" ]] ||
    { echo "ERROR: run-specific NVMe cache destination already exists" >&2; exit 1; }
STAGE_PORT_HASH="$(printf '%s' "$THROUGHPUT_RUN_ID-nvme-stage" | cksum)"
STAGE_PORT_HASH="${STAGE_PORT_HASH%% *}"
STAGE_MASTER_PORT="$((20000 + STAGE_PORT_HASH % 10000))"
"$TORCHRUN" \
    --standalone \
    --nproc_per_node="$NPROC_REQUIRED" \
    --master_port="$STAGE_MASTER_PORT" \
    "$HELPER" stage-nvme \
    --source "$GPFS_LATENT_CACHE_ROOT" \
    --destination "$NVME_LATENT_CACHE_ROOT" \
    --evidence-dir "$EVIDENCE_DIR/nvme-stage"

R6_ITER1000_CHECKPOINT_PATH="$(realpath -e -- "$R6_ITER1000_CHECKPOINT_PATH")"
[[ "$(basename "$R6_ITER1000_CHECKPOINT_PATH")" == iter_000001000 ]] ||
    { echo "ERROR: R6 resume source must be iter_000001000" >&2; exit 1; }
for component in model optim scheduler trainer; do
    test -d "$R6_ITER1000_CHECKPOINT_PATH/$component" ||
        { echo "ERROR: R6 resume source is missing $component" >&2; exit 1; }
done
[[ ! -e "$R6_ITER1000_CHECKPOINT_PATH/dataloader" ]] ||
    { echo "ERROR: R6 checkpoint unexpectedly has dataloader state; protocol expects explicit stream replay" >&2; exit 1; }

start_gpu_sampler() {
    local output="$1"
    local stderr_output="$2"
    "$PYTHON" "$HELPER" sample-gpu \
        --output "$output" \
        --interval-seconds "$GPU_SAMPLE_INTERVAL_SECONDS" \
        --expected-gpus "$NPROC_REQUIRED" \
        >"${output%.ndjson}.sampler.stdout" 2>"$stderr_output" &
    GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
    local sampler_status=0
    if [[ -n "${GPU_SAMPLER_PID:-}" ]]; then
        kill "$GPU_SAMPLER_PID" 2>/dev/null || true
        wait "$GPU_SAMPLER_PID" 2>/dev/null || sampler_status=$?
        GPU_SAMPLER_PID=
    fi
    return "$sampler_status"
}

EXPECTED_CELL_IDS=()
CELL_FAILURES=0
CELL_INFEASIBLE=0
run_cell() {
    local workers="$1"
    local storage="$2"
    local activation_checkpointing="$3"
    local cell_id="workers-${workers}__storage-${storage}__ac-${activation_checkpointing}"
    local CELL_ROOT CELL_LATENT_CACHE_ROOT CELL_TRAINING_OUTPUT CELL_RUN_NAME CELL_RUN_DIR
    local CELL_LOG CELL_STEP_NDJSON CELL_GPU_SAMPLES CELL_GPU_STDERR CELL_MEMORY_GATE_JSON
    local CELL_TRAIN_STATUS CELL_STATUS SAMPLER_STATUS CELL_TIMEOUT_SECONDS
    local -a CELL_PIPE_STATUS

    CURRENT_PHASE="cell:$cell_id"
    EXPECTED_CELL_IDS+=("$cell_id")
    CELL_ROOT="$CELLS_DIR/$cell_id"
    mkdir "$CELL_ROOT"
    if [[ "$storage" == gpfs ]]; then
        CELL_LATENT_CACHE_ROOT="$GPFS_LATENT_CACHE_ROOT"
    else
        CELL_LATENT_CACHE_ROOT="$NVME_LATENT_CACHE_ROOT"
    fi
    CELL_TIMEOUT_SECONDS=0
    if [[ "$activation_checkpointing" == none ]]; then
        CELL_TIMEOUT_SECONDS="$NO_AC_CELL_TIMEOUT_SECONDS"
    fi

    CELL_TRAINING_OUTPUT="$CELL_ROOT/training"
    CELL_RUN_NAME="$cell_id"
    CELL_RUN_DIR="$CELL_TRAINING_OUTPUT/cosmos3_action_robocasa365/action_sft/$CELL_RUN_NAME"
    CELL_LOG="$CELL_ROOT/train.log"
    CELL_STEP_NDJSON="$CELL_ROOT/optimizer-step-wall-clock.ndjson"
    CELL_GPU_SAMPLES="$CELL_ROOT/gpu-samples.ndjson"
    CELL_GPU_STDERR="$CELL_ROOT/gpu-samples.stderr"
    CELL_MEMORY_GATE_JSON="$CELL_ROOT/production_batch_memory_gate/iteration_000001001.json"
    mkdir -p "$CELL_RUN_DIR/checkpoints"
    ln -s "$R6_ITER1000_CHECKPOINT_PATH" \
        "$CELL_RUN_DIR/checkpoints/iter_000001000"
    printf '%s\n' iter_000001000 >"$CELL_RUN_DIR/checkpoints/latest_checkpoint.txt"

    "$PYTHON" "$HELPER" write-cell-contract \
        --output "$CELL_ROOT/contract.json" \
        --experiment-id "$THROUGHPUT_EXPERIMENT_ID" \
        --run-id "$THROUGHPUT_RUN_ID" \
        --cell-id "$cell_id" \
        --world-size "$NPROC_REQUIRED" \
        --workers "$workers" \
        --storage "$storage" \
        --activation-checkpointing "$activation_checkpointing" \
        --video-tokenizer-target \
            cosmos_framework.model.generator.tokenizers.precomputed_wan2pt2.PrecomputedWan2pt2LatentTokenizer \
        --cell-timeout-seconds "$CELL_TIMEOUT_SECONDS" \
        --warmup-steps "$WARMUP_STEPS" \
        --measure-steps "$MEASURE_STEPS" \
        --latent-cache-root "$CELL_LATENT_CACHE_ROOT" \
        --latent-cache-manifest-sha256 "$CACHE_MANIFEST_SHA256" \
        --fixed-index-trace-path "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" \
        --fixed-index-trace-sha256 "$FIXED_TRACE_SHA256" \
        --fixed-index-trace-manifest-sha256 "$FIXED_TRACE_MANIFEST_SHA256" \
        --wan-vae-path "$WAN_VAE_PATH" \
        --wan-vae-sha256 "$ACTUAL_WAN_VAE_SHA256" \
        --resume-mode "$PROBE_RESUME_MODE" \
        --resume-checkpoint "$R6_ITER1000_CHECKPOINT_PATH" \
        --source-root "$COSMOS_SOURCE" \
        --source-commit "$EXPECTED_SOURCE_COMMIT" \
        --snapshot-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
        --runner-sha256 "$ACTUAL_RUNNER_SHA256" \
        --helper-sha256 "$ACTUAL_HELPER_SHA256" \
        --node-evidence "$NODE_EVIDENCE_JSON" >/dev/null

    export NPROC_PER_NODE="$NPROC_REQUIRED"
    export RUN_NAME="$CELL_RUN_NAME"
    export OUTPUT_ROOT="$CELL_TRAINING_OUTPUT"
    export IMAGINAIRE_OUTPUT_ROOT="$CELL_TRAINING_OUTPUT"
    export LOG_FILENAME="$cell_id.launcher.log"
    export ROBOCASA365_ROOT
    export ROBOCASA365_SOURCE_ROOT
    export BASE_CHECKPOINT_PATH
    export WAN_VAE_PATH
    export QWEN3_VL_TOKENIZER_PATH
    export ROBOCASA365_LATENT_CACHE_ROOT="$CELL_LATENT_CACHE_ROOT"
    export ROBOCASA365_LATENT_CACHE_MODE=required
    export CWAM_METADATA_ONLY_VIDEO_TOKENIZER=1
    export ROBOCASA365_FIXED_INDEX_TRACE_PATH
    export ROBOCASA365_TRAIN_NUM_WORKERS="$workers"
    export ROBOCASA365_VAL_NUM_WORKERS
    export CWAM_ACTIVATION_CHECKPOINTING_MODE="$activation_checkpointing"
    export RESUME_EXISTING_RUN=1
    export ALLOW_DATA_STREAM_REPLAY=1
    export CWAM_MEMORY_GATE_DIR="$CELL_ROOT/production_batch_memory_gate"
    export EXTRA_TAIL_OVERRIDES="
trainer.max_iter=1060
trainer.save_final_checkpoint=false
trainer.logging_iter=1
trainer.run_validation=false
trainer.run_validation_on_start=false
trainer.validation_iter=1000000000
checkpoint.save_iter=1061
checkpoint.dcp_async_mode_enabled=false
checkpoint.load_training_state=true
checkpoint.keys_to_skip_loading=[]
checkpoint.keys_not_to_resume=[dataloader]
checkpoint.strict_resume=true
trainer.callbacks.production_batch_memory_gate.required=true
trainer.callbacks.action_gradient_evidence.every_n=1000000000
trainer.callbacks.device_monitor.every_n=1000000000
trainer.callbacks.heart_beat.every_n=1000000000
++trainer.callbacks.throughput_probe._target_=examples.h_cluster.cwam_cosmos3_latent_cache_throughput.make_throughput_probe_callback
++trainer.callbacks.throughput_probe.output_path=$CELL_STEP_NDJSON
++trainer.callbacks.throughput_probe.warmup_steps=$WARMUP_STEPS
++trainer.callbacks.throughput_probe.measure_steps=$MEASURE_STEPS
++trainer.callbacks.throughput_probe.expected_world_size=$NPROC_REQUIRED
++trainer.callbacks.throughput_probe.expected_grad_accum_iter=2
"

    GPU_SAMPLER_PID=
    start_gpu_sampler "$CELL_GPU_SAMPLES" "$CELL_GPU_STDERR"
    set +e
    if [[ "$activation_checkpointing" == none ]]; then
        timeout --signal=TERM --kill-after=120s "$NO_AC_CELL_TIMEOUT_SECONDS" \
            bash "$LAUNCHER" 2>&1 | tee "$CELL_LOG"
    else
        bash "$LAUNCHER" 2>&1 | tee "$CELL_LOG"
    fi
    CELL_PIPE_STATUS=("${PIPESTATUS[@]}")
    set -e
    SAMPLER_STATUS=0
    stop_gpu_sampler || SAMPLER_STATUS=$?
    CELL_TRAIN_STATUS="${CELL_PIPE_STATUS[0]}"
    if [[ "${CELL_PIPE_STATUS[1]}" -ne 0 && "$CELL_TRAIN_STATUS" -eq 0 ]]; then
        CELL_TRAIN_STATUS="${CELL_PIPE_STATUS[1]}"
    fi
    if [[ "$SAMPLER_STATUS" -ne 0 && "$CELL_TRAIN_STATUS" -eq 0 ]]; then
        CELL_TRAIN_STATUS="$SAMPLER_STATUS"
    fi

    "$PYTHON" "$HELPER" summarize-cell \
        --cell-id "$cell_id" \
        --workers "$workers" \
        --storage "$storage" \
        --activation-checkpointing "$activation_checkpointing" \
        --warmup-steps "$WARMUP_STEPS" \
        --measure-steps "$MEASURE_STEPS" \
        --log "$CELL_LOG" \
        --step-ndjson "$CELL_STEP_NDJSON" \
        --gpu-samples "$CELL_GPU_SAMPLES" \
        --memory-gate-json "$CELL_MEMORY_GATE_JSON" \
        --training-exit-code "$CELL_TRAIN_STATUS" \
        --expected-gpus "$NPROC_REQUIRED" \
        --output "$CELL_ROOT/metrics.json" >/dev/null

    "$PYTHON" "$HELPER" purge-probe-checkpoints \
        --cell-root "$CELL_ROOT" \
        --run-dir "$CELL_RUN_DIR" \
        --output "$CELL_ROOT/checkpoint-cleanup.json" >/dev/null

    CELL_STATUS="$(
        "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
            "$CELL_ROOT/metrics.json"
    )"
    case "$CELL_STATUS" in
        passed)
            ;;
        infeasible_oom|infeasible_memory_headroom)
            if [[ "$activation_checkpointing" == none ]]; then
                CELL_INFEASIBLE="$((CELL_INFEASIBLE + 1))"
            else
                CELL_FAILURES="$((CELL_FAILURES + 1))"
            fi
            ;;
        *)
            CELL_FAILURES="$((CELL_FAILURES + 1))"
            ;;
    esac
}

if [[ "$PROBE_MATRIX_MODE" == full16 ]]; then
    CURRENT_PHASE=full16_matrix
    for workers in "${WORKER_MATRIX[@]}"; do
        for storage in "${STORAGE_MATRIX[@]}"; do
            for activation_checkpointing in "${ACTIVATION_CHECKPOINTING_MATRIX[@]}"; do
                run_cell "$workers" "$storage" "$activation_checkpointing"
            done
        done
    done
    [[ "${#EXPECTED_CELL_IDS[@]}" -eq 16 ]] ||
        { echo "ERROR: full16 protocol did not produce exactly 16 cells" >&2; exit 1; }
else
    CURRENT_PHASE=adaptive6_worker_sweep
    WORKER_CANDIDATE_METRICS=()
    for workers in "${WORKER_MATRIX[@]}"; do
        run_cell "$workers" gpfs selective
        WORKER_CANDIDATE_METRICS+=(
            --metrics "$CELLS_DIR/workers-${workers}__storage-gpfs__ac-selective/metrics.json"
        )
    done
    WORKER_DECISION="$EVIDENCE_DIR/adaptive-worker-selection.json"
    CURRENT_PHASE=adaptive6_worker_selection
    "$PYTHON" "$HELPER" select-best-cell \
        "${WORKER_CANDIDATE_METRICS[@]}" \
        --decision workers_per_rank_on_gpfs_selective \
        --output "$WORKER_DECISION" >/dev/null
    BEST_WORKERS="$(
        "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_workers_per_rank"])' \
            "$WORKER_DECISION"
    )"

    CURRENT_PHASE=adaptive6_nvme_cell
    run_cell "$BEST_WORKERS" nvme selective
    STORAGE_DECISION="$EVIDENCE_DIR/adaptive-storage-selection.json"
    "$PYTHON" "$HELPER" select-best-cell \
        --metrics "$CELLS_DIR/workers-${BEST_WORKERS}__storage-gpfs__ac-selective/metrics.json" \
        --metrics "$CELLS_DIR/workers-${BEST_WORKERS}__storage-nvme__ac-selective/metrics.json" \
        --decision storage_at_selected_workers_selective \
        --output "$STORAGE_DECISION" >/dev/null
    BEST_STORAGE="$(
        "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_storage"])' \
            "$STORAGE_DECISION"
    )"
    CURRENT_PHASE=adaptive6_ac_none_cell
    run_cell "$BEST_WORKERS" "$BEST_STORAGE" none
    [[ "${#EXPECTED_CELL_IDS[@]}" -eq 6 ]] ||
        { echo "ERROR: adaptive6 protocol did not produce exactly 6 cells" >&2; exit 1; }
fi

MATRIX_ARGS=()
CURRENT_PHASE=matrix_summary
for cell_id in "${EXPECTED_CELL_IDS[@]}"; do
    MATRIX_ARGS+=(--expected-cell-id "$cell_id")
done
"$PYTHON" "$HELPER" summarize-matrix \
    --run-root "$RUN_ROOT" \
    "${MATRIX_ARGS[@]}" \
    --output "$RUN_ROOT/matrix-summary.json" >/dev/null
MATRIX_STATUS="$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$RUN_ROOT/matrix-summary.json"
)"

echo "THROUGHPUT_MATRIX_COMPLETE mode=$PROBE_MATRIX_MODE run_root=$RUN_ROOT cells=${#EXPECTED_CELL_IDS[@]} failures=$CELL_FAILURES infeasible=$CELL_INFEASIBLE matrix_status=$MATRIX_STATUS"
if [[ "$CELL_FAILURES" -ne 0 ||
      ( "$MATRIX_STATUS" != passed && "$MATRIX_STATUS" != passed_with_no_ac_fallback ) ]]; then
    exit 1
fi
CURRENT_PHASE=complete
