#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Submit exactly one normal-priority, single-node, eight-H200 latent-cache
# throughput probe.  This wrapper only submits; the runner owns the fixed
# 4/8/12/16 x GPFS/NVMe x selective/none matrix and all runtime evidence.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly PINNED_OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-latent-cache-throughput"
readonly NPROC_REQUIRED=8
# H200 nodes expose 186 allocatable CPUs to Kubernetes. The 160-CPU request is
# schedulable and still leaves 1.25 cores per loader at the 8 x 16-worker cell.
readonly RJOB_CPU=160
readonly RJOB_MEMORY_MIB=1500000
readonly SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable Cosmos source snapshot.}"
: "${THROUGHPUT_EXPERIMENT_ID:?Set THROUGHPUT_EXPERIMENT_ID.}"
: "${THROUGHPUT_RUN_ID:?Set a unique THROUGHPUT_RUN_ID.}"
: "${GPFS_LATENT_CACHE_ROOT:?Set GPFS_LATENT_CACHE_ROOT.}"
: "${ROBOCASA365_FIXED_INDEX_TRACE_PATH:?Set ROBOCASA365_FIXED_INDEX_TRACE_PATH.}"
: "${NVME_MOUNT_ROOT:=/nvme}"
: "${RJOB_NAME:?Set a unique lowercase H-cluster rjob show name.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${R6_ITER1000_CHECKPOINT_PATH:?Set R6_ITER1000_CHECKPOINT_PATH.}"
: "${NPROC_PER_NODE:=8}"
: "${RJOB_PREDICT_ONLY:=1}"
: "${RJOB_IMAGE:=$PINNED_CONTAINER_IMAGE}"
: "${RJOB_QUOTA_GROUP:=$PINNED_QUOTA_GROUP}"
: "${RJOB_EXCLUDED_NODE_TAG:=$PINNED_EXCLUDED_NODE_TAG}"
: "${THROUGHPUT_OUTPUT_ROOT:=$PINNED_OUTPUT_ROOT}"
: "${PROBE_MATRIX_MODE:=adaptive6}"
: "${PROBE_RESUME_MODE:=r6_iter1000_model_optimizer}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"

[[ "$NPROC_PER_NODE" == "$NPROC_REQUIRED" ]] ||
    { echo "ERROR: NPROC_PER_NODE must be exactly 8" >&2; exit 1; }
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] ||
    { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    { echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2; exit 1; }
[[ "$THROUGHPUT_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: THROUGHPUT_EXPERIMENT_ID contains unsupported characters" >&2; exit 1; }
[[ "$THROUGHPUT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: THROUGHPUT_RUN_ID contains unsupported characters" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase Git SHA" >&2; exit 1; }
[[ "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_SNAPSHOT_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$RJOB_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
    { echo "ERROR: RJOB_IMAGE differs from the audited digest" >&2; exit 1; }
[[ "$RJOB_QUOTA_GROUP" == "$PINNED_QUOTA_GROUP" ]] ||
    { echo "ERROR: RJOB_QUOTA_GROUP differs from the audited quota group" >&2; exit 1; }
[[ "$RJOB_EXCLUDED_NODE_TAG" == "$PINNED_EXCLUDED_NODE_TAG" ]] ||
    { echo "ERROR: RJOB_EXCLUDED_NODE_TAG differs from the audited exclusion" >&2; exit 1; }
[[ "$PROBE_MATRIX_MODE" == adaptive6 || "$PROBE_MATRIX_MODE" == full16 ]] ||
    { echo "ERROR: PROBE_MATRIX_MODE must be adaptive6 or full16" >&2; exit 1; }
[[ "$PROBE_RESUME_MODE" == r6_iter1000_model_optimizer ]] ||
    { echo "ERROR: probe cells must share the R6 iter1000 state" >&2; exit 1; }
[[ "$NVME_MOUNT_ROOT" == /nvme ]] ||
    { echo "ERROR: --store-host-nvme exposes exactly /nvme on H" >&2; exit 1; }
[[ "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" == *.npy ]] ||
    { echo "ERROR: ROBOCASA365_FIXED_INDEX_TRACE_PATH must end in .npy" >&2; exit 1; }

RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_latent_cache_throughput_probe.sh"
HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_cosmos3_latent_cache_throughput.py"
PROVENANCE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"
test -s "$RUNNER"
test -s "$HELPER"
test -s "$PROVENANCE_HELPER"
test -s "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
test -d "$GPFS_LATENT_CACHE_ROOT"
test -s "$GPFS_LATENT_CACHE_ROOT/_SUCCESS.json"
test -s "$GPFS_LATENT_CACHE_ROOT/manifest.json"
FIXED_INDEX_TRACE_MANIFEST="${ROBOCASA365_FIXED_INDEX_TRACE_PATH%.npy}.manifest.json"
if [[
    -e "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" ||
    -L "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" ||
    -e "$FIXED_INDEX_TRACE_MANIFEST" ||
    -L "$FIXED_INDEX_TRACE_MANIFEST"
]]; then
    [[
        -s "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" &&
        ! -L "$ROBOCASA365_FIXED_INDEX_TRACE_PATH" &&
        -s "$FIXED_INDEX_TRACE_MANIFEST" &&
        ! -L "$FIXED_INDEX_TRACE_MANIFEST"
    ]] || {
        echo "ERROR: fixed trace array and manifest must be complete regular files or both absent" >&2
        exit 1
    }
else
    FIXED_INDEX_TRACE_PARENT="$(dirname -- "$ROBOCASA365_FIXED_INDEX_TRACE_PATH")"
    mkdir -p "$FIXED_INDEX_TRACE_PARENT"
    [[ -d "$FIXED_INDEX_TRACE_PARENT" && -w "$FIXED_INDEX_TRACE_PARENT" ]] ||
        { echo "ERROR: fixed trace parent is not writable" >&2; exit 1; }
fi
command -v rjob >/dev/null
command -v sha256sum >/dev/null
command -v python3 >/dev/null
python3 "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null
EXPECTED_RUNNER_SHA256="$(sha256sum "$RUNNER" | awk '{print $1}')"
EXPECTED_HELPER_SHA256="$(sha256sum "$HELPER" | awk '{print $1}')"

mkdir -p "$THROUGHPUT_OUTPUT_ROOT"
THROUGHPUT_OUTPUT_ROOT="$(realpath -e -- "$THROUGHPUT_OUTPUT_ROOT")"
SUBMISSION_ROOT="$THROUGHPUT_OUTPUT_ROOT/submission-reservations"
SUBMISSION_DIR="$SUBMISSION_ROOT/$THROUGHPUT_RUN_ID"
SUBMISSION_PENDING="$SUBMISSION_DIR/submission-pending.json"
SUBMISSION_LOG="$SUBMISSION_DIR/rjob-submit.log"
SUBMISSION_ACCEPTED="$SUBMISSION_DIR/submission-accepted.json"
SCHEDULER_READBACK="$SUBMISSION_DIR/rjob-get.log"

printf -v Q_COSMOS_SOURCE %q "$COSMOS_SOURCE"
printf -v Q_THROUGHPUT_EXPERIMENT_ID %q "$THROUGHPUT_EXPERIMENT_ID"
printf -v Q_THROUGHPUT_RUN_ID %q "$THROUGHPUT_RUN_ID"
printf -v Q_THROUGHPUT_OUTPUT_ROOT %q "$THROUGHPUT_OUTPUT_ROOT"
printf -v Q_GPFS_LATENT_CACHE_ROOT %q "$GPFS_LATENT_CACHE_ROOT"
printf -v Q_FIXED_INDEX_TRACE_PATH %q "$ROBOCASA365_FIXED_INDEX_TRACE_PATH"
printf -v Q_NVME_MOUNT_ROOT %q "$NVME_MOUNT_ROOT"
printf -v Q_RJOB_NAME %q "$RJOB_NAME"
printf -v Q_EXPECTED_SOURCE_COMMIT %q "$EXPECTED_SOURCE_COMMIT"
printf -v Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256 %q "$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
printf -v Q_EXPECTED_RUNNER_SHA256 %q "$EXPECTED_RUNNER_SHA256"
printf -v Q_EXPECTED_HELPER_SHA256 %q "$EXPECTED_HELPER_SHA256"
printf -v Q_PROBE_RESUME_MODE %q "$PROBE_RESUME_MODE"
printf -v Q_PROBE_MATRIX_MODE %q "$PROBE_MATRIX_MODE"
printf -v Q_R6_CHECKPOINT %q "$R6_ITER1000_CHECKPOINT_PATH"
printf -v Q_COSMOS_ENV %q "$COSMOS_ENV"
printf -v Q_ROBOCASA365_ROOT %q "$ROBOCASA365_ROOT"
printf -v Q_ROBOCASA365_SOURCE_ROOT %q "$ROBOCASA365_SOURCE_ROOT"
printf -v Q_BASE_CHECKPOINT_PATH %q "$BASE_CHECKPOINT_PATH"
printf -v Q_WAN_VAE_PATH %q "$WAN_VAE_PATH"
printf -v Q_QWEN3_VL_TOKENIZER_PATH %q "$QWEN3_VL_TOKENIZER_PATH"
printf -v Q_RUNNER %q "$RUNNER"

PROBE_COMMAND="$(
    printf '%s\n' \
        "cd $Q_COSMOS_SOURCE" \
        "export COSMOS_SOURCE=$Q_COSMOS_SOURCE" \
        "export THROUGHPUT_EXPERIMENT_ID=$Q_THROUGHPUT_EXPERIMENT_ID" \
        "export THROUGHPUT_RUN_ID=$Q_THROUGHPUT_RUN_ID" \
        "export THROUGHPUT_OUTPUT_ROOT=$Q_THROUGHPUT_OUTPUT_ROOT" \
        "export GPFS_LATENT_CACHE_ROOT=$Q_GPFS_LATENT_CACHE_ROOT" \
        "export ROBOCASA365_FIXED_INDEX_TRACE_PATH=$Q_FIXED_INDEX_TRACE_PATH" \
        "export NVME_MOUNT_ROOT=$Q_NVME_MOUNT_ROOT" \
        "export RJOB_NAME=$Q_RJOB_NAME" \
        "export EXPECTED_SOURCE_COMMIT=$Q_EXPECTED_SOURCE_COMMIT" \
        "export EXPECTED_SNAPSHOT_MANIFEST_SHA256=$Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
        "export EXPECTED_RUNNER_SHA256=$Q_EXPECTED_RUNNER_SHA256" \
        "export EXPECTED_HELPER_SHA256=$Q_EXPECTED_HELPER_SHA256" \
        "export PROBE_RESUME_MODE=$Q_PROBE_RESUME_MODE" \
        "export PROBE_MATRIX_MODE=$Q_PROBE_MATRIX_MODE" \
        "export R6_ITER1000_CHECKPOINT_PATH=$Q_R6_CHECKPOINT" \
        "export COSMOS_ENV=$Q_COSMOS_ENV" \
        "export ROBOCASA365_ROOT=$Q_ROBOCASA365_ROOT" \
        "export ROBOCASA365_SOURCE_ROOT=$Q_ROBOCASA365_SOURCE_ROOT" \
        "export BASE_CHECKPOINT_PATH=$Q_BASE_CHECKPOINT_PATH" \
        "export WAN_VAE_PATH=$Q_WAN_VAE_PATH" \
        "export QWEN3_VL_TOKENIZER_PATH=$Q_QWEN3_VL_TOKENIZER_PATH" \
        "export NPROC_PER_NODE=8" \
        "bash $Q_RUNNER"
)"
PROBE_COMMAND_SHA256="$(printf %s "$PROBE_COMMAND" | sha256sum | awk '{print $1}')"

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
    --store-host-nvme true
    --mount
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    SUBMIT_ARGS+=(--predict-only true)
fi

echo "RJOB_NAME=$RJOB_NAME"
echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "NPROC_PER_NODE=$NPROC_REQUIRED"
echo "PROBE_MATRIX_MODE=$PROBE_MATRIX_MODE"
echo "ADAPTIVE6=gpfs-selective-workers[4,8,12,16]+best-worker-nvme-selective+best-pair-ac-none"
echo "FULL16_OPTION=workers[4,8,12,16]xstorage[gpfs,nvme]xac[selective,none]"
echo "CELL_WINDOW=warmup10+measure50"
echo "RUNNER_SHA256=$EXPECTED_RUNNER_SHA256"
echo "HELPER_SHA256=$EXPECTED_HELPER_SHA256"
echo "PROBE_COMMAND_SHA256=$PROBE_COMMAND_SHA256"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    rjob "${SUBMIT_ARGS[@]}" -- \
        /usr/bin/env -u BASH_ENV -u ENV /usr/bin/bash --noprofile --norc -c "$PROBE_COMMAND"
    echo "THROUGHPUT_SUBMISSION_PREDICT_ONLY reservation_created=false"
    exit 0
fi

mkdir -p "$SUBMISSION_ROOT"
if ! mkdir "$SUBMISSION_DIR"; then
    echo "ERROR: duplicate submission reservation exists: $SUBMISSION_DIR" >&2
    exit 1
fi
python3 -c '
import json, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "submitting",
    "experiment_id": sys.argv[2],
    "run_id": sys.argv[3],
    "rjob_show_name": sys.argv[4],
    "gpu": 8,
    "cpu": 160,
    "memory_mib": 1500000,
    "private_machine": "group",
    "container_image": sys.argv[5],
    "source_commit": sys.argv[6],
    "snapshot_manifest_sha256": sys.argv[7],
    "runner_sha256": sys.argv[8],
    "helper_sha256": sys.argv[9],
    "command_sha256": sys.argv[10],
    "probe_resume_mode": sys.argv[11],
    "probe_matrix_mode": sys.argv[12],
}
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
' \
    "$SUBMISSION_PENDING" \
    "$THROUGHPUT_EXPERIMENT_ID" \
    "$THROUGHPUT_RUN_ID" \
    "$RJOB_NAME" \
    "$RJOB_IMAGE" \
    "$EXPECTED_SOURCE_COMMIT" \
    "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$EXPECTED_RUNNER_SHA256" \
    "$EXPECTED_HELPER_SHA256" \
    "$PROBE_COMMAND_SHA256" \
    "$PROBE_RESUME_MODE" \
    "$PROBE_MATRIX_MODE"

set +e
rjob "${SUBMIT_ARGS[@]}" -- \
    /usr/bin/env -u BASH_ENV -u ENV /usr/bin/bash --noprofile --norc -c "$PROBE_COMMAND" \
    2>&1 | tee "$SUBMISSION_LOG"
SUBMIT_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
if [[ "${SUBMIT_PIPE_STATUS[0]}" -ne 0 || "${SUBMIT_PIPE_STATUS[1]}" -ne 0 ]]; then
    echo "ERROR: rjob submission failed or became ambiguous; reservation retained at $SUBMISSION_DIR" >&2
    exit 1
fi
SCHEDULER_RJOB_NAME="$(
    python3 -c '
import re, sys
text = open(sys.argv[1], errors="replace").read()
matches = re.findall(r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)", text)
if len(matches) != 1 or not matches[0].startswith(sys.argv[2] + "-"):
    raise SystemExit(f"cannot bind one scheduler rjob name: {matches!r}")
print(matches[0])
' "$SUBMISSION_LOG" "$RJOB_NAME"
)"

READBACK_OK=0
for _ in {1..10}; do
    set +e
    READBACK_CONTENT="$(rjob get "$SCHEDULER_RJOB_NAME" 2>&1)"
    READBACK_STATUS=$?
    set -e
    if [[ "$READBACK_STATUS" -eq 0 ]]; then
        printf '%s\n' "$READBACK_CONTENT" >"$SCHEDULER_READBACK"
        READBACK_OK=1
        break
    fi
    sleep 1
done
[[ "$READBACK_OK" == 1 ]] ||
    { echo "ERROR: accepted rjob has no scheduler readback; reservation retained" >&2; exit 1; }

python3 -c '
import hashlib, json, pathlib, sys
pending = pathlib.Path(sys.argv[1])
payload = json.loads(pending.read_text())
payload.update({
    "status": "accepted",
    "scheduler_rjob_name": sys.argv[3],
    "submission_log": sys.argv[4],
    "submission_log_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest(),
    "scheduler_readback": sys.argv[5],
    "scheduler_readback_sha256": hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest(),
})
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
' \
    "$SUBMISSION_PENDING" \
    "$SUBMISSION_ACCEPTED" \
    "$SCHEDULER_RJOB_NAME" \
    "$SUBMISSION_LOG" \
    "$SCHEDULER_READBACK"
echo "THROUGHPUT_SUBMISSION_ACCEPTED scheduler_rjob=$SCHEDULER_RJOB_NAME reservation=$SUBMISSION_ACCEPTED"
