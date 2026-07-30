#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Submit exactly one raw, non-formal, one-GPU quick-eval RJob.
#
# Unlike the formal submitter, this intentionally performs neither predict nor
# authenticated scheduler readback. The single `rjob submit` call below is the
# opt-in action. Its result can never qualify as formal or prior-gate evidence.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly PINNED_PERSONAL_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly SNAPSHOT_ALLOWED_ROOT="$PINNED_PERSONAL_GPFS_ROOT/run_snapshots"
readonly PINNED_QUICK_OUTPUT_PARENT="$PINNED_PERSONAL_GPFS_ROOT/outputs/cwam-robocasa365-quick-eval/runs"
readonly PERSONAL_GPFS_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS_MOUNT="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly PINNED_EGL_VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
readonly SUBMITTER_REL="examples/h_cluster/submit_cwam_cosmos3_robocasa365_quick_eval_rjob.sh"
readonly RUNNER_REL="examples/h_cluster/run_cwam_cosmos3_robocasa365_quick_eval.sh"
readonly HELPER_REL="examples/h_cluster/cwam_robocasa365_quick_eval.py"
readonly PROVENANCE_HELPER_REL="examples/h_cluster/cwam_formal_provenance.py"
readonly RJOB_BIN="/usr/local/bin/rjob"
readonly BASH_BIN="/usr/bin/bash"
readonly ENV_BIN="/usr/bin/env"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

require_nonempty_env() {
    local name
    for name in "$@"; do
        [[ -v "$name" && -n "${!name}" ]] ||
            die "required environment variable is missing or empty: $name"
    done
}

require_safe_id() {
    local label="$1"
    local value="$2"
    [[ "${#value}" -le 128 &&
        "$value" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ &&
        "$value" != "." &&
        "$value" != ".." ]] ||
        die "$label must be a strict path-safe identity"
}

readonly CALLER_INPUTS=(
    COSMOS_SOURCE
    COSMOS_ENV
    ROBOCASA_ENV
    DEMIAN_REPO
    ROBOCASA_REPO
    ROBOSUITE_REPO
    SERVING_CHECKPOINT_PATH
    CHECKPOINT_MANIFEST
    TRAINING_CONFIG
    ACTION_STATS
    COSMOS_ENVIRONMENT_MANIFEST
    ROBOCASA_ENVIRONMENT_MANIFEST
    SOURCE_DCP_EXPORT_RECEIPT
    EVAL_EXPERIMENT_ID
    EVAL_LOGICAL_RUN_ID
    EVAL_TASKS_CSV
    EXPECTED_SOURCE_COMMIT
    EXPECTED_SOURCE_MANIFEST_SHA256
    EXPECTED_CHECKPOINT_MANIFEST_SHA256
    EXPECTED_TRAINING_CONFIG_SHA256
    EXPECTED_ACTION_STATS_SHA256
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256
    EXPECTED_EGL_VENDOR_JSON_SHA256
    EXPECTED_ROBOCASA_COMMIT
    EXPECTED_ROBOSUITE_COMMIT
    RJOB_NAME
)
require_nonempty_env "${CALLER_INPUTS[@]}"

[[ "$EVAL_EXPERIMENT_ID" == QUICK-NONFORMAL-* ]] ||
    die "EVAL_EXPERIMENT_ID must start with QUICK-NONFORMAL-"
require_safe_id EVAL_EXPERIMENT_ID "$EVAL_EXPERIMENT_ID"
require_safe_id EVAL_LOGICAL_RUN_ID "$EVAL_LOGICAL_RUN_ID"
[[ "$EVAL_LOGICAL_RUN_ID" == "$EVAL_EXPERIMENT_ID"-* ]] ||
    die "EVAL_LOGICAL_RUN_ID must be namespaced below EVAL_EXPERIMENT_ID"
[[ "$EVAL_TASKS_CSV" =~ ^[A-Za-z0-9_]+$ && "$EVAL_TASKS_CSV" != *,* ]] ||
    die "EVAL_TASKS_CSV must contain exactly one task"
[[ "${#RJOB_NAME}" -le 50 && "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] ||
    die "RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters"
for commit_name in EXPECTED_SOURCE_COMMIT EXPECTED_ROBOCASA_COMMIT EXPECTED_ROBOSUITE_COMMIT; do
    [[ "${!commit_name}" =~ ^[0-9a-f]{40}$ ]] || die "$commit_name must be a full lowercase Git SHA"
done
for digest_name in \
    EXPECTED_SOURCE_MANIFEST_SHA256 \
    EXPECTED_CHECKPOINT_MANIFEST_SHA256 \
    EXPECTED_TRAINING_CONFIG_SHA256 \
    EXPECTED_ACTION_STATS_SHA256 \
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256 \
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256 \
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256 \
    EXPECTED_EGL_VENDOR_JSON_SHA256; do
    [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] || die "$digest_name must be a lowercase SHA-256"
done

COSMOS_SOURCE="$(realpath -e -- "$COSMOS_SOURCE")"
case "$COSMOS_SOURCE/" in
    "$SNAPSHOT_ALLOWED_ROOT"/*/) ;;
    *) die "COSMOS_SOURCE must be an immutable snapshot below $SNAPSHOT_ALLOWED_ROOT" ;;
esac
[[ -d "$COSMOS_SOURCE" && ! -L "$COSMOS_SOURCE" ]] || die "COSMOS_SOURCE is not a safe directory"
[[ "$(stat -c %a -- "$COSMOS_SOURCE")" == 555 ]] || die "COSMOS_SOURCE snapshot root must be mode 0555"
readonly SOURCE_MANIFEST="$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
readonly SUBMITTER="$COSMOS_SOURCE/$SUBMITTER_REL"
readonly RUNNER="$COSMOS_SOURCE/$RUNNER_REL"
readonly QUICK_HELPER="$COSMOS_SOURCE/$HELPER_REL"
readonly PROVENANCE_HELPER="$COSMOS_SOURCE/$PROVENANCE_HELPER_REL"
[[ "$(realpath -e -- "${BASH_SOURCE[0]}")" == "$SUBMITTER" ]] ||
    die "quick submitter must execute from the exact immutable COSMOS_SOURCE snapshot"
for source_file in "$SOURCE_MANIFEST" "$SUBMITTER" "$RUNNER" "$QUICK_HELPER" "$PROVENANCE_HELPER"; do
    [[ -f "$source_file" && ! -L "$source_file" ]] || die "snapshot file is missing: $source_file"
    [[ "$(stat -c %a -- "$source_file")" == 444 ]] || die "snapshot file must be mode 0444: $source_file"
done
[[ "$(sha256_file "$SOURCE_MANIFEST")" == "$EXPECTED_SOURCE_MANIFEST_SHA256" ]] ||
    die "SOURCE_MANIFEST SHA-256 mismatch"

/usr/bin/python3 -I -B - \
    "$SOURCE_MANIFEST" \
    "$EXPECTED_SOURCE_COMMIT" \
    "$SUBMITTER_REL" \
    "$RUNNER_REL" \
    "$HELPER_REL" \
    "$PROVENANCE_HELPER_REL" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
root = manifest_path.parent
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1:
    raise SystemExit("snapshot manifest schema_version must be 1")
if manifest.get("source_head") != sys.argv[2] or manifest.get("source_dirty") is not False:
    raise SystemExit("snapshot manifest does not bind the expected clean commit")
files = manifest.get("files")
if not isinstance(files, list) or manifest.get("file_count") != len(files):
    raise SystemExit("snapshot manifest file inventory is malformed")

descriptors = {}
for value in files:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise SystemExit(f"snapshot manifest file descriptor is malformed: {value!r}")
    relative = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    if not isinstance(relative, str) or not isinstance(digest, str) or type(size) is not int:
        raise SystemExit(f"snapshot manifest file descriptor types are malformed: {value!r}")
    if relative in descriptors:
        raise SystemExit(f"snapshot manifest has a duplicate path: {relative}")
    descriptors[relative] = (digest, size)
for relative in sys.argv[3:]:
    payload = (root / relative).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if descriptors.get(relative) != (actual, len(payload)):
        raise SystemExit(f"snapshot manifest does not bind quick entrypoint: {relative}")
PY
"/usr/bin/python3" "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" >/dev/null

mkdir -p -- "$PINNED_QUICK_OUTPUT_PARENT"
[[ "$(realpath -e -- "$PINNED_QUICK_OUTPUT_PARENT")" == "$PINNED_QUICK_OUTPUT_PARENT" ]] ||
    die "quick output parent is not canonical"
readonly EVAL_RUN_NAME="$EVAL_LOGICAL_RUN_ID"
readonly EVAL_OUTPUT_ROOT="$PINNED_QUICK_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
readonly EVAL_EGL_VENDOR_JSON="$PINNED_EGL_VENDOR_JSON"
readonly EVAL_NVME_ROOT="/nvme/.cwam-robocasa365-quick-eval/$EVAL_LOGICAL_RUN_ID"
readonly CWAM_QUICK_EVAL=1

readonly WORKER_ENV_NAMES=(
    CWAM_QUICK_EVAL
    COSMOS_SOURCE
    COSMOS_ENV
    ROBOCASA_ENV
    DEMIAN_REPO
    ROBOCASA_REPO
    ROBOSUITE_REPO
    SERVING_CHECKPOINT_PATH
    CHECKPOINT_MANIFEST
    TRAINING_CONFIG
    ACTION_STATS
    SOURCE_MANIFEST
    COSMOS_ENVIRONMENT_MANIFEST
    ROBOCASA_ENVIRONMENT_MANIFEST
    SOURCE_DCP_EXPORT_RECEIPT
    EVAL_OUTPUT_ROOT
    EVAL_EXPERIMENT_ID
    EVAL_LOGICAL_RUN_ID
    EVAL_RUN_NAME
    EVAL_TASKS_CSV
    EXPECTED_SOURCE_COMMIT
    EXPECTED_SOURCE_MANIFEST_SHA256
    EXPECTED_CHECKPOINT_MANIFEST_SHA256
    EXPECTED_TRAINING_CONFIG_SHA256
    EXPECTED_ACTION_STATS_SHA256
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256
    EXPECTED_EGL_VENDOR_JSON_SHA256
    EXPECTED_ROBOCASA_COMMIT
    EXPECTED_ROBOSUITE_COMMIT
    EVAL_EGL_VENDOR_JSON
    EVAL_NVME_ROOT
)
WORKER_SCRIPT=$'set -Eeuo pipefail\n'
WORKER_SCRIPT+=$'unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE PYTHONPATH PYTHONUSERBASE PYTHONBREAKPOINT LD_PRELOAD LD_AUDIT __EGL_VENDOR_LIBRARY_DIRS LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE LIBGL_DRIVERS_PATH\n'
WORKER_SCRIPT+=$'export __GLX_VENDOR_LIBRARY_NAME=nvidia\n'
WORKER_SCRIPT+="cd $(printf '%q' "$COSMOS_SOURCE")"$'\n'
for name in "${WORKER_ENV_NAMES[@]}"; do
    printf -v quoted '%q' "${!name}"
    WORKER_SCRIPT+="export $name=$quoted"$'\n'
done
printf -v quoted_vendor '%q' "$PINNED_EGL_VENDOR_JSON"
WORKER_SCRIPT+="export __EGL_VENDOR_LIBRARY_FILENAMES=$quoted_vendor"$'\n'
WORKER_SCRIPT+="exec $BASH_BIN $(printf '%q' "$RUNNER")"$'\n'
readonly WORKER_COMMAND_SHA256="$(printf '%s' "$WORKER_SCRIPT" | sha256sum | awk '{print $1}')"
WORKER_COMMAND=(
    "$ENV_BIN"
    -u BASH_ENV
    -u ENV
    "$BASH_BIN"
    --noprofile
    --norc
    -c
    "$WORKER_SCRIPT"
)

SUBMIT_ARGS=(
    submit
    --name "$RJOB_NAME"
    --group "$PINNED_QUOTA_GROUP"
    --charged-group "$PINNED_QUOTA_GROUP"
    --private-machine group
    --priority 9
    --task-type normal
    --replicas 1
    --restart-policy never
    --termination-grace-period-seconds 60
    --backoff_limit 1
    --image "$PINNED_CONTAINER_IMAGE"
    --cpu 16
    --gpu 1
    --memory 187500
    --negative-tags "$PINNED_EXCLUDED_NODE_TAG"
    --host-network
    --store-host-nvme
    --mount
    "$PERSONAL_GPFS_MOUNT"
    "$PUBLIC_GPFS_MOUNT"
)

echo "QUICK_NONFORMAL_SUBMISSION formal_eligible=false asset_full_tree_verification_skipped=true"
echo "QUICK_NONFORMAL_SUBMISSION_BOUNDARY no_predict_no_scheduler_readback_no_formal_receipt"
echo "RJOB_NAME=$RJOB_NAME"
echo "EVAL_EXPERIMENT_ID=$EVAL_EXPERIMENT_ID"
echo "EVAL_LOGICAL_RUN_ID=$EVAL_LOGICAL_RUN_ID"
echo "EVAL_TASK=$EVAL_TASKS_CSV"
echo "WORKER_COMMAND_SHA256=$WORKER_COMMAND_SHA256"

# The only state-changing scheduler call in this quick submitter.
"$RJOB_BIN" "${SUBMIT_ARGS[@]}" -- "${WORKER_COMMAND[@]}"
echo "QUICK_NONFORMAL_RJOB_SUBMIT_RETURNED lifecycle=raw_quick_nonsealed"
