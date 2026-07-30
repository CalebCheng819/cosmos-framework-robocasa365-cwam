#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run exactly one deliberately non-formal RoboCasa365 episode on one H200.
#
# This is a separate opt-in path. It does not consume or emit formal submission,
# runtime, post-rollout, summary, task-success-rate, or evaluation-evidence
# artifacts. It authenticates the publisher manifest/receipt but deliberately
# skips the full live asset-tree content hash.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${CWAM_QUICK_EVAL:?Set CWAM_QUICK_EVAL=1 to opt in to non-formal quick eval}"
[[ "$CWAM_QUICK_EVAL" == 1 ]] || {
    echo "ERROR: CWAM_QUICK_EVAL must be exactly 1" >&2
    exit 1
}

for name in \
    COSMOS_SOURCE \
    COSMOS_ENV \
    ROBOCASA_ENV \
    DEMIAN_REPO \
    ROBOCASA_REPO \
    ROBOSUITE_REPO \
    SERVING_CHECKPOINT_PATH \
    CHECKPOINT_MANIFEST \
    TRAINING_CONFIG \
    ACTION_STATS \
    SOURCE_MANIFEST \
    COSMOS_ENVIRONMENT_MANIFEST \
    ROBOCASA_ENVIRONMENT_MANIFEST \
    SOURCE_DCP_EXPORT_RECEIPT \
    EVAL_OUTPUT_ROOT \
    EVAL_EXPERIMENT_ID \
    EVAL_LOGICAL_RUN_ID \
    EVAL_RUN_NAME \
    EVAL_TASKS_CSV \
    EXPECTED_SOURCE_COMMIT \
    EXPECTED_SOURCE_MANIFEST_SHA256 \
    EXPECTED_CHECKPOINT_MANIFEST_SHA256 \
    EXPECTED_TRAINING_CONFIG_SHA256 \
    EXPECTED_ACTION_STATS_SHA256 \
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256 \
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256 \
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256 \
    EXPECTED_EGL_VENDOR_JSON_SHA256 \
    EXPECTED_ROBOCASA_COMMIT \
    EXPECTED_ROBOSUITE_COMMIT \
    EVAL_EGL_VENDOR_JSON \
    EVAL_NVME_ROOT; do
    [[ -v "$name" && -n "${!name}" ]] || {
        echo "ERROR: required environment variable is missing or empty: $name" >&2
        exit 1
    }
done

readonly DEMIAN_EVALUATOR_REL="demian_wam/eval_utils/run_robocasa_eval.py"
readonly DEMIAN_TASK_REGISTRY_REL="demian_wam/eval_utils/robocasa_task_list.py"
readonly SERVER_MODULE="cosmos_framework.scripts.action_policy_server_robocasa365"
readonly SERVER_MODULE_REL="cosmos_framework/scripts/action_policy_server_robocasa365.py"
readonly SERVER_RECEIPT_HELPER_REL="examples/h_cluster/capture_robocasa365_server_receipt.py"
readonly CHECKPOINT_HELPER_REL="examples/h_cluster/cwam_serving_checkpoint_manifest.py"
readonly QUICK_HELPER_REL="examples/h_cluster/cwam_robocasa365_quick_eval.py"
readonly EXPECTED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"
readonly EXPECTED_DEMIAN_EVALUATOR_SHA256="c38ed1f4049d079b84c185cb572667a47eaf984c2c22b492efc8985401119836"
readonly EXPECTED_DEMIAN_TASK_REGISTRY_SHA256="e902311cf435986859ae01920548aba169e085ad6d26187cad9fb0d1f9c315de"
readonly PINNED_ASSET_MANIFEST_SHA256="3dc62249192bf2a8fd4bef1a31c6de9124cbc10421aa91b6769dee8e39e4d3c0"
readonly PINNED_ASSET_RECEIPT_SHA256="dc25dff483b1f42c904926cf72f91369f6ac94f0a4f50ab54e4c776299f93dca"
readonly PINNED_ASSET_ROBOCASA_COMMIT="29f7ce8814c1547f5af762a0997fbd4b64848dd7"
readonly PINNED_EGL_VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
readonly PINNED_QUICK_OUTPUT_PARENT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-quick-eval/runs"
readonly H_NVIDIA_DRIVER_BIN="/usr/local/nvidia/bin"
readonly H_NVIDIA_DRIVER_LIB="/usr/local/nvidia/lib"
readonly H_NVIDIA_DRIVER_LIB64="/usr/local/nvidia/lib64"
readonly H_RUNTIME_PATH="$H_NVIDIA_DRIVER_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
readonly H_WORKER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
readonly EVAL_STAGE=one_episode_gate
readonly EVAL_EVALUATION_CLASS=exploratory
readonly NUM_SHARDS=1
readonly EPISODES_PER_TASK=1
readonly BASE_SEED=42
readonly MAX_STEPS=400
readonly OPEN_LOOP_HORIZON=12
readonly POLICY_SAMPLER=unipc
readonly POLICY_SEED=42
readonly POLICY_GUIDANCE=1.0
readonly POLICY_NUM_STEPS=30
readonly BASE_PORT=16000
readonly SERVER_READY_TIMEOUT_SECONDS=1800
readonly CLEANUP_GRACE_SECONDS=30

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

verify_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA-256 is malformed"
    [[ -f "$path" && ! -L "$path" ]] || die "$label is not a regular non-symlink file: $path"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] ||
        die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

canonical_existing() {
    realpath -e -- "$1"
}

verify_git_checkout() {
    local label="$1"
    local root="$2"
    local expected_commit="$3"
    local top_level actual_commit
    top_level="$(git -c safe.directory="$root" -C "$root" rev-parse --show-toplevel)"
    [[ "$top_level" == "$root" ]] || die "$label Git root mismatch: $top_level"
    actual_commit="$(git -c safe.directory="$root" -C "$root" rev-parse HEAD)"
    [[ "$actual_commit" == "$expected_commit" ]] ||
        die "$label commit mismatch: expected=$expected_commit actual=$actual_commit"
    [[ -z "$(git -c safe.directory="$root" -C "$root" status --porcelain --untracked-files=all)" ]] ||
        die "$label checkout is dirty: $root"
}

echo "QUICK_NONFORMAL_MODE stage=$EVAL_STAGE evaluation_class=$EVAL_EVALUATION_CLASS formal_eligible=false asset_full_tree_verification_skipped=true lifecycle=raw_quick_nonsealed"
echo "QUICK_NONFORMAL_DISCLAIMER single_episode_only_not_formal_evidence_not_prior_gate_eligible"

[[ "$EVAL_EXPERIMENT_ID" == QUICK-NONFORMAL-* ]] ||
    die "EVAL_EXPERIMENT_ID must start with QUICK-NONFORMAL-"
[[ "$EVAL_LOGICAL_RUN_ID" == "$EVAL_EXPERIMENT_ID"-* ]] ||
    die "EVAL_LOGICAL_RUN_ID must be namespaced below EVAL_EXPERIMENT_ID"
[[ "$EVAL_RUN_NAME" == "$EVAL_LOGICAL_RUN_ID" ]] ||
    die "EVAL_RUN_NAME must equal EVAL_LOGICAL_RUN_ID"
for identity_name in EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME; do
    identity_value="${!identity_name}"
    [[ "${#identity_value}" -le 128 &&
        "$identity_value" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ &&
        "$identity_value" != "." &&
        "$identity_value" != ".." ]] ||
        die "$identity_name is not a strict path-safe identity"
done
[[ "$EVAL_TASKS_CSV" =~ ^[A-Za-z0-9_]+$ && "$EVAL_TASKS_CSV" != *,* ]] ||
    die "EVAL_TASKS_CSV must contain exactly one path-safe task name"
for commit_name in EXPECTED_SOURCE_COMMIT EXPECTED_ROBOCASA_COMMIT EXPECTED_ROBOSUITE_COMMIT; do
    [[ "${!commit_name}" =~ ^[0-9a-f]{40}$ ]] || die "$commit_name must be a full lowercase Git SHA"
done
[[ "$EXPECTED_ROBOCASA_COMMIT" == "$PINNED_ASSET_ROBOCASA_COMMIT" ]] ||
    die "EXPECTED_ROBOCASA_COMMIT differs from the published asset checkout"
[[ "$EVAL_EGL_VENDOR_JSON" == "$PINNED_EGL_VENDOR_JSON" ]] ||
    die "EVAL_EGL_VENDOR_JSON differs from the audited NVIDIA vendor JSON"

for forbidden_loader_variable in \
    LD_PRELOAD \
    LD_AUDIT \
    __EGL_VENDOR_LIBRARY_DIRS \
    LIBGL_ALWAYS_SOFTWARE \
    MESA_LOADER_DRIVER_OVERRIDE \
    LIBGL_DRIVERS_PATH; do
    [[ -z "${!forbidden_loader_variable-}" ]] ||
        die "$forbidden_loader_variable must be unset"
done
[[ "${__GLX_VENDOR_LIBRARY_NAME-}" == nvidia ]] ||
    die "__GLX_VENDOR_LIBRARY_NAME must be exactly nvidia"
[[ "${__EGL_VENDOR_LIBRARY_FILENAMES-}" == "$EVAL_EGL_VENDOR_JSON" ]] ||
    die "__EGL_VENDOR_LIBRARY_FILENAMES must equal EVAL_EGL_VENDOR_JSON"
[[ -n "$H_WORKER_LD_LIBRARY_PATH" ]] || die "H worker-init did not provide LD_LIBRARY_PATH"
IFS=: read -r -a driver_entries <<<"$H_WORKER_LD_LIBRARY_PATH"
for driver_entry in "${driver_entries[@]}"; do
    case "$driver_entry" in
        "$H_NVIDIA_DRIVER_LIB" | "$H_NVIDIA_DRIVER_LIB64") ;;
        *) die "H worker-init LD_LIBRARY_PATH contains an unaudited entry: $driver_entry" ;;
    esac
done
export LD_LIBRARY_PATH="$H_NVIDIA_DRIVER_LIB:$H_NVIDIA_DRIVER_LIB64"

COSMOS_SOURCE="$(canonical_existing "$COSMOS_SOURCE")"
COSMOS_ENV="$(canonical_existing "$COSMOS_ENV")"
ROBOCASA_ENV="$(canonical_existing "$ROBOCASA_ENV")"
DEMIAN_REPO="$(canonical_existing "$DEMIAN_REPO")"
ROBOCASA_REPO="$(canonical_existing "$ROBOCASA_REPO")"
ROBOSUITE_REPO="$(canonical_existing "$ROBOSUITE_REPO")"
SERVING_CHECKPOINT_PATH="$(canonical_existing "$SERVING_CHECKPOINT_PATH")"
if [[ -d "$SERVING_CHECKPOINT_PATH/model" ]]; then
    SERVING_CHECKPOINT_PATH="$(canonical_existing "$SERVING_CHECKPOINT_PATH/model")"
fi
CHECKPOINT_MANIFEST="$(canonical_existing "$CHECKPOINT_MANIFEST")"
TRAINING_CONFIG="$(canonical_existing "$TRAINING_CONFIG")"
ACTION_STATS="$(canonical_existing "$ACTION_STATS")"
SOURCE_MANIFEST="$(canonical_existing "$SOURCE_MANIFEST")"
COSMOS_ENVIRONMENT_MANIFEST="$(canonical_existing "$COSMOS_ENVIRONMENT_MANIFEST")"
ROBOCASA_ENVIRONMENT_MANIFEST="$(canonical_existing "$ROBOCASA_ENVIRONMENT_MANIFEST")"
SOURCE_DCP_EXPORT_RECEIPT="$(canonical_existing "$SOURCE_DCP_EXPORT_RECEIPT")"
EVAL_EGL_VENDOR_JSON="$(canonical_existing "$EVAL_EGL_VENDOR_JSON")"

readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"
[[ -x "$COSMOS_PY" && -x "$ROBOCASA_PY" ]] || die "pinned Python environments are not executable"
for required_file in \
    "$COSMOS_SOURCE/$SERVER_MODULE_REL" \
    "$COSMOS_SOURCE/$SERVER_RECEIPT_HELPER_REL" \
    "$COSMOS_SOURCE/$CHECKPOINT_HELPER_REL" \
    "$COSMOS_SOURCE/$QUICK_HELPER_REL" \
    "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" \
    "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL"; do
    [[ -f "$required_file" && ! -L "$required_file" ]] || die "required source file is missing: $required_file"
done
[[ "$SOURCE_MANIFEST" == "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" ]] ||
    die "SOURCE_MANIFEST must be the exact immutable snapshot manifest"
[[ "$(stat -c %a -- "$SOURCE_MANIFEST")" == 444 ]] ||
    die "SOURCE_MANIFEST must be mode 0444"

verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"
verify_sha256 checkpoint_manifest "$CHECKPOINT_MANIFEST" "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"
verify_sha256 training_config "$TRAINING_CONFIG" "$EXPECTED_TRAINING_CONFIG_SHA256"
verify_sha256 action_stats "$ACTION_STATS" "$EXPECTED_ACTION_STATS_SHA256"
verify_sha256 cosmos_environment_manifest \
    "$COSMOS_ENVIRONMENT_MANIFEST" "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256"
verify_sha256 robocasa_environment_manifest \
    "$ROBOCASA_ENVIRONMENT_MANIFEST" "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256"
verify_sha256 source_dcp_export_receipt \
    "$SOURCE_DCP_EXPORT_RECEIPT" "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"
verify_sha256 egl_vendor_json "$EVAL_EGL_VENDOR_JSON" "$EXPECTED_EGL_VENDOR_JSON_SHA256"
verify_sha256 demian_evaluator \
    "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" "$EXPECTED_DEMIAN_EVALUATOR_SHA256"
verify_sha256 demian_task_registry \
    "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL" "$EXPECTED_DEMIAN_TASK_REGISTRY_SHA256"

"$COSMOS_PY" -I -B - "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1:
    raise SystemExit("source snapshot manifest schema_version must be 1")
if manifest.get("source_head") != sys.argv[2] or manifest.get("source_dirty") is not False:
    raise SystemExit("source snapshot manifest does not bind the expected clean commit")
files = manifest.get("files")
if not isinstance(files, list) or manifest.get("file_count") != len(files):
    raise SystemExit("source snapshot manifest file inventory is malformed")
PY

if [[ -e "$COSMOS_SOURCE/.git" || -L "$COSMOS_SOURCE/.git" ]]; then
    verify_git_checkout cosmos "$COSMOS_SOURCE" "$EXPECTED_SOURCE_COMMIT"
fi
verify_git_checkout demian "$DEMIAN_REPO" "$EXPECTED_DEMIAN_COMMIT"
verify_git_checkout robocasa "$ROBOCASA_REPO" "$EXPECTED_ROBOCASA_COMMIT"
verify_git_checkout robosuite "$ROBOSUITE_REPO" "$EXPECTED_ROBOSUITE_COMMIT"

readonly ROBOCASA_ASSET_TREE_MANIFEST="$ROBOCASA_REPO/.cwam_provenance/asset-tree-manifest.jsonl"
readonly ROBOCASA_ASSET_TREE_RECEIPT="$ROBOCASA_REPO/.cwam_provenance/asset-tree-receipt.json"
verify_sha256 asset_manifest "$ROBOCASA_ASSET_TREE_MANIFEST" "$PINNED_ASSET_MANIFEST_SHA256"
verify_sha256 asset_publisher_receipt "$ROBOCASA_ASSET_TREE_RECEIPT" "$PINNED_ASSET_RECEIPT_SHA256"

readonly ALLOWED_OUTPUT_PARENT="$PINNED_QUICK_OUTPUT_PARENT"
[[ -d "$ALLOWED_OUTPUT_PARENT" && ! -L "$ALLOWED_OUTPUT_PARENT" ]] ||
    die "pinned quick output parent does not exist: $ALLOWED_OUTPUT_PARENT"
[[ "$(canonical_existing "$ALLOWED_OUTPUT_PARENT")" == "$ALLOWED_OUTPUT_PARENT" ]] ||
    die "pinned quick output parent is not canonical"
EVAL_OUTPUT_ROOT="$(realpath -m -- "$EVAL_OUTPUT_ROOT")"
readonly EXPERIMENT_OUTPUT_ROOT="$ALLOWED_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID"
readonly EXPECTED_EVAL_OUTPUT_ROOT="$EXPERIMENT_OUTPUT_ROOT/$EVAL_LOGICAL_RUN_ID"
[[ "$EVAL_OUTPUT_ROOT" == "$EXPECTED_EVAL_OUTPUT_ROOT" ]] ||
    die "EVAL_OUTPUT_ROOT must derive exactly from the quick experiment and logical IDs"
[[ ! -e "$EXPERIMENT_OUTPUT_ROOT" ]] || die "refusing to reuse quick experiment output root"
mkdir -- "$EXPERIMENT_OUTPUT_ROOT"
mkdir -- "$EVAL_OUTPUT_ROOT"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/quick-nonformal-control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/quick-nonformal-results"
readonly RESULTS_DIR="$RESULTS_ROOT/$EVAL_RUN_NAME"
readonly QUICK_ASSET_RECEIPT="$CONTROL_DIR/quick-asset-metadata-receipt.json"
readonly CHECKPOINT_VERIFICATION_RECEIPT="$CONTROL_DIR/quick-checkpoint-verification.json"
readonly SERVER_RECEIPT="$CONTROL_DIR/quick-server-runtime-receipt.json"
readonly SERVER_LOG="$CONTROL_DIR/quick-server.log"
readonly SERVER_RECEIPT_LOG="$CONTROL_DIR/quick-server-receipt.log"
readonly EVALUATOR_LOG="$CONTROL_DIR/quick-evaluator.log"
readonly QUICK_RESULT="$RESULTS_DIR/quick-result.json"
mkdir -- "$CONTROL_DIR" "$RESULTS_ROOT"

PYTHONPATH="$COSMOS_SOURCE/examples/h_cluster" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$QUICK_HELPER_REL" verify-assets \
    --asset-root "$ROBOCASA_REPO" \
    --expected-manifest-sha256 "$PINNED_ASSET_MANIFEST_SHA256" \
    --expected-publisher-receipt-sha256 "$PINNED_ASSET_RECEIPT_SHA256" \
    --expected-robocasa-commit "$EXPECTED_ROBOCASA_COMMIT" \
    --required-entry directory:robocasa \
    --required-entry file:robocasa/__init__.py \
    --required-entry directory:robocasa/environments \
    --required-entry directory:robocasa/models/assets \
    --output "$QUICK_ASSET_RECEIPT"

checkpoint_verification_started_seconds=$SECONDS
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$CHECKPOINT_HELPER_REL" \
    verify \
    --manifest "$CHECKPOINT_MANIFEST" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    >"$CHECKPOINT_VERIFICATION_RECEIPT"
echo "QUICK_CHECKPOINT_VERIFIED elapsed_seconds=$((SECONDS - checkpoint_verification_started_seconds))"

SERVER_PID=
EVALUATOR_PID=
NVME_CREATED=0
stop_child_pids() {
    local deadline pid
    local -a child_pids=("$@")
    for pid in "${child_pids[@]}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    deadline=$((SECONDS + CLEANUP_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        local running=0
        for pid in "${child_pids[@]}"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                running=1
            fi
        done
        ((running == 1)) || break
        sleep 1
    done
    for pid in "${child_pids[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
}
cleanup_nvme() {
    local expected="/nvme/.cwam-robocasa365-quick-eval/$EVAL_LOGICAL_RUN_ID"
    [[ "$NVME_CREATED" == 1 ]] || return 0
    [[ "$EVAL_NVME_ROOT" == "$expected" && ! -L "$EVAL_NVME_ROOT" ]] ||
        die "refusing to clean unexpected quick NVMe path: $EVAL_NVME_ROOT"
    [[ ! -d "$EVAL_NVME_ROOT" ]] || /usr/bin/rm --one-file-system -rf -- "$EVAL_NVME_ROOT"
    [[ ! -e "$EVAL_NVME_ROOT" && ! -L "$EVAL_NVME_ROOT" ]] ||
        die "scoped quick NVMe cleanup failed: $EVAL_NVME_ROOT"
    NVME_CREATED=0
}
cleanup() {
    local status="$1"
    trap - EXIT INT TERM
    stop_child_pids "$EVALUATOR_PID" "$SERVER_PID"
    cleanup_nvme || status=1
    exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

EVAL_NVME_ROOT="$(realpath -m -- "$EVAL_NVME_ROOT")"
[[ "$EVAL_NVME_ROOT" == "/nvme/.cwam-robocasa365-quick-eval/$EVAL_LOGICAL_RUN_ID" ]] ||
    die "EVAL_NVME_ROOT must derive exactly from EVAL_LOGICAL_RUN_ID"
[[ "$(findmnt -n -T /nvme -o TARGET)" == /nvme ]] || die "/nvme is not a distinct mount target"
[[ "$(findmnt -n -T /nvme -o SOURCE)" == /dev/nvme1n1 ]] || die "/nvme is not backed by /dev/nvme1n1"
[[ "$(findmnt -n -T /nvme -o FSTYPE)" == xfs ]] || die "/nvme is not xfs"
[[ ! -e "$EVAL_NVME_ROOT" ]] || die "refusing to reuse quick NVMe root"
mkdir -- "$EVAL_NVME_ROOT"
NVME_CREATED=1

"$COSMOS_PY" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"quick eval requires exactly one visible GPU, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "H200" not in name:
    raise SystemExit(f"quick eval requires one H200, got {name!r}")
print(f"QUICK_EVAL_CUDA_READY name={name} torch={torch.__version__} cuda={torch.version.cuda}")
PY

"$COSMOS_PY" - "$BASE_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", port))
print(f"QUICK_EVAL_PORT_READY port={port}")
PY

readonly SERVER_OUTPUT="$EVAL_NVME_ROOT/server-output"
readonly COMPILE_CACHE="$EVAL_NVME_ROOT/torchinductor"
LAUNCH_NONCE="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
mkdir -- "$COMPILE_CACHE"
(
    echo "QUICK_NONFORMAL_ARTIFACT formal_eligible=false asset_full_tree_verification_skipped=true lifecycle=raw_quick_nonsealed"
    cd "$COSMOS_SOURCE"
    exec env \
        -u WORLD_SIZE \
        -u RANK \
        -u LOCAL_RANK \
        -u MASTER_ADDR \
        -u MASTER_PORT \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH="$POLICY_PYTHONPATH" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE" \
        "$COSMOS_PY" -m "$SERVER_MODULE" \
        --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
        --config-file "$TRAINING_CONFIG" \
        --output-dir "$SERVER_OUTPUT" \
        --action-stats-path "$ACTION_STATS" \
        --host 127.0.0.1 \
        --port "$BASE_PORT" \
        --sampler "$POLICY_SAMPLER" \
        --seed "$POLICY_SEED" \
        --guidance "$POLICY_GUIDANCE" \
        --num-steps "$POLICY_NUM_STEPS" \
        --launch-nonce "$LAUNCH_NONCE"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

{
    echo "QUICK_NONFORMAL_ARTIFACT formal_eligible=false asset_full_tree_verification_skipped=true lifecycle=raw_quick_nonsealed"
    PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$SERVER_RECEIPT_HELPER_REL" \
        --uri "ws://127.0.0.1:$BASE_PORT" \
        --output "$SERVER_RECEIPT" \
        --timeout-seconds "$SERVER_READY_TIMEOUT_SECONDS" \
        --checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
        --training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
        --action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
        --source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --demian-commit "$EXPECTED_DEMIAN_COMMIT" \
        --shard-index 0 \
        --expected-launch-nonce "$LAUNCH_NONCE" \
        --expected-server-pid "$SERVER_PID" \
        --watch-pid "$SERVER_PID"
} >"$SERVER_RECEIPT_LOG"
echo "QUICK_POLICY_SERVER_READY"

(
    echo "QUICK_NONFORMAL_ARTIFACT formal_eligible=false asset_full_tree_verification_skipped=true lifecycle=raw_quick_nonsealed"
    exec env \
        -u LD_PRELOAD \
        -u LD_AUDIT \
        -u __EGL_VENDOR_LIBRARY_DIRS \
        -u LIBGL_ALWAYS_SOFTWARE \
        -u MESA_LOADER_DRIVER_OVERRIDE \
        -u LIBGL_DRIVERS_PATH \
        CUDA_VISIBLE_DEVICES=0 \
        MUJOCO_EGL_DEVICE_ID=0 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON" \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        PYTHONPATH="$EVAL_PYTHONPATH" \
        "$ROBOCASA_PY" "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" \
        --task-shard 0 \
        --num-shards "$NUM_SHARDS" \
        --episodes "$EPISODES_PER_TASK" \
        --episodes-start 0 \
        --max-steps "$MAX_STEPS" \
        --open-loop-horizon "$OPEN_LOOP_HORIZON" \
        --host 127.0.0.1 \
        --port "$BASE_PORT" \
        --checkpoint "$SERVING_CHECKPOINT_PATH" \
        --results-dir "$RESULTS_ROOT" \
        --run-name "$EVAL_RUN_NAME" \
        --no-save-video \
        --headless \
        --camera-height 256 \
        --camera-width 256 \
        --seed "$BASE_SEED" \
        --tasks "$EVAL_TASKS_CSV"
) >"$EVALUATOR_LOG" 2>&1 &
EVALUATOR_PID="$!"
if ! wait "$EVALUATOR_PID"; then
    tail -200 "$EVALUATOR_LOG" >&2 || true
    die "quick evaluator failed"
fi
EVALUATOR_PID=
stop_child_pids "$SERVER_PID"
SERVER_PID=

# Recheck cheap mutable provenance after rollout. Full asset content verification
# remains intentionally skipped.
if [[ -e "$COSMOS_SOURCE/.git" || -L "$COSMOS_SOURCE/.git" ]]; then
    verify_git_checkout cosmos "$COSMOS_SOURCE" "$EXPECTED_SOURCE_COMMIT"
fi
verify_git_checkout demian "$DEMIAN_REPO" "$EXPECTED_DEMIAN_COMMIT"
verify_git_checkout robocasa "$ROBOCASA_REPO" "$EXPECTED_ROBOCASA_COMMIT"
verify_git_checkout robosuite "$ROBOSUITE_REPO" "$EXPECTED_ROBOSUITE_COMMIT"

chmod 0444 \
    "$QUICK_ASSET_RECEIPT" \
    "$CHECKPOINT_VERIFICATION_RECEIPT" \
    "$SERVER_RECEIPT" \
    "$SERVER_LOG" \
    "$SERVER_RECEIPT_LOG" \
    "$EVALUATOR_LOG"

SEAL_ARGS=(
    seal-result
    --args-json "$RESULTS_DIR/args.json"
    --results-json "$RESULTS_DIR/results_shard0.json"
    --output "$QUICK_RESULT"
    --expected-task "$EVAL_TASKS_CSV"
    --expected-seed "$BASE_SEED"
    --expected-run-name "$EVAL_RUN_NAME"
    --expected-results-root "$RESULTS_ROOT"
    --expected-checkpoint "$SERVING_CHECKPOINT_PATH"
    --experiment-id "$EVAL_EXPERIMENT_ID"
    --logical-run-id "$EVAL_LOGICAL_RUN_ID"
    --input source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"
    --input checkpoint_manifest "$CHECKPOINT_MANIFEST" "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"
    --input training_config "$TRAINING_CONFIG" "$EXPECTED_TRAINING_CONFIG_SHA256"
    --input action_stats "$ACTION_STATS" "$EXPECTED_ACTION_STATS_SHA256"
    --input cosmos_environment_manifest \
        "$COSMOS_ENVIRONMENT_MANIFEST" "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256"
    --input robocasa_environment_manifest \
        "$ROBOCASA_ENVIRONMENT_MANIFEST" "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256"
    --input source_dcp_export_receipt \
        "$SOURCE_DCP_EXPORT_RECEIPT" "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"
    --input egl_vendor_json "$EVAL_EGL_VENDOR_JSON" "$EXPECTED_EGL_VENDOR_JSON_SHA256"
    --input asset_manifest "$ROBOCASA_ASSET_TREE_MANIFEST" "$PINNED_ASSET_MANIFEST_SHA256"
    --input asset_publisher_receipt "$ROBOCASA_ASSET_TREE_RECEIPT" "$PINNED_ASSET_RECEIPT_SHA256"
    --input quick_asset_metadata_receipt \
        "$QUICK_ASSET_RECEIPT" "$(sha256_file "$QUICK_ASSET_RECEIPT")"
    --input quick_checkpoint_verification \
        "$CHECKPOINT_VERIFICATION_RECEIPT" "$(sha256_file "$CHECKPOINT_VERIFICATION_RECEIPT")"
    --input quick_server_runtime_receipt "$SERVER_RECEIPT" "$(sha256_file "$SERVER_RECEIPT")"
)
PYTHONPATH="$COSMOS_SOURCE/examples/h_cluster" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$QUICK_HELPER_REL" "${SEAL_ARGS[@]}"

chmod 0555 \
    "$CONTROL_DIR" \
    "$RESULTS_ROOT" \
    "$EVAL_OUTPUT_ROOT" \
    "$EXPERIMENT_OUTPUT_ROOT"
echo "QUICK_NONFORMAL_EVAL_EXECUTION_SUCCESS formal_eligible=false asset_full_tree_verification_skipped=true"
echo "QUICK_RESULT=$QUICK_RESULT sha256=$(sha256_file "$QUICK_RESULT") lifecycle=raw_quick_nonsealed"
