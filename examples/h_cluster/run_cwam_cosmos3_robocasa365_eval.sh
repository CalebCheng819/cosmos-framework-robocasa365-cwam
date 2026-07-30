#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run one immutable CWAM/Cosmos3 RoboCasa365 gate or formal evaluation.
#
# The wrapper launches one policy server and one simulator shard per visible
# H200, captures every live server's metadata before rollouts, creates the
# post-server-start eval contract, and fails closed unless the two required CSV
# artifacts pass completeness/provenance validation.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot}"
: "${COSMOS_ENV:?Set COSMOS_ENV to the pinned Cosmos CUDA environment}"
: "${ROBOCASA_ENV:?Set ROBOCASA_ENV to the pinned RoboCasa evaluation environment}"
: "${DEMIAN_REPO:?Set DEMIAN_REPO to the pinned DeMiAn checkout}"
: "${ROBOCASA_REPO:?Set ROBOCASA_REPO to the pinned RoboCasa checkout}"
: "${ROBOSUITE_REPO:?Set ROBOSUITE_REPO to the pinned robosuite checkout}"
: "${SERVING_CHECKPOINT_PATH:?Set SERVING_CHECKPOINT_PATH to a consolidated checkpoint directory}"
: "${CHECKPOINT_MANIFEST:?Set CHECKPOINT_MANIFEST to the serving export manifest}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG to the exact training config YAML}"
: "${ACTION_STATS:?Set ACTION_STATS to the RoboCasa365 quantile_rot stats JSON}"
: "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to the immutable source manifest}"
: "${COSMOS_ENVIRONMENT_MANIFEST:?Set COSMOS_ENVIRONMENT_MANIFEST to the exact Cosmos environment manifest}"
: "${ROBOCASA_ENVIRONMENT_MANIFEST:?Set ROBOCASA_ENVIRONMENT_MANIFEST to the exact RoboCasa environment manifest}"
: "${EVAL_INTENT_MANIFEST:?Set EVAL_INTENT_MANIFEST to the immutable preregistered intent}"
: "${SOURCE_DCP_EXPORT_RECEIPT:?Set SOURCE_DCP_EXPORT_RECEIPT to the selected serving-DCP receipt}"
: "${EVAL_OUTPUT_ROOT:?Set EVAL_OUTPUT_ROOT to a new per-experiment directory}"
: "${EVAL_EXPERIMENT_ID:?Set EVAL_EXPERIMENT_ID}"
: "${EVAL_LOGICAL_RUN_ID:?Set EVAL_LOGICAL_RUN_ID}"
: "${EVAL_RUN_NAME:?Set EVAL_RUN_NAME}"
: "${EVAL_EVALUATION_CLASS:?Set EVAL_EVALUATION_CLASS}"
: "${SOURCE_GIT_PUSHED:?Set SOURCE_GIT_PUSHED to true or false}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?Set EXPECTED_SOURCE_MANIFEST_SHA256}"
: "${EXPECTED_CHECKPOINT_MANIFEST_SHA256:?Set EXPECTED_CHECKPOINT_MANIFEST_SHA256}"
: "${EXPECTED_TRAINING_CONFIG_SHA256:?Set EXPECTED_TRAINING_CONFIG_SHA256}"
: "${EXPECTED_ACTION_STATS_SHA256:?Set EXPECTED_ACTION_STATS_SHA256}"
: "${EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256:?Set the audited Cosmos environment manifest SHA256}"
: "${EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256:?Set the audited RoboCasa environment manifest SHA256}"
: "${EXPECTED_EVAL_INTENT_MANIFEST_SHA256:?Set the preregistered intent SHA256}"
: "${EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256:?Set the serving-DCP receipt SHA256}"
: "${EXPECTED_EGL_VENDOR_JSON_SHA256:?Set the pinned EGL vendor JSON SHA256}"
: "${EXPECTED_ROBOCASA_COMMIT:?Set EXPECTED_ROBOCASA_COMMIT}"
: "${EXPECTED_ROBOSUITE_COMMIT:?Set EXPECTED_ROBOSUITE_COMMIT}"
: "${MINIMUM_MEAN_SUCCESS_RATE:?Pre-register MINIMUM_MEAN_SUCCESS_RATE}"
: "${MINIMUM_SUCCESSFUL_TASKS:?Pre-register MINIMUM_SUCCESSFUL_TASKS}"
: "${ROBOCASA_ASSET_ROOT:?Set the exact published RoboCasa asset root}"
: "${ROBOCASA_ASSET_TREE_MANIFEST:?Set the exact asset-tree manifest}"
: "${ROBOCASA_ASSET_TREE_RECEIPT:?Set the exact asset publisher receipt}"
: "${EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256:?Set the audited asset manifest SHA256}"
: "${EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256:?Set the audited asset receipt SHA256}"
: "${EVAL_SUBMISSION_RESERVATION_PATH:?Set the submit-side reservation path}"
: "${EXPECTED_EVAL_SUBMISSION_PENDING_SHA256:?Set the submit-side reservation SHA256}"
: "${EVAL_SCHEDULER_READBACK_RECEIPT:?Set the accepted scheduler readback receipt path}"
: "${EVAL_SUBMISSION_UNKNOWN_PATH:?Set the mutually exclusive submission-unknown path}"
: "${EVAL_RUNTIME_CAPTURE_NONCE:?Set the submitter-generated runtime capture nonce}"
: "${EVAL_SUBMITTED_AT_UTC:?Set the submitter-generated UTC timestamp}"
: "${EVAL_SUBMISSION_MAX_AGE_SECONDS:?Set the pinned submission freshness window}"
: "${EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS:?Set the pinned future clock-skew window}"
: "${EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS:?Set the pinned scheduler-start window}"
: "${EVAL_EGL_VENDOR_JSON:?Set the exact NVIDIA EGL vendor JSON path}"
: "${EXPECTED_CONTAINER_IMAGE:?Set the exact digest-pinned container image}"
: "${RJOB_NAME:?Set the submit-side RJob showname}"
: "${KUBEBRAIN_RJOB_NAME:?H scheduler must inject KUBEBRAIN_RJOB_NAME}"
: "${KUBEBRAIN_RJOB_UID:?H scheduler must inject KUBEBRAIN_RJOB_UID}"

: "${EVAL_STAGE:?Set EVAL_STAGE explicitly}"
: "${EVAL_SPLIT:?Set EVAL_SPLIT explicitly}"
[[ -v EVAL_TASKS_CSV ]] || {
    echo "ERROR: Set EVAL_TASKS_CSV explicitly (empty only for formal)" >&2
    exit 1
}
[[ -v PRIOR_GATE_EVIDENCE ]] || {
    echo "ERROR: Set PRIOR_GATE_EVIDENCE explicitly (empty only for one_episode_gate)" >&2
    exit 1
}
[[ -v EXPECTED_PRIOR_GATE_EVIDENCE_SHA256 ]] || {
    echo "ERROR: Set EXPECTED_PRIOR_GATE_EVIDENCE_SHA256 explicitly (empty only for one_episode_gate)" >&2
    exit 1
}
: "${EPISODES_PER_TASK:?Set EPISODES_PER_TASK explicitly}"
: "${NUM_SHARDS:?Set NUM_SHARDS explicitly}"
: "${BASE_SEED:?Set BASE_SEED explicitly}"
: "${MAX_STEPS:?Set MAX_STEPS explicitly}"
: "${OPEN_LOOP_HORIZON:?Set OPEN_LOOP_HORIZON explicitly}"
: "${POLICY_SAMPLER:?Set POLICY_SAMPLER explicitly}"
: "${POLICY_SEED:?Set POLICY_SEED explicitly}"
: "${POLICY_GUIDANCE:?Set POLICY_GUIDANCE explicitly}"
: "${POLICY_NUM_STEPS:?Set POLICY_NUM_STEPS explicitly}"
: "${BASE_PORT:?Set BASE_PORT explicitly}"
: "${SERVER_READY_TIMEOUT_SECONDS:?Set SERVER_READY_TIMEOUT_SECONDS explicitly}"
: "${ALLOWED_OUTPUT_PARENT:?Set ALLOWED_OUTPUT_PARENT explicitly}"
: "${EVAL_NVME_ROOT:?Set EVAL_NVME_ROOT explicitly}"

readonly DEMIAN_EVALUATOR_REL="demian_wam/eval_utils/run_robocasa_eval.py"
readonly DEMIAN_TASK_REGISTRY_REL="demian_wam/eval_utils/robocasa_task_list.py"
readonly SERVER_MODULE="cosmos_framework.scripts.action_policy_server_robocasa365"
readonly SERVER_MODULE_REL="cosmos_framework/scripts/action_policy_server_robocasa365.py"
readonly RECEIPT_HELPER_REL="examples/h_cluster/capture_robocasa365_server_receipt.py"
readonly EVIDENCE_HELPER_REL="examples/h_cluster/cwam_robocasa365_eval_evidence.py"
readonly CHECKPOINT_HELPER_REL="examples/h_cluster/cwam_serving_checkpoint_manifest.py"
readonly INTENT_HELPER_REL="examples/h_cluster/cwam_robocasa365_eval_intent.py"
readonly RUNTIME_RECEIPT_HELPER_REL="examples/h_cluster/capture_robocasa365_eval_runtime.py"
readonly SUBMISSION_VERIFY_HELPER_REL="examples/h_cluster/verify_cwam_robocasa365_eval_submission.py"
readonly ASSET_VERIFY_HELPER_REL="examples/h_cluster/cwam_robocasa_asset_manifest.py"
readonly EXPECTED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"
readonly EXPECTED_DEMIAN_EVALUATOR_SHA256="c38ed1f4049d079b84c185cb572667a47eaf984c2c22b492efc8985401119836"
readonly EXPECTED_DEMIAN_TASK_REGISTRY_SHA256="e902311cf435986859ae01920548aba169e085ad6d26187cad9fb0d1f9c315de"
readonly H_NVIDIA_DRIVER_BIN="/usr/local/nvidia/bin"
readonly H_NVIDIA_DRIVER_LIB="/usr/local/nvidia/lib"
readonly H_NVIDIA_DRIVER_LIB64="/usr/local/nvidia/lib64"
readonly H_RUNTIME_PATH="$H_NVIDIA_DRIVER_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
readonly H_WORKER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
readonly PINNED_EVAL_OUTPUT_PARENT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-eval/runs"
readonly PINNED_EGL_VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_ASSET_MANIFEST_SHA256="3dc62249192bf2a8fd4bef1a31c6de9124cbc10421aa91b6769dee8e39e4d3c0"
readonly PINNED_ASSET_RECEIPT_SHA256="dc25dff483b1f42c904926cf72f91369f6ac94f0a4f50ab54e4c776299f93dca"
readonly PINNED_ASSET_ROBOCASA_COMMIT="29f7ce8814c1547f5af762a0997fbd4b64848dd7"
readonly SCHEDULER_READBACK_WAIT_SECONDS=900
readonly CLEANUP_GRACE_SECONDS=30

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_positive_int() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer, got $value"
}

require_nonnegative_int() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$label must be a non-negative integer, got $value"
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

verify_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA256 is malformed"
    [[ -f "$path" ]] || die "$label file does not exist: $path"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] ||
        die "$label SHA256 mismatch: expected=$expected actual=$actual path=$path"
}

canonical_existing() {
    realpath -e -- "$1"
}

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
[[ "${__GLX_VENDOR_LIBRARY_NAME-}" == "nvidia" ]] ||
    die "__GLX_VENDOR_LIBRARY_NAME must be exactly nvidia"
[[ "${__EGL_VENDOR_LIBRARY_FILENAMES-}" == "$EVAL_EGL_VENDOR_JSON" ]] ||
    die "__EGL_VENDOR_LIBRARY_FILENAMES must equal EVAL_EGL_VENDOR_JSON"
[[ "$EVAL_EGL_VENDOR_JSON" == "$PINNED_EGL_VENDOR_JSON" ]] ||
    die "EVAL_EGL_VENDOR_JSON differs from the audited NVIDIA vendor JSON path"
[[ "$EXPECTED_CONTAINER_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
    die "EXPECTED_CONTAINER_IMAGE differs from the audited image digest"

[[ -n "$H_WORKER_LD_LIBRARY_PATH" ]] || die "H worker-init did not provide LD_LIBRARY_PATH"
[[ "$H_WORKER_LD_LIBRARY_PATH" != :* &&
    "$H_WORKER_LD_LIBRARY_PATH" != *: &&
    "$H_WORKER_LD_LIBRARY_PATH" != *::* ]] ||
    die "H worker-init LD_LIBRARY_PATH contains an empty entry"
IFS=: read -r -a H_WORKER_DRIVER_LIBRARY_ENTRIES <<<"$H_WORKER_LD_LIBRARY_PATH"
[[ "${#H_WORKER_DRIVER_LIBRARY_ENTRIES[@]}" -gt 0 ]] ||
    die "H worker-init LD_LIBRARY_PATH is empty"
for driver_library_entry in "${H_WORKER_DRIVER_LIBRARY_ENTRIES[@]}"; do
    case "$driver_library_entry" in
        "$H_NVIDIA_DRIVER_LIB" | "$H_NVIDIA_DRIVER_LIB64") ;;
        *) die "H worker-init LD_LIBRARY_PATH contains an unaudited entry: $driver_library_entry" ;;
    esac
done
export LD_LIBRARY_PATH="$H_NVIDIA_DRIVER_LIB:$H_NVIDIA_DRIVER_LIB64"

require_positive_int NUM_SHARDS "$NUM_SHARDS"
require_positive_int EPISODES_PER_TASK "$EPISODES_PER_TASK"
require_positive_int MAX_STEPS "$MAX_STEPS"
require_positive_int OPEN_LOOP_HORIZON "$OPEN_LOOP_HORIZON"
require_positive_int POLICY_NUM_STEPS "$POLICY_NUM_STEPS"
require_positive_int SERVER_READY_TIMEOUT_SECONDS "$SERVER_READY_TIMEOUT_SECONDS"
require_nonnegative_int BASE_SEED "$BASE_SEED"
require_nonnegative_int POLICY_SEED "$POLICY_SEED"
require_nonnegative_int MINIMUM_SUCCESSFUL_TASKS "$MINIMUM_SUCCESSFUL_TASKS"
[[ "$EVAL_STAGE" == "one_episode_gate" || "$EVAL_STAGE" == "formal" ]] ||
    die "EVAL_STAGE must be one_episode_gate or formal"
[[ "$EVAL_EVALUATION_CLASS" == exploratory || "$EVAL_EVALUATION_CLASS" == confirmatory ]] ||
    die "EVAL_EVALUATION_CLASS must be exploratory or confirmatory"
[[ "$SOURCE_GIT_PUSHED" == true || "$SOURCE_GIT_PUSHED" == false ]] ||
    die "SOURCE_GIT_PUSHED must be true or false"
[[ "$EVAL_EVALUATION_CLASS" != confirmatory || "$SOURCE_GIT_PUSHED" == true ]] ||
    die "confirmatory evaluation requires SOURCE_GIT_PUSHED=true"
[[ "$EVAL_SPLIT" == "atomic" || "$EVAL_SPLIT" == "composite" ]] ||
    die "EVAL_SPLIT must be atomic or composite"
[[ "$POLICY_SAMPLER" == "unipc" || "$POLICY_SAMPLER" == "edm" ]] ||
    die "POLICY_SAMPLER must be unipc or edm"
[[ "$EVAL_SPLIT" == "atomic" ]] || die "Primary eval requires EVAL_SPLIT=atomic"
[[ "$BASE_SEED" == 42 ]] || die "BASE_SEED must be exactly 42"
[[ "$MAX_STEPS" == 400 ]] || die "MAX_STEPS must be exactly 400"
[[ "$OPEN_LOOP_HORIZON" == 12 ]] || die "OPEN_LOOP_HORIZON must be exactly 12"
[[ "$POLICY_SAMPLER" == "unipc" ]] || die "POLICY_SAMPLER must be exactly unipc"
[[ "$POLICY_SEED" == 42 ]] || die "POLICY_SEED must be exactly 42"
[[ "$POLICY_GUIDANCE" == "1.0" ]] || die "POLICY_GUIDANCE must be exactly 1.0"
[[ "$POLICY_NUM_STEPS" == 30 ]] || die "POLICY_NUM_STEPS must be exactly 30"
[[ "$BASE_PORT" == 16000 ]] || die "BASE_PORT must be exactly 16000"
[[ "$SERVER_READY_TIMEOUT_SECONDS" == 1800 ]] ||
    die "SERVER_READY_TIMEOUT_SECONDS must be exactly 1800"
[[ "$EVAL_SUBMISSION_MAX_AGE_SECONDS" == 259200 ]] ||
    die "EVAL_SUBMISSION_MAX_AGE_SECONDS must be exactly 259200"
[[ "$EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS" == 300 ]] ||
    die "EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS must be exactly 300"
[[ "$EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS" == 7200 ]] ||
    die "EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS must be exactly 7200"
[[ "$EVAL_RUNTIME_CAPTURE_NONCE" =~ ^[0-9a-f]{64}$ ]] ||
    die "EVAL_RUNTIME_CAPTURE_NONCE must be a submitter-generated lowercase nonce"
[[ "$EXPECTED_EVAL_SUBMISSION_PENDING_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    die "EXPECTED_EVAL_SUBMISSION_PENDING_SHA256 must be a lowercase SHA256"
for digest_name in \
    EXPECTED_EVAL_INTENT_MANIFEST_SHA256 \
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256 \
    EXPECTED_EGL_VENDOR_JSON_SHA256; do
    [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] ||
        die "$digest_name must be a lowercase SHA256"
done
[[ "$EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256" == "$PINNED_ASSET_MANIFEST_SHA256" ]] ||
    die "Asset manifest SHA256 differs from the audited publication"
[[ "$EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256" == "$PINNED_ASSET_RECEIPT_SHA256" ]] ||
    die "Asset receipt SHA256 differs from the audited publication"
[[ "$EXPECTED_ROBOCASA_COMMIT" == "$PINNED_ASSET_ROBOCASA_COMMIT" ]] ||
    die "RoboCasa commit differs from the asset-expanded audited checkout"
for commit in \
    "$EXPECTED_SOURCE_COMMIT" \
    "$EXPECTED_DEMIAN_COMMIT" \
    "$EXPECTED_ROBOCASA_COMMIT" \
    "$EXPECTED_ROBOSUITE_COMMIT"; do
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "Expected Git commits must be full lowercase SHA-1 values"
done
if [[ "$EVAL_STAGE" == "one_episode_gate" ]]; then
    [[ "$EPISODES_PER_TASK" == 1 ]] || die "one_episode_gate requires EPISODES_PER_TASK=1"
    [[ "$NUM_SHARDS" == 1 ]] || die "one_episode_gate requires NUM_SHARDS=1"
    [[ -n "$EVAL_TASKS_CSV" && "$EVAL_TASKS_CSV" != *,* ]] ||
        die "one_episode_gate requires exactly one EVAL_TASKS_CSV task"
    [[ "$MINIMUM_MEAN_SUCCESS_RATE" == "0.0" ]] ||
        die "one_episode_gate MINIMUM_MEAN_SUCCESS_RATE must be 0.0"
    [[ "$MINIMUM_SUCCESSFUL_TASKS" == 0 ]] ||
        die "one_episode_gate MINIMUM_SUCCESSFUL_TASKS must be 0"
    [[ -z "$PRIOR_GATE_EVIDENCE" && -z "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" ]] ||
        die "one_episode_gate must not bind prior gate evidence"
else
    [[ "$EPISODES_PER_TASK" == 20 ]] || die "formal evaluation requires EPISODES_PER_TASK=20"
    [[ "$NUM_SHARDS" == 8 ]] || die "formal evaluation requires NUM_SHARDS=8"
    [[ -z "$EVAL_TASKS_CSV" ]] || die "formal evaluation must use the complete canonical split"
    [[ "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
        die "formal EXPECTED_PRIOR_GATE_EVIDENCE_SHA256 must be a lowercase SHA256"
    PRIOR_GATE_EVIDENCE="$(canonical_existing "$PRIOR_GATE_EVIDENCE")"
    [[ -f "$PRIOR_GATE_EVIDENCE" && ! -L "$PRIOR_GATE_EVIDENCE" ]] ||
        die "formal PRIOR_GATE_EVIDENCE must be a canonical regular file"
    [[ "$(stat -c %a -- "$PRIOR_GATE_EVIDENCE")" == 444 ]] ||
        die "formal PRIOR_GATE_EVIDENCE must be immutable mode 0444"
    verify_sha256 \
        prior_gate_evidence \
        "$PRIOR_GATE_EVIDENCE" \
        "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256"
fi
for identity_name in EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME; do
    identity_value="${!identity_name}"
    [[ "${#identity_value}" -le 128 &&
        "$identity_value" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ &&
        "$identity_value" != "." &&
        "$identity_value" != ".." ]] ||
        die "$identity_name is not a strict path-safe identity"
done
[[ "$EVAL_LOGICAL_RUN_ID" == "$EVAL_EXPERIMENT_ID"-* ]] ||
    die "EVAL_LOGICAL_RUN_ID must be namespaced below EVAL_EXPERIMENT_ID"
[[ "$EVAL_RUN_NAME" == "$EVAL_LOGICAL_RUN_ID" ]] ||
    die "EVAL_RUN_NAME must equal EVAL_LOGICAL_RUN_ID"
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    die "RJOB_NAME is not a strict H RJob showname"
[[ "$KUBEBRAIN_RJOB_NAME" == "$RJOB_NAME"-* ]] ||
    die "KUBEBRAIN_RJOB_NAME does not derive from RJOB_NAME"
[[ "$KUBEBRAIN_RJOB_UID" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89aAbB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$ ]] ||
    die "KUBEBRAIN_RJOB_UID is not a UUID"

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
EVAL_INTENT_MANIFEST="$(canonical_existing "$EVAL_INTENT_MANIFEST")"
SOURCE_DCP_EXPORT_RECEIPT="$(canonical_existing "$SOURCE_DCP_EXPORT_RECEIPT")"
ALLOWED_OUTPUT_PARENT="$(canonical_existing "$ALLOWED_OUTPUT_PARENT")"
ROBOCASA_ASSET_ROOT="$(canonical_existing "$ROBOCASA_ASSET_ROOT")"
ROBOCASA_ASSET_TREE_MANIFEST="$(canonical_existing "$ROBOCASA_ASSET_TREE_MANIFEST")"
ROBOCASA_ASSET_TREE_RECEIPT="$(canonical_existing "$ROBOCASA_ASSET_TREE_RECEIPT")"
[[ -f "$EVAL_EGL_VENDOR_JSON" ]] || die "Pinned NVIDIA EGL vendor JSON does not exist"

[[ -x "$COSMOS_ENV/bin/python" ]] || die "COSMOS_ENV Python is not executable"
[[ -x "$ROBOCASA_ENV/bin/python" ]] || die "ROBOCASA_ENV Python is not executable"
[[ -d "$SERVING_CHECKPOINT_PATH" ]] || die "SERVING_CHECKPOINT_PATH must be a directory"
[[ -f "$COSMOS_SOURCE/$SERVER_MODULE_REL" ]] || die "Missing RoboCasa365 policy server module"
[[ -f "$COSMOS_SOURCE/$RECEIPT_HELPER_REL" ]] || die "Missing server receipt helper"
[[ -f "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" ]] || die "Missing eval evidence helper"
[[ -f "$COSMOS_SOURCE/$CHECKPOINT_HELPER_REL" ]] || die "Missing serving checkpoint manifest helper"
[[ -f "$COSMOS_SOURCE/$INTENT_HELPER_REL" ]] || die "Missing eval intent helper"
[[ -f "$COSMOS_SOURCE/$RUNTIME_RECEIPT_HELPER_REL" ]] || die "Missing eval runtime receipt helper"
[[ -f "$COSMOS_SOURCE/$SUBMISSION_VERIFY_HELPER_REL" ]] || die "Missing submission verification helper"
[[ -f "$COSMOS_SOURCE/$ASSET_VERIFY_HELPER_REL" ]] || die "Missing RoboCasa asset verification helper"
[[ -f "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" ]] || die "Missing DeMiAn evaluator"
[[ -f "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL" ]] || die "Missing DeMiAn task registry"
[[ "$ROBOCASA_ASSET_ROOT" == "$ROBOCASA_REPO" ]] ||
    die "ROBOCASA_ASSET_ROOT must equal the exact RoboCasa source checkout"
[[ "$ROBOCASA_ASSET_TREE_MANIFEST" == "$ROBOCASA_REPO/.cwam_provenance/asset-tree-manifest.jsonl" ]] ||
    die "ROBOCASA_ASSET_TREE_MANIFEST is outside the exact asset provenance directory"
[[ "$ROBOCASA_ASSET_TREE_RECEIPT" == "$ROBOCASA_REPO/.cwam_provenance/asset-tree-receipt.json" ]] ||
    die "ROBOCASA_ASSET_TREE_RECEIPT is outside the exact asset provenance directory"

[[ "$SOURCE_MANIFEST" == "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" ]] ||
    die "SOURCE_MANIFEST must be the exact SNAPSHOT_MANIFEST.json under COSMOS_SOURCE"
[[ "$(stat -c %a -- "$SOURCE_MANIFEST")" == 444 ]] ||
    die "SOURCE_MANIFEST must be immutable mode 0444"
verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"
"$COSMOS_ENV/bin/python" -I -B - "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1:
    raise SystemExit("ERROR: source snapshot manifest schema_version must be 1")
if manifest.get("source_head") != sys.argv[2]:
    raise SystemExit("ERROR: source snapshot manifest does not bind EXPECTED_SOURCE_COMMIT")
if manifest.get("source_dirty") is not False:
    raise SystemExit("ERROR: source snapshot manifest must assert source_dirty=false")
files = manifest.get("files")
if not isinstance(files, list) or manifest.get("file_count") != len(files):
    raise SystemExit("ERROR: source snapshot manifest file inventory is malformed")
PY

if [[ -e "$COSMOS_SOURCE/.git" || -L "$COSMOS_SOURCE/.git" ]]; then
    SOURCE_GIT_TOPLEVEL="$(
        git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse --show-toplevel
    )"
    [[ "$SOURCE_GIT_TOPLEVEL" == "$COSMOS_SOURCE" ]] ||
        die "COSMOS_SOURCE .git does not describe the exact source root"
    SOURCE_COMMIT="$(git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse HEAD)"
    [[ "$SOURCE_COMMIT" == "$EXPECTED_SOURCE_COMMIT" ]] ||
        die "Cosmos source commit mismatch: expected=$EXPECTED_SOURCE_COMMIT actual=$SOURCE_COMMIT"
    [[ -z "$(git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" status --porcelain --untracked-files=all)" ]] ||
        die "Pinned source checkout is dirty: $COSMOS_SOURCE"
fi
DEMIAN_COMMIT="$(git -c safe.directory="$DEMIAN_REPO" -C "$DEMIAN_REPO" rev-parse HEAD)"
ROBOCASA_COMMIT="$(git -c safe.directory="$ROBOCASA_REPO" -C "$ROBOCASA_REPO" rev-parse HEAD)"
ROBOSUITE_COMMIT="$(git -c safe.directory="$ROBOSUITE_REPO" -C "$ROBOSUITE_REPO" rev-parse HEAD)"
[[ "$DEMIAN_COMMIT" == "$EXPECTED_DEMIAN_COMMIT" ]] ||
    die "DeMiAn commit mismatch: expected=$EXPECTED_DEMIAN_COMMIT actual=$DEMIAN_COMMIT"
[[ "$ROBOCASA_COMMIT" == "$EXPECTED_ROBOCASA_COMMIT" ]] ||
    die "RoboCasa commit mismatch: expected=$EXPECTED_ROBOCASA_COMMIT actual=$ROBOCASA_COMMIT"
[[ "$ROBOSUITE_COMMIT" == "$EXPECTED_ROBOSUITE_COMMIT" ]] ||
    die "robosuite commit mismatch: expected=$EXPECTED_ROBOSUITE_COMMIT actual=$ROBOSUITE_COMMIT"
for repo in "$DEMIAN_REPO" "$ROBOCASA_REPO" "$ROBOSUITE_REPO"; do
    [[ -z "$(git -c safe.directory="$repo" -C "$repo" status --porcelain --untracked-files=all)" ]] ||
        die "Pinned source checkout is dirty: $repo"
done

verify_sha256 checkpoint_manifest "$CHECKPOINT_MANIFEST" "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"
verify_sha256 training_config "$TRAINING_CONFIG" "$EXPECTED_TRAINING_CONFIG_SHA256"
verify_sha256 action_stats "$ACTION_STATS" "$EXPECTED_ACTION_STATS_SHA256"
verify_sha256 \
    cosmos_environment_manifest \
    "$COSMOS_ENVIRONMENT_MANIFEST" \
    "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256"
verify_sha256 \
    robocasa_environment_manifest \
    "$ROBOCASA_ENVIRONMENT_MANIFEST" \
    "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256"
verify_sha256 eval_intent_manifest "$EVAL_INTENT_MANIFEST" "$EXPECTED_EVAL_INTENT_MANIFEST_SHA256"
verify_sha256 \
    source_dcp_export_receipt \
    "$SOURCE_DCP_EXPORT_RECEIPT" \
    "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"
verify_sha256 egl_vendor_json "$EVAL_EGL_VENDOR_JSON" "$EXPECTED_EGL_VENDOR_JSON_SHA256"
[[ "$(stat -c %a -- "$EVAL_INTENT_MANIFEST")" == 444 ]] ||
    die "EVAL_INTENT_MANIFEST must be immutable mode 0444"
[[ "$(stat -c %a -- "$SOURCE_DCP_EXPORT_RECEIPT")" == 444 ]] ||
    die "SOURCE_DCP_EXPORT_RECEIPT must be immutable mode 0444"
verify_sha256 demian_evaluator \
    "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" \
    "$EXPECTED_DEMIAN_EVALUATOR_SHA256"
verify_sha256 demian_task_registry \
    "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL" \
    "$EXPECTED_DEMIAN_TASK_REGISTRY_SHA256"
verify_sha256 \
    robocasa_asset_manifest \
    "$ROBOCASA_ASSET_TREE_MANIFEST" \
    "$EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256"
verify_sha256 \
    robocasa_asset_receipt \
    "$ROBOCASA_ASSET_TREE_RECEIPT" \
    "$EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256"
if [[ "$EVAL_STAGE" == one_episode_gate ]]; then
    INTENT_CPU=16
    INTENT_MEMORY_MIB=187500
    INTENT_GPU=1
else
    INTENT_CPU=128
    INTENT_MEMORY_MIB=1500000
    INTENT_GPU=8
fi
INTENT_VERIFY_ARGS=(
    verify
    --manifest "$EVAL_INTENT_MANIFEST"
    --expected-manifest-sha256 "$EXPECTED_EVAL_INTENT_MANIFEST_SHA256"
    --experiment-id "$EVAL_EXPERIMENT_ID"
    --logical-run-id "$EVAL_LOGICAL_RUN_ID"
    --run-name "$EVAL_RUN_NAME"
    --stage "$EVAL_STAGE"
    --evaluation-class "$EVAL_EVALUATION_CLASS"
    --git-pushed "$SOURCE_GIT_PUSHED"
    --split "$EVAL_SPLIT"
    --tasks-csv "$EVAL_TASKS_CSV"
    --episodes-per-task "$EPISODES_PER_TASK"
    --num-shards "$NUM_SHARDS"
    --base-seed "$BASE_SEED"
    --max-steps "$MAX_STEPS"
    --open-loop-horizon "$OPEN_LOOP_HORIZON"
    --policy-sampler "$POLICY_SAMPLER"
    --policy-seed "$POLICY_SEED"
    --policy-guidance "$POLICY_GUIDANCE"
    --policy-num-steps "$POLICY_NUM_STEPS"
    --base-port "$BASE_PORT"
    --server-ready-timeout-seconds "$SERVER_READY_TIMEOUT_SECONDS"
    --minimum-mean-success-rate "$MINIMUM_MEAN_SUCCESS_RATE"
    --minimum-successful-tasks "$MINIMUM_SUCCESSFUL_TASKS"
    --source-root "$COSMOS_SOURCE"
    --source-commit "$EXPECTED_SOURCE_COMMIT"
    --source-manifest "$SOURCE_MANIFEST"
    --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256"
    --serving-checkpoint-path "$SERVING_CHECKPOINT_PATH"
    --checkpoint-manifest "$CHECKPOINT_MANIFEST"
    --expected-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"
    --source-dcp-export-receipt "$SOURCE_DCP_EXPORT_RECEIPT"
    --expected-source-dcp-export-receipt-sha256 "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256"
    --training-config "$TRAINING_CONFIG"
    --expected-training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256"
    --action-stats "$ACTION_STATS"
    --expected-action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256"
    --cosmos-env-root "$COSMOS_ENV"
    --cosmos-environment-manifest "$COSMOS_ENVIRONMENT_MANIFEST"
    --expected-cosmos-environment-manifest-sha256 "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256"
    --robocasa-env-root "$ROBOCASA_ENV"
    --robocasa-environment-manifest "$ROBOCASA_ENVIRONMENT_MANIFEST"
    --expected-robocasa-environment-manifest-sha256 "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256"
    --robocasa-asset-root "$ROBOCASA_ASSET_ROOT"
    --robocasa-asset-manifest "$ROBOCASA_ASSET_TREE_MANIFEST"
    --expected-robocasa-asset-manifest-sha256 "$EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256"
    --robocasa-asset-receipt "$ROBOCASA_ASSET_TREE_RECEIPT"
    --expected-robocasa-asset-receipt-sha256 "$EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256"
    --demian-root "$DEMIAN_REPO"
    --demian-commit "$EXPECTED_DEMIAN_COMMIT"
    --robocasa-root "$ROBOCASA_REPO"
    --robocasa-commit "$EXPECTED_ROBOCASA_COMMIT"
    --robosuite-root "$ROBOSUITE_REPO"
    --robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT"
    --container-image "$EXPECTED_CONTAINER_IMAGE"
    --egl-vendor-json "$EVAL_EGL_VENDOR_JSON"
    --expected-egl-vendor-json-sha256 "$EXPECTED_EGL_VENDOR_JSON_SHA256"
    --cpu "$INTENT_CPU"
    --memory-mib "$INTENT_MEMORY_MIB"
    --gpu "$INTENT_GPU"
    --quota-group eailabagent_gpu
    --negative-tag node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn
    --task-type normal
    --private-machine group
    --priority 9
    --restart-policy never
    --backoff-limit 1
    --replicas 1
)
if [[ "$EVAL_STAGE" == formal ]]; then
    INTENT_VERIFY_ARGS+=(
        --prior-gate-evidence "$PRIOR_GATE_EVIDENCE"
        --expected-prior-gate-evidence-sha256 "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256"
    )
fi
PYTHONPATH="$COSMOS_SOURCE/examples/h_cluster" \
    "$COSMOS_ENV/bin/python" "$COSMOS_SOURCE/$INTENT_HELPER_REL" \
    "${INTENT_VERIFY_ARGS[@]}" >/dev/null
if [[ "$EVAL_STAGE" == "formal" ]]; then
    PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_ENV/bin/python" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
        verify-prior-gate \
        --gate-evidence "$PRIOR_GATE_EVIDENCE" \
        --expected-gate-evidence-sha256 "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" \
        --current-source-root "$COSMOS_SOURCE" \
        --current-source-commit "$EXPECTED_SOURCE_COMMIT" \
        --current-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --current-checkpoint-path "$SERVING_CHECKPOINT_PATH" \
        --current-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
        --current-source-dcp-export-receipt-sha256 "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256" \
        --current-asset-manifest-sha256 "$EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256" \
        --current-asset-publisher-receipt-sha256 "$EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256" \
        --current-training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
        --current-action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
        --current-cosmos-environment-manifest-sha256 "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256" \
        --current-robocasa-environment-manifest-sha256 "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256" \
        --current-demian-commit "$EXPECTED_DEMIAN_COMMIT" \
        --current-robocasa-commit "$EXPECTED_ROBOCASA_COMMIT" \
        --current-robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT" \
        >/dev/null
fi

readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"

EVAL_OUTPUT_ROOT="$(realpath -m -- "$EVAL_OUTPUT_ROOT")"
[[ "$ALLOWED_OUTPUT_PARENT" == "$PINNED_EVAL_OUTPUT_PARENT" ]] ||
    die "ALLOWED_OUTPUT_PARENT differs from the pinned eval runs root"
EXPECTED_EVAL_OUTPUT_ROOT="$ALLOWED_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
[[ "$EVAL_OUTPUT_ROOT" == "$EXPECTED_EVAL_OUTPUT_ROOT" ]] ||
    die "EVAL_OUTPUT_ROOT must be derived exactly from experiment and logical run IDs"
case "$EVAL_OUTPUT_ROOT/" in
    "$ALLOWED_OUTPUT_PARENT"/*/) ;;
    *) die "EVAL_OUTPUT_ROOT must be a strict child of $ALLOWED_OUTPUT_PARENT" ;;
esac
readonly EXPERIMENT_OUTPUT_ROOT="$ALLOWED_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID"
[[ ! -e "$EXPERIMENT_OUTPUT_ROOT" ]] ||
    die "Refusing to reuse experiment output root: $EXPERIMENT_OUTPUT_ROOT"
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || die "Refusing to reuse EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
mkdir -- "$EXPERIMENT_OUTPUT_ROOT"
mkdir -- "$EVAL_OUTPUT_ROOT"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly RESULTS_DIR="$RESULTS_ROOT/$EVAL_RUN_NAME"
readonly CONTRACT_PATH="$CONTROL_DIR/eval-contract.json"
readonly SUMMARY_CSV="$RESULTS_DIR/summary.csv"
readonly TASK_SUCCESS_RATES_CSV="$RESULTS_DIR/task_success_rates.csv"
readonly EVIDENCE_JSON="$RESULTS_DIR/evaluation_evidence.json"
readonly CHECKPOINT_VERIFICATION_RECEIPT="$CONTROL_DIR/checkpoint-verification.json"
readonly EVAL_RUNTIME_RECEIPT="$CONTROL_DIR/eval-runtime-receipt.json"
readonly SUBMISSION_VERIFICATION_RECEIPT="$CONTROL_DIR/submission-verification.json"
readonly ASSET_VERIFICATION_RECEIPT="$CONTROL_DIR/asset-tree-verification.json"
readonly POST_ROLLOUT_VERIFICATION_RECEIPT="$CONTROL_DIR/post-rollout-verification.json"
readonly SEALED_EVIDENCE_VERIFICATION_RECEIPT="$CONTROL_DIR/sealed-evidence-verification.json"
mkdir -- "$CONTROL_DIR" "$RESULTS_ROOT"

EVAL_SCHEDULER_READBACK_RECEIPT="$(realpath -m -- "$EVAL_SCHEDULER_READBACK_RECEIPT")"
EVAL_SUBMISSION_UNKNOWN_PATH="$(realpath -m -- "$EVAL_SUBMISSION_UNKNOWN_PATH")"
[[ "$(basename -- "$EVAL_SCHEDULER_READBACK_RECEIPT")" == scheduler-readback-receipt.json ]] ||
    die "EVAL_SCHEDULER_READBACK_RECEIPT has a non-canonical terminal filename"
[[ "$(basename -- "$EVAL_SUBMISSION_UNKNOWN_PATH")" == submission-unknown.json ]] ||
    die "EVAL_SUBMISSION_UNKNOWN_PATH has a non-canonical terminal filename"
[[ "$(dirname -- "$EVAL_SCHEDULER_READBACK_RECEIPT")" == "$(dirname -- "$EVAL_SUBMISSION_UNKNOWN_PATH")" ]] ||
    die "accepted and unknown terminal records must share one reservation directory"

require_accepted_submission_terminal_state() {
    local terminal_parent
    terminal_parent="$(dirname -- "$EVAL_SCHEDULER_READBACK_RECEIPT")"
    [[ -d "$terminal_parent" && ! -L "$terminal_parent" ]] ||
        die "submission terminal-state parent is not a regular directory"
    [[ "$(realpath -e -- "$terminal_parent")" == "$terminal_parent" ]] ||
        die "submission terminal-state parent is not canonical"
    [[ ! -e "$EVAL_SUBMISSION_UNKNOWN_PATH" && ! -L "$EVAL_SUBMISSION_UNKNOWN_PATH" ]] ||
        die "submission-unknown terminal record exists"
    [[ -f "$EVAL_SCHEDULER_READBACK_RECEIPT" && ! -L "$EVAL_SCHEDULER_READBACK_RECEIPT" ]] ||
        die "accepted scheduler readback receipt is not a regular file"
    [[ "$(stat -c %a -- "$EVAL_SCHEDULER_READBACK_RECEIPT")" == 444 ]] ||
        die "accepted scheduler readback receipt is not sealed mode 0444"
    [[ "$(stat -c %a -- "$terminal_parent")" == 555 ]] ||
        die "submission reservation directory is not sealed mode 0555"
}

for ((waited = 0; waited < SCHEDULER_READBACK_WAIT_SECONDS; waited++)); do
    if [[ ( -e "$EVAL_SCHEDULER_READBACK_RECEIPT" || -L "$EVAL_SCHEDULER_READBACK_RECEIPT" ) &&
        ( -e "$EVAL_SUBMISSION_UNKNOWN_PATH" || -L "$EVAL_SUBMISSION_UNKNOWN_PATH" ) ]]; then
        die "accepted and unknown submission terminal records both exist"
    fi
    [[ ! -e "$EVAL_SUBMISSION_UNKNOWN_PATH" && ! -L "$EVAL_SUBMISSION_UNKNOWN_PATH" ]] ||
        die "submission became unknown before worker verification"
    if [[ -s "$EVAL_SCHEDULER_READBACK_RECEIPT" &&
        ! -L "$EVAL_SCHEDULER_READBACK_RECEIPT" &&
        "$(stat -c %a -- "$EVAL_SCHEDULER_READBACK_RECEIPT")" == 444 &&
        "$(stat -c %a -- "$(dirname -- "$EVAL_SCHEDULER_READBACK_RECEIPT")")" == 555 ]]; then
        break
    fi
    sleep 1
done
[[ -s "$EVAL_SCHEDULER_READBACK_RECEIPT" ]] ||
    die "Timed out waiting for the accepted scheduler readback receipt"
require_accepted_submission_terminal_state

/usr/bin/env \
    PATH="$H_RUNTIME_PATH" \
    PYTHONPATH= \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SUBMISSION_VERIFY_HELPER_REL" \
    --reservation "$EVAL_SUBMISSION_RESERVATION_PATH" \
    --expected-reservation-sha256 "$EXPECTED_EVAL_SUBMISSION_PENDING_SHA256" \
    --scheduler-readback-receipt "$EVAL_SCHEDULER_READBACK_RECEIPT" \
    --submission-unknown "$EVAL_SUBMISSION_UNKNOWN_PATH" \
    --experiment-id "$EVAL_EXPERIMENT_ID" \
    --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --run-name "$EVAL_RUN_NAME" \
    --submission-nonce "$EVAL_RUNTIME_CAPTURE_NONCE" \
    --submitted-at-utc "$EVAL_SUBMITTED_AT_UTC" \
    --observed-scheduler-rjob-name "$KUBEBRAIN_RJOB_NAME" \
    --observed-scheduler-rjob-uid "$KUBEBRAIN_RJOB_UID" \
    --max-age-seconds "$EVAL_SUBMISSION_MAX_AGE_SECONDS" \
    --max-future-skew-seconds "$EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS" \
    --max-start-to-verification-seconds "$EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS" \
    --output "$SUBMISSION_VERIFICATION_RECEIPT"
echo "EVAL_SUBMISSION_VERIFIED receipt=$SUBMISSION_VERIFICATION_RECEIPT"
require_accepted_submission_terminal_state

asset_verification_started_seconds=$SECONDS
/usr/bin/env \
    PATH="$H_RUNTIME_PATH" \
    PYTHONPATH= \
    "$COSMOS_PY" "$COSMOS_SOURCE/$ASSET_VERIFY_HELPER_REL" \
    verify \
    --asset-root "$ROBOCASA_ASSET_ROOT" \
    --output "$ASSET_VERIFICATION_RECEIPT"
"$COSMOS_PY" - \
    "$ASSET_VERIFICATION_RECEIPT" \
    "$ROBOCASA_ASSET_ROOT" \
    "$EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256" \
    "$EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256" \
    "$EXPECTED_ROBOCASA_COMMIT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {
    "asset_root": sys.argv[2],
    "manifest_sha256": sys.argv[3],
    "publisher_receipt_sha256": sys.argv[4],
    "robocasa_git_commit": sys.argv[5],
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(
            f"asset verification receipt {key} mismatch: expected={value!r}, "
            f"actual={receipt.get(key)!r}"
        )
if receipt.get("status") != "passed":
    raise SystemExit("asset verification receipt did not pass")
PY
asset_verification_elapsed_seconds=$((SECONDS - asset_verification_started_seconds))
echo "ROBOCASA_ASSET_TREE_VERIFIED" \
    "receipt=$ASSET_VERIFICATION_RECEIPT" \
    "elapsed_seconds=$asset_verification_elapsed_seconds"

checkpoint_verification_started_seconds=$SECONDS
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$CHECKPOINT_HELPER_REL" \
    verify \
    --manifest "$CHECKPOINT_MANIFEST" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    >"$CHECKPOINT_VERIFICATION_RECEIPT"
checkpoint_verification_elapsed_seconds=$((SECONDS - checkpoint_verification_started_seconds))
echo "SERVING_CHECKPOINT_VERIFIED" \
    "receipt=$CHECKPOINT_VERIFICATION_RECEIPT" \
    "elapsed_seconds=$checkpoint_verification_elapsed_seconds"

/usr/bin/env \
    PATH="$H_RUNTIME_PATH" \
    PYTHONPATH= \
    "$COSMOS_PY" "$COSMOS_SOURCE/$RUNTIME_RECEIPT_HELPER_REL" \
    --cosmos-python "$COSMOS_PY" \
    --robocasa-python "$ROBOCASA_PY" \
    --cosmos-environment-manifest "$COSMOS_ENVIRONMENT_MANIFEST" \
    --expected-cosmos-environment-manifest-sha256 "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256" \
    --robocasa-environment-manifest "$ROBOCASA_ENVIRONMENT_MANIFEST" \
    --expected-robocasa-environment-manifest-sha256 "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256" \
    --cosmos-source-root "$COSMOS_SOURCE" \
    --expected-cosmos-commit "$EXPECTED_SOURCE_COMMIT" \
    --cosmos-source-manifest "$SOURCE_MANIFEST" \
    --expected-cosmos-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --demian-source-root "$DEMIAN_REPO" \
    --expected-demian-commit "$EXPECTED_DEMIAN_COMMIT" \
    --robocasa-source-root "$ROBOCASA_REPO" \
    --expected-robocasa-commit "$EXPECTED_ROBOCASA_COMMIT" \
    --robosuite-source-root "$ROBOSUITE_REPO" \
    --expected-robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT" \
    --experiment-id "$EVAL_EXPERIMENT_ID" \
    --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --capture-nonce "$EVAL_RUNTIME_CAPTURE_NONCE" \
    --submitted-at-utc "$EVAL_SUBMITTED_AT_UTC" \
    --submission-verification-receipt "$SUBMISSION_VERIFICATION_RECEIPT" \
    --expected-submission-verification-receipt-sha256 "$(sha256_file "$SUBMISSION_VERIFICATION_RECEIPT")" \
    --asset-verification-receipt "$ASSET_VERIFICATION_RECEIPT" \
    --expected-asset-verification-receipt-sha256 "$(sha256_file "$ASSET_VERIFICATION_RECEIPT")" \
    --egl-vendor-json "$EVAL_EGL_VENDOR_JSON" \
    --max-start-to-capture-seconds "$EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS" \
    --max-future-skew-seconds "$EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS" \
    --num-gpus "$NUM_SHARDS" \
    --output "$EVAL_RUNTIME_RECEIPT"
echo "EVAL_RUNTIME_RECEIPT_READY path=$EVAL_RUNTIME_RECEIPT sha256=$(sha256_file "$EVAL_RUNTIME_RECEIPT")"

SERVER_PIDS=()
EVALUATOR_PIDS=()
LAUNCH_NONCES=()
NVME_CREATED=0
stop_child_pids() {
    local deadline
    local still_running
    local pid
    local -a child_pids=("$@")
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    deadline=$((SECONDS + CLEANUP_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        still_running=0
        for pid in "${child_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running=1
            fi
        done
        ((still_running == 1)) || break
        sleep 1
    done
    for pid in "${child_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${child_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}
cleanup_nvme() {
    local expected_nvme_root="/nvme/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID"
    [[ "$NVME_CREATED" == 1 ]] || return 0
    [[ "$EVAL_NVME_ROOT" == "$expected_nvme_root" ]] ||
        die "refusing to clean an unexpected NVMe path: $EVAL_NVME_ROOT"
    [[ ! -L "$EVAL_NVME_ROOT" ]] ||
        die "refusing to clean a symlinked NVMe path: $EVAL_NVME_ROOT"
    if [[ -d "$EVAL_NVME_ROOT" ]]; then
        /usr/bin/rm --one-file-system -rf -- "$EVAL_NVME_ROOT"
    fi
    [[ ! -e "$EVAL_NVME_ROOT" && ! -L "$EVAL_NVME_ROOT" ]] ||
        die "scoped NVMe cleanup did not remove $EVAL_NVME_ROOT"
    NVME_CREATED=0
}
cleanup() {
    local status="$1"
    trap - EXIT INT TERM
    stop_child_pids "${EVALUATOR_PIDS[@]}" "${SERVER_PIDS[@]}"
    if ! cleanup_nvme; then
        status=1
    fi
    exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

EVAL_NVME_ROOT="$(realpath -m -- "$EVAL_NVME_ROOT")"
[[ "$EVAL_NVME_ROOT" == "/nvme/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID" ]] ||
    die "EVAL_NVME_ROOT must be derived exactly from EVAL_LOGICAL_RUN_ID"
case "$EVAL_NVME_ROOT/" in
    /nvme/*/) ;;
    *) die "EVAL_NVME_ROOT must be a strict child of /nvme" ;;
esac
[[ "$(findmnt -n -T /nvme -o TARGET)" == /nvme ]] || die "/nvme is not a distinct mount target"
[[ "$(findmnt -n -T /nvme -o SOURCE)" == /dev/nvme1n1 ]] || die "/nvme is not backed by /dev/nvme1n1"
[[ "$(findmnt -n -T /nvme -o FSTYPE)" == xfs ]] || die "/nvme is not xfs"
[[ ! -e "$EVAL_NVME_ROOT" ]] || die "Refusing to reuse EVAL_NVME_ROOT: $EVAL_NVME_ROOT"
mkdir -- "$EVAL_NVME_ROOT"
NVME_CREATED=1

"$COSMOS_PY" - "$NUM_SHARDS" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != expected:
    raise SystemExit(f"visible GPU count mismatch: expected={expected}, actual={torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(expected)]
if any("H200" not in name for name in names):
    raise SystemExit(f"all visible GPUs must be H200: {names}")
print(f"EVAL_CUDA_READY count={expected} names={names} torch={torch.__version__} cuda={torch.version.cuda}")
PY

"$COSMOS_PY" - "$BASE_PORT" "$NUM_SHARDS" <<'PY'
import socket
import sys

base_port = int(sys.argv[1])
count = int(sys.argv[2])
sockets = []
try:
    for port in range(base_port, base_port + count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
print(f"EVAL_PORTS_READY base={base_port} count={count}")
PY

require_accepted_submission_terminal_state
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    port=$((BASE_PORT + shard))
    server_log="$CONTROL_DIR/server-shard${shard}.log"
    server_output="$EVAL_NVME_ROOT/server-output-shard${shard}"
    compile_cache="$EVAL_NVME_ROOT/torchinductor-shard${shard}"
    launch_nonce="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
    LAUNCH_NONCES+=("$launch_nonce")
    mkdir -- "$compile_cache"
    (
        cd "$COSMOS_SOURCE"
        exec env \
            -u WORLD_SIZE \
            -u RANK \
            -u LOCAL_RANK \
            -u MASTER_ADDR \
            -u MASTER_PORT \
            CUDA_VISIBLE_DEVICES="$shard" \
            PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$compile_cache" \
            "$COSMOS_PY" -m "$SERVER_MODULE" \
            --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
            --config-file "$TRAINING_CONFIG" \
            --output-dir "$server_output" \
            --action-stats-path "$ACTION_STATS" \
            --host 127.0.0.1 \
            --port "$port" \
            --sampler "$POLICY_SAMPLER" \
            --seed "$POLICY_SEED" \
            --guidance "$POLICY_GUIDANCE" \
            --num-steps "$POLICY_NUM_STEPS" \
            --launch-nonce "$launch_nonce"
    ) >"$server_log" 2>&1 &
    SERVER_PIDS+=("$!")
done

RECEIPTS=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    port=$((BASE_PORT + shard))
    receipt="$CONTROL_DIR/server-runtime-receipt-shard${shard}.json"
    RECEIPTS+=("$receipt")
    PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$RECEIPT_HELPER_REL" \
        --uri "ws://127.0.0.1:$port" \
        --output "$receipt" \
        --timeout-seconds "$SERVER_READY_TIMEOUT_SECONDS" \
        --checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
        --training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
        --action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
        --source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --demian-commit "$EXPECTED_DEMIAN_COMMIT" \
        --shard-index "$shard" \
        --expected-launch-nonce "${LAUNCH_NONCES[$shard]}" \
        --expected-server-pid "${SERVER_PIDS[$shard]}" \
        --watch-pid "${SERVER_PIDS[$shard]}" \
        >"$CONTROL_DIR/server-receipt-shard${shard}.log"
done
echo "ALL_POLICY_SERVERS_READY count=$NUM_SHARDS"

CONTRACT_ARGS=(
    create-contract
    --output "$CONTRACT_PATH"
    --experiment-id "$EVAL_EXPERIMENT_ID"
    --logical-run-id "$EVAL_LOGICAL_RUN_ID"
    --source-commit "$EXPECTED_SOURCE_COMMIT"
    --source-root "$COSMOS_SOURCE"
    --source-manifest "$SOURCE_MANIFEST"
    --cosmos-server-module "$COSMOS_SOURCE/$SERVER_MODULE_REL"
    --checkpoint-path "$SERVING_CHECKPOINT_PATH"
    --checkpoint-manifest "$CHECKPOINT_MANIFEST"
    --checkpoint-verification-receipt "$CHECKPOINT_VERIFICATION_RECEIPT"
    --source-dcp-export-receipt "$SOURCE_DCP_EXPORT_RECEIPT"
    --training-config "$TRAINING_CONFIG"
    --action-stats "$ACTION_STATS"
    --cosmos-environment-manifest "$COSMOS_ENVIRONMENT_MANIFEST"
    --robocasa-environment-manifest "$ROBOCASA_ENVIRONMENT_MANIFEST"
    --eval-intent-manifest "$EVAL_INTENT_MANIFEST"
    --submission-verification-receipt "$SUBMISSION_VERIFICATION_RECEIPT"
    --asset-verification-receipt "$ASSET_VERIFICATION_RECEIPT"
    --eval-runtime-receipt "$EVAL_RUNTIME_RECEIPT"
    --demian-repo "$DEMIAN_REPO"
    --demian-commit "$EXPECTED_DEMIAN_COMMIT"
    --demian-evaluator "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL"
    --demian-task-registry "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL"
    --robocasa-repo "$ROBOCASA_REPO"
    --robocasa-commit "$EXPECTED_ROBOCASA_COMMIT"
    --robosuite-repo "$ROBOSUITE_REPO"
    --robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT"
    --stage "$EVAL_STAGE"
    --split "$EVAL_SPLIT"
    --episodes-per-task "$EPISODES_PER_TASK"
    --base-seed "$BASE_SEED"
    --max-steps "$MAX_STEPS"
    --open-loop-horizon "$OPEN_LOOP_HORIZON"
    --num-shards "$NUM_SHARDS"
    --policy-sampler "$POLICY_SAMPLER"
    --policy-seed "$POLICY_SEED"
    --policy-guidance "$POLICY_GUIDANCE"
    --policy-num-steps "$POLICY_NUM_STEPS"
    --results-dir "$RESULTS_DIR"
    --run-name "$EVAL_RUN_NAME"
    --minimum-mean-success-rate "$MINIMUM_MEAN_SUCCESS_RATE"
    --minimum-successful-tasks "$MINIMUM_SUCCESSFUL_TASKS"
)
for receipt in "${RECEIPTS[@]}"; do
    CONTRACT_ARGS+=(--server-runtime-receipt "$receipt")
done
if [[ "$EVAL_STAGE" == "one_episode_gate" ]]; then
    IFS=',' read -r -a EVAL_TASKS <<<"$EVAL_TASKS_CSV"
    CONTRACT_ARGS+=(--tasks "${EVAL_TASKS[@]}")
else
    CONTRACT_ARGS+=(
        --prior-gate-evidence "$PRIOR_GATE_EVIDENCE"
        --expected-prior-gate-evidence-sha256 "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256"
    )
fi
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
    "${CONTRACT_ARGS[@]}" >"$CONTROL_DIR/create-contract.log"
echo "EVAL_CONTRACT_READY path=$CONTRACT_PATH sha256=$(sha256_file "$CONTRACT_PATH")"

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    port=$((BASE_PORT + shard))
    evaluator_log="$CONTROL_DIR/evaluator-shard${shard}.log"
    EVALUATOR_ARGS=(
        --task-shard "$shard"
        --num-shards "$NUM_SHARDS"
        --episodes "$EPISODES_PER_TASK"
        --episodes-start 0
        --max-steps "$MAX_STEPS"
        --open-loop-horizon "$OPEN_LOOP_HORIZON"
        --host 127.0.0.1
        --port "$port"
        --checkpoint "$SERVING_CHECKPOINT_PATH"
        --results-dir "$RESULTS_ROOT"
        --run-name "$EVAL_RUN_NAME"
        --no-save-video
        --headless
        --camera-height 256
        --camera-width 256
        --seed "$BASE_SEED"
    )
    if [[ "$EVAL_STAGE" == "one_episode_gate" ]]; then
        EVALUATOR_ARGS+=(--tasks "$EVAL_TASKS_CSV")
    else
        EVALUATOR_ARGS+=(--split "$EVAL_SPLIT")
    fi
    env \
        -u LD_PRELOAD \
        -u LD_AUDIT \
        -u __EGL_VENDOR_LIBRARY_DIRS \
        -u LIBGL_ALWAYS_SOFTWARE \
        -u MESA_LOADER_DRIVER_OVERRIDE \
        -u LIBGL_DRIVERS_PATH \
        CUDA_VISIBLE_DEVICES="$shard" \
        MUJOCO_EGL_DEVICE_ID="$shard" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON" \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        PYTHONPATH="$EVAL_PYTHONPATH" \
        "$ROBOCASA_PY" "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" \
        "${EVALUATOR_ARGS[@]}" >"$evaluator_log" 2>&1 &
    EVALUATOR_PIDS+=("$!")
done

eval_failed=0
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    if ! wait "${EVALUATOR_PIDS[$shard]}"; then
        echo "ERROR: evaluator shard $shard failed; tail follows" >&2
        tail -200 "$CONTROL_DIR/evaluator-shard${shard}.log" >&2 || true
        eval_failed=1
    fi
done
EVALUATOR_PIDS=()
[[ "$eval_failed" == 0 ]] || die "One or more RoboCasa evaluator shards failed"
require_accepted_submission_terminal_state
stop_child_pids "${SERVER_PIDS[@]}"
SERVER_PIDS=()

PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
    verify-post-rollout \
    --contract "$CONTRACT_PATH" \
    --output "$POST_ROLLOUT_VERIFICATION_RECEIPT" \
    >"$CONTROL_DIR/post-rollout-verification.log"
echo "POST_ROLLOUT_INPUTS_VERIFIED receipt=$POST_ROLLOUT_VERIFICATION_RECEIPT"
require_accepted_submission_terminal_state

SUMMARIZE_ARGS=(
    summarize
    --contract "$CONTRACT_PATH"
    --post-rollout-verification-receipt "$POST_ROLLOUT_VERIFICATION_RECEIPT"
    --results-dir "$RESULTS_DIR"
    --summary-csv "$SUMMARY_CSV"
    --task-success-rates-csv "$TASK_SUCCESS_RATES_CSV"
    --evidence-json "$EVIDENCE_JSON"
)
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
    "${SUMMARIZE_ARGS[@]}" | tee "$CONTROL_DIR/summarize.log"

[[ -s "$SUMMARY_CSV" ]] || die "summary.csv is missing or empty"
[[ -s "$TASK_SUCCESS_RATES_CSV" ]] || die "task_success_rates.csv is missing or empty"
[[ -s "$EVIDENCE_JSON" ]] || die "evaluation_evidence.json is missing or empty"
RAW_RESULT_FILES=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    if [[ "$NUM_SHARDS" == 1 ]]; then
        RAW_RESULT_FILES+=("$RESULTS_DIR/args.json")
    else
        RAW_RESULT_FILES+=("$RESULTS_DIR/args_shard${shard}.json")
    fi
    RAW_RESULT_FILES+=("$RESULTS_DIR/results_shard${shard}.json")
done
chmod 0444 \
    "$CONTRACT_PATH" \
    "$CHECKPOINT_VERIFICATION_RECEIPT" \
    "$SUBMISSION_VERIFICATION_RECEIPT" \
    "$ASSET_VERIFICATION_RECEIPT" \
    "$EVAL_RUNTIME_RECEIPT" \
    "$POST_ROLLOUT_VERIFICATION_RECEIPT" \
    "${RECEIPTS[@]}" \
    "${RAW_RESULT_FILES[@]}" \
    "$SUMMARY_CSV" \
    "$TASK_SUCCESS_RATES_CSV" \
    "$EVIDENCE_JSON"
chmod 0555 "$RESULTS_DIR"
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
    verify-sealed-evidence \
    --contract "$CONTRACT_PATH" \
    --post-rollout-verification-receipt "$POST_ROLLOUT_VERIFICATION_RECEIPT" \
    --results-dir "$RESULTS_DIR" \
    --summary-csv "$SUMMARY_CSV" \
    --task-success-rates-csv "$TASK_SUCCESS_RATES_CSV" \
    --evidence-json "$EVIDENCE_JSON" \
    --output "$SEALED_EVIDENCE_VERIFICATION_RECEIPT" \
    >"$CONTROL_DIR/sealed-evidence-verification.log"
[[ -s "$SEALED_EVIDENCE_VERIFICATION_RECEIPT" ]] ||
    die "sealed evidence verification receipt is missing or empty"
[[ "$(stat -c %a -- "$SEALED_EVIDENCE_VERIFICATION_RECEIPT")" == 444 ]] ||
    die "sealed evidence verification receipt is not mode 0444"
SEALED_EVIDENCE_VERIFICATION_SHA256="$(sha256_file "$SEALED_EVIDENCE_VERIFICATION_RECEIPT")"
chmod 0444 "$CONTROL_DIR/sealed-evidence-verification.log"
require_accepted_submission_terminal_state
chmod 0555 \
    "$CONTROL_DIR" \
    "$RESULTS_ROOT" \
    "$EVAL_OUTPUT_ROOT" \
    "$EXPERIMENT_OUTPUT_ROOT"
EFFECT_PASS="$(
    "$COSMOS_PY" - \
        "$SEALED_EVIDENCE_VERIFICATION_RECEIPT" \
        "$SEALED_EVIDENCE_VERIFICATION_SHA256" \
        "$CONTROL_DIR" \
        "$RESULTS_ROOT" \
        "$EVAL_OUTPUT_ROOT" \
        "$EXPERIMENT_OUTPUT_ROOT" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_sha256 = sys.argv[2]
directories = [pathlib.Path(value) for value in sys.argv[3:]]
for directory in directories:
    metadata = directory.lstat()
    if directory.resolve(strict=True) != directory or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"sealed evaluation directory is unsafe: {directory}")
    if stat.S_IMODE(metadata.st_mode) != 0o555:
        raise SystemExit(f"sealed evaluation directory is not mode 0555: {directory}")
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

before = path.lstat()
if path.resolve(strict=True) != path or not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444:
    raise SystemExit("sealed evidence verification receipt is unsafe or not mode 0444")
descriptor = os.open(
    path,
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
)
try:
    opened = os.fstat(descriptor)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
final = path.lstat()
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if not (identity(before) == identity(opened) == identity(after) == identity(final)):
    raise SystemExit("sealed evidence verification receipt changed during final verdict read")
payload = b"".join(chunks)
if hashlib.sha256(payload).hexdigest() != expected_sha256:
    raise SystemExit("sealed evidence verification receipt SHA-256 changed after directory sealing")
receipt = json.loads(payload)
if (
    receipt.get("schema_version") != 1
    or receipt.get("kind") != "cwam_robocasa365_sealed_evidence_verification_receipt"
    or receipt.get("status") != "passed"
):
    raise SystemExit("sealed evidence verification receipt terminal verdict is invalid")
validation = receipt.get("validation")
if not isinstance(validation, dict) or not validation or any(value is not True for value in validation.values()):
    raise SystemExit("sealed evidence verification receipt is not an all-true validation")
effect_pass = receipt.get("evaluation", {}).get("effect_pass")
if type(effect_pass) is not bool:
    raise SystemExit("sealed evidence verification receipt effect_pass is not boolean")
print(str(effect_pass).lower())
PY
)"
require_accepted_submission_terminal_state
echo "ROBOCASA365_EVAL_EXECUTION_SUCCESS stage=$EVAL_STAGE"
echo "SUMMARY_CSV=$SUMMARY_CSV sha256=$(sha256_file "$SUMMARY_CSV")"
echo "TASK_SUCCESS_RATES_CSV=$TASK_SUCCESS_RATES_CSV sha256=$(sha256_file "$TASK_SUCCESS_RATES_CSV")"
echo "EVIDENCE_JSON=$EVIDENCE_JSON sha256=$(sha256_file "$EVIDENCE_JSON")"
echo "SEALED_EVIDENCE_VERIFICATION_RECEIPT=$SEALED_EVIDENCE_VERIFICATION_RECEIPT sha256=$SEALED_EVIDENCE_VERIFICATION_SHA256"
if [[ "$EVAL_STAGE" == "formal" ]]; then
    if [[ "$EFFECT_PASS" == true ]]; then
        echo "ROBOCASA365_EVAL_EFFECT_GATE_PASSED"
    else
        die "Formal evaluation completed and was sealed, but the preregistered effect gate failed"
    fi
fi
