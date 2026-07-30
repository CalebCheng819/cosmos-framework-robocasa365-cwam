#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Submit one phase-exact CWAM/Cosmos3 RoboCasa365 closed-loop evaluation.
# Prediction-only is the default and creates no reservation.  Setting
# RJOB_PREDICT_ONLY=0 deliberately creates immutable experiment and logical-run
# reservations, submits one H RJob, and refuses to report acceptance until the
# authoritative scheduler objects and a started replica have been validated.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly PINNED_PERSONAL_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly PINNED_OUTPUT_PARENT="$PINNED_PERSONAL_GPFS_ROOT/outputs/cwam-robocasa365-eval"
readonly PINNED_RUNS_ROOT="$PINNED_OUTPUT_PARENT/runs"
readonly PINNED_SUBMISSION_ROOT="$PINNED_OUTPUT_PARENT/submission-reservations"
readonly SNAPSHOT_ALLOWED_ROOT="$PINNED_PERSONAL_GPFS_ROOT/run_snapshots"
readonly PERSONAL_GPFS_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS_MOUNT="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly PINNED_ROBOCASA_COMMIT="29f7ce8814c1547f5af762a0997fbd4b64848dd7"
readonly PINNED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"
readonly PINNED_ASSET_MANIFEST_SHA256="3dc62249192bf2a8fd4bef1a31c6de9124cbc10421aa91b6769dee8e39e4d3c0"
readonly PINNED_ASSET_RECEIPT_SHA256="dc25dff483b1f42c904926cf72f91369f6ac94f0a4f50ab54e4c776299f93dca"
readonly PINNED_EGL_VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
readonly SUBMISSION_MAX_AGE_SECONDS=259200
readonly SUBMISSION_MAX_FUTURE_SKEW_SECONDS=300
readonly SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS=7200
readonly SCHEDULER_RECEIPT_ATTEMPTS=86400
readonly SCHEDULER_RECEIPT_RETRY_SECONDS=1
readonly SCHEDULER_RECEIPT_TIMEOUT_SECONDS=90000
readonly RJOB_REPLICAS=1
readonly REALPATH_BIN="/usr/bin/realpath"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly PYTHON3_BIN="/usr/bin/python3"
readonly BASH_BIN="/usr/bin/bash"
readonly ENV_BIN="/usr/bin/env"
readonly TEE_BIN="/usr/bin/tee"
readonly TIMEOUT_BIN="/usr/bin/timeout"
readonly SLEEP_BIN="/usr/bin/sleep"
readonly CHMOD_BIN="/usr/bin/chmod"
readonly STAT_BIN="/usr/bin/stat"
readonly GIT_BIN="/usr/bin/git"
readonly RJOB_BIN="/usr/local/bin/rjob"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_nonempty_env() {
    local name
    for name in "$@"; do
        [[ -v "$name" ]] || die "Required environment variable is not set: $name"
        [[ -n "${!name}" ]] || die "Required environment variable is empty: $name"
    done
}

require_safe_id() {
    local label="$1"
    local value="$2"
    [[ "${#value}" -le 128 &&
        "$value" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ ]] ||
        die "$label must be 1-128 path-safe characters with alphanumeric endpoints"
    [[ "$value" != "." && "$value" != ".." ]] || die "$label must not be '.' or '..'"
}

sha256_file() {
    local digest remainder
    read -r digest remainder < <("$SHA256SUM_BIN" "$1")
    printf '%s\n' "$digest"
}

initialize_output_infrastructure() {
    "$PYTHON3_BIN" -I - \
        "$PINNED_PERSONAL_GPFS_ROOT" \
        "$PINNED_OUTPUT_PARENT" \
        "$PINNED_RUNS_ROOT" \
        "$PINNED_SUBMISSION_ROOT" <<'PY'
import os
import pathlib
import stat
import sys

DIRECTORY_MODE = 0o755


def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def open_existing_directory(path, label):
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SystemExit(f"{label} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"{label} must be a non-symlink directory: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"{label} cannot be resolved safely: {path}: {error}") from error
    if resolved != path:
        raise SystemExit(f"{label} must be an exact canonical directory: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SystemExit(f"{label} cannot be opened safely: {path}: {error}") from error
    opened = os.fstat(descriptor)
    final = path.lstat()
    if identity(metadata) != identity(opened) or identity(opened) != identity(final):
        os.close(descriptor)
        raise SystemExit(f"{label} changed while it was opened: {path}")
    return descriptor


def open_or_create_managed_directory(parent_fd, name, path, expected_uid, expected_gid):
    created = False
    try:
        os.mkdir(name, DIRECTORY_MODE, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    except OSError as error:
        raise SystemExit(f"cannot create fixed eval infrastructure directory {path}: {error}") from error

    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise SystemExit(f"cannot inspect fixed eval infrastructure directory {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"fixed eval infrastructure path must be a non-symlink directory: {path}")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise SystemExit(f"cannot open fixed eval infrastructure directory {path}: {error}") from error

    try:
        opened = os.fstat(descriptor)
        if identity(metadata) != identity(opened):
            raise SystemExit(f"fixed eval infrastructure directory changed while opening: {path}")
        if created:
            os.fchmod(descriptor, DIRECTORY_MODE)
            os.fsync(parent_fd)
            opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) != DIRECTORY_MODE:
            raise SystemExit(
                f"fixed eval infrastructure directory has unsafe mode: "
                f"path={path} expected={DIRECTORY_MODE:04o} actual={stat.S_IMODE(opened.st_mode):04o}"
            )
        if (opened.st_uid, opened.st_gid) != (expected_uid, expected_gid):
            raise SystemExit(
                f"fixed eval infrastructure directory has unsafe ownership: "
                f"path={path} expected={expected_uid}:{expected_gid} "
                f"actual={opened.st_uid}:{opened.st_gid}"
            )
        final = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if identity(opened) != identity(final):
            raise SystemExit(f"fixed eval infrastructure directory changed after validation: {path}")
        os.fsync(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def initialize(personal_root, output_parent, runs_root, submission_root, *, expected_uid=None, expected_gid=None):
    personal_root = pathlib.Path(personal_root)
    output_parent = pathlib.Path(output_parent)
    runs_root = pathlib.Path(runs_root)
    submission_root = pathlib.Path(submission_root)
    outputs_root = personal_root / "outputs"
    expected_paths = (
        personal_root / "outputs" / "cwam-robocasa365-eval",
        personal_root / "outputs" / "cwam-robocasa365-eval" / "runs",
        personal_root / "outputs" / "cwam-robocasa365-eval" / "submission-reservations",
    )
    if (output_parent, runs_root, submission_root) != expected_paths:
        raise SystemExit("fixed eval infrastructure paths differ from the pinned hierarchy")
    if expected_uid is None:
        expected_uid = os.geteuid()
    if expected_gid is None:
        expected_gid = os.getegid()

    descriptors = []
    try:
        personal_fd = open_existing_directory(personal_root, "pinned personal GPFS root")
        descriptors.append(personal_fd)
        outputs_fd = open_or_create_managed_directory(
            personal_fd,
            "outputs",
            outputs_root,
            expected_uid,
            expected_gid,
        )
        descriptors.append(outputs_fd)
        output_fd = open_or_create_managed_directory(
            outputs_fd,
            "cwam-robocasa365-eval",
            output_parent,
            expected_uid,
            expected_gid,
        )
        descriptors.append(output_fd)
        runs_fd = open_or_create_managed_directory(
            output_fd,
            "runs",
            runs_root,
            expected_uid,
            expected_gid,
        )
        descriptors.append(runs_fd)
        submission_fd = open_or_create_managed_directory(
            output_fd,
            "submission-reservations",
            submission_root,
            expected_uid,
            expected_gid,
        )
        descriptors.append(submission_fd)
        for name in ("by-experiment", "by-logical-run"):
            descriptor = open_or_create_managed_directory(
                submission_fd,
                name,
                submission_root / name,
                expected_uid,
                expected_gid,
            )
            descriptors.append(descriptor)

        managed = (
            outputs_root,
            output_parent,
            runs_root,
            submission_root,
            submission_root / "by-experiment",
            submission_root / "by-logical-run",
        )
        for path, descriptor in zip(managed, descriptors[1:], strict=True):
            metadata = path.lstat()
            opened = os.fstat(descriptor)
            if (
                path.resolve(strict=True) != path
                or identity(metadata) != identity(opened)
                or stat.S_IMODE(metadata.st_mode) != DIRECTORY_MODE
                or (metadata.st_uid, metadata.st_gid) != (expected_uid, expected_gid)
            ):
                raise SystemExit(f"fixed eval infrastructure directory failed final validation: {path}")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: <personal-root> <output-parent> <runs-root> <submission-root>")
    initialize(*sys.argv[1:])
PY
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
    SOURCE_MANIFEST
    COSMOS_ENVIRONMENT_MANIFEST
    ROBOCASA_ENVIRONMENT_MANIFEST
    EVAL_INTENT_MANIFEST
    SOURCE_DCP_EXPORT_RECEIPT
    EVAL_EXPERIMENT_ID
    EVAL_LOGICAL_RUN_ID
    EVAL_EVALUATION_CLASS
    SOURCE_GIT_PUSHED
    EXPECTED_SOURCE_COMMIT
    EXPECTED_SNAPSHOT_MANIFEST_SHA256
    EXPECTED_SOURCE_MANIFEST_SHA256
    EXPECTED_CHECKPOINT_MANIFEST_SHA256
    EXPECTED_TRAINING_CONFIG_SHA256
    EXPECTED_ACTION_STATS_SHA256
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_EVAL_INTENT_MANIFEST_SHA256
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256
    EXPECTED_EGL_VENDOR_JSON_SHA256
    EXPECTED_ROBOCASA_COMMIT
    EXPECTED_ROBOSUITE_COMMIT
    MINIMUM_MEAN_SUCCESS_RATE
    MINIMUM_SUCCESSFUL_TASKS
    EVAL_STAGE
    EVAL_SPLIT
    EPISODES_PER_TASK
    NUM_SHARDS
    BASE_SEED
    MAX_STEPS
    OPEN_LOOP_HORIZON
    POLICY_SAMPLER
    POLICY_SEED
    POLICY_GUIDANCE
    POLICY_NUM_STEPS
    BASE_PORT
    SERVER_READY_TIMEOUT_SECONDS
    RJOB_NAME
)
require_nonempty_env "${CALLER_INPUTS[@]}"
[[ -v EVAL_TASKS_CSV ]] || die "EVAL_TASKS_CSV must be explicitly set (empty only for formal)"
[[ -v PRIOR_GATE_EVIDENCE ]] ||
    die "PRIOR_GATE_EVIDENCE must be explicitly set (empty only for one_episode_gate)"
[[ -v EXPECTED_PRIOR_GATE_EVIDENCE_SHA256 ]] ||
    die "EXPECTED_PRIOR_GATE_EVIDENCE_SHA256 must be explicitly set (empty only for one_episode_gate)"
: "${RJOB_PREDICT_ONLY:=1}"

[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] || die "RJOB_PREDICT_ONLY must be 0 or 1"
[[ "$EVAL_EVALUATION_CLASS" == exploratory || "$EVAL_EVALUATION_CLASS" == confirmatory ]] ||
    die "EVAL_EVALUATION_CLASS must be exploratory or confirmatory"
[[ "$SOURCE_GIT_PUSHED" == true || "$SOURCE_GIT_PUSHED" == false ]] ||
    die "SOURCE_GIT_PUSHED must be true or false"
[[ "$EVAL_EVALUATION_CLASS" != confirmatory || "$SOURCE_GIT_PUSHED" == true ]] ||
    die "confirmatory evaluation requires SOURCE_GIT_PUSHED=true"
[[ "${#RJOB_NAME}" -le 50 &&
    "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] ||
    die "RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters"
require_safe_id EVAL_EXPERIMENT_ID "$EVAL_EXPERIMENT_ID"
require_safe_id EVAL_LOGICAL_RUN_ID "$EVAL_LOGICAL_RUN_ID"
[[ "$EVAL_LOGICAL_RUN_ID" == "$EVAL_EXPERIMENT_ID"-* ]] ||
    die "EVAL_LOGICAL_RUN_ID must be namespaced below EVAL_EXPERIMENT_ID"

for digest_name in \
    EXPECTED_SNAPSHOT_MANIFEST_SHA256 \
    EXPECTED_SOURCE_MANIFEST_SHA256 \
    EXPECTED_CHECKPOINT_MANIFEST_SHA256 \
    EXPECTED_TRAINING_CONFIG_SHA256 \
    EXPECTED_ACTION_STATS_SHA256 \
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256 \
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256 \
    EXPECTED_EVAL_INTENT_MANIFEST_SHA256 \
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256 \
    EXPECTED_EGL_VENDOR_JSON_SHA256; do
    [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] ||
        die "$digest_name must be a lowercase SHA-256"
done
for commit_name in EXPECTED_SOURCE_COMMIT EXPECTED_ROBOCASA_COMMIT EXPECTED_ROBOSUITE_COMMIT; do
    [[ "${!commit_name}" =~ ^[0-9a-f]{40}$ ]] ||
        die "$commit_name must be a full lowercase Git SHA"
done
[[ "$EXPECTED_SOURCE_MANIFEST_SHA256" == "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" ]] ||
    die "SOURCE_MANIFEST must be the immutable snapshot manifest"
[[ "$EXPECTED_ROBOCASA_COMMIT" == "$PINNED_ROBOCASA_COMMIT" ]] ||
    die "EXPECTED_ROBOCASA_COMMIT differs from the asset-expanded audited checkout"

[[ "$EVAL_SPLIT" == "atomic" ]] || die "This primary submitter requires EVAL_SPLIT=atomic"
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

case "$EVAL_STAGE" in
    one_episode_gate)
        [[ "$EPISODES_PER_TASK" == 1 ]] ||
            die "one_episode_gate requires EPISODES_PER_TASK=1"
        [[ "$NUM_SHARDS" == 1 ]] || die "one_episode_gate requires NUM_SHARDS=1"
        [[ -n "$EVAL_TASKS_CSV" && "$EVAL_TASKS_CSV" != *,* ]] ||
            die "one_episode_gate requires exactly one preregistered task"
        [[ "$MINIMUM_MEAN_SUCCESS_RATE" == "0.0" ]] ||
            die "one_episode_gate MINIMUM_MEAN_SUCCESS_RATE must be 0.0"
        [[ "$MINIMUM_SUCCESSFUL_TASKS" == 0 ]] ||
            die "one_episode_gate MINIMUM_SUCCESSFUL_TASKS must be 0"
        [[ -z "$PRIOR_GATE_EVIDENCE" && -z "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" ]] ||
            die "one_episode_gate must not bind prior gate evidence"
        RJOB_CPU=16
        RJOB_MEMORY_MIB=187500
        RJOB_GPU=1
        RJOB_HOST_NETWORK=false
        ;;
    formal)
        [[ "$EPISODES_PER_TASK" == 20 ]] ||
            die "formal requires EPISODES_PER_TASK=20"
        [[ "$NUM_SHARDS" == 8 ]] || die "formal requires NUM_SHARDS=8"
        [[ -z "$EVAL_TASKS_CSV" ]] ||
            die "formal must use the complete canonical atomic split"
        [[ -n "$PRIOR_GATE_EVIDENCE" ]] ||
            die "formal requires PRIOR_GATE_EVIDENCE"
        [[ "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
            die "formal EXPECTED_PRIOR_GATE_EVIDENCE_SHA256 must be a lowercase SHA-256"
        "$PYTHON3_BIN" - "$MINIMUM_MEAN_SUCCESS_RATE" "$MINIMUM_SUCCESSFUL_TASKS" <<'PY'
import math
import sys

try:
    mean_rate = float(sys.argv[1])
    successful_tasks = int(sys.argv[2])
except ValueError as error:
    raise SystemExit(f"formal effect thresholds are malformed: {error}")
if not math.isfinite(mean_rate) or not 0.0 <= mean_rate <= 1.0:
    raise SystemExit("formal MINIMUM_MEAN_SUCCESS_RATE must be finite and in [0,1]")
if not 0 <= successful_tasks <= 17:
    raise SystemExit("formal MINIMUM_SUCCESSFUL_TASKS must be in [0,17]")
PY
        RJOB_CPU=128
        RJOB_MEMORY_MIB=1500000
        RJOB_GPU=8
        RJOB_HOST_NETWORK=true
        ;;
    *)
        die "EVAL_STAGE must be one_episode_gate or formal"
        ;;
esac
readonly RJOB_CPU RJOB_MEMORY_MIB RJOB_GPU

for required_tool in \
    "$REALPATH_BIN" \
    "$SHA256SUM_BIN" \
    "$PYTHON3_BIN" \
    "$BASH_BIN" \
    "$ENV_BIN" \
    "$TEE_BIN" \
    "$TIMEOUT_BIN" \
    "$SLEEP_BIN" \
    "$CHMOD_BIN" \
    "$STAT_BIN" \
    "$GIT_BIN" \
    "$RJOB_BIN"; do
    [[ -x "$required_tool" ]] || die "Required absolute tool is unavailable: $required_tool"
done
[[ "$("$REALPATH_BIN" -e -- "$RJOB_BIN")" == "$RJOB_BIN" ]] ||
    die "rjob client must be the canonical $RJOB_BIN"
[[ "$("$REALPATH_BIN" -e -- /proc/"$$"/exe)" == "$BASH_BIN" ]] ||
    die "Submitter must run with the audited $BASH_BIN"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

COSMOS_SOURCE_REAL="$("$REALPATH_BIN" -e -- "$COSMOS_SOURCE")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REAL" && -d "$COSMOS_SOURCE_REAL" ]] ||
    die "COSMOS_SOURCE must be an exact canonical directory"
case "$COSMOS_SOURCE_REAL/" in
    "$SNAPSHOT_ALLOWED_ROOT"/*/) ;;
    *) die "COSMOS_SOURCE must be a strict child of $SNAPSHOT_ALLOWED_ROOT" ;;
esac
[[ "$COSMOS_SOURCE_REAL" != *".incomplete"* ]] ||
    die "COSMOS_SOURCE must not contain .incomplete"
readonly SUBMITTER="$COSMOS_SOURCE_REAL/examples/h_cluster/submit_cwam_cosmos3_robocasa365_eval_rjob.sh"
readonly RUNNER="$COSMOS_SOURCE_REAL/examples/h_cluster/run_cwam_cosmos3_robocasa365_eval.sh"
readonly PROVENANCE_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_formal_provenance.py"
readonly SCHEDULER_CAPTURE_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/capture_cwam_formal_rjob_receipt.py"
readonly SUBMISSION_VERIFY_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/verify_cwam_robocasa365_eval_submission.py"
readonly ASSET_VERIFY_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_robocasa_asset_manifest.py"
readonly RUNTIME_RECEIPT_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/capture_robocasa365_eval_runtime.py"
readonly EVIDENCE_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_robocasa365_eval_evidence.py"
readonly CHECKPOINT_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_serving_checkpoint_manifest.py"
readonly INTENT_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_robocasa365_eval_intent.py"
readonly SNAPSHOT_MANIFEST="$COSMOS_SOURCE_REAL/SNAPSHOT_MANIFEST.json"
[[ "$("$REALPATH_BIN" -e -- "${BASH_SOURCE[0]}")" == "$SUBMITTER" ]] ||
    die "Submitter must execute from the exact immutable COSMOS_SOURCE snapshot"
for required_file in \
    "$SUBMITTER" \
    "$RUNNER" \
    "$PROVENANCE_HELPER" \
    "$SCHEDULER_CAPTURE_HELPER" \
    "$SUBMISSION_VERIFY_HELPER" \
    "$ASSET_VERIFY_HELPER" \
    "$RUNTIME_RECEIPT_HELPER" \
    "$EVIDENCE_HELPER" \
    "$CHECKPOINT_HELPER" \
    "$INTENT_HELPER" \
    "$SNAPSHOT_MANIFEST"; do
    [[ -s "$required_file" && ! -L "$required_file" ]] ||
        die "Required immutable snapshot file is missing, empty, or a symlink: $required_file"
done
[[ "$(sha256_file "$SNAPSHOT_MANIFEST")" == "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" ]] ||
    die "Snapshot manifest SHA-256 mismatch"
[[ "$SOURCE_MANIFEST" == "$SNAPSHOT_MANIFEST" ]] ||
    die "SOURCE_MANIFEST must be the exact snapshot manifest path"

readonly BOOTSTRAP_RELATIVE_FILES=(
    "examples/h_cluster/submit_cwam_cosmos3_robocasa365_eval_rjob.sh"
    "examples/h_cluster/run_cwam_cosmos3_robocasa365_eval.sh"
    "examples/h_cluster/cwam_formal_provenance.py"
    "examples/h_cluster/capture_cwam_formal_rjob_receipt.py"
    "examples/h_cluster/verify_cwam_robocasa365_eval_submission.py"
    "examples/h_cluster/cwam_robocasa_asset_manifest.py"
    "examples/h_cluster/capture_robocasa365_eval_runtime.py"
    "examples/h_cluster/cwam_robocasa365_eval_evidence.py"
    "examples/h_cluster/cwam_serving_checkpoint_manifest.py"
    "examples/h_cluster/cwam_robocasa365_eval_intent.py"
)
BOOTSTRAP_ARGUMENTS=()
for relative in "${BOOTSTRAP_RELATIVE_FILES[@]}"; do
    path="$COSMOS_SOURCE_REAL/$relative"
    BOOTSTRAP_ARGUMENTS+=("$relative" "$(sha256_file "$path")" "$("$STAT_BIN" -c %s -- "$path")")
done
"$PYTHON3_BIN" - "$SNAPSHOT_MANIFEST" "$EXPECTED_SOURCE_COMMIT" "${BOOTSTRAP_ARGUMENTS[@]}" <<'PY'
import json
import pathlib
import re
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("schema_version") != 1 or manifest.get("source_head") != sys.argv[2]:
    raise SystemExit("snapshot bootstrap schema/source commit mismatch")
if manifest.get("source_dirty") is not False:
    raise SystemExit("snapshot bootstrap requires source_dirty=false")
entries = manifest.get("files")
if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
    raise SystemExit("snapshot manifest file inventory is malformed")
index = {}
for entry in entries:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
        raise SystemExit(f"snapshot manifest entry schema drift: {entry!r}")
    path = entry["path"]
    if not isinstance(path, str) or path in index or re.fullmatch(r"[^\x00-\x1f\x7f]+", path) is None:
        raise SystemExit(f"snapshot manifest path is invalid or duplicated: {path!r}")
    index[path] = entry
arguments = sys.argv[3:]
if len(arguments) % 3:
    raise SystemExit("internal bootstrap argument mismatch")
for offset in range(0, len(arguments), 3):
    relative, digest, size = arguments[offset : offset + 3]
    entry = index.get(relative)
    if entry is None or (entry["sha256"], entry["bytes"]) != (digest, int(size)):
        raise SystemExit(f"snapshot does not authenticate {relative}")
PY
"$PYTHON3_BIN" "$PROVENANCE_HELPER" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE_REAL" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null

for path_name in \
    COSMOS_ENV \
    ROBOCASA_ENV \
    DEMIAN_REPO \
    ROBOCASA_REPO \
    ROBOSUITE_REPO \
    SERVING_CHECKPOINT_PATH \
    CHECKPOINT_MANIFEST \
    TRAINING_CONFIG \
    ACTION_STATS \
    COSMOS_ENVIRONMENT_MANIFEST \
    ROBOCASA_ENVIRONMENT_MANIFEST \
    EVAL_INTENT_MANIFEST \
    SOURCE_DCP_EXPORT_RECEIPT; do
    supplied="${!path_name}"
    resolved="$("$REALPATH_BIN" -e -- "$supplied")"
    [[ "$supplied" == "$resolved" && "$resolved" != *".incomplete"* ]] ||
        die "$path_name must be an exact canonical path without .incomplete"
done
for expected_pair in \
    "checkpoint_manifest|$CHECKPOINT_MANIFEST|$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
    "training_config|$TRAINING_CONFIG|$EXPECTED_TRAINING_CONFIG_SHA256" \
    "action_stats|$ACTION_STATS|$EXPECTED_ACTION_STATS_SHA256" \
    "cosmos_environment_manifest|$COSMOS_ENVIRONMENT_MANIFEST|$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256" \
    "robocasa_environment_manifest|$ROBOCASA_ENVIRONMENT_MANIFEST|$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256" \
    "eval_intent_manifest|$EVAL_INTENT_MANIFEST|$EXPECTED_EVAL_INTENT_MANIFEST_SHA256" \
    "source_dcp_export_receipt|$SOURCE_DCP_EXPORT_RECEIPT|$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256" \
    "egl_vendor_json|$PINNED_EGL_VENDOR_JSON|$EXPECTED_EGL_VENDOR_JSON_SHA256"; do
    IFS='|' read -r expected_label expected_path expected_sha256 <<<"$expected_pair"
    actual_sha256="$(sha256_file "$expected_path")"
    [[ "$actual_sha256" == "$expected_sha256" ]] ||
        die "$expected_label SHA-256 mismatch: expected=$expected_sha256 actual=$actual_sha256"
done
[[ "$("$STAT_BIN" -c %a -- "$EVAL_INTENT_MANIFEST")" == 444 ]] ||
    die "EVAL_INTENT_MANIFEST must be immutable mode 0444"
[[ "$("$STAT_BIN" -c %a -- "$SOURCE_DCP_EXPORT_RECEIPT")" == 444 ]] ||
    die "SOURCE_DCP_EXPORT_RECEIPT must be immutable mode 0444"
"$PYTHON3_BIN" "$CHECKPOINT_HELPER" verify \
    --manifest "$CHECKPOINT_MANIFEST" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    >/dev/null
for repo_contract in \
    "DeMiAn|$DEMIAN_REPO|$PINNED_DEMIAN_COMMIT" \
    "RoboCasa|$ROBOCASA_REPO|$EXPECTED_ROBOCASA_COMMIT" \
    "robosuite|$ROBOSUITE_REPO|$EXPECTED_ROBOSUITE_COMMIT"; do
    IFS='|' read -r repo_label repo_path expected_commit <<<"$repo_contract"
    actual_commit="$("$GIT_BIN" -c safe.directory="$repo_path" -C "$repo_path" rev-parse --verify HEAD)"
    [[ "$actual_commit" == "$expected_commit" ]] ||
        die "$repo_label commit mismatch: expected=$expected_commit actual=$actual_commit"
    [[ -z "$("$GIT_BIN" -c safe.directory="$repo_path" -C "$repo_path" status --porcelain --untracked-files=all)" ]] ||
        die "$repo_label checkout is dirty"
done
[[ -d "$ROBOCASA_REPO" ]] || die "ROBOCASA_REPO must be a directory"
readonly ROBOCASA_ASSET_TREE_MANIFEST="$ROBOCASA_REPO/.cwam_provenance/asset-tree-manifest.jsonl"
readonly ROBOCASA_ASSET_TREE_RECEIPT="$ROBOCASA_REPO/.cwam_provenance/asset-tree-receipt.json"
[[ "$(sha256_file "$ROBOCASA_ASSET_TREE_MANIFEST")" == "$PINNED_ASSET_MANIFEST_SHA256" ]] ||
    die "RoboCasa asset manifest differs from the audited publication"
[[ "$(sha256_file "$ROBOCASA_ASSET_TREE_RECEIPT")" == "$PINNED_ASSET_RECEIPT_SHA256" ]] ||
    die "RoboCasa asset receipt differs from the audited publication"
ASSET_PREFLIGHT_JSON="$(
    "$PYTHON3_BIN" "$ASSET_VERIFY_HELPER" verify \
        --asset-root "$ROBOCASA_REPO"
)"
TRACKED_REPOSITORY_PREFLIGHT_JSON="$(
    "$PYTHON3_BIN" - \
        "$EVIDENCE_HELPER" \
        "$DEMIAN_REPO" \
        "$PINNED_DEMIAN_COMMIT" \
        "$ROBOCASA_REPO" \
        "$EXPECTED_ROBOCASA_COMMIT" \
        "$ROBOSUITE_REPO" \
        "$EXPECTED_ROBOSUITE_COMMIT" <<'PY'
import importlib.util
import json
import pathlib
import sys

helper = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(helper.parent))
spec = importlib.util.spec_from_file_location("cwam_eval_preflight_evidence", helper)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import authenticated evidence helper")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
arguments = sys.argv[2:]
reports = {}
for offset in range(0, len(arguments), 2):
    root = pathlib.Path(arguments[offset])
    expected_commit = arguments[offset + 1]
    name = ("demian", "robocasa", "robosuite")[offset // 2]
    reports[name] = module._verify_git_repository(
        name=name,
        root=root,
        expected_commit=expected_commit,
        verify_tracked_files=True,
    )
print(json.dumps(reports, separators=(",", ":"), sort_keys=True, allow_nan=False))
PY
)"
SUBMISSION_PREFLIGHT_JSON="$(
    "$PYTHON3_BIN" -I - "$ASSET_PREFLIGHT_JSON" "$TRACKED_REPOSITORY_PREFLIGHT_JSON" <<'PY'
import json
import sys

asset = json.loads(sys.argv[1])
repositories = json.loads(sys.argv[2])
if asset.get("status") != "passed":
    raise SystemExit("RoboCasa asset preflight did not pass")
if set(repositories) != {"demian", "robocasa", "robosuite"}:
    raise SystemExit("tracked repository preflight coverage is incomplete")
for name, report in repositories.items():
    tracked = report.get("tracked_files", {})
    if (
        report.get("clean") is not True
        or tracked.get("all_tracked_regular_files_match_head_blobs") is not True
    ):
        raise SystemExit(f"tracked repository preflight did not pass: {name}")
print(json.dumps({
    "asset_tree": asset,
    "repositories": repositories,
    "validation": {
        "asset_tree_full_hash_match": True,
        "repository_heads_clean": True,
        "tracked_regular_files_match_head_blobs": True,
    },
}, separators=(",", ":"), sort_keys=True, allow_nan=False))
PY
)"
SUBMISSION_PREFLIGHT_SHA256="$(printf '%s' "$SUBMISSION_PREFLIGHT_JSON" | "$SHA256SUM_BIN" | {
    read -r digest remainder
    printf '%s\n' "$digest"
})"
INTENT_VERIFY_ARGS=(
    verify
    --manifest "$EVAL_INTENT_MANIFEST"
    --expected-manifest-sha256 "$EXPECTED_EVAL_INTENT_MANIFEST_SHA256"
    --experiment-id "$EVAL_EXPERIMENT_ID"
    --logical-run-id "$EVAL_LOGICAL_RUN_ID"
    --run-name "$EVAL_LOGICAL_RUN_ID"
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
    --source-root "$COSMOS_SOURCE_REAL"
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
    --robocasa-asset-root "$ROBOCASA_REPO"
    --robocasa-asset-manifest "$ROBOCASA_ASSET_TREE_MANIFEST"
    --expected-robocasa-asset-manifest-sha256 "$PINNED_ASSET_MANIFEST_SHA256"
    --robocasa-asset-receipt "$ROBOCASA_ASSET_TREE_RECEIPT"
    --expected-robocasa-asset-receipt-sha256 "$PINNED_ASSET_RECEIPT_SHA256"
    --demian-root "$DEMIAN_REPO"
    --demian-commit "$PINNED_DEMIAN_COMMIT"
    --robocasa-root "$ROBOCASA_REPO"
    --robocasa-commit "$EXPECTED_ROBOCASA_COMMIT"
    --robosuite-root "$ROBOSUITE_REPO"
    --robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT"
    --container-image "$PINNED_CONTAINER_IMAGE"
    --egl-vendor-json "$PINNED_EGL_VENDOR_JSON"
    --expected-egl-vendor-json-sha256 "$EXPECTED_EGL_VENDOR_JSON_SHA256"
    --cpu "$RJOB_CPU"
    --memory-mib "$RJOB_MEMORY_MIB"
    --gpu "$RJOB_GPU"
    --quota-group "$PINNED_QUOTA_GROUP"
    --negative-tag "$PINNED_EXCLUDED_NODE_TAG"
    --task-type normal
    --private-machine group
    --priority 9
    --restart-policy never
    --backoff-limit 1
    --replicas "$RJOB_REPLICAS"
)
if [[ "$EVAL_STAGE" == formal ]]; then
    INTENT_VERIFY_ARGS+=(
        --prior-gate-evidence "$PRIOR_GATE_EVIDENCE"
        --expected-prior-gate-evidence-sha256 "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256"
    )
fi
PYTHONPATH="$COSMOS_SOURCE_REAL/examples/h_cluster" \
    "$PYTHON3_BIN" "$INTENT_HELPER" "${INTENT_VERIFY_ARGS[@]}" >/dev/null
EVAL_INTENT_DESCRIPTOR_JSON="$(
    "$PYTHON3_BIN" -I - \
        "$EVAL_INTENT_MANIFEST" \
        "$EXPECTED_EVAL_INTENT_MANIFEST_SHA256" \
        "$("$STAT_BIN" -c %s -- "$EVAL_INTENT_MANIFEST")" <<'PY'
import json
import sys

print(json.dumps({
    "path": sys.argv[1],
    "sha256": sys.argv[2],
    "size_bytes": int(sys.argv[3]),
}, separators=(",", ":"), sort_keys=True))
PY
)"
if [[ "$EVAL_STAGE" == "formal" ]]; then
    resolved_prior_gate="$("$REALPATH_BIN" -e -- "$PRIOR_GATE_EVIDENCE")"
    [[ "$PRIOR_GATE_EVIDENCE" == "$resolved_prior_gate" &&
        "$resolved_prior_gate" != *".incomplete"* &&
        -f "$resolved_prior_gate" &&
        ! -L "$resolved_prior_gate" ]] ||
        die "PRIOR_GATE_EVIDENCE must be an exact canonical regular path without .incomplete"
    [[ "$(sha256_file "$PRIOR_GATE_EVIDENCE")" == "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" ]] ||
        die "PRIOR_GATE_EVIDENCE SHA-256 mismatch"
    [[ "$("$STAT_BIN" -c %a -- "$PRIOR_GATE_EVIDENCE")" == 444 ]] ||
        die "PRIOR_GATE_EVIDENCE must be immutable mode 0444"
fi

"$PYTHON3_BIN" - "$COSMOS_SOURCE_REAL" "$EVAL_TASKS_CSV" "$EVAL_STAGE" <<'PY'
import importlib.util
import pathlib
import sys

helper = pathlib.Path(sys.argv[1]) / "examples/h_cluster/cwam_robocasa365_eval_evidence.py"
sys.path.insert(0, str(helper.parent))
spec = importlib.util.spec_from_file_location("cwam_eval_submit_contract", helper)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import authenticated evidence helper")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tasks_csv, stage = sys.argv[2:]
if stage == "one_episode_gate" and tasks_csv not in module.ATOMIC_TASKS:
    raise SystemExit(f"gate task is not in the canonical atomic registry: {tasks_csv!r}")
if stage == "formal" and len(module.ATOMIC_TASKS) != 17:
    raise SystemExit(f"formal atomic registry must contain 17 tasks, got {len(module.ATOMIC_TASKS)}")
PY

if [[ "$EVAL_STAGE" == "formal" ]]; then
    PYTHONPATH="$COSMOS_SOURCE_REAL" "$PYTHON3_BIN" "$EVIDENCE_HELPER" \
        verify-prior-gate \
        --gate-evidence "$PRIOR_GATE_EVIDENCE" \
        --expected-gate-evidence-sha256 "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" \
        --current-source-root "$COSMOS_SOURCE_REAL" \
        --current-source-commit "$EXPECTED_SOURCE_COMMIT" \
        --current-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --current-checkpoint-path "$SERVING_CHECKPOINT_PATH" \
        --current-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
        --current-source-dcp-export-receipt-sha256 "$EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256" \
        --current-asset-manifest-sha256 "$PINNED_ASSET_MANIFEST_SHA256" \
        --current-asset-publisher-receipt-sha256 "$PINNED_ASSET_RECEIPT_SHA256" \
        --current-training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
        --current-action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
        --current-cosmos-environment-manifest-sha256 "$EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256" \
        --current-robocasa-environment-manifest-sha256 "$EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256" \
        --current-demian-commit "$PINNED_DEMIAN_COMMIT" \
        --current-robocasa-commit "$EXPECTED_ROBOCASA_COMMIT" \
        --current-robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT" \
        >/dev/null
    PRIOR_GATE_EVIDENCE_DESCRIPTOR_JSON="$(
        "$PYTHON3_BIN" -I - \
            "$PRIOR_GATE_EVIDENCE" \
            "$EXPECTED_PRIOR_GATE_EVIDENCE_SHA256" \
            "$("$STAT_BIN" -c %s -- "$PRIOR_GATE_EVIDENCE")" <<'PY'
import json
import sys

print(json.dumps({
    "path": sys.argv[1],
    "sha256": sys.argv[2],
    "size_bytes": int(sys.argv[3]),
}, separators=(",", ":"), sort_keys=True))
PY
    )"
else
    PRIOR_GATE_EVIDENCE_DESCRIPTOR_JSON=null
fi

readonly EVAL_RUN_NAME="$EVAL_LOGICAL_RUN_ID"
readonly ALLOWED_OUTPUT_PARENT="$PINNED_RUNS_ROOT"
readonly EXPERIMENT_OUTPUT_ROOT="$PINNED_RUNS_ROOT/$EVAL_EXPERIMENT_ID"
readonly EVAL_OUTPUT_ROOT="$PINNED_RUNS_ROOT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
readonly EVAL_NVME_ROOT="/nvme/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID"
[[ ! -e "$EXPERIMENT_OUTPUT_ROOT" ]] ||
    die "Refusing to submit against an existing experiment output root: $EXPERIMENT_OUTPUT_ROOT"
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || die "Refusing to reuse EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"

SUBMITTED_AT_UTC="$("$PYTHON3_BIN" -I -c \
    'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat().replace("+00:00","Z"))')"
SUBMISSION_NONCE="$("$PYTHON3_BIN" -I -c 'import secrets; print(secrets.token_hex(32))')"
[[ "$SUBMISSION_NONCE" =~ ^[0-9a-f]{64}$ ]] || die "Failed to generate submission nonce"
EXPIRES_AT_UTC="$("$PYTHON3_BIN" -I - "$SUBMITTED_AT_UTC" "$SUBMISSION_MAX_AGE_SECONDS" <<'PY'
import datetime
import sys

submitted = datetime.datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
print((submitted + datetime.timedelta(seconds=int(sys.argv[2]))).isoformat().replace("+00:00", "Z"))
PY
)"

SCHEDULER_EXPECTED_JSON="$(
    "$PYTHON3_BIN" -I - \
        "$RJOB_NAME" \
        "$PINNED_CONTAINER_IMAGE" \
        "$RJOB_CPU" \
        "$RJOB_MEMORY_MIB" \
        "$RJOB_GPU" \
        "$RJOB_HOST_NETWORK" \
        "$PINNED_EXCLUDED_NODE_TAG" \
        "$PERSONAL_GPFS_MOUNT" \
        "$PUBLIC_GPFS_MOUNT" <<'PY'
import json
import sys

print(json.dumps({
    "showname": sys.argv[1],
    "quota_group": "eailabagent_gpu",
    "private_machine": "group",
    "task_type": "normal",
    "priority": 9,
    "restart_policy": "Never",
    "termination_grace_period_seconds": 60,
    "backoff_limit": 1,
    "replicas": 1,
    "image": sys.argv[2],
    "cpu": int(sys.argv[3]),
    "memory_mib": int(sys.argv[4]),
    "gpu": int(sys.argv[5]),
    "host_network": {"false": False, "true": True}[sys.argv[6]],
    "negative_tags": [sys.argv[7]],
    "store_host_nvme": True,
    "mounts": [sys.argv[8], sys.argv[9]],
}, separators=(",", ":"), sort_keys=True))
PY
)"

mark_submission_unknown() {
    local reason="$1"
    local scheduler_name="${2:-}"
    "$PYTHON3_BIN" - \
        "$SUBMISSION_PENDING" \
        "$SCHEDULER_READBACK_RECEIPT" \
        "$SUBMISSION_UNKNOWN" \
        "$EXPERIMENT_RESERVATION_DIR" \
        "$reason" \
        "$scheduler_name" \
        "$SUBMISSION_LOG" \
        "$SCHEDULER_EXPECTED_CONTRACT" \
        "$WORKER_ENV_CONTRACT" \
        "$SCHEDULER_CAPTURE_LOG" \
        "$SCHEDULER_STRUCTURED_RECEIPT" \
        "$SCHEDULER_TEXT_READBACK" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import stat
import sys
import uuid

pending = pathlib.Path(sys.argv[1])
accepted = pathlib.Path(sys.argv[2])
target = pathlib.Path(sys.argv[3])
parent = pathlib.Path(sys.argv[4])
artifacts = [pending, *(pathlib.Path(value) for value in sys.argv[7:])]
if (
    parent.resolve(strict=True) != parent
    or target.parent != parent
    or accepted.parent != parent
    or any(path.parent != parent for path in artifacts)
):
    raise SystemExit("submission terminal paths do not share the canonical reservation parent")
if accepted.exists() or accepted.is_symlink():
    raise SystemExit("refusing to publish unknown after accepted")
if target.exists() or target.is_symlink():
    raise SystemExit("submission unknown terminal record already exists")


def stable_digest(path: pathlib.Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"cannot seal non-regular submission artifact: {path}")
    os.chmod(path, 0o444, follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            payload.extend(chunk)
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
    if identity(before)[:2] != identity(opened)[:2] or identity(opened) != identity(after) or identity(after) != identity(final):
        raise RuntimeError(f"submission artifact changed while sealing: {path}")
    if stat.S_IMODE(final.st_mode) != 0o444:
        raise RuntimeError(f"submission artifact was not sealed mode 0444: {path}")
    return hashlib.sha256(payload).hexdigest()


available = {}
for path in artifacts:
    if path.exists() or path.is_symlink():
        available[str(path)] = stable_digest(path)
submission_reservation = (
    {"path": str(pending), "sha256": available[str(pending)]}
    if str(pending) in available
    else None
)
payload = {
    "schema_version": 1,
    "status": "unknown",
    "unknown_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "reason": sys.argv[5],
    "scheduler_rjob_name": sys.argv[6] or None,
    "submission_reservation": submission_reservation,
    "available_artifacts": available,
}
temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(
    temporary,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
published = False
try:
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444, follow_symlinks=False)
    if accepted.exists() or accepted.is_symlink():
        raise RuntimeError("accepted appeared before unknown terminal publication")
    os.link(temporary, target, follow_symlinks=False)
    published = True
    temporary.unlink()
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(parent, 0o555)
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if accepted.exists() or accepted.is_symlink():
        raise RuntimeError("accepted appeared while publishing unknown")
    if target.is_symlink() or not target.is_file() or stat.S_IMODE(target.stat().st_mode) != 0o444:
        raise RuntimeError("submission unknown terminal record was not sealed")
    if stat.S_IMODE(parent.stat().st_mode) != 0o555:
        raise RuntimeError("submission reservation directory was not sealed")
except Exception:
    os.chmod(parent, 0o700)
    if published:
        target.unlink(missing_ok=True)
    raise
finally:
    temporary.unlink(missing_ok=True)
PY
}

SUBMISSION_TERMINAL_COMPLETE=0
SUBMISSION_EXIT_REASON=unexpected_submission_exit
SCHEDULER_RJOB_NAME=""

submission_exit() {
    local status="$1"
    trap - EXIT HUP INT TERM
    if [[ "$SUBMISSION_TERMINAL_COMPLETE" != 1 ]]; then
        if [[ -e "$SCHEDULER_READBACK_RECEIPT" || -L "$SCHEDULER_READBACK_RECEIPT" ]]; then
            if [[ -e "$SUBMISSION_UNKNOWN" || -L "$SUBMISSION_UNKNOWN" ]]; then
                echo "ERROR: accepted and unknown submission terminal records both exist" >&2
                status=1
            fi
        elif [[ ! -e "$SUBMISSION_UNKNOWN" && ! -L "$SUBMISSION_UNKNOWN" ]]; then
            mark_submission_unknown "$SUBMISSION_EXIT_REASON" "$SCHEDULER_RJOB_NAME" || {
                echo "ERROR: failed to publish durable submission-unknown terminal record" >&2
                status=1
            }
        fi
        [[ "$status" -ne 0 ]] || status=1
    fi
    exit "$status"
}

submission_signal() {
    SUBMISSION_EXIT_REASON="signal_$1_during_submission"
    exit "$2"
}

if [[ "$RJOB_PREDICT_ONLY" == 0 ]]; then
    initialize_output_infrastructure
    EXPERIMENT_RESERVATION_DIR="$PINNED_SUBMISSION_ROOT/by-experiment/$EVAL_EXPERIMENT_ID"
    RUN_RESERVATION_DIR="$PINNED_SUBMISSION_ROOT/by-logical-run/$EVAL_LOGICAL_RUN_ID"
    if ! /usr/bin/mkdir --mode=0700 -- "$EXPERIMENT_RESERVATION_DIR"; then
        die "EVAL_EXPERIMENT_ID already has a submit-side reservation"
    fi
    SUBMISSION_PENDING="$EXPERIMENT_RESERVATION_DIR/submission-pending.json"
    SCHEDULER_READBACK_RECEIPT="$EXPERIMENT_RESERVATION_DIR/scheduler-readback-receipt.json"
    SUBMISSION_UNKNOWN="$EXPERIMENT_RESERVATION_DIR/submission-unknown.json"
    SUBMISSION_LOG="$EXPERIMENT_RESERVATION_DIR/rjob-submit.log"
    SCHEDULER_EXPECTED_CONTRACT="$EXPERIMENT_RESERVATION_DIR/rjob-expected-contract.json"
    WORKER_ENV_CONTRACT="$EXPERIMENT_RESERVATION_DIR/worker-env-contract.json"
    SCHEDULER_CAPTURE_LOG="$EXPERIMENT_RESERVATION_DIR/rjob-scheduler-capture.log"
    SCHEDULER_STRUCTURED_RECEIPT="$EXPERIMENT_RESERVATION_DIR/rjob-scheduler-receipt.json"
    SCHEDULER_TEXT_READBACK="$EXPERIMENT_RESERVATION_DIR/rjob-get.log"
    readonly \
        SUBMISSION_PENDING \
        SCHEDULER_READBACK_RECEIPT \
        SUBMISSION_UNKNOWN \
        SUBMISSION_LOG \
        SCHEDULER_EXPECTED_CONTRACT \
        WORKER_ENV_CONTRACT \
        SCHEDULER_CAPTURE_LOG \
        SCHEDULER_STRUCTURED_RECEIPT \
        SCHEDULER_TEXT_READBACK
    trap 'submission_exit "$?"' EXIT
    trap 'submission_signal HUP 129' HUP
    trap 'submission_signal INT 130' INT
    trap 'submission_signal TERM 143' TERM
    if ! /usr/bin/mkdir --mode=0700 -- "$RUN_RESERVATION_DIR"; then
        SUBMISSION_EXIT_REASON=logical_run_reservation_collision_after_experiment_lock
        die "EVAL_LOGICAL_RUN_ID already has a submit-side reservation; experiment tombstone retained"
    fi
    "$PYTHON3_BIN" - \
        "$SUBMISSION_PENDING" \
        "$SUBMITTED_AT_UTC" \
        "$EXPIRES_AT_UTC" \
        "$EVAL_EXPERIMENT_ID" \
        "$EVAL_LOGICAL_RUN_ID" \
        "$EVAL_RUN_NAME" \
        "$SUBMISSION_NONCE" \
        "$RJOB_NAME" \
        "$EXPECTED_SOURCE_COMMIT" \
        "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
        "$EVAL_STAGE" \
        "$EVAL_SPLIT" \
        "$EVAL_TASKS_CSV" \
        "$EPISODES_PER_TASK" \
        "$NUM_SHARDS" \
        "$BASE_SEED" \
        "$PRIOR_GATE_EVIDENCE_DESCRIPTOR_JSON" \
        "$SUBMISSION_PREFLIGHT_JSON" \
        "$SUBMISSION_PREFLIGHT_SHA256" \
        "$SCHEDULER_EXPECTED_JSON" \
        "$EVAL_INTENT_DESCRIPTOR_JSON" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "cwam_robocasa365_eval_submission_reservation",
    "status": "submitting",
    "submitted_at_utc": sys.argv[2],
    "expires_at_utc": sys.argv[3],
    "experiment_id": sys.argv[4],
    "logical_run_id": sys.argv[5],
    "run_name": sys.argv[6],
    "submission_nonce": sys.argv[7],
    "rjob_showname": sys.argv[8],
    "source_commit": sys.argv[9],
    "source_manifest_sha256": sys.argv[10],
    "evaluation": {
        "stage": sys.argv[11],
        "split": sys.argv[12],
        "tasks_csv": sys.argv[13],
        "episodes_per_task": int(sys.argv[14]),
        "num_shards": int(sys.argv[15]),
        "base_seed": int(sys.argv[16]),
        "prior_gate_evidence": json.loads(sys.argv[17]),
    },
    "preflight": json.loads(sys.argv[18]),
    "preflight_sha256": sys.argv[19],
    "scheduler_expected": json.loads(sys.argv[20]),
    "eval_intent_manifest": json.loads(sys.argv[21]),
}
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    EXPECTED_SUBMISSION_PENDING_SHA256="$(sha256_file "$SUBMISSION_PENDING")"
    "$PYTHON3_BIN" - \
        "$RUN_RESERVATION_DIR/reservation-pointer.json" \
        "$EVAL_EXPERIMENT_ID" \
        "$EVAL_LOGICAL_RUN_ID" \
        "$SUBMISSION_PENDING" \
        "$EXPECTED_SUBMISSION_PENDING_SHA256" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "cwam_robocasa365_eval_logical_run_reservation_pointer",
    "experiment_id": sys.argv[2],
    "logical_run_id": sys.argv[3],
    "reservation_path": sys.argv[4],
    "reservation_sha256": sys.argv[5],
}
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    "$CHMOD_BIN" 0555 "$RUN_RESERVATION_DIR"
else
    EXPERIMENT_RESERVATION_DIR=""
    SUBMISSION_PENDING="/__cwam_predict_only__/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID/submission-pending.json"
    SCHEDULER_READBACK_RECEIPT="/__cwam_predict_only__/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID/scheduler-readback-receipt.json"
    SUBMISSION_UNKNOWN="/__cwam_predict_only__/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID/submission-unknown.json"
    EXPECTED_SUBMISSION_PENDING_SHA256="$(printf '0%.0s' {1..64})"
fi

export EVAL_RUN_NAME EVAL_OUTPUT_ROOT ALLOWED_OUTPUT_PARENT EVAL_NVME_ROOT
export ROBOCASA_ASSET_ROOT="$ROBOCASA_REPO"
export ROBOCASA_ASSET_TREE_MANIFEST ROBOCASA_ASSET_TREE_RECEIPT
export EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256="$PINNED_ASSET_MANIFEST_SHA256"
export EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256="$PINNED_ASSET_RECEIPT_SHA256"
export EVAL_SUBMISSION_RESERVATION_PATH="$SUBMISSION_PENDING"
export EXPECTED_EVAL_SUBMISSION_PENDING_SHA256="$EXPECTED_SUBMISSION_PENDING_SHA256"
export EVAL_SCHEDULER_READBACK_RECEIPT="$SCHEDULER_READBACK_RECEIPT"
export EVAL_SUBMISSION_UNKNOWN_PATH="$SUBMISSION_UNKNOWN"
export EVAL_RUNTIME_CAPTURE_NONCE="$SUBMISSION_NONCE"
export EVAL_SUBMITTED_AT_UTC="$SUBMITTED_AT_UTC"
export EVAL_SUBMISSION_MAX_AGE_SECONDS="$SUBMISSION_MAX_AGE_SECONDS"
export EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS="$SUBMISSION_MAX_FUTURE_SKEW_SECONDS"
export EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS="$SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS"
export EVAL_EGL_VENDOR_JSON="$PINNED_EGL_VENDOR_JSON"
export EXPECTED_CONTAINER_IMAGE="$PINNED_CONTAINER_IMAGE"

readonly WORKER_ENV_NAMES=(
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
    EVAL_INTENT_MANIFEST
    SOURCE_DCP_EXPORT_RECEIPT
    EVAL_OUTPUT_ROOT
    EVAL_EXPERIMENT_ID
    EVAL_LOGICAL_RUN_ID
    EVAL_RUN_NAME
    EVAL_EVALUATION_CLASS
    SOURCE_GIT_PUSHED
    EXPECTED_SOURCE_COMMIT
    EXPECTED_SOURCE_MANIFEST_SHA256
    EXPECTED_CHECKPOINT_MANIFEST_SHA256
    EXPECTED_TRAINING_CONFIG_SHA256
    EXPECTED_ACTION_STATS_SHA256
    EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256
    EXPECTED_EVAL_INTENT_MANIFEST_SHA256
    EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256
    EXPECTED_EGL_VENDOR_JSON_SHA256
    PRIOR_GATE_EVIDENCE
    EXPECTED_PRIOR_GATE_EVIDENCE_SHA256
    EXPECTED_ROBOCASA_COMMIT
    EXPECTED_ROBOSUITE_COMMIT
    MINIMUM_MEAN_SUCCESS_RATE
    MINIMUM_SUCCESSFUL_TASKS
    EVAL_STAGE
    EVAL_SPLIT
    EVAL_TASKS_CSV
    EPISODES_PER_TASK
    NUM_SHARDS
    BASE_SEED
    MAX_STEPS
    OPEN_LOOP_HORIZON
    POLICY_SAMPLER
    POLICY_SEED
    POLICY_GUIDANCE
    POLICY_NUM_STEPS
    BASE_PORT
    SERVER_READY_TIMEOUT_SECONDS
    ALLOWED_OUTPUT_PARENT
    EVAL_NVME_ROOT
    ROBOCASA_ASSET_ROOT
    ROBOCASA_ASSET_TREE_MANIFEST
    ROBOCASA_ASSET_TREE_RECEIPT
    EXPECTED_ROBOCASA_ASSET_MANIFEST_SHA256
    EXPECTED_ROBOCASA_ASSET_RECEIPT_SHA256
    EVAL_SUBMISSION_RESERVATION_PATH
    EXPECTED_EVAL_SUBMISSION_PENDING_SHA256
    EVAL_SCHEDULER_READBACK_RECEIPT
    EVAL_SUBMISSION_UNKNOWN_PATH
    EVAL_RUNTIME_CAPTURE_NONCE
    EVAL_SUBMITTED_AT_UTC
    EVAL_SUBMISSION_MAX_AGE_SECONDS
    EVAL_SUBMISSION_MAX_FUTURE_SKEW_SECONDS
    EVAL_SUBMISSION_MAX_START_TO_VERIFICATION_SECONDS
    EVAL_EGL_VENDOR_JSON
    EXPECTED_CONTAINER_IMAGE
    RJOB_NAME
)
WORKER_SCRIPT=$'set -Eeuo pipefail\n'
WORKER_SCRIPT+=$'unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE PYTHONPATH PYTHONUSERBASE PYTHONBREAKPOINT LD_PRELOAD LD_AUDIT __EGL_VENDOR_LIBRARY_DIRS LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE LIBGL_DRIVERS_PATH\n'
WORKER_SCRIPT+=$'export __GLX_VENDOR_LIBRARY_NAME=nvidia\n'
WORKER_SCRIPT+="cd $(printf '%q' "$COSMOS_SOURCE_REAL")"$'\n'
for name in "${WORKER_ENV_NAMES[@]}"; do
    printf -v quoted '%q' "${!name}"
    WORKER_SCRIPT+="export $name=$quoted"$'\n'
done
printf -v quoted_vendor '%q' "$PINNED_EGL_VENDOR_JSON"
WORKER_SCRIPT+="export __EGL_VENDOR_LIBRARY_FILENAMES=$quoted_vendor"$'\n'
WORKER_SCRIPT+="exec $BASH_BIN $(printf '%q' "$RUNNER")"$'\n'
WORKER_COMMAND_SHA256="$(printf '%s' "$WORKER_SCRIPT" | "$SHA256SUM_BIN" | {
    read -r digest remainder
    printf '%s\n' "$digest"
})"
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
    --replicas "$RJOB_REPLICAS"
    --restart-policy never
    --termination-grace-period-seconds 60
    --backoff_limit 1
    --image "$PINNED_CONTAINER_IMAGE"
    --cpu "$RJOB_CPU"
    --gpu "$RJOB_GPU"
    --memory "$RJOB_MEMORY_MIB"
    --negative-tags "$PINNED_EXCLUDED_NODE_TAG"
)
if [[ "$RJOB_HOST_NETWORK" == true ]]; then
    SUBMIT_ARGS+=(--host-network)
fi
SUBMIT_ARGS+=(
    --store-host-nvme
    --mount
    "$PERSONAL_GPFS_MOUNT"
    "$PUBLIC_GPFS_MOUNT"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    SUBMIT_ARGS+=(--predict-only true)
fi

echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "EVAL_STAGE=$EVAL_STAGE"
echo "EVAL_EXPERIMENT_ID=$EVAL_EXPERIMENT_ID"
echo "EVAL_LOGICAL_RUN_ID=$EVAL_LOGICAL_RUN_ID"
echo "RJOB_RESOURCES=cpu=$RJOB_CPU memory_mib=$RJOB_MEMORY_MIB gpu=$RJOB_GPU"
echo "RJOB_HOST_NETWORK=$RJOB_HOST_NETWORK"
echo "RJOB_IMAGE=$PINNED_CONTAINER_IMAGE"
echo "RJOB_MOUNTS=$PERSONAL_GPFS_MOUNT;$PUBLIC_GPFS_MOUNT"
echo "RJOB_STORE_HOST_NVME=true"
echo "WORKER_COMMAND_SHA256=$WORKER_COMMAND_SHA256"
echo "SUBMISSION_NONCE_SHA256=$(printf '%s' "$SUBMISSION_NONCE" | "$SHA256SUM_BIN" | awk '{print $1}')"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    "$RJOB_BIN" "${SUBMIT_ARGS[@]}" -- "${WORKER_COMMAND[@]}"
    echo "ROBOCASA365_EVAL_SUBMISSION_PREDICT_ONLY reservation_created=false"
    exit 0
fi

set +e
"$TIMEOUT_BIN" --foreground --kill-after=10s 180s \
    "$RJOB_BIN" "${SUBMIT_ARGS[@]}" -- "${WORKER_COMMAND[@]}" 2>&1 |
    "$TEE_BIN" "$SUBMISSION_LOG"
SUBMIT_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
if [[ "${SUBMIT_PIPE_STATUS[0]}" -ne 0 || "${SUBMIT_PIPE_STATUS[1]}" -ne 0 ]]; then
    SUBMISSION_EXIT_REASON=rjob_submit_failed_or_ambiguous
    mark_submission_unknown "rjob_submit_failed_or_ambiguous"
    die "RJob submission failed or became ambiguous; reservation retained"
fi

"$CHMOD_BIN" 0444 "$SUBMISSION_LOG"

set +e
SCHEDULER_RJOB_NAME="$(
    "$PYTHON3_BIN" - "$SUBMISSION_LOG" "$RJOB_NAME" <<'PY'
import pathlib
import re
import sys

matches = re.findall(
    r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)",
    pathlib.Path(sys.argv[1]).read_text(errors="replace"),
)
if len(matches) != 1 or not matches[0].startswith(sys.argv[2] + "-"):
    raise SystemExit(f"cannot bind exactly one scheduler RJob name: {matches!r}")
print(matches[0])
PY
)"
PARSE_STATUS=$?
set -e
if [[ "$PARSE_STATUS" -ne 0 || -z "$SCHEDULER_RJOB_NAME" ]]; then
    mark_submission_unknown "accepted_cli_without_unique_scheduler_name"
    die "Cannot bind accepted CLI output to one scheduler RJob"
fi

"$PYTHON3_BIN" - \
    "$SCHEDULER_EXPECTED_CONTRACT" \
    "$SCHEDULER_RJOB_NAME" \
    "$SCHEDULER_EXPECTED_JSON" \
    "$COSMOS_SOURCE_REAL" \
    "$RUNNER" \
    "$EVAL_EXPERIMENT_ID" \
    "$WORKER_COMMAND_SHA256" <<'PY'
import json
import os
import pathlib
import sys

scheduler = json.loads(sys.argv[3])
payload = {
    "schema_version": 1,
    "scheduler_rjob_name": sys.argv[2],
    "showname": scheduler["showname"],
    "quota_group": scheduler["quota_group"],
    "task_type": scheduler["task_type"],
    "priority": scheduler["priority"],
    "backoff_limit": scheduler["backoff_limit"],
    "task_name": "t0",
    "replicas": scheduler["replicas"],
    "image": scheduler["image"],
    "cpu": scheduler["cpu"],
    "memory_mib": scheduler["memory_mib"],
    "gpu": scheduler["gpu"],
    "negative_tags": scheduler["negative_tags"],
    "private_machine": scheduler["private_machine"],
    "restart_policy": scheduler["restart_policy"],
    "termination_grace_period_seconds": scheduler["termination_grace_period_seconds"],
    "mounts": scheduler["mounts"],
    "snapshot_root": sys.argv[4],
    "wrapper_path": sys.argv[5],
    "experiment_id": sys.argv[6],
    "command_sha256": sys.argv[7],
}
target = pathlib.Path(sys.argv[1])
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
"$PYTHON3_BIN" - \
    "$WORKER_ENV_CONTRACT" \
    "$WORKER_COMMAND_SHA256" \
    "$RUNNER" \
    "${WORKER_ENV_NAMES[@]}" <<'PY'
import json
import os
import pathlib
import sys

target, command_sha256, runner = sys.argv[1:4]
names = sys.argv[4:]
payload = {
    "schema_version": 1,
    "command_sha256": command_sha256,
    "runner": runner,
    "environment": {name: os.environ[name] for name in names},
    "unset_environment": [
        "BASH_ENV",
        "ENV",
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "PYTHONBREAKPOINT",
        "LD_PRELOAD",
        "LD_AUDIT",
        "__EGL_VENDOR_LIBRARY_DIRS",
        "LIBGL_ALWAYS_SOFTWARE",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "LIBGL_DRIVERS_PATH",
    ],
    "pinned_environment": {
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    },
}
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

set +e
"$TIMEOUT_BIN" --foreground --kill-after=10s "${SCHEDULER_RECEIPT_TIMEOUT_SECONDS}s" \
    "$PYTHON3_BIN" - \
        "$SCHEDULER_CAPTURE_HELPER" \
        "$SCHEDULER_RJOB_NAME" \
        "$SCHEDULER_EXPECTED_CONTRACT" \
        "$WORKER_ENV_CONTRACT" \
        "$SCHEDULER_STRUCTURED_RECEIPT" \
        "$SCHEDULER_RECEIPT_ATTEMPTS" \
        "$SCHEDULER_RECEIPT_RETRY_SECONDS" >"$SCHEDULER_CAPTURE_LOG" 2>&1 <<'PY'
import datetime
import hashlib
import importlib.util
import json
import pathlib
import shlex
import sys

helper_path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("cwam_eval_scheduler_receipt", helper_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load authenticated scheduler receipt helper")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
worker_contract = json.loads(pathlib.Path(sys.argv[4]).read_text())
module._FORMAL_WRAPPER_RELATIVE_PATH = pathlib.PurePosixPath(
    "examples/h_cluster/run_cwam_cosmos3_robocasa365_eval.sh"
)
module._RECEIPT_KIND = "cwam_robocasa365_eval_rjob_scheduler_receipt"
clean_prefix = [
    "/usr/bin/env", "-u", "BASH_ENV", "-u", "ENV",
    "/usr/bin/bash", "--noprofile", "--norc", "-c",
]


def validate_eval_command(command_value, expected):
    command = module._require_list(command_value, "scheduler eval command")
    if len(command) != len(clean_prefix) + 1 or command[: len(clean_prefix)] != clean_prefix:
        raise module.ReceiptValidationError("scheduler command is not the audited clean-env command")
    script = module._require_string(command[-1], "scheduler eval shell script")
    digest = hashlib.sha256(script.encode()).hexdigest()
    module._require_exact(digest, expected["command_sha256"], "scheduler eval command SHA-256")
    module._require_exact(digest, worker_contract["command_sha256"], "worker environment command SHA-256")
    exports = module._parse_exports(script)
    for name, value in worker_contract["environment"].items():
        module._require_one_export(exports, name, value)
    for name, value in worker_contract["pinned_environment"].items():
        module._require_one_export(exports, name, value)
    invocations = []
    for line_number, line in enumerate(script.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as error:
            raise module.ReceiptValidationError(
                f"scheduler command line {line_number} is invalid shell"
            ) from error
        if tokens == ["exec", "/usr/bin/bash", worker_contract["runner"]]:
            invocations.append(tokens)
    if len(invocations) != 1:
        raise module.ReceiptValidationError("scheduler command must exec the exact runner once")
    for name in worker_contract["unset_environment"]:
        expected_line = (
            "unset " + " ".join(worker_contract["unset_environment"])
        )
        if expected_line not in script:
            raise module.ReceiptValidationError(
                f"scheduler command does not clear the graphics-loader environment: {name}"
            )
    return [str(item) for item in command]


module._validate_command = validate_eval_command
original_validate = module.validate_and_normalize_receipt
host_nvme_annotation = "kubebrain.brainpp.cn/writeread-hostpath-mounts"
host_nvme_contract = "/nvme:/mnt/nvme1n1,/cfs:/mnt/cfs"


def validate_started_eval_receipt(raw_rjob, raw_replicas, expected, *, captured_at=None):
    rjob = module._require_mapping(raw_rjob, "rjob")
    task = module._require_mapping(
        module._require_mapping(
            module._require_mapping(rjob.get("spec"), "rjob.spec").get("taskSpecs"),
            "rjob.spec.taskSpecs",
        ).get("t0"),
        "rjob.spec.taskSpecs.t0",
    )
    template = module._require_mapping(task.get("template"), "rjob.spec.taskSpecs.t0.template")
    annotations = module._require_mapping(
        module._require_mapping(template.get("metadata"), "template.metadata").get("annotations"),
        "template.metadata.annotations",
    )
    module._require_exact(
        annotations.get(host_nvme_annotation),
        host_nvme_contract,
        host_nvme_annotation,
    )
    receipt = original_validate(raw_rjob, raw_replicas, expected, captured_at=captured_at)
    start_time = receipt["replicas"][0]["status"]["start_time"]
    if not isinstance(start_time, str) or not start_time:
        raise module.ReceiptNotReady("eval replica has no scheduler start_time yet")
    parsed = datetime.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise module.ReceiptValidationError("eval replica start_time is not timezone-aware")
    receipt["spec"]["task"]["store_host_nvme"] = {
        "annotation": host_nvme_annotation,
        "value": host_nvme_contract,
        "container_target": "/nvme",
        "host_source": "/mnt/nvme1n1",
    }
    receipt["validation"]["replica_started"] = True
    return receipt


module.validate_and_normalize_receipt = validate_started_eval_receipt
receipt, digest = module.run_capture(
    scheduler_rjob_name=sys.argv[2],
    expected_json=pathlib.Path(sys.argv[3]),
    output=pathlib.Path(sys.argv[5]),
    attempts=int(sys.argv[6]),
    retry_seconds=float(sys.argv[7]),
)
print(
    "SCHEDULER_STRUCTURED_RECEIPT_VALID",
    f"rjob={receipt['metadata']['name']}",
    f"uid={receipt['metadata']['uid']}",
    f"start={receipt['replicas'][0]['status']['start_time']}",
    f"sha256={digest}",
)
PY
SCHEDULER_CAPTURE_STATUS=$?
set -e
"$CHMOD_BIN" 0444 "$SCHEDULER_EXPECTED_CONTRACT" "$WORKER_ENV_CONTRACT" "$SCHEDULER_CAPTURE_LOG"
if [[ "$SCHEDULER_CAPTURE_STATUS" -ne 0 || ! -s "$SCHEDULER_STRUCTURED_RECEIPT" ]]; then
    mark_submission_unknown "accepted_cli_without_valid_started_scheduler_contract" "$SCHEDULER_RJOB_NAME"
    die "Scheduler CRD validation failed or replica did not start"
fi

READBACK_OK=0
READBACK_CONTENT=""
for _ in {1..10}; do
    set +e
    READBACK_CONTENT="$(
        "$TIMEOUT_BIN" --foreground --kill-after=5s 20s \
            "$RJOB_BIN" get "$SCHEDULER_RJOB_NAME" 2>&1
    )"
    READBACK_STATUS=$?
    set -e
    if [[ "$READBACK_STATUS" -eq 0 &&
        "$READBACK_CONTENT" == *"rjob $SCHEDULER_RJOB_NAME (showname=$RJOB_NAME):"* ]]; then
        printf '%s\n' "$READBACK_CONTENT" >"$SCHEDULER_TEXT_READBACK"
        READBACK_OK=1
        break
    fi
    "$SLEEP_BIN" 1
done
if [[ "$READBACK_OK" != 1 ]]; then
    printf '%s\n' "$READBACK_CONTENT" >"$SCHEDULER_TEXT_READBACK"
    mark_submission_unknown "validated_scheduler_without_text_readback" "$SCHEDULER_RJOB_NAME"
    die "Accepted RJob has no exact text readback"
fi
"$CHMOD_BIN" 0444 "$SCHEDULER_TEXT_READBACK"

"$PYTHON3_BIN" - \
    "$SCHEDULER_READBACK_RECEIPT" \
    "$SUBMISSION_PENDING" \
    "$EXPECTED_SUBMISSION_PENDING_SHA256" \
    "$SCHEDULER_RJOB_NAME" \
    "$WORKER_COMMAND_SHA256" \
    "$SCHEDULER_EXPECTED_JSON" \
    "$SCHEDULER_STRUCTURED_RECEIPT" \
    "$SUBMISSION_LOG" \
    "$SCHEDULER_EXPECTED_CONTRACT" \
    "$SCHEDULER_CAPTURE_LOG" \
    "$SCHEDULER_TEXT_READBACK" \
    "$SUBMISSION_UNKNOWN" \
    "$EXPERIMENT_RESERVATION_DIR" \
    "$WORKER_ENV_CONTRACT" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys
import uuid

target = pathlib.Path(sys.argv[1])
pending_path = pathlib.Path(sys.argv[2])
unknown_path = pathlib.Path(sys.argv[12])
parent = pathlib.Path(sys.argv[13])
worker_env_contract = pathlib.Path(sys.argv[14])
if target.parent != parent or unknown_path.parent != parent or parent.resolve(strict=True) != parent:
    raise SystemExit("submission terminal paths do not share the canonical reservation parent")
if target.exists() or target.is_symlink():
    raise SystemExit("accepted terminal record already exists")
if unknown_path.exists() or unknown_path.is_symlink():
    raise SystemExit("refusing to publish accepted after submission-unknown")
pending = json.loads(pending_path.read_text())
structured_path = pathlib.Path(sys.argv[7])
structured = json.loads(structured_path.read_text())
replicas = structured["replicas"]
if len(replicas) != 1:
    raise SystemExit("structured scheduler receipt must contain exactly one replica")


def descriptor(path):
    value = pathlib.Path(path)
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
    }


payload = {
    "schema_version": 1,
    "kind": "cwam_robocasa365_eval_scheduler_readback_receipt",
    "status": "accepted",
    "accepted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "experiment_id": pending["experiment_id"],
    "logical_run_id": pending["logical_run_id"],
    "run_name": pending["run_name"],
    "submission_nonce": pending["submission_nonce"],
    "submitted_at_utc": pending["submitted_at_utc"],
    "expires_at_utc": pending["expires_at_utc"],
    "scheduler_rjob_name": sys.argv[4],
    "scheduler_rjob_uid": structured["metadata"]["uid"],
    "scheduler_started_at_utc": replicas[0]["status"]["start_time"],
    "worker_command_sha256": sys.argv[5],
    "submission_reservation": {
        "path": str(pending_path),
        "sha256": sys.argv[3],
    },
    "scheduler_expected": json.loads(sys.argv[6]),
    "artifacts": {
        "scheduler_structured_receipt": descriptor(structured_path),
        "submission_log": descriptor(sys.argv[8]),
        "scheduler_expected_contract": descriptor(sys.argv[9]),
        "scheduler_capture_log": descriptor(sys.argv[10]),
        "scheduler_text_readback": descriptor(sys.argv[11]),
    },
}
sealed_artifacts = [
    pending_path,
    structured_path,
    pathlib.Path(sys.argv[8]),
    pathlib.Path(sys.argv[9]),
    pathlib.Path(sys.argv[10]),
    pathlib.Path(sys.argv[11]),
    worker_env_contract,
]
for path in sealed_artifacts:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"cannot seal non-regular submission artifact: {path}")
    os.chmod(path, 0o444, follow_symlinks=False)
    if (path.stat().st_mode & 0o777) != 0o444:
        raise SystemExit(f"submission artifact was not sealed mode 0444: {path}")
temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
descriptor_fd = os.open(
    temporary,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o444,
)
published = False
try:
    with os.fdopen(descriptor_fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, target, follow_symlinks=False)
    published = True
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(target, 0o444, follow_symlinks=False)
    os.chmod(parent, 0o555)
    directory_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if unknown_path.exists() or unknown_path.is_symlink():
        raise RuntimeError("submission-unknown appeared while publishing accepted")
    if target.is_symlink() or not target.is_file() or (target.stat().st_mode & 0o777) != 0o444:
        raise RuntimeError("accepted terminal record was not sealed")
    if (parent.stat().st_mode & 0o777) != 0o555:
        raise RuntimeError("submission reservation directory was not sealed")
except Exception:
    os.chmod(parent, 0o700)
    if published:
        target.unlink(missing_ok=True)
    raise
finally:
    temporary.unlink(missing_ok=True)
PY
SUBMISSION_TERMINAL_COMPLETE=1
trap - EXIT HUP INT TERM
echo "ROBOCASA365_EVAL_SUBMISSION_ACCEPTED scheduler_rjob=$SCHEDULER_RJOB_NAME receipt=$SCHEDULER_READBACK_RECEIPT"
