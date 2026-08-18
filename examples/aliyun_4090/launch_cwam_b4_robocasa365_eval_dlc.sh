#!/usr/bin/env bash
# Materialize the audited B4 inputs on worker NVMe and run the formal evaluation.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${EVAL_CONTROL_COMMIT:?Set EVAL_CONTROL_COMMIT}"
: "${EVAL_EXPERIMENT_ID:?Set EVAL_EXPERIMENT_ID}"
: "${EVAL_LOGICAL_RUN_ID:?Set EVAL_LOGICAL_RUN_ID}"

readonly SOURCE_COMMIT=e7775ffcea0a2d577597fbeb05c8bb92905aba0f
readonly CONTROL_ROOT="/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/scripts/$EVAL_EXPERIMENT_ID-control-v3"
readonly SOURCE_ROOT=/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/source/CWAM-B4-e7775ff-20260817
readonly INPUT_ROOT=/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs
readonly OSS_ROOT=/oss-chengjuntao/cpfs-user-chengjuntao/cwam_cosmos3_robocasa365_eval
readonly CHECKPOINT_NAME=B4-iter_000020000-serving-ema-bf16-metadata-v1
readonly CHECKPOINT_SOURCE="$OSS_ROOT/inputs/checkpoints/$CHECKPOINT_NAME"
readonly SOURCE_ENV_ARCHIVE="$OSS_ROOT/inputs/envs/archives/cosmos-framework-09f23119-cu128-train-r22-complete.tar"
readonly PUBLICATION_ENV_ARCHIVE="$INPUT_ROOT/envs/archives/cosmos-framework-09f23119-cu128-train-r22-complete.tar"
readonly SOURCE_ENV_RECEIPT="$OSS_ROOT/control/receipts/cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json"
readonly SOURCE_ROBOCASA_ARCHIVE="$OSS_ROOT/inputs/archives/robocasa-29f7ce8814c1-assets-source-package-20260729.tar"
readonly SOURCE_ROBOCASA_OVERLAY="$OSS_ROOT/inputs/archives/robocasa-29f7ce8814c1-assets-legacy-fixture-overlay-20260817.tar"
readonly DEMIAN_REPO="$INPUT_ROOT/demian-2600ece5bf0e"
readonly ROBOSUITE_REPO="$INPUT_ROOT/robosuite-232ce7d4a6ed"
readonly ROBOCASA_ENV="$INPUT_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
readonly ACTION_STATS="$INPUT_ROOT/config/robocasa365_native_frame_wise_relative_rot6d.json"
readonly VAE_SOURCE="$INPUT_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly QWEN_SOURCE="$INPUT_ROOT/models/Qwen3-VL-8B-Instruct-tokenizer"
readonly EGL_FRONTEND_SOURCE="$INPUT_ROOT/runtime/glvnd/libEGL.so.1.1.0"
readonly RUNTIME_CLOSURE_SOURCE="$INPUT_ROOT/runtime/cosmos-r21-dlc-runtime-v1"
readonly RUNNER="$CONTROL_ROOT/run_cwam_b4_robocasa365_eval_dlc.sh"
readonly SHARD_GROUP_RUNNER="$CONTROL_ROOT/run_cwam_b4_robocasa365_eval_shard_group.sh"
readonly B4_SHELL_UTILS="$CONTROL_ROOT/cwam_b4_eval_shell_utils.sh"
readonly B4_EVAL_HELPER="$CONTROL_ROOT/cwam_b4_robocasa365_eval.py"
readonly ENV_HELPER="$CONTROL_ROOT/cwam_cosmos_env_archive.py"
readonly ASSET_HELPER="$CONTROL_ROOT/cwam_robocasa_asset_source_package.py"
readonly EGL_PROBE="$CONTROL_ROOT/probe_mujoco_egl_mapping.py"
readonly PERSIST_OUTPUT_HELPER="$CONTROL_ROOT/cwam_dlc_persist_eval_output.py"
readonly SCRATCH_ROOT="/tmp/cwam_b4_eval_$EVAL_EXPERIMENT_ID"
readonly COSMOS_ENV=/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs/envs/cosmos-framework-09f23119-cu128-train
readonly LOCAL_ENV_ARCHIVE="$SCRATCH_ROOT/inputs/archives/cosmos-framework-09f23119-cu128-train-r22-complete.tar"
readonly LOCAL_ENV_RECEIPT="$SCRATCH_ROOT/control/receipts/cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json"
readonly LOCAL_ROBOCASA_ARCHIVE="$SCRATCH_ROOT/inputs/archives/robocasa-29f7ce8814c1-assets-source-package-20260729.tar"
readonly LOCAL_ROBOCASA_OVERLAY="$SCRATCH_ROOT/inputs/archives/robocasa-29f7ce8814c1-assets-legacy-fixture-overlay-20260817.tar"
readonly ROBOCASA_REPO="$SCRATCH_ROOT/inputs/repos/robocasa-29f7ce8814c1"
readonly CHECKPOINT_ARTIFACT_ROOT="$SCRATCH_ROOT/inputs/checkpoints/$CHECKPOINT_NAME"
readonly SERVING_CHECKPOINT_PATH="$CHECKPOINT_ARTIFACT_ROOT/model"
readonly WAN2PT2_VAE="$SCRATCH_ROOT/inputs/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly QWEN_TOKENIZER="$SCRATCH_ROOT/inputs/models/Qwen3-VL-8B-Instruct-tokenizer"
readonly EGL_ROOT="$SCRATCH_ROOT/runtime/glvnd"
readonly LOCAL_RUNTIME_STAGING_ROOT=/tmp/cwam_cosmos3_robocasa365_eval_nvme/staging
readonly LOCAL_PRIVATE_GLIBC="$LOCAL_RUNTIME_STAGING_ROOT/cosmos-r21-private-glibc235-v1"
readonly LOCAL_GCC_RUNTIME="$LOCAL_RUNTIME_STAGING_ROOT/cosmos-r21-gcc13-runtime-v3"
readonly EVAL_EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
readonly DLC_LOCAL_FORMAL_PARENT=/tmp/cwam_b4_eval_formal
readonly EVAL_LOCAL_OUTPUT_ROOT="$DLC_LOCAL_FORMAL_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
readonly EVAL_PERSISTENT_OUTPUT_ROOT="$OSS_ROOT/runs/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
readonly EVAL_RUN_NAME=b4-iter20000-robocasa365-seed42
readonly LAUNCHER_TERMINAL="$EVAL_LOCAL_OUTPUT_ROOT/launcher-terminal.json"
readonly REQUIRED_LOCAL_BYTES=161061273600

STARTED_AT="$(date -u +%FT%TZ)"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" && ! -L "$1" ]] || die "missing regular file: $1"
}

require_control_file() {
    require_file "$1"
    [[ "$(stat -c %a "$1")" == 444 ]] || die "control file must have mode 0444: $1"
}

copy_regular_file() {
    local source="$1"
    local destination="$2"
    local expected_bytes="$3"
    [[ -f "$source" && ! -L "$source" ]] || die "source file is missing or unsafe: $source"
    [[ "$(stat -c %s -- "$source")" == "$expected_bytes" ]] ||
        die "source file size mismatch: $source"
    mkdir --parents --mode=0755 -- "$(dirname -- "$destination")"
    [[ ! -e "$destination" && ! -L "$destination" ]] ||
        die "refusing to replace local staged file: $destination"
    cp --reflink=auto --sparse=always --no-preserve=ownership,mode,timestamps -- \
        "$source" "$destination"
    chmod 0444 -- "$destination"
    [[ "$(stat -c %s -- "$destination")" == "$expected_bytes" ]] ||
        die "local staged file size mismatch: $destination"
    printf 'B4_LOCAL_FILE_READY bytes=%s destination=%s\n' "$expected_bytes" "$destination"
}

prepare_empty_stage() {
    local stage="$1"
    [[ ! -e "$stage" && ! -L "$stage" ]] ||
        die "refusing to replace extraction stage: $stage"
    mkdir --parents --mode=0755 -- "$stage"
    [[ -d "$stage" && ! -L "$stage" ]] ||
        die "extraction stage is not a real directory: $stage"
    [[ -z "$(find "$stage" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
        die "extraction stage is not empty: $stage"
    printf 'B4_EMPTY_STAGE_READY destination=%s\n' "$stage"
}

stage_runtime_directory() {
    local label="$1"
    local source="$2"
    local destination="$3"
    local parent
    local stage

    [[ -d "$source" && ! -L "$source" ]] ||
        die "$label source is missing or unsafe: $source"
    [[ ! -e "$destination" && ! -L "$destination" ]] ||
        die "$label destination already exists: $destination"
    parent="$(dirname -- "$destination")"
    stage="${destination}.incomplete-${EVAL_EXPERIMENT_ID}-$$"
    [[ ! -e "$stage" && ! -L "$stage" ]] ||
        die "$label stage already exists: $stage"
    mkdir --parents --mode=0755 -- "$parent"
    chmod 0755 -- "$(dirname -- "$parent")" "$parent"
    mkdir --mode=0755 -- "$stage"
    chmod 0755 -- "$stage"
    cp --archive --reflink=auto -- "$source/." "$stage/"
    sync -f -- "$stage"
    mv -T -- "$stage" "$destination"
    [[ -d "$destination" && ! -L "$destination" ]] ||
        die "$label atomic local publication failed"
    printf 'B4_RUNTIME_STAGE_READY label=%s source=%s destination=%s\n' \
        "$label" "$source" "$destination"
}

git_revision() {
    git -c safe.directory="$1" -C "$1" rev-parse HEAD
}

write_launcher_terminal() {
    local status="$1"
    local exit_code="$2"
    local completed_at
    completed_at="$(date -u +%FT%TZ)"
    python3 - "$LAUNCHER_TERMINAL" "$status" "$exit_code" "$completed_at" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "metadata_no_hash_v1",
    "kind": "cwam_b4_dlc_launcher_terminal",
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "completed_at_utc": sys.argv[4],
    "experiment_id": os.environ["EVAL_EXPERIMENT_ID"],
    "logical_run_id": os.environ["EVAL_LOGICAL_RUN_ID"],
    "dlc_job_id": os.environ.get("DLC_JOB_ID"),
    "control_commit": os.environ["EVAL_CONTROL_COMMIT"],
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
    temporary = pathlib.Path(stream.name)
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.chmod(temporary, 0o444)
os.replace(temporary, path)
PY
}

persist_formal_output() {
    local formal_exit_code="$?"
    local persistence_exit_code
    local terminal_exit_code
    trap - EXIT INT TERM
    set +e
    if [[ -d "$EVAL_LOCAL_OUTPUT_ROOT" ]]; then
        if ((formal_exit_code == 0)); then
            write_launcher_terminal COMPLETE 0
            terminal_exit_code=$?
            if ((terminal_exit_code != 0)); then
                printf 'ERROR: failed to write successful launcher terminal receipt\n' >&2
                formal_exit_code=1
            fi
        else
            write_launcher_terminal FAILED "$formal_exit_code" || true
        fi
    fi
    python3 "$PERSIST_OUTPUT_HELPER" \
        --local-output-parent "$DLC_LOCAL_FORMAL_PARENT" \
        --local-output-root "$EVAL_LOCAL_OUTPUT_ROOT" \
        --persistent-output-parent "$OSS_ROOT/runs" \
        --persistent-output-root "$EVAL_PERSISTENT_OUTPUT_ROOT" \
        --experiment-id "$EVAL_EXPERIMENT_ID" \
        --run-id "$EVAL_LOGICAL_RUN_ID" \
        --control-commit "$EVAL_CONTROL_COMMIT" \
        --launcher-exit-code "$formal_exit_code" \
        --job-id "${DLC_JOB_ID:-${PAI_JOB_ID:-unknown-at-runtime}}"
    persistence_exit_code=$?
    if ((persistence_exit_code != 0)); then
        printf 'ERROR B4_PERSISTENCE_FAILED formal_exit_code=%s persistence_exit_code=%s\n' \
            "$formal_exit_code" "$persistence_exit_code" >&2
        exit 90
    fi
    printf 'B4_PERSISTENCE_COMPLETE formal_exit_code=%s root=%s\n' \
        "$formal_exit_code" "$EVAL_PERSISTENT_OUTPUT_ROOT"
    exit "$formal_exit_code"
}

[[ "$EVAL_CONTROL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid control commit"
[[ "$(findmnt -n -T /mnt/workspace -o TARGET)" == /mnt/workspace ]] || die "CPFS mount missing"
[[ "$(findmnt -n -T /oss-chengjuntao -o TARGET)" == /oss-chengjuntao ]] || die "OSS mount missing"
[[ -w /oss-chengjuntao ]] || die "OSS mount is not writable"
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || die "source snapshot missing"
[[ "$(git_revision "$SOURCE_ROOT")" == "$SOURCE_COMMIT" ]] || die "source snapshot revision mismatch"
[[ -z "$(git -c safe.directory="$SOURCE_ROOT" -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || die "source snapshot is dirty"
for file in "$RUNNER" "$SHARD_GROUP_RUNNER" "$B4_SHELL_UTILS" "$B4_EVAL_HELPER" \
    "$ENV_HELPER" "$ASSET_HELPER" "$EGL_PROBE" "$PERSIST_OUTPUT_HELPER"; do
    require_control_file "$file"
done
for file in "$SOURCE_ENV_ARCHIVE" "$SOURCE_ENV_RECEIPT" \
    "$SOURCE_ROBOCASA_ARCHIVE" "$SOURCE_ROBOCASA_OVERLAY" \
    "$ACTION_STATS" "$VAE_SOURCE" "$EGL_FRONTEND_SOURCE"; do
    require_file "$file"
done
for directory in "$CHECKPOINT_SOURCE" "$DEMIAN_REPO" "$ROBOSUITE_REPO" "$ROBOCASA_ENV" \
    "$QWEN_SOURCE" "$RUNTIME_CLOSURE_SOURCE"; do
    [[ -d "$directory" && ! -L "$directory" ]] || die "missing input directory: $directory"
done
[[ "$(stat -c %s "$SOURCE_ENV_ARCHIVE")" == 14736947200 ]] || die "environment archive size mismatch"
[[ "$(stat -c %s "$SOURCE_ENV_RECEIPT")" == 2124 ]] || die "environment receipt size mismatch"
[[ "$(stat -c %s "$SOURCE_ROBOCASA_ARCHIVE")" == 11101859840 ]] || die "RoboCasa archive size mismatch"
[[ "$(stat -c %s "$SOURCE_ROBOCASA_OVERLAY")" == 2181120 ]] || die "RoboCasa overlay size mismatch"
[[ "$(stat -c %s "$VAE_SOURCE")" == 2818839170 ]] || die "VAE size mismatch"
[[ ! -e "$EVAL_PERSISTENT_OUTPUT_ROOT" && ! -L "$EVAL_PERSISTENT_OUTPUT_ROOT" ]] || \
    die "persistent formal output already exists"
mkdir --parents --mode=0700 -- "$DLC_LOCAL_FORMAL_PARENT"
[[ ! -e "$DLC_LOCAL_FORMAL_PARENT/$EVAL_EXPERIMENT_ID" && \
    ! -L "$DLC_LOCAL_FORMAL_PARENT/$EVAL_EXPERIMENT_ID" ]] || \
    die "refusing to reuse local formal experiment output"
mkdir --parents --mode=0700 -- "$EVAL_LOCAL_OUTPUT_ROOT"
trap persist_formal_output EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf "$SCRATCH_ROOT" /tmp/cwam_cosmos3_robocasa365_eval_nvme
available_local_bytes="$(LC_ALL=C df -B1 --output=avail /tmp | awk 'NR == 2 { print $1 }')"
[[ "$available_local_bytes" =~ ^[0-9]+$ && "$available_local_bytes" -ge "$REQUIRED_LOCAL_BYTES" ]] ||
    die "worker-local /tmp has less than 150 GiB available"
mkdir -p "$SCRATCH_ROOT/inputs/repos" "$SCRATCH_ROOT/inputs/checkpoints" \
    "$(dirname "$WAN2PT2_VAE")" "$(dirname "$COSMOS_ENV")" "$EGL_ROOT"
prepare_empty_stage "$COSMOS_ENV"
prepare_empty_stage "$ROBOCASA_REPO"
stage_runtime_directory \
    cosmos_private_glibc \
    "$RUNTIME_CLOSURE_SOURCE/cosmos-r21-private-glibc235-v1" \
    "$LOCAL_PRIVATE_GLIBC"
stage_runtime_directory \
    cosmos_gcc13_runtime \
    "$RUNTIME_CLOSURE_SOURCE/cosmos-r21-gcc13-runtime-v3" \
    "$LOCAL_GCC_RUNTIME"
require_file "$LOCAL_PRIVATE_GLIBC/root/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
require_file "$LOCAL_GCC_RUNTIME/runtime-root/lib/libstdc++.so.6.0.32"

copy_regular_file "$SOURCE_ENV_ARCHIVE" "$LOCAL_ENV_ARCHIVE" 14736947200
copy_regular_file "$SOURCE_ENV_RECEIPT" "$LOCAL_ENV_RECEIPT" 2124
copy_regular_file "$SOURCE_ROBOCASA_ARCHIVE" "$LOCAL_ROBOCASA_ARCHIVE" 11101859840
copy_regular_file "$SOURCE_ROBOCASA_OVERLAY" "$LOCAL_ROBOCASA_OVERLAY" 2181120

python3 "$ENV_HELPER" verify-extract-metadata \
    --archive "$LOCAL_ENV_ARCHIVE" \
    --receipt "$LOCAL_ENV_RECEIPT" \
    --expected-publication-archive "$PUBLICATION_ENV_ARCHIVE" \
    --expected-archive-bytes 14736947200 \
    --expected-receipt-bytes 2124 \
    --expected-ready-marker-bytes 973 \
    --expected-regular-files 72662 \
    --expected-directories 11389 \
    --expected-symlinks 4 \
    --expected-regular-file-bytes 14660695665 \
    --expected-source-commit 09f23119ea92c707207bba55565e7a09d16896a2 \
    --stage "$COSMOS_ENV"

python3 "$ASSET_HELPER" verify-extract \
    --archive "$LOCAL_ROBOCASA_ARCHIVE" --expected-archive-bytes 11101859840 \
    --overlay "$LOCAL_ROBOCASA_OVERLAY" --expected-overlay-bytes 2181120 \
    --stage "$ROBOCASA_REPO" \
    --receipt-output "$SCRATCH_ROOT/robocasa-asset-extraction.json"

cp -a "$CHECKPOINT_SOURCE" "$CHECKPOINT_ARTIFACT_ROOT"
cp -a "$VAE_SOURCE" "$WAN2PT2_VAE"
cp -a "$QWEN_SOURCE" "$QWEN_TOKENIZER"
cp -a "$EGL_FRONTEND_SOURCE" "$EGL_ROOT/libEGL.so.1.1.0"
ln -s libEGL.so.1.1.0 "$EGL_ROOT/libEGL.so.1"
ln -s libEGL.so.1 "$EGL_ROOT/libEGL.so"
export LD_LIBRARY_PATH="$EGL_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$(dirname "$EVAL_EGL_VENDOR_JSON")"
if [[ ! -e "$EVAL_EGL_VENDOR_JSON" ]]; then
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' >"$EVAL_EGL_VENDOR_JSON"
fi
require_file "$EVAL_EGL_VENDOR_JSON"

python3 - "$EGL_ROOT/libEGL.so.1" <<'PY'
import ctypes
import sys
ctypes.CDLL(sys.argv[1])
ctypes.CDLL("libEGL_nvidia.so.0")
print("EGL libraries loaded")
PY

env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
    CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON" \
    PYTHONPATH="$ROBOCASA_REPO:$ROBOSUITE_REPO" \
    "$ROBOCASA_ENV/bin/python" "$EGL_PROBE" --hold-seconds 1 \
    >"$EVAL_LOCAL_OUTPUT_ROOT/egl-one-gpu-smoke.log" 2>&1
printf 'B4_ONE_GPU_EGL_SMOKE_PASSED log=%s\n' \
    "$EVAL_LOCAL_OUTPUT_ROOT/egl-one-gpu-smoke.log"

export COSMOS_SOURCE="$SOURCE_ROOT"
export COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO ROBOCASA_REPO ROBOSUITE_REPO
export CHECKPOINT_ARTIFACT_ROOT SERVING_CHECKPOINT_PATH ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER
export TRAINING_CONFIG="$CHECKPOINT_ARTIFACT_ROOT/config.yaml"
export EVAL_OUTPUT_ROOT="$EVAL_LOCAL_OUTPUT_ROOT"
export EVAL_WORKER_SCRATCH="$SCRATCH_ROOT/runtime/eval"
export EVAL_RUN_NAME EVAL_EGL_VENDOR_JSON B4_EVAL_HELPER SHARD_GROUP_RUNNER B4_SHELL_UTILS

bash "$RUNNER"
[[ -s "$EVAL_LOCAL_OUTPUT_ROOT/evaluation_evidence.json" ]] || die "terminal evaluation evidence missing"
[[ -s "$EVAL_LOCAL_OUTPUT_ROOT/status.json" ]] || die "terminal evaluation status missing"
python3 - "$EVAL_LOCAL_OUTPUT_ROOT/evaluation_evidence.json" "$EVAL_LOCAL_OUTPUT_ROOT/status.json" <<'PY'
import json
import pathlib
import sys
for name in sys.argv[1:]:
    value = json.loads(pathlib.Path(name).read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        raise SystemExit(f"non-terminal formal artifact: {name}: {value.get('status')}")
PY
