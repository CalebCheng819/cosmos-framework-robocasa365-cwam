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
readonly ENV_ARCHIVE="$INPUT_ROOT/envs/archives/cosmos-framework-09f23119-cu128-train-r22-complete.tar"
readonly ENV_RECEIPT="$OSS_ROOT/control/receipts/cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json"
readonly ROBOCASA_ARCHIVE="$OSS_ROOT/inputs/archives/robocasa-29f7ce8814c1-assets-source-package-20260729.tar"
readonly ROBOCASA_OVERLAY="$OSS_ROOT/inputs/archives/robocasa-29f7ce8814c1-assets-legacy-fixture-overlay-20260817.tar"
readonly DEMIAN_REPO="$INPUT_ROOT/demian-2600ece5bf0e"
readonly ROBOSUITE_REPO="$INPUT_ROOT/robosuite-232ce7d4a6ed"
readonly ROBOCASA_ENV="$INPUT_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
readonly ACTION_STATS="$INPUT_ROOT/config/robocasa365_native_frame_wise_relative_rot6d.json"
readonly VAE_SOURCE="$INPUT_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly QWEN_SOURCE="$INPUT_ROOT/models/Qwen3-VL-8B-Instruct-tokenizer"
readonly EGL_FRONTEND_SOURCE="$INPUT_ROOT/runtime/glvnd/libEGL.so.1.1.0"
readonly RUNNER="$CONTROL_ROOT/run_cwam_b4_robocasa365_eval_dlc.sh"
readonly SHARD_GROUP_RUNNER="$CONTROL_ROOT/run_cwam_b4_robocasa365_eval_shard_group.sh"
readonly B4_EVAL_HELPER="$CONTROL_ROOT/cwam_b4_robocasa365_eval.py"
readonly ENV_HELPER="$CONTROL_ROOT/cwam_cosmos_env_archive.py"
readonly ASSET_HELPER="$CONTROL_ROOT/cwam_robocasa_asset_source_package.py"
readonly EGL_PROBE="$CONTROL_ROOT/probe_mujoco_egl_mapping.py"
readonly SCRATCH_ROOT="/tmp/cwam_b4_eval_$EVAL_EXPERIMENT_ID"
readonly COSMOS_ENV=/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs/envs/cosmos-framework-09f23119-cu128-train
readonly ROBOCASA_REPO="$SCRATCH_ROOT/inputs/repos/robocasa-29f7ce8814c1"
readonly CHECKPOINT_ARTIFACT_ROOT="$SCRATCH_ROOT/inputs/checkpoints/$CHECKPOINT_NAME"
readonly SERVING_CHECKPOINT_PATH="$CHECKPOINT_ARTIFACT_ROOT/model"
readonly WAN2PT2_VAE="$SCRATCH_ROOT/inputs/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly QWEN_TOKENIZER="$SCRATCH_ROOT/inputs/models/Qwen3-VL-8B-Instruct-tokenizer"
readonly EGL_ROOT="$SCRATCH_ROOT/runtime/glvnd"
readonly EVAL_EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
readonly OUTPUT_PARENT="$OSS_ROOT/runs/$EVAL_EXPERIMENT_ID"
readonly OUTPUT_FINAL="$OUTPUT_PARENT/$EVAL_LOGICAL_RUN_ID"
readonly OUTPUT_STAGE="$OUTPUT_PARENT/.$EVAL_LOGICAL_RUN_ID.incomplete-${DLC_JOB_ID:-unknown}-$$"
readonly EVAL_RUN_NAME=b4-iter20000-robocasa365-seed42
readonly LAUNCHER_TERMINAL="$OUTPUT_STAGE/launcher-terminal.json"

OUTPUT_PUBLISHED=0
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

on_exit() {
    local exit_code="$?"
    trap - EXIT INT TERM
    if [[ -d "$OUTPUT_STAGE" ]]; then
        if ((exit_code == 0)); then
            write_launcher_terminal COMPLETE 0 || exit_code=1
        else
            write_launcher_terminal FAILED "$exit_code" || true
        fi
        if [[ ! -e "$OUTPUT_FINAL" ]]; then
            mv -T "$OUTPUT_STAGE" "$OUTPUT_FINAL" || true
            OUTPUT_PUBLISHED=1
        fi
    fi
    exit "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$EVAL_CONTROL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid control commit"
[[ "$(findmnt -n -T /mnt/workspace -o TARGET)" == /mnt/workspace ]] || die "CPFS mount missing"
[[ "$(findmnt -n -T /oss-chengjuntao -o TARGET)" == /oss-chengjuntao ]] || die "OSS mount missing"
[[ -w /oss-chengjuntao ]] || die "OSS mount is not writable"
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || die "source snapshot missing"
[[ "$(git_revision "$SOURCE_ROOT")" == "$SOURCE_COMMIT" ]] || die "source snapshot revision mismatch"
[[ -z "$(git -c safe.directory="$SOURCE_ROOT" -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || die "source snapshot is dirty"
for file in "$RUNNER" "$SHARD_GROUP_RUNNER" "$B4_EVAL_HELPER" "$ENV_HELPER" "$ASSET_HELPER" "$EGL_PROBE"; do
    require_control_file "$file"
done
for file in "$ENV_ARCHIVE" "$ENV_RECEIPT" "$ROBOCASA_ARCHIVE" "$ROBOCASA_OVERLAY" \
    "$ACTION_STATS" "$VAE_SOURCE" "$EGL_FRONTEND_SOURCE"; do
    require_file "$file"
done
for directory in "$CHECKPOINT_SOURCE" "$DEMIAN_REPO" "$ROBOSUITE_REPO" "$ROBOCASA_ENV" "$QWEN_SOURCE"; do
    [[ -d "$directory" && ! -L "$directory" ]] || die "missing input directory: $directory"
done
[[ "$(stat -c %s "$ENV_ARCHIVE")" == 14736947200 ]] || die "environment archive size mismatch"
[[ "$(stat -c %s "$ENV_RECEIPT")" == 2124 ]] || die "environment receipt size mismatch"
[[ "$(stat -c %s "$ROBOCASA_ARCHIVE")" == 11101859840 ]] || die "RoboCasa archive size mismatch"
[[ "$(stat -c %s "$ROBOCASA_OVERLAY")" == 2181120 ]] || die "RoboCasa overlay size mismatch"
[[ "$(stat -c %s "$VAE_SOURCE")" == 2818839170 ]] || die "VAE size mismatch"
[[ ! -e "$OUTPUT_FINAL" && ! -e "$OUTPUT_STAGE" ]] || die "formal output already exists"
mkdir -p "$OUTPUT_PARENT" "$OUTPUT_STAGE"

rm -rf "$SCRATCH_ROOT" /tmp/cwam_cosmos3_robocasa365_eval_nvme
mkdir -p "$SCRATCH_ROOT/inputs/repos" "$SCRATCH_ROOT/inputs/checkpoints" \
    "$(dirname "$WAN2PT2_VAE")" "$(dirname "$COSMOS_ENV")" "$EGL_ROOT"

python3 "$ENV_HELPER" verify-extract-metadata \
    --archive "$ENV_ARCHIVE" \
    --receipt "$ENV_RECEIPT" \
    --expected-publication-archive "$ENV_ARCHIVE" \
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
    --archive "$ROBOCASA_ARCHIVE" --expected-archive-bytes 11101859840 \
    --overlay "$ROBOCASA_OVERLAY" --expected-overlay-bytes 2181120 \
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
    >"$OUTPUT_STAGE/egl-one-gpu-smoke.log" 2>&1

export COSMOS_SOURCE="$SOURCE_ROOT"
export COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO ROBOCASA_REPO ROBOSUITE_REPO
export CHECKPOINT_ARTIFACT_ROOT SERVING_CHECKPOINT_PATH ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER
export TRAINING_CONFIG="$CHECKPOINT_ARTIFACT_ROOT/config.yaml"
export EVAL_OUTPUT_ROOT="$OUTPUT_STAGE"
export EVAL_WORKER_SCRATCH="$SCRATCH_ROOT/runtime/eval"
export EVAL_RUN_NAME EVAL_EGL_VENDOR_JSON B4_EVAL_HELPER SHARD_GROUP_RUNNER

bash "$RUNNER"
[[ -s "$OUTPUT_STAGE/evaluation_evidence.json" ]] || die "terminal evaluation evidence missing"
[[ -s "$OUTPUT_STAGE/status.json" ]] || die "terminal evaluation status missing"
python3 - "$OUTPUT_STAGE/evaluation_evidence.json" "$OUTPUT_STAGE/status.json" <<'PY'
import json
import pathlib
import sys
for name in sys.argv[1:]:
    value = json.loads(pathlib.Path(name).read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        raise SystemExit(f"non-terminal formal artifact: {name}: {value.get('status')}")
PY
