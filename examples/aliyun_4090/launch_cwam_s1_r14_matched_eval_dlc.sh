#!/usr/bin/env bash
# Stage OSS-backed assets on worker-local storage, publish S1, and launch the
# fixed two-task S1-versus-R14 evaluation protocol on one eight-GPU DLC worker.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${EVAL_CONTROL_COMMIT:?Set the immutable evaluation-control Git commit}"
: "${EVAL_EXPERIMENT_ID:?Set the formal evaluation Experiment ID}"
: "${EVAL_LOGICAL_RUN_ID:?Set the formal evaluation Run ID}"
[[ "$EVAL_CONTROL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'ERROR EVAL_CONTROL_COMMIT must be a lowercase 40-character Git commit\n' >&2
    exit 2
}
[[ "$EVAL_EXPERIMENT_ID" =~ ^CWAM-DLC-ROB365-S1-VS-R14-MATCHED2-8GPU-S42-R[0-9]+-[0-9]{8}$ ]] || {
    printf 'ERROR EVAL_EXPERIMENT_ID has an invalid formal ID\n' >&2
    exit 2
}
[[ "$EVAL_LOGICAL_RUN_ID" =~ ^CWAM-DLC-ROB365-COSMOS3-NANO-S1-EMA-10000-S42-MATCHED2-R[0-9]+-[0-9]{8}$ ]] || {
    printf 'ERROR EVAL_LOGICAL_RUN_ID has an invalid formal ID\n' >&2
    exit 2
}

readonly EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID
readonly R14_BASELINE_RUN_ID=CWAM-DSW960-ROB365-COSMOS3-NANO-R14-EMA-340EP-S42-FRESH-RERUN2-20260809
readonly PROJECT_ROOT=/mnt/workspace/cwam_cosmos3_robocasa365_eval
readonly INPUTS_ROOT="$PROJECT_ROOT/inputs"
readonly OSS_PROJECT_ROOT=/oss-chengjuntao/cpfs-user-chengjuntao/cwam_cosmos3_robocasa365_eval
readonly CONTROL_DEPLOY_ROOT="$PROJECT_ROOT/control/scripts/$EVAL_LOGICAL_RUN_ID"
readonly R14_BASELINE_ROOT="$PROJECT_ROOT/control/baselines/$R14_BASELINE_RUN_ID"
readonly SOURCE_SNAPSHOT_INPUT_ROOT="$INPUTS_ROOT/source/CWAM-DLC-ROB365-COSMOS3-NANO-S1-20260803-dddd8fd3f61a"
readonly ROBOCASA_ENV_ROOT="$INPUTS_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
readonly LOCAL_ASSET_ROOT=/tmp/cwam_s1_r14_matched_assets
readonly LOCAL_BUILD_ROOT=/tmp/.cwam_s1_r14_matched_assets.incomplete
readonly LOCAL_CHECKPOINT_PARENT="$LOCAL_ASSET_ROOT/checkpoints"
readonly LOCAL_CHECKPOINT_STAGE="$LOCAL_CHECKPOINT_PARENT/.S1-R3-iter_000010000-serving-ema-bf16-v1.incomplete-transfer-20260803"
readonly LOCAL_CHECKPOINT_ROOT="$LOCAL_CHECKPOINT_PARENT/S1-R3-iter_000010000-serving-ema-bf16-v1"
readonly SOURCE_CHECKPOINT_ROOT="$OSS_PROJECT_ROOT/inputs/checkpoints/S1-R3-iter_000010000-serving-ema-bf16-v1"
readonly SOURCE_ENV_ARCHIVE="$OSS_PROJECT_ROOT/inputs/envs/archives/cosmos-framework-09f23119-cu128-train-r22-complete.tar"
readonly SOURCE_ENV_RECEIPT="$OSS_PROJECT_ROOT/control/receipts/cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json"
readonly SOURCE_RUNTIME_RECEIPT="$OSS_PROJECT_ROOT/control/receipts/cosmos-r21-dlc-runtime-v1-publication-v2.json"
readonly SOURCE_WAN_VAE="$OSS_PROJECT_ROOT/inputs/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly SOURCE_QWEN_TOKENIZER="$OSS_PROJECT_ROOT/inputs/models/Qwen3-VL-8B-Instruct-tokenizer"
readonly LOCAL_ENV_ARCHIVE="$LOCAL_ASSET_ROOT/envs/cosmos-framework-09f23119-cu128-train-r22-complete.tar"
readonly LOCAL_ENV_RECEIPT="$LOCAL_ASSET_ROOT/control/cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json"
readonly LOCAL_RUNTIME_RECEIPT="$LOCAL_ASSET_ROOT/control/cosmos-r21-dlc-runtime-v1-publication-v2.json"
readonly LOCAL_CHECKPOINT_RECEIPT="$LOCAL_ASSET_ROOT/control/S1-R3-iter_000010000-serving-ema-bf16-v1-atomic-publication.json"
readonly LOCAL_CHECKPOINT_LOCK="$LOCAL_ASSET_ROOT/control/S1-R3-iter_000010000-serving-ema-bf16-v1-atomic-publication.lock"
readonly LOCAL_WAN_VAE="$LOCAL_ASSET_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly LOCAL_QWEN_TOKENIZER="$LOCAL_ASSET_ROOT/models/Qwen3-VL-8B-Instruct-tokenizer"
readonly FORMAL_RUNNER_OVERRIDE="$CONTROL_DEPLOY_ROOT/run_cwam_cosmos3_robocasa365_eval_dlc.sh"
readonly FORMAL_LAUNCHER="$CONTROL_DEPLOY_ROOT/launch_cwam_cosmos3_robocasa365_eval_dlc.sh"
readonly EGL_MATRIX_OVERRIDE="$CONTROL_DEPLOY_ROOT/run_mujoco_egl_mapping_matrix_dlc.sh"
readonly S1_R14_MATCHED_HELPER="$CONTROL_DEPLOY_ROOT/cwam_s1_r14_matched_eval.py"
readonly COSMOS_ENV_ARCHIVE_METADATA_HELPER="$CONTROL_DEPLOY_ROOT/cwam_cosmos_env_archive.py"
readonly CHECKPOINT_PUBLISHER="$CONTROL_DEPLOY_ROOT/publish_cwam_ema_checkpoint_dsw.py"
readonly CHECKPOINT_CONTRACT="$CONTROL_DEPLOY_ROOT/cwam_s1_local_checkpoint_publication_contract.json"
readonly PERSIST_OUTPUT_HELPER="$CONTROL_DEPLOY_ROOT/cwam_dlc_persist_eval_output.py"
readonly SOURCE_STAGE_HELPER="$CONTROL_DEPLOY_ROOT/cwam_stage_source_snapshot.py"
readonly DLC_LOCAL_SCRATCH_ROOT=/tmp/cwam_cosmos3_robocasa365_eval
readonly DLC_LOCAL_SOURCE_PARENT="$DLC_LOCAL_SCRATCH_ROOT/inputs/source"
readonly SOURCE_SNAPSHOT_ROOT="$DLC_LOCAL_SOURCE_PARENT/CWAM-DLC-ROB365-COSMOS3-NANO-S1-20260803-dddd8fd3f61a"
readonly EGL_SMOKE_PROBE="$SOURCE_SNAPSHOT_ROOT/examples/aliyun_4090/probe_mujoco_egl_mapping.py"
readonly R14_BASELINE_ARGS="$R14_BASELINE_ROOT/args_shard0.json"
readonly R14_BASELINE_RESULTS="$R14_BASELINE_ROOT/results_shard0.json"
readonly EGL_GLVD_FRONTEND_SOURCE="$INPUTS_ROOT/runtime/glvnd/libEGL.so.1.1.0"
readonly EGL_GLVD_RUNTIME_ROOT=/tmp/cwam-glvnd
readonly EGL_GLVD_FRONTEND="$EGL_GLVD_RUNTIME_ROOT/libEGL.so.1.1.0"
readonly EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
readonly EGL_VENDOR_DOCUMENT='{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'
readonly REQUIRED_LOCAL_BYTES=161061273600
readonly DLC_LOCAL_FORMAL_PARENT="$DLC_LOCAL_SCRATCH_ROOT/formal-runs"
readonly EVAL_LOCAL_OUTPUT_ROOT="$DLC_LOCAL_FORMAL_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
readonly EVAL_PERSISTENT_OUTPUT_ROOT="$OSS_PROJECT_ROOT/runs/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_control_file() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || die "missing immutable control file: $path"
    [[ "$(realpath -e -- "$path")" == "$path" ]] || die "control file is not canonical: $path"
    [[ "$(stat -c %a -- "$path")" == 444 ]] || die "control file must have mode 0444: $path"
}

prepare_nvidia_egl_runtime() {
    local egl_vendor_dir
    local temporary_vendor_json

    case ",${NVIDIA_DRIVER_CAPABILITIES:-}," in
        *,all,* | *,graphics,*) ;;
        *) die "NVIDIA_DRIVER_CAPABILITIES must include graphics or all" ;;
    esac
    [[ -f "$EGL_GLVD_FRONTEND_SOURCE" && ! -L "$EGL_GLVD_FRONTEND_SOURCE" ]] ||
        die "controlled GLVND EGL frontend is missing or unsafe"
    [[ "$(realpath -e -- "$EGL_GLVD_FRONTEND_SOURCE")" == "$EGL_GLVD_FRONTEND_SOURCE" ]] ||
        die "controlled GLVND EGL frontend is not canonical"
    [[ "$(stat -c %a -- "$EGL_GLVD_FRONTEND_SOURCE")" == 444 ]] ||
        die "controlled GLVND EGL frontend must have mode 0444"
    [[ "$(stat -c %s -- "$EGL_GLVD_FRONTEND_SOURCE")" == 72352 ]] ||
        die "controlled GLVND EGL frontend size mismatch"
    [[ ! -e "$EGL_GLVD_RUNTIME_ROOT" && ! -L "$EGL_GLVD_RUNTIME_ROOT" ]] ||
        die "refusing to reuse worker-local GLVND runtime"
    install -d -m 0755 -- "$EGL_GLVD_RUNTIME_ROOT"
    install -m 0444 -- "$EGL_GLVD_FRONTEND_SOURCE" "$EGL_GLVD_FRONTEND"
    ln -s -- "$(basename -- "$EGL_GLVD_FRONTEND")" "$EGL_GLVD_RUNTIME_ROOT/libEGL.so.1"
    ln -s -- "$(basename -- "$EGL_GLVD_FRONTEND")" "$EGL_GLVD_RUNTIME_ROOT/libEGL.so"
    export LD_LIBRARY_PATH="$EGL_GLVD_RUNTIME_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    python3 - "$EGL_GLVD_FRONTEND" <<'PY' || die "GLVND EGL frontend or NVIDIA backend is unavailable"
import ctypes
import sys

frontend = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
getattr(frontend, "eglQueryString")
ctypes.CDLL("libEGL_nvidia.so.0", mode=ctypes.RTLD_GLOBAL)
PY

    egl_vendor_dir="$(dirname -- "$EGL_VENDOR_JSON")"
    if [[ ! -e "$EGL_VENDOR_JSON" && ! -L "$EGL_VENDOR_JSON" ]]; then
        install -d -m 0755 -- "$egl_vendor_dir"
        temporary_vendor_json="$(mktemp "$egl_vendor_dir/.10_nvidia.json.XXXXXX")"
        printf '%s\n' "$EGL_VENDOR_DOCUMENT" > "$temporary_vendor_json"
        chmod 0644 -- "$temporary_vendor_json"
        mv -T -- "$temporary_vendor_json" "$EGL_VENDOR_JSON"
    fi
    [[ -f "$EGL_VENDOR_JSON" && ! -L "$EGL_VENDOR_JSON" ]] ||
        die "NVIDIA EGL vendor JSON is missing or unsafe"
    [[ "$(stat -c %a -- "$EGL_VENDOR_JSON")" == 644 ]] ||
        die "NVIDIA EGL vendor JSON must have mode 0644"
    python3 - "$EGL_VENDOR_JSON" <<'PY' || die "NVIDIA EGL vendor JSON has unexpected content"
import json
import pathlib
import sys

document = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "file_format_version": "1.0.0",
    "ICD": {"library_path": "libEGL_nvidia.so.0"},
}
if document != expected:
    raise SystemExit(1)
PY
    printf 'S1_MATCHED_NVIDIA_EGL_RUNTIME_READY frontend=%s vendor=%s capabilities=%s\n' \
        "$EGL_GLVD_FRONTEND" "$EGL_VENDOR_JSON" "$NVIDIA_DRIVER_CAPABILITIES"
}

preflight_mujoco_egl_runtime() {
    [[ -x "$ROBOCASA_ENV_ROOT/bin/python" ]] || die "RoboCasa Python is not executable"
    [[ -f "$EGL_SMOKE_PROBE" && ! -L "$EGL_SMOKE_PROBE" ]] ||
        die "MuJoCo EGL smoke probe is missing or unsafe"
    [[ "$(stat -c %a -- "$EGL_SMOKE_PROBE")" == 444 ]] ||
        die "MuJoCo EGL smoke probe must have mode 0444"
    env \
        -u LD_PRELOAD \
        CUDA_VISIBLE_DEVICES=0 \
        MUJOCO_EGL_DEVICE_ID=0 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
        "$ROBOCASA_ENV_ROOT/bin/python" "$EGL_SMOKE_PROBE" --hold-seconds 1
    printf 'S1_MATCHED_PRECOPY_MUJOCO_EGL_SMOKE_PASSED gpu=0\n'
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
    printf 'S1_MATCHED_LOCAL_FILE_READY bytes=%s destination=%s\n' \
        "$expected_bytes" "$destination"
}

copy_qwen_tokenizer() {
    local source="$1"
    local destination="$2"
    local unsafe
    [[ -d "$source" && ! -L "$source" ]] || die "Qwen tokenizer source is missing or unsafe"
    unsafe="$(find "$source" -mindepth 1 ! -type d ! -type f -print -quit)"
    [[ -z "$unsafe" ]] || die "Qwen tokenizer contains unsupported entry: $unsafe"
    [[ ! -e "$destination" && ! -L "$destination" ]] ||
        die "refusing to replace local Qwen tokenizer"
    mkdir --parents --mode=0755 -- "$destination"
    cp --archive --no-preserve=ownership,mode,timestamps -- "$source/." "$destination/"
    find "$destination" -type d -exec chmod 0755 {} +
    find "$destination" -type f -exec chmod 0444 {} +
    printf 'S1_MATCHED_LOCAL_DIRECTORY_READY files=%s destination=%s\n' \
        "$(find "$destination" -type f | wc -l)" "$destination"
}

copy_checkpoint_closure() {
    local source="$1"
    local destination="$2"
    local names=(
        H-SHA256SUMS
        config.yaml
        h-serving-checkpoint-manifest.json
        ema-bf16-export-receipt.json
        ema-bf16-key-manifest.json
        ema-bf16-validation.json
        source-dcp-full-verification.json
        ema-serving-dcp-transfer-manifest.json
    )
    local sizes=(1303 21049 1822 2873 5474802 2001 3635 11166)
    local model_names=(
        .metadata
        __0_0.distcp
        __1_0.distcp
        __2_0.distcp
        __3_0.distcp
        __4_0.distcp
        __5_0.distcp
        __6_0.distcp
        __7_0.distcp
    )
    local model_sizes=(
        1356571
        3796677222
        3796677158
        3796675645
        3796675645
        3796675645
        3796675645
        3796675645
        3796675645
    )
    local index

    [[ -d "$source" && ! -L "$source" ]] || die "S1 source checkpoint is missing or unsafe"
    mkdir --parents --mode=0755 -- "$destination/model"
    for index in "${!names[@]}"; do
        copy_regular_file \
            "$source/${names[$index]}" \
            "$destination/${names[$index]}" \
            "${sizes[$index]}"
    done
    for index in "${!model_names[@]}"; do
        copy_regular_file \
            "$source/model/${model_names[$index]}" \
            "$destination/model/${model_names[$index]}" \
            "${model_sizes[$index]}"
    done
}

stage_source_snapshot() {
    python3 -B "$SOURCE_STAGE_HELPER" \
        --source-root "$SOURCE_SNAPSHOT_INPUT_ROOT" \
        --manifest-path "$SOURCE_SNAPSHOT_INPUT_ROOT/SNAPSHOT_MANIFEST.json" \
        --destination-root "$SOURCE_SNAPSHOT_ROOT"
    [[ -d "$SOURCE_SNAPSHOT_ROOT" && ! -L "$SOURCE_SNAPSHOT_ROOT" ]] ||
        die "worker-local source snapshot publication failed"
    [[ "$(stat -c %a -- "$SOURCE_SNAPSHOT_ROOT")" == 555 ]] ||
        die "worker-local source snapshot root must be mode 0555"
    printf 'S1_MATCHED_LOCAL_SOURCE_READY source=%s destination=%s\n' \
        "$SOURCE_SNAPSHOT_INPUT_ROOT" "$SOURCE_SNAPSHOT_ROOT"
}

persist_formal_output() {
    local formal_exit_code=$?
    local persistence_exit_code
    trap - EXIT
    set +e
    python3 "$PERSIST_OUTPUT_HELPER" \
        --local-output-parent "$DLC_LOCAL_FORMAL_PARENT" \
        --local-output-root "$EVAL_LOCAL_OUTPUT_ROOT" \
        --persistent-output-parent "$OSS_PROJECT_ROOT/runs" \
        --persistent-output-root "$EVAL_PERSISTENT_OUTPUT_ROOT" \
        --experiment-id "$EVAL_EXPERIMENT_ID" \
        --run-id "$EVAL_LOGICAL_RUN_ID" \
        --control-commit "$EVAL_CONTROL_COMMIT" \
        --launcher-exit-code "$formal_exit_code" \
        --job-id "${DLC_JOB_ID:-${PAI_JOB_ID:-unknown-at-runtime}}"
    persistence_exit_code=$?
    if [[ "$persistence_exit_code" -ne 0 ]]; then
        printf 'ERROR S1_MATCHED_PERSISTENCE_FAILED formal_exit_code=%s persistence_exit_code=%s\n' \
            "$formal_exit_code" "$persistence_exit_code" >&2
        exit 90
    fi
    printf 'S1_MATCHED_PERSISTENCE_COMPLETE formal_exit_code=%s root=%s\n' \
        "$formal_exit_code" "$EVAL_PERSISTENT_OUTPUT_ROOT"
    exit "$formal_exit_code"
}

for control_file in \
    "$FORMAL_RUNNER_OVERRIDE" \
    "$FORMAL_LAUNCHER" \
    "$EGL_MATRIX_OVERRIDE" \
    "$S1_R14_MATCHED_HELPER" \
    "$COSMOS_ENV_ARCHIVE_METADATA_HELPER" \
    "$CHECKPOINT_PUBLISHER" \
    "$CHECKPOINT_CONTRACT" \
    "$PERSIST_OUTPUT_HELPER" \
    "$SOURCE_STAGE_HELPER" \
    "$R14_BASELINE_ARGS" \
    "$R14_BASELINE_RESULTS"; do
    require_control_file "$control_file"
done

[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" == 8 ]] ||
    die "matched evaluation requires exactly eight GPUs"
prepare_nvidia_egl_runtime
[[ "$(findmnt -n -T /mnt/workspace -o TARGET)" == /mnt/workspace ]] ||
    die "CPFS is not mounted at /mnt/workspace"
[[ "$(findmnt -n -T /oss-chengjuntao -o TARGET)" == /oss-chengjuntao ]] ||
    die "OSS is not mounted at /oss-chengjuntao"
[[ -L "$PROJECT_ROOT/runs" ]] ||
    die "persistent runs root must be the controlled OSS symlink"
[[ "$(realpath -e -- "$PROJECT_ROOT/runs")" == "$OSS_PROJECT_ROOT/runs" ]] ||
    die "persistent runs root does not resolve to the fixed OSS namespace"
[[ "$(findmnt -n -T "$OSS_PROJECT_ROOT/runs" -o TARGET)" == /oss-chengjuntao ]] ||
    die "persistent runs root is not on the /oss-chengjuntao mount"
[[ -w "$OSS_PROJECT_ROOT/runs" ]] ||
    die "persistent runs root is not writable"
[[ ! -e "$EVAL_PERSISTENT_OUTPUT_ROOT" && ! -L "$EVAL_PERSISTENT_OUTPUT_ROOT" ]] ||
    die "refusing to replace persistent run root: $EVAL_PERSISTENT_OUTPUT_ROOT"
mkdir --parents --mode=0700 -- "$DLC_LOCAL_FORMAL_PARENT"
[[ ! -e "$DLC_LOCAL_FORMAL_PARENT/$EVAL_EXPERIMENT_ID" && \
    ! -L "$DLC_LOCAL_FORMAL_PARENT/$EVAL_EXPERIMENT_ID" ]] ||
    die "refusing to reuse local experiment output root"
trap persist_formal_output EXIT
stage_source_snapshot
preflight_mujoco_egl_runtime
available_local_bytes="$(LC_ALL=C df -B1 --output=avail /tmp | awk 'NR == 2 { print $1 }')"
[[ "$available_local_bytes" =~ ^[0-9]+$ && "$available_local_bytes" -ge "$REQUIRED_LOCAL_BYTES" ]] ||
    die "worker-local /tmp has less than 150 GiB available"
[[ ! -e "$LOCAL_ASSET_ROOT" && ! -L "$LOCAL_ASSET_ROOT" ]] ||
    die "refusing to reuse local asset root: $LOCAL_ASSET_ROOT"
[[ ! -e "$LOCAL_BUILD_ROOT" && ! -L "$LOCAL_BUILD_ROOT" ]] ||
    die "refusing to reuse local asset build root: $LOCAL_BUILD_ROOT"

mkdir --mode=0755 -- "$LOCAL_BUILD_ROOT"
mkdir --parents --mode=0755 -- \
    "$LOCAL_BUILD_ROOT/control" \
    "$LOCAL_BUILD_ROOT/checkpoints" \
    "$LOCAL_BUILD_ROOT/envs" \
    "$LOCAL_BUILD_ROOT/models/Wan2.2-TI2V-5B"

copy_regular_file \
    "$SOURCE_ENV_ARCHIVE" \
    "$LOCAL_BUILD_ROOT/envs/cosmos-framework-09f23119-cu128-train-r22-complete.tar" \
    14736947200
copy_regular_file \
    "$SOURCE_ENV_RECEIPT" \
    "$LOCAL_BUILD_ROOT/control/cosmos-framework-09f23119-cu128-train-r22-complete-archive-publication.json" \
    2124
copy_regular_file \
    "$SOURCE_RUNTIME_RECEIPT" \
    "$LOCAL_BUILD_ROOT/control/cosmos-r21-dlc-runtime-v1-publication-v2.json" \
    992
copy_regular_file \
    "$SOURCE_WAN_VAE" \
    "$LOCAL_BUILD_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
    2818839170
copy_qwen_tokenizer \
    "$SOURCE_QWEN_TOKENIZER" \
    "$LOCAL_BUILD_ROOT/models/Qwen3-VL-8B-Instruct-tokenizer"
copy_checkpoint_closure \
    "$SOURCE_CHECKPOINT_ROOT" \
    "$LOCAL_BUILD_ROOT/checkpoints/.S1-R3-iter_000010000-serving-ema-bf16-v1.incomplete-transfer-20260803"

find "$LOCAL_BUILD_ROOT" -type d -exec chmod 0755 {} +
sync -f -- "$LOCAL_BUILD_ROOT"
mv -T -- "$LOCAL_BUILD_ROOT" "$LOCAL_ASSET_ROOT"
[[ -d "$LOCAL_ASSET_ROOT" && ! -L "$LOCAL_ASSET_ROOT" ]] ||
    die "worker-local asset publication failed"
printf 'S1_MATCHED_LOCAL_ASSET_ROOT_READY root=%s\n' "$LOCAL_ASSET_ROOT"

checkpoint_contract_sha256="$(sha256sum -- "$CHECKPOINT_CONTRACT" | awk '{print $1}')"
python3 "$CHECKPOINT_PUBLISHER" \
    --stage-root "$LOCAL_CHECKPOINT_STAGE" \
    --final-root "$LOCAL_CHECKPOINT_ROOT" \
    --receipt-output "$LOCAL_CHECKPOINT_RECEIPT" \
    --lock-path "$LOCAL_CHECKPOINT_LOCK" \
    --contract "$CHECKPOINT_CONTRACT" \
    --expected-contract-sha256 "$checkpoint_contract_sha256" \
    --allowed-checkpoint-parent "$LOCAL_CHECKPOINT_PARENT" \
    --publish
readonly EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256="$(
    sha256sum -- "$LOCAL_CHECKPOINT_RECEIPT" | awk '{print $1}'
)"

mkdir --parents --mode=0755 -- /tmp/cwam_cosmos3_robocasa365_eval

export S1_R14_MATCHED_MODE=1
export FORMAL_MATCHED_LOCAL_ASSET_ROOT_OVERRIDE="$LOCAL_ASSET_ROOT"
export FORMAL_CHECKPOINTS_ROOT_OVERRIDE="$LOCAL_CHECKPOINT_PARENT"
export FORMAL_RUNNER_OVERRIDE
export EGL_MATRIX_OVERRIDE
export S1_R14_MATCHED_HELPER
export COSMOS_ENV_ARCHIVE_METADATA_HELPER
export EVAL_CONTROL_COMMIT
export R14_BASELINE_ARGS
export R14_BASELINE_RESULTS
export R14_BASELINE_RUN_ID

export COSMOS_SOURCE="$SOURCE_SNAPSHOT_ROOT"
export SOURCE_MANIFEST="$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
export EXPECTED_SOURCE_COMMIT=dddd8fd3f61a336bfe359168eab2520048db9adb
export EXPECTED_SOURCE_MANIFEST_SHA256=c8112fc6da2256ccd26eb7fbd40d126214e72f626f9456e9c6ae4ef598213067
export EXPECTED_COSMOS_ENV_SOURCE_COMMIT=09f23119ea92c707207bba55565e7a09d16896a2

export COSMOS_ENV=/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs/envs/cosmos-framework-09f23119-cu128-train
export COSMOS_ENV_ARCHIVE="$LOCAL_ENV_ARCHIVE"
export COSMOS_ENV_ARCHIVE_PUBLICATION_RECEIPT="$LOCAL_ENV_RECEIPT"
export COSMOS_ENV_READY_MARKER="$COSMOS_ENV/.cwam_env_ready.json"
export EXPECTED_COSMOS_ENV_READY_MARKER_SHA256=30d569ca042cefcddb8f4729d9edbcf3ae7989d747c61c7d18aba6aee98c367d

export ROBOCASA_ENV="$ROBOCASA_ENV_ROOT"
export ROBOCASA_ENV_READY_MARKER="$ROBOCASA_ENV/.cwam_env_ready.json"
export EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256=d9a60a1b268d2feb2f1f07ae4e6b9cb2cd2e5605ba30315b66e31f31b66adae2
export ROBOCASA_TORCHCODEC_RELAY_RECEIPT="$INPUTS_ROOT/config/robocasa-torchcodec-plain-wheel-relay-receipt.json"
export EXPECTED_ROBOCASA_TORCHCODEC_RELAY_RECEIPT_SHA256=1b16cc5419b7eb8558d5c0bd6dc4099493df629d50756ad3116b3c1da47f5ce4

export DEMIAN_REPO="$INPUTS_ROOT/demian-2600ece5bf0e"
export ROBOCASA_REPO=/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs/repos/robocasa-29f7ce8814c1
export ROBOCASA_ASSET_ROOT="$ROBOCASA_REPO"
export ROBOSUITE_REPO="$INPUTS_ROOT/robosuite-232ce7d4a6ed"
export EXPECTED_DEMIAN_COMMIT=2600ece5bf0eeacf010a95f084dcda2de6d30d59
export EXPECTED_ROBOCASA_COMMIT=29f7ce8814c1547f5af762a0997fbd4b64848dd7
export EXPECTED_ROBOSUITE_COMMIT=232ce7d4a6ed89c949a9aba024a05c8c32fdd08b

export CHECKPOINT_ARTIFACT_ROOT="$LOCAL_CHECKPOINT_ROOT"
export SERVING_CHECKPOINT_PATH="$LOCAL_CHECKPOINT_ROOT/model"
export H_SERVING_CHECKPOINT_MANIFEST="$LOCAL_CHECKPOINT_ROOT/h-serving-checkpoint-manifest.json"
export EMA_EXPORT_RECEIPT="$LOCAL_CHECKPOINT_ROOT/ema-bf16-export-receipt.json"
export EMA_KEY_MANIFEST="$LOCAL_CHECKPOINT_ROOT/ema-bf16-key-manifest.json"
export EMA_VALIDATION="$LOCAL_CHECKPOINT_ROOT/ema-bf16-validation.json"
export EMA_SOURCE_DCP_FULL_RECEIPT="$LOCAL_CHECKPOINT_ROOT/source-dcp-full-verification.json"
export EMA_TRANSFER_MANIFEST="$LOCAL_CHECKPOINT_ROOT/ema-serving-dcp-transfer-manifest.json"
export CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT="$LOCAL_CHECKPOINT_RECEIPT"
export EXPECTED_H_SERVING_CHECKPOINT_MANIFEST_SHA256=60b841044ad6d0d8940411f1c9964f4d687b18288e060216695ee6444229ad59
export EXPECTED_SERVING_CHECKPOINT_TREE_SHA256=87e2fd424ce82c309d3fe06eaab268f905586cbf41dbcc31b8dc8fd690770b9f
export EXPECTED_EMA_EXPORT_RECEIPT_SHA256=f548369f09576b139223e1bbd09c49e62d56e5932db996cff20daebe2b63dd11
export EXPECTED_EMA_KEY_MANIFEST_SHA256=05f2ce5e73e92d49743685ec7c3f5abf753e5913f1a3edb6ca2a5bda86f769c0
export EXPECTED_EMA_VALIDATION_SHA256=1ae32013dd10b20f4ed7ce8ebd43856a45807eb93dc3ea1d6c2381277f15ad20
export EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256=70699cec3ff5d27f25710834408f1d5135140a1cc9319a167612163ab4f4e160
export EXPECTED_EMA_TRANSFER_MANIFEST_SHA256=a6a7c36959932ab3e859df08f986ab483e3469af41d4aec57067aa648b669ae8
export EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256
export EXPECTED_EMA_KEY_COUNT=822
export EXPECTED_EMA_CHUNK_COUNT=6570

export TRAINING_CONFIG="$LOCAL_CHECKPOINT_ROOT/config.yaml"
export ACTION_STATS="$INPUTS_ROOT/config/robocasa365_native_frame_wise_relative_rot6d.json"
export WAN2PT2_VAE="$LOCAL_WAN_VAE"
export QWEN_TOKENIZER="$LOCAL_QWEN_TOKENIZER"
export QWEN_TOKENIZER_MANIFEST="$INPUTS_ROOT/config/Qwen3-VL-8B-Instruct-tokenizer-manifest.json"
export ASSET_ENV_CLOSURE_MANIFEST="$INPUTS_ROOT/config/asset-env-closure-manifest.json"
export ASSET_ENV_CLOSURE_H_VERIFICATION="$INPUTS_ROOT/config/asset-env-closure-h-verification.json"
export EXPECTED_TRAINING_CONFIG_SHA256=f5da79c66713923dee2bf8cb8c254e93666cc1dac575c5fe28311d5ecd5951db
export EXPECTED_ACTION_STATS_SHA256=2b48eb8a42e0c9c7befc45034d152282dad328f36389c10cafdc5be2635b67df
export EXPECTED_WAN2PT2_VAE_SHA256=20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
export EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256=c71787b714a6876114b69d249e88bfa50de1262d33f46f8285dd00da729d4978
export EXPECTED_ASSET_ENV_CLOSURE_MANIFEST_SHA256=08b73f93a2ef191e2d82783e0a35689d1416a6d05e0384145e78551b3103f5bb
export EXPECTED_ASSET_ENV_CLOSURE_H_VERIFICATION_SHA256=60263e2de95efeeab89bc04b0b176331440a46ef8f027c8d708f8a9ecf739d45

export RUNTIME_CLOSURE_MANIFEST="$INPUTS_ROOT/config/cosmos-r21-dlc-runtime-v1-manifest.json"
export RUNTIME_CLOSURE_PUBLICATION_RECEIPT="$LOCAL_RUNTIME_RECEIPT"
export EXPECTED_RUNTIME_CLOSURE_MANIFEST_SHA256=162608b6a857123f88b0471b993f6c8dfcd09d439bbb9e68b6ad699b8b3850be
export EXPECTED_RUNTIME_CLOSURE_TREE_SHA256=bdd0f5b2b92f96f137086d22ad4c25d8c72d3c04cfa529b7f0170e394c2e4612
export EXPECTED_RUNTIME_CLOSURE_PUBLICATION_RECEIPT_SHA256=b2212009c6dde171f5c99f75401eab25b0e8d622fab8a5920950b1fa50ba5cc8

export DLC_CPFS_MOUNT=/mnt/workspace
export DLC_CPFS_OUTPUT_PARENT="$PROJECT_ROOT/runs"
export DLC_LOCAL_SCRATCH_PARENT="$DLC_LOCAL_SCRATCH_ROOT"
export DLC_LOCAL_OUTPUT_PARENT="$DLC_LOCAL_FORMAL_PARENT"
export EVAL_EXPERIMENT_ID
export EVAL_LOGICAL_RUN_ID
export EVAL_RUN_NAME="$EVAL_LOGICAL_RUN_ID"
export EVAL_OUTPUT_ROOT="$EVAL_LOCAL_OUTPUT_ROOT"

printf 'S1_MATCHED_DLC_BOOTSTRAP_READY experiment_id=%s logical_run_id=%s control_commit=%s\n' \
    "$EVAL_EXPERIMENT_ID" "$EVAL_LOGICAL_RUN_ID" "$EVAL_CONTROL_COMMIT"
bash "$FORMAL_LAUNCHER"
