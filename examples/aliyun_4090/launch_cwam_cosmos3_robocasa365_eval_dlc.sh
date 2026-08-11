#!/usr/bin/env bash
# Bootstrap host-bound EGL evidence, then run the exact formal DLC protocol.
# Workflow iteration: ITER-2026-08-03-E35S1-DLC-EVAL.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot}"
: "${ROBOCASA_ENV:?Set ROBOCASA_ENV to the pinned RoboCasa environment}"
: "${EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256:?Set the ready-marker SHA256}"
: "${EVAL_EXPERIMENT_ID:?Set EVAL_EXPERIMENT_ID}"

readonly GPU_COUNT=8
readonly PROJECT_ROOT=/mnt/workspace/cwam_cosmos3_robocasa365_eval
readonly EGL_CONTROL_ROOT="$PROJECT_ROOT/control/dlc-egl/$EVAL_EXPERIMENT_ID"
readonly EVIDENCE_ROOT="$EGL_CONTROL_ROOT/egl-mapping-evidence-$EVAL_EXPERIMENT_ID"
readonly EGL_MAPPING_RECEIPT="$EGL_CONTROL_ROOT/robocasa-egl-device-mapping.json"
readonly EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
readonly EGL_PROBE="$COSMOS_SOURCE/examples/aliyun_4090/probe_mujoco_egl_mapping.py"
readonly EGL_MATRIX="$COSMOS_SOURCE/examples/aliyun_4090/run_mujoco_egl_mapping_matrix_dlc.sh"
readonly EGL_RECEIPT_GENERATOR="$COSMOS_SOURCE/examples/aliyun_4090/generate_cwam_dlc_egl_mapping_receipt.py"
readonly S1_R14_MATCHED_MODE="${S1_R14_MATCHED_MODE:-0}"
case "$S1_R14_MATCHED_MODE" in
    0)
        [[ -z "${FORMAL_RUNNER_OVERRIDE:-}" ]] || {
            printf 'ERROR FORMAL_RUNNER_OVERRIDE is reserved for S1/R14 matched mode\n' >&2
            exit 2
        }
        FORMAL_RUNNER="$COSMOS_SOURCE/examples/aliyun_4090/run_cwam_cosmos3_robocasa365_eval_dlc.sh"
        ;;
    1)
        : "${FORMAL_RUNNER_OVERRIDE:?Set FORMAL_RUNNER_OVERRIDE in S1/R14 matched mode}"
        FORMAL_RUNNER="$FORMAL_RUNNER_OVERRIDE"
        ;;
    *)
        printf 'ERROR S1_R14_MATCHED_MODE must be 0 or 1\n' >&2
        exit 2
        ;;
esac
readonly FORMAL_RUNNER

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

require_immutable_source_file() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] ||
        die "immutable source file is missing or unsafe: $path"
    [[ "$(stat -c %a -- "$path")" == 444 ]] ||
        die "immutable source file must have mode 0444: $path"
}

[[ "$(realpath -e -- "$COSMOS_SOURCE")" == "$COSMOS_SOURCE" ]] ||
    die "COSMOS_SOURCE must be canonical"
[[ -x "$ROBOCASA_ENV/bin/python" ]] || die "RoboCasa Python is not executable"
require_immutable_source_file "$EGL_PROBE"
require_immutable_source_file "$EGL_MATRIX"
require_immutable_source_file "$EGL_RECEIPT_GENERATOR"
require_immutable_source_file "$FORMAL_RUNNER"
[[ "$(realpath -e -- "$FORMAL_RUNNER")" == "$FORMAL_RUNNER" ]] ||
    die "FORMAL_RUNNER must be canonical"
[[ -f "$EGL_VENDOR_JSON" && ! -L "$EGL_VENDOR_JSON" ]] ||
    die "NVIDIA EGL vendor JSON is missing"
[[ "$(stat -c %a -- "$EGL_VENDOR_JSON")" == 644 ]] ||
    die "NVIDIA EGL vendor JSON must have mode 0644"
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" == "$GPU_COUNT" ]] ||
    die "DLC worker must expose exactly eight GPUs"
[[ ! -e "$EGL_CONTROL_ROOT" && ! -L "$EGL_CONTROL_ROOT" ]] ||
    die "refusing to reuse EGL control root: $EGL_CONTROL_ROOT"

mkdir --parents --mode=0755 -- "$(dirname -- "$EGL_CONTROL_ROOT")"
mkdir --mode=0755 -- "$EGL_CONTROL_ROOT"

export EGL_CONTROL_ROOT
export EVIDENCE_ROOT
export EGL_PROBE
export EGL_VENDOR_JSON
export ROBOCASA_READY_MARKER_SHA256="$EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256"
export EXPECTED_EGL_PROBE_SHA256="$(sha256_file "$EGL_PROBE")"
export EXPECTED_EGL_VENDOR_JSON_SHA256="$(sha256_file "$EGL_VENDOR_JSON")"

bash "$EGL_MATRIX" | tee "$EGL_CONTROL_ROOT/egl-mapping-matrix.log"
python3 "$EGL_RECEIPT_GENERATOR" \
    --evidence-root "$EVIDENCE_ROOT" \
    --output "$EGL_MAPPING_RECEIPT" \
    --environment "$ROBOCASA_ENV" \
    --ready-marker-sha256 "$ROBOCASA_READY_MARKER_SHA256" \
    --probe "$EGL_PROBE" \
    --probe-sha256 "$EXPECTED_EGL_PROBE_SHA256" \
    --vendor "$EGL_VENDOR_JSON" \
    --vendor-sha256 "$EXPECTED_EGL_VENDOR_JSON_SHA256" \
    --gpu-count "$GPU_COUNT" | tee "$EGL_CONTROL_ROOT/egl-mapping-receipt.log"
chmod 0444 "$EGL_CONTROL_ROOT/egl-mapping-matrix.log" "$EGL_CONTROL_ROOT/egl-mapping-receipt.log"

export EVAL_EGL_VENDOR_JSON="$EGL_VENDOR_JSON"
export EGL_MAPPING_RECEIPT
export EXPECTED_EGL_MAPPING_RECEIPT_SHA256="$(sha256_file "$EGL_MAPPING_RECEIPT")"

echo "DLC_EGL_PREFLIGHT_PASSED receipt=$EGL_MAPPING_RECEIPT sha256=$EXPECTED_EGL_MAPPING_RECEIPT_SHA256"
exec bash "$FORMAL_RUNNER"
