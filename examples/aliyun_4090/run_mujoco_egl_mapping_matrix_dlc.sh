#!/usr/bin/env bash
# Generate per-GPU MuJoCo/EGL attribution evidence on one PAI DLC worker.
# Workflow iteration: ITER-2026-08-03-E35S1-DLC-EVAL.

set -Eeuo pipefail
umask 022

: "${EGL_CONTROL_ROOT:?Set EGL_CONTROL_ROOT to the unique DLC control directory}"
: "${ROBOCASA_ENV:?Set ROBOCASA_ENV to the pinned RoboCasa environment}"
: "${ROBOCASA_READY_MARKER_SHA256:?Set ROBOCASA_READY_MARKER_SHA256}"
: "${EGL_PROBE:?Set EGL_PROBE to the immutable MuJoCo probe script}"
: "${EXPECTED_EGL_PROBE_SHA256:?Set EXPECTED_EGL_PROBE_SHA256}"
: "${EGL_VENDOR_JSON:?Set EGL_VENDOR_JSON to the live NVIDIA vendor JSON}"
: "${EXPECTED_EGL_VENDOR_JSON_SHA256:?Set EXPECTED_EGL_VENDOR_JSON_SHA256}"
: "${EVIDENCE_ROOT:?Set EVIDENCE_ROOT to a unique control evidence directory}"

readonly GPU_COUNT=8
readonly ENVIRONMENT="$(realpath -e -- "$ROBOCASA_ENV")"
readonly MARKER="$ENVIRONMENT/.cwam_env_ready.json"
readonly PROBE="$(realpath -e -- "$EGL_PROBE")"
readonly EGL_VENDOR_JSON="$(realpath -e -- "$EGL_VENDOR_JSON")"
readonly GPU_INDICES="$(seq 0 $((GPU_COUNT - 1)))"
readonly ALL_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
readonly INVENTORY="$EVIDENCE_ROOT/gpu-inventory.csv"
readonly SUMMARY="$EVIDENCE_ROOT/mapping-summary.tsv"

EGL_CONTROL_ROOT="$(realpath -m -- "$EGL_CONTROL_ROOT")"
EVIDENCE_ROOT="$(realpath -m -- "$EVIDENCE_ROOT")"
[[ -d "$EGL_CONTROL_ROOT" && ! -L "$EGL_CONTROL_ROOT" ]]
[[ "$EVIDENCE_ROOT" == "$EGL_CONTROL_ROOT/egl-mapping-evidence-"* ]]
[[ ! -e "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" ]]
[[ -x "$ENVIRONMENT/bin/python" ]]
[[ -f "$MARKER" && "$(stat -c %a "$MARKER")" == 444 ]]
[[ "$(sha256sum "$MARKER" | awk '{print $1}')" == "$ROBOCASA_READY_MARKER_SHA256" ]]
[[ -x "$PROBE" && "$(stat -c %a "$PROBE")" == 555 ]]
[[ "$(sha256sum "$PROBE" | awk '{print $1}')" == "$EXPECTED_EGL_PROBE_SHA256" ]]
[[ -f "$EGL_VENDOR_JSON" && "$(stat -c %a "$EGL_VENDOR_JSON")" == 644 ]]
[[ "$(sha256sum "$EGL_VENDOR_JSON" | awk '{print $1}')" == "$EXPECTED_EGL_VENDOR_JSON_SHA256" ]]

mkdir -m 0755 "$EVIDENCE_ROOT"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader >"$INVENTORY"
[[ "$(wc -l <"$INVENTORY")" == "$GPU_COUNT" ]]
printf 'scheme\tphysical_index\tcuda_visible_devices\tmujoco_egl_device_id\tprobe_pid\trenderer_status\tobserved_physical_index\texpected_gpu_uuid\tobserved_gpu_uuid\tpmon_matched_pid\tmapping_matches_expected\n' >"$SUMMARY"

gpu_uuid() {
    local physical_index="$1"
    awk -F, -v wanted="$physical_index" '
        {
            index = $1
            uuid = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", index)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
            if (index == wanted) {
                print uuid
                found = 1
            }
        }
        END {
            if (!found) {
                exit 1
            }
        }
    ' "$INVENTORY"
}

run_one() {
    local scheme="$1"
    local physical_index="$2"
    local cuda_visible_devices="$3"
    local egl_device_id="$4"
    local prefix="$EVIDENCE_ROOT/${scheme}-gpu${physical_index}"
    local probe_log="${prefix}-probe.log"
    local pmon_log="${prefix}-nvidia-smi-pmon.log"
    local snapshot_log="${prefix}-nvidia-smi-snapshot.log"
    local compute_log="${prefix}-nvidia-smi-compute-apps.log"
    local pid
    local ready=0
    local renderer_status
    local observed_indexes
    local observed_physical_index
    local expected_uuid
    local observed_uuid
    local mapping_matches_expected

    for path in "$probe_log" "$pmon_log" "$snapshot_log" "$compute_log"; do
        [[ ! -e "$path" && ! -L "$path" ]]
    done
    env \
        -u LD_PRELOAD \
        -u LD_AUDIT \
        -u __EGL_VENDOR_LIBRARY_DIRS \
        -u LIBGL_ALWAYS_SOFTWARE \
        -u MESA_LOADER_DRIVER_OVERRIDE \
        -u LIBGL_DRIVERS_PATH \
        CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
        MUJOCO_EGL_DEVICE_ID="$egl_device_id" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONNOUSERSITE=1 \
        "$ENVIRONMENT/bin/python" "$PROBE" --hold-seconds 12 \
        >"$probe_log" 2>&1 &
    pid=$!

    for _ in $(seq 1 100); do
        if grep -q '"event": "egl_ready"' "$probe_log"; then
            ready=1
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ "$ready" != 1 ]]; then
        wait "$pid" || true
        chmod 0444 "$probe_log"
        echo "ERROR: EGL probe did not become ready: $scheme gpu=$physical_index" >&2
        return 1
    fi

    nvidia-smi pmon -s um -c 2 -d 1 >"$pmon_log"
    nvidia-smi >"$snapshot_log"
    nvidia-smi \
        --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
        --format=csv,noheader >"$compute_log" 2>&1 || true

    if wait "$pid"; then
        renderer_status=passed
    else
        renderer_status=failed
    fi
    [[ "$renderer_status" == passed ]]
    grep -q '"event": "egl_closed"' "$probe_log"

    observed_indexes="$(
        awk -v wanted_pid="$pid" '
            $1 ~ /^[0-9]+$/ && $2 == wanted_pid {
                print $1
            }
        ' "$pmon_log" | sort -nu
    )"
    [[ "$(printf '%s\n' "$observed_indexes" | sed '/^$/d' | wc -l)" == 1 ]]
    observed_physical_index="$observed_indexes"
    expected_uuid="$(gpu_uuid "$physical_index")"
    observed_uuid="$(gpu_uuid "$observed_physical_index")"
    if [[ "$observed_physical_index" == "$physical_index" &&
        "$observed_uuid" == "$expected_uuid" ]]; then
        mapping_matches_expected=true
    else
        mapping_matches_expected=false
    fi

    chmod 0444 "$probe_log" "$pmon_log" "$snapshot_log" "$compute_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\ttrue\t%s\n' \
        "$scheme" \
        "$physical_index" \
        "$cuda_visible_devices" \
        "$egl_device_id" \
        "$pid" \
        "$renderer_status" \
        "$observed_physical_index" \
        "$expected_uuid" \
        "$observed_uuid" \
        "$mapping_matches_expected" \
        >>"$SUMMARY"
}

for physical_index in $GPU_INDICES; do
    run_one \
        isolated_physical_identity \
        "$physical_index" \
        "$physical_index" \
        "$physical_index"
done
for physical_index in $GPU_INDICES; do
    run_one \
        visible_all_physical_index \
        "$physical_index" \
        "$ALL_VISIBLE_DEVICES" \
        "$physical_index"
done
for physical_index in $GPU_INDICES; do
    run_one \
        isolated_relative_zero \
        "$physical_index" \
        "$physical_index" \
        0
done

chmod 0444 "$INVENTORY" "$SUMMARY"
printf 'EVIDENCE_ROOT=%s\n' "$EVIDENCE_ROOT"
printf 'SUMMARY_SHA256=%s\n' "$(sha256sum "$SUMMARY" | awk '{print $1}')"
cat "$SUMMARY"
