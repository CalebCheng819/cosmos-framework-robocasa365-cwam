#!/usr/bin/env bash
# Run the fixed-R13 horizon and C-intervention matrix on one 8-H200 worker.

set -Eeuo pipefail
umask 077

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${RUN_ID:=CWAM-COSMOS3-C1-DIAG-20260818-R1}"
: "${EXPECTED_HOST:=gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"
: "${OSS_ROOT:=eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1-DIAG/$RUN_ID}"
: "${EPISODES_PER_TASK:=20}"

readonly COSMOS_ENV="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train"
readonly ROBOCASA_ENV="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
readonly DEMIAN_REPO="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/code/demian-2600ece5bf0e"
readonly ROBOCASA_REPO="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/code/robocasa-29f7ce8814c1"
readonly ROBOSUITE_REPO="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/code/robosuite-232ce7d4a6ed"
readonly CHECKPOINT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/staging/cwam-serving-checkpoints/C1-R13-iter_000020000-serving-ema-bf16-metadata-v1"
readonly ACTION_STATS="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857/stats/robocasa365_native_frame_wise_relative_rot6d.json"
readonly WAN_VAE="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly EGL_VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly ROOT="/nvme/cwam-c1-diagnostics/$RUN_ID"
readonly CONTROL="$ROOT/control"
readonly RESULTS="$ROOT/conditions"
readonly TRACE_ROOT="$ROOT/c-traces/normal_h12"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

upload_tree() {
    local source=$1
    local remote=$2
    rclone copy --transfers 8 --checkers 16 --stats 30s "$source" "$remote"
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    jobs -pr | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    if [[ -d "$ROOT" ]]; then
        upload_tree "$ROOT" "$OSS_ROOT" || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

[[ "$(hostname -f)" == "$EXPECTED_HOST" ]] || die "expected $EXPECTED_HOST, got $(hostname -f)"
[[ -x "$COSMOS_PY" ]] || die "missing Cosmos Python: $COSMOS_PY"
[[ -x "$ROBOCASA_PY" ]] || die "missing RoboCasa Python: $ROBOCASA_PY"
[[ -d "$CHECKPOINT_ROOT/model" && -f "$CHECKPOINT_ROOT/config.yaml" ]] || die "R13 checkpoint incomplete"
[[ -f "$ACTION_STATS" && -f "$WAN_VAE" ]] || die "evaluation assets are incomplete"
[[ -f "$EGL_VENDOR_JSON" ]] || die "NVIDIA EGL vendor file is missing"
[[ "$(findmnt -n -T /nvme -o TARGET)" == /nvme ]] || die "/nvme is not mounted"
[[ ! -e "$ROOT" ]] || die "refusing to reuse $ROOT"
command -v rclone >/dev/null || die "rclone is unavailable"

mkdir -p "$CONTROL" "$RESULTS" "$TRACE_ROOT"
"$COSMOS_PY" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected eight visible CUDA GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if any("H200" not in name for name in names):
    raise SystemExit(f"Expected only H200 GPUs, got {names}")
print(f"DIAGNOSTIC_CUDA_READY names={names}", flush=True)
PY

(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
        sleep 30
    done
) >"$CONTROL/gpu-monitor.log" 2>&1 &
monitor_pid=$!

declare -a server_pids=()
declare -a evaluator_pids=()

wait_for_server() {
    local port=$1
    local nonce=$2
    local horizon=$3
    local intervention=$4
    local pid=$5
    PYTHONPATH="$DEMIAN_REPO/openpi/packages/openpi-client/src" "$ROBOCASA_PY" - "$port" "$nonce" "$horizon" "$intervention" "$pid" <<'PY'
import os
import sys
import time

from openpi_client.websocket_client_policy import WebsocketClientPolicy

port, nonce, horizon, intervention, pid = sys.argv[1:]
deadline = time.monotonic() + 1200
last = None
while time.monotonic() < deadline:
    if not os.path.exists(f"/proc/{pid}"):
        raise SystemExit(f"policy server exited before readiness: pid={pid}")
    try:
        metadata = WebsocketClientPolicy("127.0.0.1", int(port)).get_server_metadata()
        if metadata.get("launch_nonce") != nonce:
            raise RuntimeError("launch nonce mismatch")
        if metadata.get("open_loop_horizon") != int(horizon):
            raise RuntimeError(f"horizon mismatch: {metadata.get('open_loop_horizon')}")
        if metadata.get("c_intervention") != intervention:
            raise RuntimeError(f"intervention mismatch: {metadata.get('c_intervention')}")
        if metadata.get("control_protocol", {}).get("reset") != "nonce_bound_websocket_control_v1":
            raise RuntimeError("server reset protocol is absent")
        print(f"SERVER_READY port={port} horizon={horizon} intervention={intervention}", flush=True)
        raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(5)
raise SystemExit(f"server readiness timed out: {last!r}")
PY
}

run_condition() {
    local gpu=$1
    local condition=$2
    local horizon=$3
    local intervention=$4
    local trace_root=$5
    local port=$((18100 + gpu))
    local condition_root="$RESULTS/$condition"
    local server_output="$condition_root/server-output"
    local run_name="$RUN_ID-$condition"
    local nonce
    local -a intervention_args=(
        --c-intervention "$intervention"
        --c-intervention-selected-slots 8
    )
    if [[ "$intervention" == normal || "$intervention" == shuffle || "$intervention" == cross_episode ]]; then
        intervention_args+=(--c-intervention-trace-root "$trace_root")
    fi
    nonce=$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(24))')
    mkdir -p "$condition_root/results" "$server_output" "$condition_root/torchinductor"

    (
        cd "$COSMOS_SOURCE"
        exec env \
            -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$condition_root/torchinductor" \
            "$COSMOS_PY" -m cosmos_framework.scripts.action_policy_server_robocasa365 \
                --checkpoint-path "$CHECKPOINT_ROOT/model" \
                --config-file "$CHECKPOINT_ROOT/config.yaml" \
                --output-dir "$server_output" \
                --action-stats-path "$ACTION_STATS" \
                --wan2pt2-vae-path "$WAN_VAE" \
                --expected-ema-key-count 1119 \
                --host 127.0.0.1 \
                --port "$port" \
                --sampler unipc \
                --seed 42 \
                --guidance 1.0 \
                --num-steps 30 \
                --launch-nonce "$nonce" \
                --open-loop-horizon "$horizon" \
                "${intervention_args[@]}"
    ) >"$condition_root/server.log" 2>&1 &
    local server_pid=$!
    server_pids+=("$server_pid")
    wait_for_server "$port" "$nonce" "$horizon" "$intervention" "$server_pid" \
        >"$condition_root/server-ready.log" 2>&1

    env \
        -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
        CUDA_VISIBLE_DEVICES="$gpu" \
        MUJOCO_EGL_DEVICE_ID="$gpu" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        PYTHONPATH="$EVAL_PYTHONPATH" \
        CWAM_DIAGNOSTIC_CONDITION="$condition" \
        CWAM_DIAGNOSTIC_HORIZON="$horizon" \
        "$ROBOCASA_PY" "$COSMOS_SOURCE/examples/h_cluster/run_cwam_c1_stage_eval.py" \
            --tasks CloseBlenderLid,PickPlaceCounterToStove \
            --episodes "$EPISODES_PER_TASK" \
            --episodes-start 0 \
            --max-steps 400 \
            --open-loop-horizon "$horizon" \
            --host 127.0.0.1 \
            --port "$port" \
            --checkpoint "$CHECKPOINT_ROOT/model" \
            --results-dir "$condition_root/results" \
            --run-name "$run_name" \
            --no-save-video \
            --headless \
            --camera-height 256 \
            --camera-width 256 \
            --seed 42 >"$condition_root/evaluator.log" 2>&1

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    upload_tree "$condition_root" "$OSS_ROOT/conditions/$condition"
}

phase_a=(
    "0 normal_h1 1 normal $ROOT/c-traces/normal_h1"
    "1 normal_h4 4 normal $ROOT/c-traces/normal_h4"
    "2 normal_h12 12 normal $TRACE_ROOT"
    "3 zero_h12 12 zero $ROOT/c-traces/unused-zero"
    "4 selected_h12 12 selected $ROOT/c-traces/unused-selected"
    "5 complement_h12 12 complement $ROOT/c-traces/unused-complement"
)
for spec in "${phase_a[@]}"; do
    read -r gpu condition horizon intervention trace_root <<<"$spec"
    run_condition "$gpu" "$condition" "$horizon" "$intervention" "$trace_root" &
    evaluator_pids+=("$!")
done
for pid in "${evaluator_pids[@]}"; do
    wait "$pid"
done
evaluator_pids=()

[[ -n "$(find "$TRACE_ROOT" -name 'replan_*.pt' -print -quit)" ]] || die "normal_h12 produced no C traces"

phase_b=(
    "6 shuffle_h12 12 shuffle $TRACE_ROOT"
    "7 cross_episode_h12 12 cross_episode $TRACE_ROOT"
)
for spec in "${phase_b[@]}"; do
    read -r gpu condition horizon intervention trace_root <<<"$spec"
    run_condition "$gpu" "$condition" "$horizon" "$intervention" "$trace_root" &
    evaluator_pids+=("$!")
done
for pid in "${evaluator_pids[@]}"; do
    wait "$pid"
done
evaluator_pids=()

PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" \
    "$COSMOS_SOURCE/examples/h_cluster/summarize_cwam_c1_diagnostics.py" \
    --results-root "$RESULTS" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --output-json "$CONTROL/diagnostic-summary.json" \
    --output-csv "$CONTROL/diagnostic-stage-rates.csv" \
    >"$CONTROL/summary.log"

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
upload_tree "$ROOT" "$OSS_ROOT"
touch "$CONTROL/DIAGNOSTICS_COMPLETE"
upload_tree "$CONTROL" "$OSS_ROOT/control"
echo "CWAM_C1_DIAGNOSTICS_COMPLETE run_id=$RUN_ID oss_root=$OSS_ROOT"
