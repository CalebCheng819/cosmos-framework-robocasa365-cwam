#!/usr/bin/env bash
# Run the fixed 17-task x 20-episode B4 RoboCasa365 evaluation on one 8x4090 worker.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

required=(
    COSMOS_SOURCE COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO ROBOCASA_REPO
    ROBOSUITE_REPO CHECKPOINT_ARTIFACT_ROOT SERVING_CHECKPOINT_PATH
    TRAINING_CONFIG ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER
    EVAL_OUTPUT_ROOT EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME
    EVAL_EGL_VENDOR_JSON B4_EVAL_HELPER B4_SHELL_UTILS SHARD_GROUP_RUNNER
    EVAL_WORKER_SCRATCH
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done
[[ -f "$B4_SHELL_UTILS" && ! -L "$B4_SHELL_UTILS" ]] || {
    printf 'ERROR: missing B4 shell utilities\n' >&2
    exit 2
}
# shellcheck source=examples/aliyun_4090/cwam_b4_eval_shell_utils.sh
source "$B4_SHELL_UTILS"

readonly SOURCE_COMMIT=e7775ffcea0a2d577597fbeb05c8bb92905aba0f
readonly DEMIAN_COMMIT=2600ece5bf0eeacf010a95f084dcda2de6d30d59
readonly ROBOCASA_COMMIT=29f7ce8814c1547f5af762a0997fbd4b64848dd7
readonly ROBOSUITE_COMMIT=232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
readonly SERVER_READY_SECONDS=1800
readonly EVALUATOR_TIMEOUT_SECONDS=129600
readonly SERVER_EXIT_SECONDS=180
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly RESULTS_DIR="$RESULTS_ROOT/$EVAL_RUN_NAME"
readonly GATE_ROOT="$EVAL_OUTPUT_ROOT/one-episode-gates"
readonly SCRATCH_DIR="$EVAL_WORKER_SCRATCH"
readonly HELPER="$B4_EVAL_HELPER"
readonly SERVER_MODULE=cosmos_framework.scripts.action_policy_server_robocasa365
readonly EVALUATOR="$DEMIAN_REPO/demian_wam/eval_utils/run_robocasa_eval.py"
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly TORCHRUN="$COSMOS_ENV/bin/torchrun"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"

readonly -a SERVER_LABELS=(server-a server-b)
readonly -a SERVER_DEVICES=(0,1,2,3 4,5,6,7)
readonly -a SERVER_PORTS=(16000 16001)
readonly -a SERVER_GATE_GPUS=(0 4)
declare -a SERVER_PIDS=("" "")
declare -a SERVER_PGIDS=("" "")
declare -a SERVER_NONCES=("" "")
declare -a SERVER_EXIT_CODES=("" "")
declare -a EVALUATOR_PIDS=("" "")
declare -a EVALUATOR_PGIDS=("" "")
TERMINAL_WRITTEN=0

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

git_revision() {
    git -c safe.directory="$1" -C "$1" rev-parse HEAD
}

write_status() {
    local status="$1"
    local detail="${2:-}"
    "$COSMOS_PY" "$HELPER" write-status \
        --experiment-id "$EVAL_EXPERIMENT_ID" \
        --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
        --status "$status" \
        --detail "$detail" \
        --output "$EVAL_OUTPUT_ROOT/status.json" >/dev/null
}

process_alive_non_zombie() {
    local pid="$1"
    local state
    kill -0 "$pid" 2>/dev/null || return 1
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$state" && "$state" != Z* ]]
}

stop_process_group() {
    local pid="$1"
    local pgid="$2"
    [[ -n "$pid" ]] || return 0
    if [[ -n "$pgid" ]] && process_alive_non_zombie "$pid"; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
        for _ in {1..30}; do
            process_alive_non_zombie "$pid" || break
            sleep 1
        done
        process_alive_non_zombie "$pid" && kill -KILL -- "-$pgid" 2>/dev/null || true
    else
        kill -TERM "$pid" 2>/dev/null || true
    fi
}

stop_evaluators() {
    local index
    for index in 0 1; do
        stop_process_group "${EVALUATOR_PIDS[$index]}" "${EVALUATOR_PGIDS[$index]}"
    done
}

stop_servers() {
    local index
    for index in 0 1; do
        stop_process_group "${SERVER_PIDS[$index]}" "${SERVER_PGIDS[$index]}"
    done
}

on_exit() {
    local status="$?"
    trap - EXIT INT TERM
    if ((status != 0)); then
        stop_evaluators
        stop_servers
        if [[ -x "$COSMOS_PY" && -f "$HELPER" && -d "$EVAL_OUTPUT_ROOT" ]]; then
            write_status FAILED "runner_exit_code=$status" || true
        fi
    elif ((TERMINAL_WRITTEN == 0)); then
        status=1
        write_status FAILED "runner exited without COMPLETE evidence" || true
    fi
    exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for directory in "$COSMOS_SOURCE" "$COSMOS_ENV" "$ROBOCASA_ENV" "$DEMIAN_REPO" \
    "$ROBOCASA_REPO" "$ROBOSUITE_REPO" "$CHECKPOINT_ARTIFACT_ROOT" \
    "$SERVING_CHECKPOINT_PATH" "$QWEN_TOKENIZER"; do
    [[ -d "$directory" && ! -L "$directory" ]] || die "missing directory: $directory"
done
for file in "$TRAINING_CONFIG" "$ACTION_STATS" "$WAN2PT2_VAE" \
    "$EVAL_EGL_VENDOR_JSON" "$HELPER" "$SHARD_GROUP_RUNNER" "$EVALUATOR"; do
    [[ -f "$file" && ! -L "$file" ]] || die "missing regular file: $file"
done
cwam_require_trusted_executable "$COSMOS_PY" "$COSMOS_ENV" ||
    die "untrusted Cosmos Python entrypoint"
cwam_require_trusted_executable "$TORCHRUN" "$COSMOS_ENV" ||
    die "untrusted torchrun entrypoint"
cwam_require_trusted_executable "$ROBOCASA_PY" "$(dirname -- "$ROBOCASA_ENV")" ||
    die "untrusted RoboCasa Python entrypoint"
[[ "$(git_revision "$COSMOS_SOURCE")" == "$SOURCE_COMMIT" ]] || die "Cosmos source revision mismatch"
[[ "$(git_revision "$DEMIAN_REPO")" == "$DEMIAN_COMMIT" ]] || die "DeMiAn revision mismatch"
[[ "$(git_revision "$ROBOCASA_REPO")" == "$ROBOCASA_COMMIT" ]] || die "RoboCasa revision mismatch"
[[ "$(git_revision "$ROBOSUITE_REPO")" == "$ROBOSUITE_COMMIT" ]] || die "robosuite revision mismatch"
[[ "$(findmnt -n -T "$EVAL_OUTPUT_ROOT" -o TARGET)" != / ]] || die "evaluation output is not on persistent storage"
[[ "$SCRATCH_DIR" == /tmp/* ]] || die "worker scratch must be under /tmp: $SCRATCH_DIR"
scratch_mount="$(findmnt -n -T /tmp -o TARGET)"
[[ "$scratch_mount" != /mnt/workspace && "$scratch_mount" != /oss-chengjuntao ]] || \
    die "worker scratch resolves to persistent shared storage: $scratch_mount"
[[ ! -e "$CONTROL_DIR" ]] || die "refusing to reuse evaluation output: $EVAL_OUTPUT_ROOT"
[[ ! -e "$SCRATCH_DIR" ]] || die "refusing to reuse worker scratch: $SCRATCH_DIR"
mkdir -p "$CONTROL_DIR" "$RESULTS_ROOT" "$GATE_ROOT" \
    "$SCRATCH_DIR/server-a" "$SCRATCH_DIR/server-b" \
    "$SCRATCH_DIR/torchinductor-a" "$SCRATCH_DIR/torchinductor-b"
write_status RUNNING "checkpoint_load_gate_pending"

"$COSMOS_PY" "$HELPER" verify-checkpoint \
    --root "$CHECKPOINT_ARTIFACT_ROOT" \
    --output "$CONTROL_DIR/checkpoint-verification.json" \
    >"$CONTROL_DIR/checkpoint-verification.log"

"$COSMOS_PY" - <<'PY'
import os
import torch

if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
devices = []
for index in range(8):
    props = torch.cuda.get_device_properties(index)
    devices.append((index, props.name, props.major, props.minor))
if any("4090" not in name or (major, minor) != (8, 9) for _, name, major, minor in devices):
    raise SystemExit(f"expected 8 RTX 4090 sm89 GPUs, got {devices}")
if len(os.sched_getaffinity(0)) != 126:
    raise SystemExit(f"expected 126 effective CPUs, got {len(os.sched_getaffinity(0))}")
print(devices)
PY

"$COSMOS_PY" - "${SERVER_PORTS[@]}" <<'PY'
import socket
import sys

sockets = []
try:
    for raw_port in sys.argv[1:]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", int(raw_port)))
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
PY

launch_server() {
    local index="$1"
    local label="${SERVER_LABELS[$index]}"
    local devices="${SERVER_DEVICES[$index]}"
    local port="${SERVER_PORTS[$index]}"
    local nonce
    local pid
    local pgid=
    nonce="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(24))')"
    SERVER_NONCES[$index]="$nonce"
    (
        cd "$COSMOS_SOURCE"
        exec setsid env \
            -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
            CUDA_VISIBLE_DEVICES="$devices" \
            PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$SCRATCH_DIR/torchinductor-${label#server-}" \
            "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
            -m "$SERVER_MODULE" \
            --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
            --config-file "$TRAINING_CONFIG" \
            --output-dir "$SCRATCH_DIR/$label" \
            --action-stats-path "$ACTION_STATS" \
            --wan2pt2-vae-path "$WAN2PT2_VAE" \
            --qwen-tokenizer-path "$QWEN_TOKENIZER" \
            --host 127.0.0.1 --port "$port" \
            --sampler unipc --seed 42 --guidance 1.0 --num-steps 30 \
            --expected-ema-key-count 986 --launch-nonce "$nonce"
    ) >"$CONTROL_DIR/policy-$label.log" 2>&1 &
    pid=$!
    SERVER_PIDS[$index]="$pid"
    for _ in {1..100}; do
        kill -0 "$pid" 2>/dev/null || die "$label exited before process-group setup"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$label did not enter an isolated process group"
    SERVER_PGIDS[$index]="$pgid"
}

capture_server_runtime() {
    local index="$1"
    local label="${SERVER_LABELS[$index]}"
    local port="${SERVER_PORTS[$index]}"
    "$COSMOS_PY" "$HELPER" capture-runtime \
        --uri "ws://127.0.0.1:$port" --output "$CONTROL_DIR/runtime-$label.json" \
        --timeout-seconds "$SERVER_READY_SECONDS" --launcher-pid "${SERVER_PIDS[$index]}" \
        --server-label "$label" --visible-devices "${SERVER_DEVICES[$index]}" \
        --launch-nonce "${SERVER_NONCES[$index]}" --checkpoint "$SERVING_CHECKPOINT_PATH" \
        --config "$TRAINING_CONFIG" --action-stats "$ACTION_STATS" \
        --vae "$WAN2PT2_VAE" --tokenizer "$QWEN_TOKENIZER" \
        --source-commit "$SOURCE_COMMIT" --demian-commit "$DEMIAN_COMMIT" \
        >"$CONTROL_DIR/runtime-$label.log"
}

launch_server 0
launch_server 1
capture_server_runtime 0
capture_server_runtime 1
write_status RUNNING "two_policy_servers_loaded_and_runtime_verified"

eval_env=(
    env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON"
    __GLX_VENDOR_LIBRARY_NAME=nvidia PYTHONPATH="$EVAL_PYTHONPATH"
)

run_gate() {
    local index="$1"
    local label="${SERVER_LABELS[$index]}"
    local port="${SERVER_PORTS[$index]}"
    local gpu="${SERVER_GATE_GPUS[$index]}"
    local gate_run="gate-close-blender-lid-$label"
    local gate_dir="$GATE_ROOT/$gate_run"
    kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null || die "$label exited before graphics gate"
    timeout --signal=TERM --kill-after=30 3600 \
        "${eval_env[@]}" CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
        "$ROBOCASA_PY" "$EVALUATOR" --tasks CloseBlenderLid \
        --task-shard 0 --num-shards 1 --episodes 1 \
        --results-dir "$GATE_ROOT" --run-name "$gate_run" \
        --episodes-start 0 --max-steps 400 --open-loop-horizon 12 \
        --host 127.0.0.1 --port "$port" --checkpoint "$SERVING_CHECKPOINT_PATH" \
        --no-save-video --headless --camera-height 256 --camera-width 256 \
        --seed 42 --split atomic >"$CONTROL_DIR/gate-$label.log" 2>&1
    "$COSMOS_PY" "$HELPER" verify-gate \
        --results-dir "$gate_dir" --checkpoint "$SERVING_CHECKPOINT_PATH" \
        --run-name "$gate_run" --server-label "$label" --port "$port" \
        --output "$CONTROL_DIR/gate-$label.json" \
        >"$CONTROL_DIR/gate-$label-receipt.log"
}

run_gate 0
run_gate 1
write_status RUNNING "two_graphics_gates_passed_formal_340_running"

export ROBOCASA_PY ROBOCASA_ENV EVALUATOR EVAL_PYTHONPATH EVAL_EGL_VENDOR_JSON B4_SHELL_UTILS
export SERVING_CHECKPOINT_PATH RESULTS_ROOT RESULTS_DIR EVAL_RUN_NAME
export EVALUATOR_TIMEOUT_SECONDS

launch_shard_group() {
    local index="$1"
    shift
    local label="${SERVER_LABELS[$index]}"
    local port="${SERVER_PORTS[$index]}"
    local group_control="$CONTROL_DIR/$label"
    local pid
    local pgid=
    mkdir -p "$group_control"
    (
        exec setsid env EVAL_GROUP_CONTROL_DIR="$group_control" \
            bash "$SHARD_GROUP_RUNNER" "$label" "$port" "${SERVER_PIDS[$index]}" "$@"
    ) >"$CONTROL_DIR/$label-shard-group.log" 2>&1 &
    pid=$!
    EVALUATOR_PIDS[$index]="$pid"
    for _ in {1..100}; do
        kill -0 "$pid" 2>/dev/null || die "$label evaluator exited before process-group setup"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$label evaluator did not enter an isolated process group"
    EVALUATOR_PGIDS[$index]="$pgid"
}

launch_shard_group 0 0 2 4 6
launch_shard_group 1 1 3 5 7

active_pids=("${EVALUATOR_PIDS[@]}")
while ((${#active_pids[@]})); do
    completed_pid=
    set +e
    wait -n -p completed_pid "${active_pids[@]}"
    group_exit_code=$?
    set -e
    [[ -n "$completed_pid" ]] || die "wait did not identify a completed shard group"
    remaining_pids=()
    for pid in "${active_pids[@]}"; do
        [[ "$pid" == "$completed_pid" ]] || remaining_pids+=("$pid")
    done
    active_pids=("${remaining_pids[@]}")
    if ((group_exit_code != 0)); then
        stop_evaluators
        die "shard group PID $completed_pid exited with code $group_exit_code"
    fi
done
EVALUATOR_PIDS=("" "")
EVALUATOR_PGIDS=("" "")

shutdown_server() {
    local index="$1"
    local label="${SERVER_LABELS[$index]}"
    local port="${SERVER_PORTS[$index]}"
    local deadline
    "$COSMOS_PY" "$HELPER" shutdown-server \
        --uri "ws://127.0.0.1:$port" --output "$CONTROL_DIR/shutdown-$label.json" \
        --timeout-seconds 60 --launch-nonce "${SERVER_NONCES[$index]}" \
        --runtime-receipt "$CONTROL_DIR/runtime-$label.json" \
        >"$CONTROL_DIR/shutdown-$label.log"
    deadline=$((SECONDS + SERVER_EXIT_SECONDS))
    while process_alive_non_zombie "${SERVER_PIDS[$index]}" && ((SECONDS < deadline)); do sleep 1; done
    process_alive_non_zombie "${SERVER_PIDS[$index]}" && die "$label did not exit after controlled shutdown"
    set +e
    wait "${SERVER_PIDS[$index]}"
    SERVER_EXIT_CODES[$index]=$?
    set -e
    SERVER_PIDS[$index]=
    SERVER_PGIDS[$index]=
}

shutdown_server 0
shutdown_server 1

"$COSMOS_PY" "$HELPER" summarize \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --source-commit "$SOURCE_COMMIT" --demian-commit "$DEMIAN_COMMIT" \
    --run-name "$EVAL_RUN_NAME" --results-dir "$RESULTS_DIR" \
    --checkpoint "$SERVING_CHECKPOINT_PATH" \
    --runtime-receipt "$CONTROL_DIR/runtime-server-a.json" \
    --runtime-receipt "$CONTROL_DIR/runtime-server-b.json" \
    --gate-receipt "$CONTROL_DIR/gate-server-a.json" \
    --gate-receipt "$CONTROL_DIR/gate-server-b.json" \
    --shutdown-receipt "$CONTROL_DIR/shutdown-server-a.json" \
    --shutdown-receipt "$CONTROL_DIR/shutdown-server-b.json" \
    --server-log "$CONTROL_DIR/policy-server-a.log" \
    --server-log "$CONTROL_DIR/policy-server-b.log" \
    --server-exit-code "${SERVER_EXIT_CODES[0]}" \
    --server-exit-code "${SERVER_EXIT_CODES[1]}" \
    --summary-csv "$EVAL_OUTPUT_ROOT/summary.csv" \
    --task-csv "$EVAL_OUTPUT_ROOT/task_success_rates.csv" \
    --evidence-json "$EVAL_OUTPUT_ROOT/evaluation_evidence.json" \
    --status-json "$EVAL_OUTPUT_ROOT/status.json" \
    >"$CONTROL_DIR/summarize.log"
TERMINAL_WRITTEN=1
printf 'B4_ROBOCASA365_EVAL_COMPLETE evidence=%s\n' "$EVAL_OUTPUT_ROOT/evaluation_evidence.json"
