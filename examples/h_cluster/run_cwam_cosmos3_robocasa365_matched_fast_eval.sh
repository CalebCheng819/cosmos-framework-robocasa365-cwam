#!/usr/bin/env bash
# Run two matched RoboCasa365 tasks with two four-rank policy servers on one H200 node.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

required=(
    EVAL_CONTROL_SOURCE COSMOS_SOURCE COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO
    ROBOCASA_REPO ROBOSUITE_REPO SERVING_CHECKPOINT_PATH TRAINING_CONFIG
    ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER EVAL_OUTPUT_ROOT
    EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME MODEL_LABEL
    MODEL_SOURCE_REVISION CONTROL_SOURCE_REVISION EXPECTED_EMA_KEY_COUNT
    EVAL_EGL_VENDOR_JSON EXPECTED_NODE_TAG RJOB_NAME
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done

readonly HELPER="$EVAL_CONTROL_SOURCE/examples/h_cluster/cwam_robocasa365_matched_fast_eval.py"
readonly SERVER_MODULE=cosmos_framework.scripts.action_policy_server_robocasa365
readonly EVALUATOR="$DEMIAN_REPO/demian_wam/eval_utils/run_robocasa_eval.py"
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly TORCHRUN="$COSMOS_ENV/bin/torchrun"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly SCRATCH_ROOT="/nvme/$EVAL_EXPERIMENT_ID"
readonly TASK_A=CloseBlenderLid
readonly TASK_B=PickPlaceCounterToStove
readonly RUN_A="$EVAL_RUN_NAME-close-blender-lid"
readonly RUN_B="$EVAL_RUN_NAME-pick-place-counter-to-stove"
readonly RESULT_A="$RESULTS_ROOT/$RUN_A/results_shard0.json"
readonly RESULT_B="$RESULTS_ROOT/$RUN_B/results_shard0.json"
readonly -a SERVER_LABELS=(server-a server-b)
readonly -a SERVER_DEVICES=(0,1,2,3 4,5,6,7)
readonly -a SERVER_PORTS=(16000 16001)
declare -a SERVER_PIDS=("" "") SERVER_PGIDS=("" "") SERVER_NONCES=("" "")
declare -a EVAL_PIDS=("" "") EVAL_PGIDS=("" "")
GPU_MONITOR_PID=
PROGRESS_PID=
TERMINAL_WRITTEN=0

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

process_alive() {
    local state
    kill -0 "$1" 2>/dev/null || return 1
    state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$state" && "$state" != Z* ]]
}

stop_group() {
    local pid="${1:-}" pgid="${2:-}"
    [[ -n "$pid" ]] || return 0
    if [[ -n "$pgid" ]] && process_alive "$pid"; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
        for _ in {1..20}; do process_alive "$pid" || return 0; sleep 1; done
        kill -KILL -- "-$pgid" 2>/dev/null || true
    else
        kill -TERM "$pid" 2>/dev/null || true
    fi
}

write_status() {
    "$COSMOS_PY" "$HELPER" write-status \
        --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
        --status "$1" --detail "${2:-}" --output "$EVAL_OUTPUT_ROOT/status.json" \
        --timeline "$CONTROL_DIR/status.ndjson" >/dev/null
}

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    [[ -z "$PROGRESS_PID" ]] || kill "$PROGRESS_PID" 2>/dev/null || true
    [[ -z "$GPU_MONITOR_PID" ]] || kill "$GPU_MONITOR_PID" 2>/dev/null || true
    stop_group "${EVAL_PIDS[0]}" "${EVAL_PGIDS[0]}"
    stop_group "${EVAL_PIDS[1]}" "${EVAL_PGIDS[1]}"
    stop_group "${SERVER_PIDS[0]}" "${SERVER_PGIDS[0]}"
    stop_group "${SERVER_PIDS[1]}" "${SERVER_PGIDS[1]}"
    if ((code != 0 || TERMINAL_WRITTEN == 0)); then
        [[ -d "$CONTROL_DIR" ]] && write_status FAILED "runner_exit_code=$code" || true
        ((code == 0)) && code=1
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for directory in "$EVAL_CONTROL_SOURCE" "$COSMOS_SOURCE" "$COSMOS_ENV" "$ROBOCASA_ENV" \
    "$DEMIAN_REPO" "$ROBOCASA_REPO" "$ROBOSUITE_REPO" "$SERVING_CHECKPOINT_PATH" "$QWEN_TOKENIZER"; do
    [[ -d "$directory" ]] || die "missing directory: $directory"
done
for file in "$HELPER" "$EVALUATOR" "$TRAINING_CONFIG" "$ACTION_STATS" "$WAN2PT2_VAE" "$EVAL_EGL_VENDOR_JSON"; do
    [[ -f "$file" ]] || die "missing file: $file"
done
[[ -x "$COSMOS_PY" && -x "$TORCHRUN" && -x "$ROBOCASA_PY" ]] || die "runtime executables are unavailable"
[[ "$(git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$MODEL_SOURCE_REVISION" ]] || die "model source revision mismatch"
[[ "$(git -c safe.directory="$EVAL_CONTROL_SOURCE" -C "$EVAL_CONTROL_SOURCE" rev-parse HEAD)" == "$CONTROL_SOURCE_REVISION" ]] || die "control source revision mismatch"
[[ "$EVAL_OUTPUT_ROOT" == /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-fast-eval/runs/* ]] || die "unexpected output root"
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || die "refusing to reuse output root"
[[ ! -e "$SCRATCH_ROOT" ]] || die "refusing to reuse scratch root"
mkdir -p "$CONTROL_DIR" "$RESULTS_ROOT" "$SCRATCH_ROOT/server-a" "$SCRATCH_ROOT/server-b" \
    "$SCRATCH_ROOT/torchinductor-a" "$SCRATCH_ROOT/torchinductor-b"
write_status STARTING "validating_eight_h200_worker"

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 8 ]] || die "expected 8 GPUs, got ${#gpu_names[@]}"
for name in "${gpu_names[@]}"; do [[ "$name" == *H200* ]] || die "non-H200 GPU visible: $name"; done
printf '%s\n' "${gpu_names[@]}" >"$CONTROL_DIR/gpu-inventory.txt"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 30 >"$CONTROL_DIR/gpu-monitor.csv" 2>"$CONTROL_DIR/gpu-monitor.err" &
GPU_MONITOR_PID=$!

"$COSMOS_PY" - 16000 16001 <<'PY'
import socket, sys
sockets = []
try:
    for raw in sys.argv[1:]:
        sock = socket.socket()
        sock.bind(("127.0.0.1", int(raw)))
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
PY

launch_server() {
    local index="$1" label="${SERVER_LABELS[$1]}" devices="${SERVER_DEVICES[$1]}" port="${SERVER_PORTS[$1]}"
    local pid pgid=
    SERVER_NONCES[$index]="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
    (
        cd "$COSMOS_SOURCE"
        exec setsid env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
            CUDA_VISIBLE_DEVICES="$devices" PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$SCRATCH_ROOT/torchinductor-${label#server-}" \
            "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
            -m "$SERVER_MODULE" --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
            --config-file "$TRAINING_CONFIG" --output-dir "$SCRATCH_ROOT/$label" \
            --action-stats-path "$ACTION_STATS" --wan2pt2-vae-path "$WAN2PT2_VAE" \
            --qwen-tokenizer-path "$QWEN_TOKENIZER" --host 127.0.0.1 --port "$port" \
            --sampler unipc --seed 42 --guidance 1.0 --num-steps 30 \
            --expected-ema-key-count "$EXPECTED_EMA_KEY_COUNT" --launch-nonce "${SERVER_NONCES[$index]}"
    ) >"$CONTROL_DIR/policy-$label.log" 2>&1 &
    pid=$!
    SERVER_PIDS[$index]="$pid"
    for _ in {1..100}; do
        process_alive "$pid" || die "$label exited during launch"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$label process group setup failed"
    SERVER_PGIDS[$index]="$pgid"
}

capture_server() {
    local index="$1" label="${SERVER_LABELS[$1]}" port="${SERVER_PORTS[$1]}"
    "$COSMOS_PY" "$HELPER" capture-runtime --uri "ws://127.0.0.1:$port" \
        --output "$CONTROL_DIR/runtime-$label.json" --timeout-seconds 1800 \
        --launcher-pid "${SERVER_PIDS[$index]}" --server-label "$label" \
        --visible-devices "${SERVER_DEVICES[$index]}" --launch-nonce "${SERVER_NONCES[$index]}" \
        --checkpoint "$SERVING_CHECKPOINT_PATH" --config "$TRAINING_CONFIG" \
        --action-stats "$ACTION_STATS" --vae "$WAN2PT2_VAE" --tokenizer "$QWEN_TOKENIZER" \
        --source-revision "$MODEL_SOURCE_REVISION" >"$CONTROL_DIR/runtime-$label.log"
}

launch_server 0
launch_server 1
capture_server 0
capture_server 1
write_status RUNNING "two_four_rank_policy_servers_ready"

eval_env=(
    env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON"
    __GLX_VENDOR_LIBRARY_NAME=nvidia PYTHONPATH="$EVAL_PYTHONPATH"
)

launch_eval() {
    local index="$1" task="$2" run_name="$3" port="$4" gpu="$5" seed="$6"
    local pid pgid=
    (
        exec setsid timeout --signal=TERM --kill-after=30 43200 \
            "${eval_env[@]}" CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
            "$ROBOCASA_PY" "$EVALUATOR" --tasks "$task" --task-shard 0 --num-shards 1 \
            --episodes 20 --results-dir "$RESULTS_ROOT" --run-name "$run_name" \
            --episodes-start 0 --max-steps 400 --open-loop-horizon 12 \
            --host 127.0.0.1 --port "$port" --checkpoint "$SERVING_CHECKPOINT_PATH" \
            --no-save-video --headless --camera-height 256 --camera-width 256 \
            --seed "$seed" --split atomic
    ) >"$CONTROL_DIR/evaluator-$task.log" 2>&1 &
    pid=$!
    EVAL_PIDS[$index]="$pid"
    for _ in {1..100}; do
        process_alive "$pid" || die "$task evaluator exited during launch"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$task evaluator process group setup failed"
    EVAL_PGIDS[$index]="$pgid"
}

launch_eval 0 "$TASK_A" "$RUN_A" 16000 0 42
launch_eval 1 "$TASK_B" "$RUN_B" 16001 4 10042
"$COSMOS_PY" "$HELPER" watch-progress \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --model-label "$MODEL_LABEL" --task-spec "$TASK_A=$RESULT_A" --task-spec "$TASK_B=$RESULT_B" \
    --pid "${EVAL_PIDS[0]}" --pid "${EVAL_PIDS[1]}" --interval-seconds 15 \
    --output "$CONTROL_DIR/progress.ndjson" >"$CONTROL_DIR/progress-watcher.log" 2>&1 &
PROGRESS_PID=$!
write_status RUNNING "rollouts_started_0_of_40"

set +e
wait "${EVAL_PIDS[0]}"; eval_a_code=$?
wait "${EVAL_PIDS[1]}"; eval_b_code=$?
set -e
EVAL_PIDS=("" "")
EVAL_PGIDS=("" "")
wait "$PROGRESS_PID" || die "progress watcher failed"
PROGRESS_PID=
[[ "$eval_a_code" -eq 0 && "$eval_b_code" -eq 0 ]] || die "evaluator exits: task_a=$eval_a_code task_b=$eval_b_code"
write_status FINALIZING "rollouts_complete_controlled_shutdown"

shutdown_server() {
    local index="$1" label="${SERVER_LABELS[$1]}" port="${SERVER_PORTS[$1]}"
    "$COSMOS_PY" "$HELPER" shutdown-server --uri "ws://127.0.0.1:$port" \
        --output "$CONTROL_DIR/shutdown-$label.json" --timeout-seconds 60 \
        --launch-nonce "${SERVER_NONCES[$index]}" --runtime-receipt "$CONTROL_DIR/runtime-$label.json" \
        >"$CONTROL_DIR/shutdown-$label.log"
    for _ in {1..180}; do process_alive "${SERVER_PIDS[$index]}" || break; sleep 1; done
    process_alive "${SERVER_PIDS[$index]}" && die "$label did not stop"
    set +e
    wait "${SERVER_PIDS[$index]}"; server_code=$?
    set -e
    [[ "$server_code" -eq 0 ]] || die "$label exited with $server_code"
    SERVER_PIDS[$index]=
    SERVER_PGIDS[$index]=
}

shutdown_server 0
shutdown_server 1
"$COSMOS_PY" "$HELPER" summarize \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --model-label "$MODEL_LABEL" --source-revision "$MODEL_SOURCE_REVISION" \
    --checkpoint "$SERVING_CHECKPOINT_PATH" --node "$EXPECTED_NODE_TAG" --rjob-name "$RJOB_NAME" \
    --task-spec "$TASK_A=$RESULT_A" --task-spec "$TASK_B=$RESULT_B" --expected-episodes 20 \
    --runtime-receipt "$CONTROL_DIR/runtime-server-a.json" --runtime-receipt "$CONTROL_DIR/runtime-server-b.json" \
    --shutdown-receipt "$CONTROL_DIR/shutdown-server-a.json" --shutdown-receipt "$CONTROL_DIR/shutdown-server-b.json" \
    --progress-log "$CONTROL_DIR/progress.ndjson" --output "$EVAL_OUTPUT_ROOT/final-summary.json" \
    --csv-output "$EVAL_OUTPUT_ROOT/task-success-rates.csv" >"$CONTROL_DIR/summarize.log"
write_status COMPLETE "40_rollouts_and_controlled_shutdown_complete"
[[ -z "$GPU_MONITOR_PID" ]] || kill "$GPU_MONITOR_PID" 2>/dev/null || true
GPU_MONITOR_PID=
TERMINAL_WRITTEN=1
printf 'CWAM_H_MATCHED_FAST_EVAL_COMPLETE summary=%s\n' "$EVAL_OUTPUT_ROOT/final-summary.json"
