#!/usr/bin/env bash
# Run the full RoboCasa365 suite with two serial queues and two four-GPU policy servers.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

required=(
    EVAL_CONTROL_SOURCE COSMOS_SOURCE COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO
    ROBOCASA_REPO ROBOSUITE_REPO SERVING_CHECKPOINT_PATH TRAINING_CONFIG
    ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER BASELINE_RECORD EVAL_OUTPUT_ROOT
    EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME MODEL_LABEL
    MODEL_SOURCE_REVISION CONTROL_SOURCE_REVISION EXPECTED_EMA_KEY_COUNT
    EVAL_EGL_VENDOR_JSON EXPECTED_NODE_TAG EVAL_GPU_COUNT RJOB_NAME
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done

readonly HELPER="$EVAL_CONTROL_SOURCE/examples/h_cluster/cwam_robocasa365_matched_fast_eval.py"
readonly SERVER_MODULE=cosmos_framework.scripts.action_policy_server_robocasa365
readonly EVALUATOR="$DEMIAN_REPO/demian_wam/eval_utils/run_robocasa_eval.py"
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly SCRATCH_ROOT="/nvme/$EVAL_EXPERIMENT_ID"
readonly EPISODES_PER_TASK=100

TASKS=(
    CloseBlenderLid
    CloseFridge
    CloseToasterOvenDoor
    CoffeeSetupMug
    OpenCabinet
    OpenDrawer
    OpenStandMixerHead
    PickPlaceCounterToCabinet
    PickPlaceCounterToStove
    PickPlaceDrawerToCounter
    PickPlaceSinkToCounter
    PickPlaceToasterToCounter
    SlideDishwasherRack
    TurnOffStove
    TurnOnElectricKettle
    TurnOnMicrowave
    TurnOnSinkFaucet
)
TASK_SLUGS=(
    close-blender-lid
    close-fridge
    close-toaster-oven-door
    coffee-setup-mug
    open-cabinet
    open-drawer
    open-stand-mixer-head
    pick-place-counter-to-cabinet
    pick-place-counter-to-stove
    pick-place-drawer-to-counter
    pick-place-sink-to-counter
    pick-place-toaster-to-counter
    slide-dishwasher-rack
    turn-off-stove
    turn-on-electric-kettle
    turn-on-microwave
    turn-on-sink-faucet
)
readonly -a TASKS TASK_SLUGS
readonly TASK_COUNT="${#TASKS[@]}"
readonly EXPECTED_TOTAL_EPISODES="$((TASK_COUNT * EPISODES_PER_TASK))"
[[ "$TASK_COUNT" -eq 17 && "${#TASK_SLUGS[@]}" -eq "$TASK_COUNT" ]] || {
    echo "ERROR: canonical task inventory mismatch" >&2
    exit 2
}
[[ "$EVAL_GPU_COUNT" == 8 ]] || { echo "ERROR: eight-GPU eval requires eight GPUs" >&2; exit 2; }

readonly -a SERVER_LABELS=(server-a server-b)
readonly -a SERVER_DEVICES=(0,1,2,3 4,5,6,7)
readonly -a EVAL_GPU_DEVICES=(0 4)
readonly POLICY_WORLD_SIZE=4
declare -a RUN_NAMES=() RESULT_PATHS=() TASK_SPECS=()
for index in "${!TASKS[@]}"; do
    run_name="$EVAL_RUN_NAME-${TASK_SLUGS[$index]}"
    result_path="$RESULTS_ROOT/$run_name/results_shard0.json"
    RUN_NAMES+=("$run_name")
    RESULT_PATHS+=("$result_path")
    TASK_SPECS+=("${TASKS[$index]}=$result_path")
done
readonly -a RUN_NAMES RESULT_PATHS TASK_SPECS

declare -a SERVER_PORTS=()
declare -a SERVER_PIDS=() SERVER_PGIDS=() SERVER_NONCES=()
declare -a EVAL_PIDS=("" "") EVAL_PGIDS=("" "") ACTIVE_TASK=(-1 -1) NEXT_TASK=(0 1)
GPU_MONITOR_PID=
PROGRESS_PID=
GIT_SAFE_CONFIG=
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
    [[ -z "$GIT_SAFE_CONFIG" ]] || rm -f "$GIT_SAFE_CONFIG"
    for index in "${!EVAL_PIDS[@]}"; do
        stop_group "${EVAL_PIDS[$index]}" "${EVAL_PGIDS[$index]}"
    done
    for index in "${!SERVER_PIDS[@]}"; do
        stop_group "${SERVER_PIDS[$index]}" "${SERVER_PGIDS[$index]}"
    done
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
for file in "$HELPER" "$EVALUATOR" "$TRAINING_CONFIG" "$ACTION_STATS" "$WAN2PT2_VAE" "$EVAL_EGL_VENDOR_JSON" "$BASELINE_RECORD"; do
    [[ -f "$file" ]] || die "missing file: $file"
done
[[ -x "$COSMOS_PY" && -x "$ROBOCASA_PY" ]] || die "runtime executables are unavailable"
GIT_SAFE_CONFIG="$(mktemp /tmp/cwam-git-safe.XXXXXX)"
export GIT_CONFIG_GLOBAL="$GIT_SAFE_CONFIG"
git config --global --add safe.directory "$COSMOS_SOURCE"
git config --global --add safe.directory "$EVAL_CONTROL_SOURCE"
[[ "$(git -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$MODEL_SOURCE_REVISION" ]] || die "model source revision mismatch"
[[ "$(git -C "$EVAL_CONTROL_SOURCE" rev-parse HEAD)" == "$CONTROL_SOURCE_REVISION" ]] || die "control source revision mismatch"
[[ "$EVAL_OUTPUT_ROOT" == /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-eight-gpu-eval/runs/* ]] || die "unexpected output root"
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || die "refusing to reuse output root"
[[ ! -e "$SCRATCH_ROOT" ]] || die "refusing to reuse scratch root"
mkdir -p "$CONTROL_DIR" "$RESULTS_ROOT"
for label in "${SERVER_LABELS[@]}"; do
    mkdir -p "$SCRATCH_ROOT/$label" "$SCRATCH_ROOT/torchinductor-${label#server-}"
done
write_status STARTING "validating_eight_h200_worker"

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 8 ]] || die "expected 8 GPUs, got ${#gpu_names[@]}"
for name in "${gpu_names[@]}"; do [[ "$name" == *H200* ]] || die "non-H200 GPU visible: $name"; done
printf '%s\n' "${gpu_names[@]}" >"$CONTROL_DIR/gpu-inventory.txt"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 30 >"$CONTROL_DIR/gpu-monitor.csv" 2>"$CONTROL_DIR/gpu-monitor.err" &
GPU_MONITOR_PID=$!

mapfile -t SERVER_PORTS < <(SERVER_COUNT="${#SERVER_LABELS[@]}" "$COSMOS_PY" - <<'PY'
import os
import socket

sockets = []
try:
    for _ in range(int(os.environ["SERVER_COUNT"])):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
    for sock in sockets:
        print(sock.getsockname()[1])
finally:
    for sock in sockets:
        sock.close()
PY
)
[[ "${#SERVER_PORTS[@]}" -eq 2 ]] || die "failed to reserve policy server ports"
for _ in "${SERVER_LABELS[@]}"; do
    SERVER_PIDS+=("")
    SERVER_PGIDS+=("")
    SERVER_NONCES+=("")
done

launch_server() {
    local index="$1" label="${SERVER_LABELS[$1]}" devices="${SERVER_DEVICES[$1]}" port="${SERVER_PORTS[$1]}"
    local pid pgid=
    local -a launcher=(
        "$COSMOS_PY" -m torch.distributed.run --standalone
        --nproc-per-node="$POLICY_WORLD_SIZE" --module "$SERVER_MODULE"
    )
    SERVER_NONCES[$index]="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
    (
        cd "$COSMOS_SOURCE"
        exec setsid env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
            -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
            CUDA_VISIBLE_DEVICES="$devices" PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$SCRATCH_ROOT/torchinductor-${label#server-}" \
            "${launcher[@]}" --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
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
        --source-revision "$MODEL_SOURCE_REVISION" --expected-gpu-count 8 \
        --expected-policy-world-size "$POLICY_WORLD_SIZE" \
        >"$CONTROL_DIR/runtime-$label.log"
}

for index in "${!SERVER_LABELS[@]}"; do launch_server "$index"; done
for index in "${!SERVER_LABELS[@]}"; do capture_server "$index"; done
write_status RUNNING "two_four_rank_policy_servers_ready ports=${SERVER_PORTS[*]}"

eval_env=(
    env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON"
    __GLX_VENDOR_LIBRARY_NAME=nvidia PYTHONPATH="$EVAL_PYTHONPATH"
)

launch_eval() {
    local slot="$1" task_index="$2"
    local task="${TASKS[$task_index]}" run_name="${RUN_NAMES[$task_index]}"
    local port="${SERVER_PORTS[$slot]}" gpu="${EVAL_GPU_DEVICES[$slot]}"
    local pid pgid=
    (
        exec setsid timeout --signal=TERM --kill-after=30 43200 \
            "${eval_env[@]}" CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
            "$ROBOCASA_PY" "$EVALUATOR" --tasks "$task" --task-shard 0 --num-shards 1 \
            --episodes "$EPISODES_PER_TASK" --results-dir "$RESULTS_ROOT" --run-name "$run_name" \
            --episodes-start 0 --max-steps 400 --open-loop-horizon 12 \
            --host 127.0.0.1 --port "$port" --checkpoint "$SERVING_CHECKPOINT_PATH" \
            --no-save-video --headless --camera-height 256 --camera-width 256 \
            --seed 42 --split atomic
    ) >"$CONTROL_DIR/evaluator-$task.log" 2>&1 &
    pid=$!
    EVAL_PIDS[$slot]="$pid"
    ACTIVE_TASK[$slot]="$task_index"
    for _ in {1..100}; do
        process_alive "$pid" || die "$task evaluator exited during launch"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$task process group setup failed"
    EVAL_PGIDS[$slot]="$pgid"
}

launch_eval 0 0
launch_eval 1 1
NEXT_TASK=(2 3)
progress_args=()
for spec in "${TASK_SPECS[@]}"; do progress_args+=(--task-spec "$spec"); done
"$COSMOS_PY" "$HELPER" watch-progress \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --model-label "$MODEL_LABEL" "${progress_args[@]}" \
    --baseline-record "$BASELINE_RECORD" \
    --comparison-output "$CONTROL_DIR/comparison-to-i10000-latest.json" \
    --pid "${SERVER_PIDS[0]}" --pid "${SERVER_PIDS[1]}" --interval-seconds 15 \
    --output "$CONTROL_DIR/progress.ndjson" >"$CONTROL_DIR/progress-watcher.log" 2>&1 &
PROGRESS_PID=$!
write_status RUNNING "rollouts_started_0_of_$EXPECTED_TOTAL_EPISODES"

tasks_complete=0
while ((tasks_complete < TASK_COUNT)); do
    for slot in 0 1; do
        pid="${EVAL_PIDS[$slot]}"
        [[ -n "$pid" ]] || continue
        process_alive "$pid" && continue
        set +e
        wait "$pid"
        eval_code=$?
        set -e
        task_index="${ACTIVE_TASK[$slot]}"
        task="${TASKS[$task_index]}"
        EVAL_PIDS[$slot]=
        EVAL_PGIDS[$slot]=
        ACTIVE_TASK[$slot]=-1
        [[ "$eval_code" -eq 0 ]] || die "$task evaluator exited with $eval_code"
        "$COSMOS_PY" "$HELPER" check-task --task "$task" \
            --result "${RESULT_PATHS[$task_index]}" --expected-episodes "$EPISODES_PER_TASK" \
            >"$CONTROL_DIR/check-$task.json"
        tasks_complete=$((tasks_complete + 1))
        episodes_complete=$((tasks_complete * EPISODES_PER_TASK))
        write_status RUNNING "task_complete=$task tasks=${tasks_complete}_of_${TASK_COUNT} episodes=${episodes_complete}_of_${EXPECTED_TOTAL_EPISODES}"
        next="${NEXT_TASK[$slot]}"
        if ((next < TASK_COUNT)); then
            launch_eval "$slot" "$next"
            NEXT_TASK[$slot]=$((next + 2))
        fi
    done
    ((tasks_complete == TASK_COUNT)) || sleep 2
done
write_status FINALIZING "${EXPECTED_TOTAL_EPISODES}_rollouts_complete_controlled_shutdown"

shutdown_server() {
    local index="$1" label="${SERVER_LABELS[$1]}" port="${SERVER_PORTS[$1]}"
    "$COSMOS_PY" "$HELPER" shutdown-server --uri "ws://127.0.0.1:$port" \
        --output "$CONTROL_DIR/shutdown-$label.json" --timeout-seconds 60 \
        --launch-nonce "${SERVER_NONCES[$index]}" --runtime-receipt "$CONTROL_DIR/runtime-$label.json" \
        >"$CONTROL_DIR/shutdown-$label.log"
    for _ in {1..180}; do process_alive "${SERVER_PIDS[$index]}" || break; sleep 1; done
    process_alive "${SERVER_PIDS[$index]}" && die "$label did not stop"
    set +e
    wait "${SERVER_PIDS[$index]}"
    server_code=$?
    set -e
    [[ "$server_code" -eq 0 ]] || die "$label exited with $server_code"
    SERVER_PIDS[$index]=
    SERVER_PGIDS[$index]=
}

for index in "${!SERVER_LABELS[@]}"; do shutdown_server "$index"; done
wait "$PROGRESS_PID" || die "progress watcher failed"
PROGRESS_PID=
summary_args=()
for spec in "${TASK_SPECS[@]}"; do summary_args+=(--task-spec "$spec"); done
for label in "${SERVER_LABELS[@]}"; do
    summary_args+=(--runtime-receipt "$CONTROL_DIR/runtime-$label.json")
    summary_args+=(--shutdown-receipt "$CONTROL_DIR/shutdown-$label.json")
done
"$COSMOS_PY" "$HELPER" summarize \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --model-label "$MODEL_LABEL" --source-revision "$MODEL_SOURCE_REVISION" \
    --checkpoint "$SERVING_CHECKPOINT_PATH" --node "$EXPECTED_NODE_TAG" --rjob-name "$RJOB_NAME" \
    "${summary_args[@]}" --expected-episodes "$EPISODES_PER_TASK" \
    --baseline-record "$BASELINE_RECORD" \
    --progress-log "$CONTROL_DIR/progress.ndjson" --output "$EVAL_OUTPUT_ROOT/final-summary.json" \
    --csv-output "$EVAL_OUTPUT_ROOT/task-success-rates.csv" >"$CONTROL_DIR/summarize.log"
write_status COMPLETE "${EXPECTED_TOTAL_EPISODES}_rollouts_and_controlled_shutdown_complete"
[[ -z "$GPU_MONITOR_PID" ]] || kill "$GPU_MONITOR_PID" 2>/dev/null || true
GPU_MONITOR_PID=
TERMINAL_WRITTEN=1
printf 'CWAM_H_FULL_FAST_EVAL_COMPLETE summary=%s\n' "$EVAL_OUTPUT_ROOT/final-summary.json"
