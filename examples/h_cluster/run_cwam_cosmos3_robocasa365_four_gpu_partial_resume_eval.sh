#!/usr/bin/env bash
# Finish two interrupted RoboCasa365 tasks with four exact episode ranges.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

required=(
    EVAL_CONTROL_SOURCE COSMOS_SOURCE COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO
    ROBOCASA_REPO ROBOSUITE_REPO SERVING_CHECKPOINT_PATH TRAINING_CONFIG
    ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER BASELINE_RECORD EVAL_OUTPUT_ROOT
    EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME MODEL_LABEL
    MODEL_SOURCE_REVISION CONTROL_SOURCE_REVISION EVALUATOR_SOURCE_REVISION
    EXPECTED_EMA_KEY_COUNT EVAL_EGL_VENDOR_JSON EXPECTED_NODE_TAG EVAL_GPU_COUNT
    RJOB_NAME EVAL_R4_SOURCE_ROOT EVAL_R4_SOURCE_RUN_NAME
    EVAL_R6_SOURCE_ROOT EVAL_R6_SOURCE_RUN_NAME
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done

readonly HELPER="$EVAL_CONTROL_SOURCE/examples/h_cluster/cwam_robocasa365_matched_fast_eval.py"
readonly EVALUATOR_PATCH="$EVAL_CONTROL_SOURCE/examples/h_cluster/patches/demian_robocasa_eval_episode_range.patch"
readonly ORIGINAL_EVALUATOR="$DEMIAN_REPO/demian_wam/eval_utils/run_robocasa_eval.py"
readonly SERVER_MODULE=cosmos_framework.scripts.action_policy_server_robocasa365
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly SCRATCH_ROOT="/nvme/$EVAL_EXPERIMENT_ID"
readonly PATCHED_DEMIAN_ROOT="$SCRATCH_ROOT/patched-demian"
readonly PATCHED_EVALUATOR="$PATCHED_DEMIAN_ROOT/demian_wam/eval_utils/run_robocasa_eval.py"
readonly EPISODES_PER_TASK=100
readonly REQUIRED_NODE_TAG=node/gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn

TASKS=(
    CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug OpenCabinet
    OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove
    PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter
    SlideDishwasherRack TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet
)
TASK_SLUGS=(
    close-blender-lid close-fridge close-toaster-oven-door coffee-setup-mug open-cabinet
    open-drawer open-stand-mixer-head pick-place-counter-to-cabinet pick-place-counter-to-stove
    pick-place-drawer-to-counter pick-place-sink-to-counter pick-place-toaster-to-counter
    slide-dishwasher-rack turn-off-stove turn-on-electric-kettle turn-on-microwave turn-on-sink-faucet
)
readonly -a TASKS TASK_SLUGS
readonly TASK_COUNT="${#TASKS[@]}"
readonly EXPECTED_TOTAL_EPISODES="$((TASK_COUNT * EPISODES_PER_TASK))"
[[ "$TASK_COUNT" -eq 17 && "${#TASK_SLUGS[@]}" -eq 17 ]] || { echo 'ERROR: task inventory mismatch' >&2; exit 2; }
[[ "$EXPECTED_NODE_TAG" == "$REQUIRED_NODE_TAG" ]] || { echo 'ERROR: partial resume must target H200-0299' >&2; exit 2; }
[[ "$EVAL_GPU_COUNT" == 4 ]] || { echo 'ERROR: partial resume requires four GPUs' >&2; exit 2; }

declare -a RESULT_PATHS=() TASK_SPECS=()
for index in "${!TASKS[@]}"; do
    if ((index < 2)); then
        result_path="$EVAL_R4_SOURCE_ROOT/results/$EVAL_R4_SOURCE_RUN_NAME-${TASK_SLUGS[$index]}/results_shard0.json"
    elif ((index < 15)); then
        result_path="$EVAL_R6_SOURCE_ROOT/results/$EVAL_R6_SOURCE_RUN_NAME-${TASK_SLUGS[$index]}/results_shard0.json"
    else
        result_path="$RESULTS_ROOT/$EVAL_RUN_NAME-${TASK_SLUGS[$index]}/results_shard0.json"
    fi
    RESULT_PATHS+=("$result_path")
    TASK_SPECS+=("${TASKS[$index]}=$result_path")
done
readonly -a RESULT_PATHS TASK_SPECS

readonly MICROWAVE_SOURCE="$EVAL_R6_SOURCE_ROOT/results/$EVAL_R6_SOURCE_RUN_NAME-turn-on-microwave/results_shard0.json"
readonly SINK_SOURCE="$EVAL_R6_SOURCE_ROOT/results/$EVAL_R6_SOURCE_RUN_NAME-turn-on-sink-faucet/results_shard0.json"
readonly -a FRAGMENT_TASKS=(TurnOnMicrowave TurnOnMicrowave TurnOnSinkFaucet TurnOnSinkFaucet)
readonly -a FRAGMENT_SLUGS=(turn-on-microwave turn-on-microwave turn-on-sink-faucet turn-on-sink-faucet)
readonly -a FRAGMENT_STARTS=(85 93 84 92)
readonly -a FRAGMENT_STOPS=(93 100 92 100)
declare -a FRAGMENT_RUN_NAMES=() FRAGMENT_PATHS=()
for slot in 0 1 2 3; do
    printf -v range '%03d-%03d' "${FRAGMENT_STARTS[$slot]}" "$((FRAGMENT_STOPS[$slot] - 1))"
    run_name="$EVAL_RUN_NAME-${FRAGMENT_SLUGS[$slot]}-ep$range"
    FRAGMENT_RUN_NAMES+=("$run_name")
    FRAGMENT_PATHS+=("$RESULTS_ROOT/$run_name/results_shard0.json")
done
readonly -a FRAGMENT_RUN_NAMES FRAGMENT_PATHS

readonly -a SERVER_LABELS=(server-a server-b server-c server-d)
readonly -a SERVER_DEVICES=(0 1 2 3)
declare -a SERVER_PORTS=() SERVER_PIDS=("" "" "" "") SERVER_PGIDS=("" "" "" "") SERVER_NONCES=("" "" "" "")
declare -a EVAL_PIDS=("" "" "" "") EVAL_PGIDS=("" "" "" "")
GPU_MONITOR_PID=
PROGRESS_PID=
GIT_SAFE_CONFIG=
TERMINAL_WRITTEN=0

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

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
    for slot in 0 1 2 3; do stop_group "${EVAL_PIDS[$slot]}" "${EVAL_PGIDS[$slot]}"; done
    for slot in 0 1 2 3; do stop_group "${SERVER_PIDS[$slot]}" "${SERVER_PGIDS[$slot]}"; done
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
    "$DEMIAN_REPO" "$ROBOCASA_REPO" "$ROBOSUITE_REPO" "$SERVING_CHECKPOINT_PATH" \
    "$QWEN_TOKENIZER" "$EVAL_R4_SOURCE_ROOT" "$EVAL_R6_SOURCE_ROOT"; do
    [[ -d "$directory" ]] || die "missing directory: $directory"
done
for file in "$HELPER" "$EVALUATOR_PATCH" "$ORIGINAL_EVALUATOR" "$TRAINING_CONFIG" \
    "$ACTION_STATS" "$WAN2PT2_VAE" "$EVAL_EGL_VENDOR_JSON" "$BASELINE_RECORD"; do
    [[ -f "$file" ]] || die "missing file: $file"
done
[[ -x "$COSMOS_PY" && -x "$ROBOCASA_PY" ]] || die 'runtime executables unavailable'
[[ "$EVAL_OUTPUT_ROOT" == /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-four-gpu-eval/runs/* ]] || die 'unexpected output root'
[[ "$EVAL_R4_SOURCE_ROOT" == /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-two-gpu-eval/runs/* ]] || die 'unexpected R4 source root'
[[ "$EVAL_R6_SOURCE_ROOT" == /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-four-gpu-eval/runs/* ]] || die 'unexpected R6 source root'
[[ ! -e "$EVAL_OUTPUT_ROOT" && ! -e "$SCRATCH_ROOT" ]] || die 'output or scratch root already exists'

GIT_SAFE_CONFIG="$(mktemp /tmp/cwam-git-safe.XXXXXX)"
export GIT_CONFIG_GLOBAL="$GIT_SAFE_CONFIG"
for directory in "$COSMOS_SOURCE" "$EVAL_CONTROL_SOURCE" "$DEMIAN_REPO"; do git config --global --add safe.directory "$directory"; done
[[ "$(git -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$MODEL_SOURCE_REVISION" ]] || die 'model source revision mismatch'
[[ "$(git -C "$EVAL_CONTROL_SOURCE" rev-parse HEAD)" == "$CONTROL_SOURCE_REVISION" ]] || die 'control source revision mismatch'
[[ "$(git -C "$DEMIAN_REPO" rev-parse HEAD)" == "$EVALUATOR_SOURCE_REVISION" ]] || die 'evaluator source revision mismatch'
[[ -z "$(git -C "$DEMIAN_REPO" status --porcelain)" ]] || die 'evaluator source is dirty'

mkdir -p "$CONTROL_DIR" "$RESULTS_ROOT" "$PATCHED_DEMIAN_ROOT/demian_wam/eval_utils"
for label in "${SERVER_LABELS[@]}"; do mkdir -p "$SCRATCH_ROOT/$label" "$SCRATCH_ROOT/torchinductor-${label#server-}"; done
cp -- "$ORIGINAL_EVALUATOR" "$PATCHED_EVALUATOR"
(cd "$PATCHED_DEMIAN_ROOT" && git apply "$EVALUATOR_PATCH")
"$ROBOCASA_PY" -m py_compile "$PATCHED_EVALUATOR"
cp -- "$EVALUATOR_PATCH" "$CONTROL_DIR/evaluator-episode-range.patch"
write_status VALIDATING 'checking_1669_source_episodes_and_four_h200_worker'

for index in $(seq 0 14); do
    "$COSMOS_PY" "$HELPER" check-task --task "${TASKS[$index]}" \
        --result "${RESULT_PATHS[$index]}" --expected-episodes "$EPISODES_PER_TASK" \
        >"$CONTROL_DIR/check-source-${TASKS[$index]}.json"
done
"$COSMOS_PY" "$HELPER" check-partial-task --task TurnOnMicrowave --result "$MICROWAVE_SOURCE" \
    --expected-episodes 85 --expected-successes 5 >"$CONTROL_DIR/check-source-TurnOnMicrowave.json"
"$COSMOS_PY" "$HELPER" check-partial-task --task TurnOnSinkFaucet --result "$SINK_SOURCE" \
    --expected-episodes 84 --expected-successes 17 >"$CONTROL_DIR/check-source-TurnOnSinkFaucet.json"

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 4 ]] || die "expected 4 GPUs, got ${#gpu_names[@]}"
for name in "${gpu_names[@]}"; do [[ "$name" == *H200* ]] || die "non-H200 GPU visible: $name"; done
printf '%s\n' "${gpu_names[@]}" >"$CONTROL_DIR/gpu-inventory.txt"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 30 >"$CONTROL_DIR/gpu-monitor.csv" 2>"$CONTROL_DIR/gpu-monitor.err" &
GPU_MONITOR_PID=$!

mapfile -t SERVER_PORTS < <(SERVER_COUNT=4 "$COSMOS_PY" - <<'PY'
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
[[ "${#SERVER_PORTS[@]}" -eq 4 ]] || die 'failed to reserve policy server ports'

launch_server() {
    local slot="$1" label="${SERVER_LABELS[$1]}" device="${SERVER_DEVICES[$1]}" port="${SERVER_PORTS[$1]}" pid pgid=
    SERVER_NONCES[$slot]="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
    (
        cd "$COSMOS_SOURCE"
        exec setsid env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
            -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
            CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$SCRATCH_ROOT/torchinductor-${label#server-}" \
            "$COSMOS_PY" -m "$SERVER_MODULE" --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
            --config-file "$TRAINING_CONFIG" --output-dir "$SCRATCH_ROOT/$label" \
            --action-stats-path "$ACTION_STATS" --wan2pt2-vae-path "$WAN2PT2_VAE" \
            --qwen-tokenizer-path "$QWEN_TOKENIZER" --host 127.0.0.1 --port "$port" \
            --sampler unipc --seed 42 --guidance 1.0 --num-steps 30 \
            --expected-ema-key-count "$EXPECTED_EMA_KEY_COUNT" --launch-nonce "${SERVER_NONCES[$slot]}"
    ) >"$CONTROL_DIR/policy-$label.log" 2>&1 &
    pid=$!
    SERVER_PIDS[$slot]="$pid"
    for _ in {1..100}; do
        process_alive "$pid" || die "$label exited during launch"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$label process group setup failed"
    SERVER_PGIDS[$slot]="$pgid"
}

capture_server() {
    local slot="$1" label="${SERVER_LABELS[$1]}" port="${SERVER_PORTS[$1]}"
    "$COSMOS_PY" "$HELPER" capture-runtime --uri "ws://127.0.0.1:$port" \
        --output "$CONTROL_DIR/runtime-$label.json" --timeout-seconds 1800 \
        --launcher-pid "${SERVER_PIDS[$slot]}" --server-label "$label" \
        --visible-devices "${SERVER_DEVICES[$slot]}" --launch-nonce "${SERVER_NONCES[$slot]}" \
        --checkpoint "$SERVING_CHECKPOINT_PATH" --config "$TRAINING_CONFIG" \
        --action-stats "$ACTION_STATS" --vae "$WAN2PT2_VAE" --tokenizer "$QWEN_TOKENIZER" \
        --source-revision "$MODEL_SOURCE_REVISION" --expected-gpu-count 4 \
        --expected-policy-world-size 1 >"$CONTROL_DIR/runtime-$label.log"
}

for slot in 0 1 2 3; do launch_server "$slot"; done
for slot in 0 1 2 3; do capture_server "$slot"; done
write_status RUNNING "source_episodes=1669 four_policy_servers_ready ports=${SERVER_PORTS[*]}"

eval_env=(
    env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON"
    __GLX_VENDOR_LIBRARY_NAME=nvidia PYTHONPATH="$EVAL_PYTHONPATH"
)

launch_eval() {
    local slot="$1" task="${FRAGMENT_TASKS[$1]}" start="${FRAGMENT_STARTS[$1]}" stop="${FRAGMENT_STOPS[$1]}"
    local run_name="${FRAGMENT_RUN_NAMES[$1]}" port="${SERVER_PORTS[$1]}" pid pgid=
    (
        exec setsid timeout --signal=TERM --kill-after=30 43200 \
            "${eval_env[@]}" CUDA_VISIBLE_DEVICES="$slot" MUJOCO_EGL_DEVICE_ID="$slot" \
            "$ROBOCASA_PY" "$PATCHED_EVALUATOR" --tasks "$task" --task-shard 0 --num-shards 1 \
            --episodes "$EPISODES_PER_TASK" --episode-range-start "$start" --episode-range-stop "$stop" \
            --results-dir "$RESULTS_ROOT" --run-name "$run_name" --episodes-start 0 \
            --max-steps 400 --open-loop-horizon 12 --host 127.0.0.1 --port "$port" \
            --checkpoint "$SERVING_CHECKPOINT_PATH" --no-save-video --headless \
            --camera-height 256 --camera-width 256 --seed 42 --split atomic
    ) >"$CONTROL_DIR/evaluator-$run_name.log" 2>&1 &
    pid=$!
    EVAL_PIDS[$slot]="$pid"
    for _ in {1..100}; do
        process_alive "$pid" || die "$run_name evaluator exited during launch"
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] && break
        sleep 0.05
    done
    [[ "$pgid" == "$pid" ]] || die "$run_name process group setup failed"
    EVAL_PGIDS[$slot]="$pgid"
}

for slot in 0 1 2 3; do launch_eval "$slot"; done
progress_args=()
for index in $(seq 0 14); do progress_args+=(--task-spec "${TASK_SPECS[$index]}"); done
progress_args+=(--resume-task-spec "TurnOnMicrowave=$MICROWAVE_SOURCE")
progress_args+=(--resume-task-spec "TurnOnSinkFaucet=$SINK_SOURCE")
for slot in 0 1 2 3; do
    progress_args+=(--fragment-spec "${FRAGMENT_TASKS[$slot]}:${FRAGMENT_STARTS[$slot]}:${FRAGMENT_STOPS[$slot]}=${FRAGMENT_PATHS[$slot]}")
done
"$COSMOS_PY" "$HELPER" watch-resume-progress \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --model-label "$MODEL_LABEL" "${progress_args[@]}" --expected-episodes 100 \
    --baseline-record "$BASELINE_RECORD" --comparison-output "$CONTROL_DIR/comparison-to-i10000-latest.json" \
    --pid "${SERVER_PIDS[0]}" --pid "${SERVER_PIDS[1]}" --pid "${SERVER_PIDS[2]}" --pid "${SERVER_PIDS[3]}" \
    --interval-seconds 15 --output "$CONTROL_DIR/progress.ndjson" >"$CONTROL_DIR/progress-watcher.log" 2>&1 &
PROGRESS_PID=$!

for slot in 0 1 2 3; do
    set +e
    wait "${EVAL_PIDS[$slot]}"
    eval_code=$?
    set -e
    EVAL_PIDS[$slot]=
    EVAL_PGIDS[$slot]=
    [[ "$eval_code" -eq 0 ]] || die "${FRAGMENT_RUN_NAMES[$slot]} evaluator exited with $eval_code"
done
write_status MERGING '31_new_rollouts_complete_validating_absolute_episode_inventory'

"$COSMOS_PY" "$HELPER" merge-resume-fragments --task TurnOnMicrowave --source "$MICROWAVE_SOURCE" \
    --fragment "85:93=${FRAGMENT_PATHS[0]}" --fragment "93:100=${FRAGMENT_PATHS[1]}" \
    --expected-episodes 100 --output "${RESULT_PATHS[15]}" \
    --receipt "$CONTROL_DIR/merge-TurnOnMicrowave.json" >"$CONTROL_DIR/merge-TurnOnMicrowave.log"
"$COSMOS_PY" "$HELPER" merge-resume-fragments --task TurnOnSinkFaucet --source "$SINK_SOURCE" \
    --fragment "84:92=${FRAGMENT_PATHS[2]}" --fragment "92:100=${FRAGMENT_PATHS[3]}" \
    --expected-episodes 100 --output "${RESULT_PATHS[16]}" \
    --receipt "$CONTROL_DIR/merge-TurnOnSinkFaucet.json" >"$CONTROL_DIR/merge-TurnOnSinkFaucet.log"
for index in 15 16; do
    "$COSMOS_PY" "$HELPER" check-task --task "${TASKS[$index]}" --result "${RESULT_PATHS[$index]}" \
        --expected-episodes 100 >"$CONTROL_DIR/check-final-${TASKS[$index]}.json"
done
write_status FINALIZING '1700_rollouts_complete_controlled_shutdown'

shutdown_server() {
    local slot="$1" label="${SERVER_LABELS[$1]}" port="${SERVER_PORTS[$1]}" server_code
    "$COSMOS_PY" "$HELPER" shutdown-server --uri "ws://127.0.0.1:$port" \
        --output "$CONTROL_DIR/shutdown-$label.json" --timeout-seconds 60 \
        --launch-nonce "${SERVER_NONCES[$slot]}" --runtime-receipt "$CONTROL_DIR/runtime-$label.json" \
        >"$CONTROL_DIR/shutdown-$label.log"
    for _ in {1..180}; do process_alive "${SERVER_PIDS[$slot]}" || break; sleep 1; done
    process_alive "${SERVER_PIDS[$slot]}" && die "$label did not stop"
    set +e
    wait "${SERVER_PIDS[$slot]}"
    server_code=$?
    set -e
    [[ "$server_code" -eq 0 ]] || die "$label exited with $server_code"
    SERVER_PIDS[$slot]=
    SERVER_PGIDS[$slot]=
}

for slot in 0 1 2 3; do shutdown_server "$slot"; done
wait "$PROGRESS_PID" || die 'progress watcher failed'
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
    "${summary_args[@]}" --expected-episodes 100 --baseline-record "$BASELINE_RECORD" \
    --progress-log "$CONTROL_DIR/progress.ndjson" --output "$EVAL_OUTPUT_ROOT/final-summary.json" \
    --csv-output "$EVAL_OUTPUT_ROOT/task-success-rates.csv" >"$CONTROL_DIR/summarize.log"
write_status COMPLETE "${EXPECTED_TOTAL_EPISODES}_rollouts_and_controlled_shutdown_complete"
[[ -z "$GPU_MONITOR_PID" ]] || kill "$GPU_MONITOR_PID" 2>/dev/null || true
GPU_MONITOR_PID=
TERMINAL_WRITTEN=1
printf 'CWAM_H_PARTIAL_RESUME_EVAL_COMPLETE summary=%s\n' "$EVAL_OUTPUT_ROOT/final-summary.json"
