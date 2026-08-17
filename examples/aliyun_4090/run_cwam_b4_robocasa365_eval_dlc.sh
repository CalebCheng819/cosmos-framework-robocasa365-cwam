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
    EVAL_EGL_VENDOR_JSON B4_EVAL_HELPER
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done

readonly SOURCE_COMMIT=e7775ffcea0a2d577597fbeb05c8bb92905aba0f
readonly DEMIAN_COMMIT=2600ece5bf0eeacf010a95f084dcda2de6d30d59
readonly ROBOCASA_COMMIT=29f7ce8814c1547f5af762a0997fbd4b64848dd7
readonly ROBOSUITE_COMMIT=232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
readonly PORT=16000
readonly POLICY_URI="ws://127.0.0.1:$PORT"
readonly SERVER_READY_SECONDS=1800
readonly EVALUATOR_TIMEOUT_SECONDS=129600
readonly SERVER_EXIT_SECONDS=180
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly RESULTS_DIR="$RESULTS_ROOT/$EVAL_RUN_NAME"
readonly GATE_ROOT="$EVAL_OUTPUT_ROOT/one-episode-gate"
readonly GATE_RUN_NAME=gate-close-blender-lid
readonly GATE_DIR="$GATE_ROOT/$GATE_RUN_NAME"
readonly SCRATCH_DIR="$EVAL_OUTPUT_ROOT/scratch"
readonly HELPER="$B4_EVAL_HELPER"
readonly SERVER_MODULE=cosmos_framework.scripts.action_policy_server_robocasa365
readonly EVALUATOR="$DEMIAN_REPO/demian_wam/eval_utils/run_robocasa_eval.py"
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly TORCHRUN="$COSMOS_ENV/bin/torchrun"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"

SERVER_PID=
SERVER_PGID=
SERVER_EXIT_CODE=
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

stop_server() {
    if [[ -n "$SERVER_PGID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    elif [[ -n "$SERVER_PID" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
}

on_exit() {
    local status="$?"
    trap - EXIT INT TERM
    if ((status != 0)); then
        stop_server
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
    "$EVAL_EGL_VENDOR_JSON" "$HELPER" "$EVALUATOR" "$COSMOS_PY" "$TORCHRUN" \
    "$ROBOCASA_PY"; do
    [[ -f "$file" && ! -L "$file" ]] || die "missing regular file: $file"
done
[[ "$(git_revision "$COSMOS_SOURCE")" == "$SOURCE_COMMIT" ]] || die "Cosmos source revision mismatch"
[[ "$(git_revision "$DEMIAN_REPO")" == "$DEMIAN_COMMIT" ]] || die "DeMiAn revision mismatch"
[[ "$(git_revision "$ROBOCASA_REPO")" == "$ROBOCASA_COMMIT" ]] || die "RoboCasa revision mismatch"
[[ "$(git_revision "$ROBOSUITE_REPO")" == "$ROBOSUITE_COMMIT" ]] || die "robosuite revision mismatch"
[[ "$(findmnt -n -T "$EVAL_OUTPUT_ROOT" -o TARGET)" != / ]] || die "evaluation output is not on persistent storage"
[[ ! -e "$CONTROL_DIR" ]] || die "refusing to reuse evaluation output: $EVAL_OUTPUT_ROOT"
mkdir -p "$CONTROL_DIR" "$RESULTS_ROOT" "$GATE_ROOT" "$SCRATCH_DIR/server" "$SCRATCH_DIR/torchinductor"
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

"$COSMOS_PY" - "$PORT" <<'PY'
import socket
import sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY

launch_nonce="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(24))')"
(
    cd "$COSMOS_SOURCE"
    exec setsid env \
        -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PYTHONPATH="$POLICY_PYTHONPATH" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        TORCHINDUCTOR_CACHE_DIR="$SCRATCH_DIR/torchinductor" \
        "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
        -m "$SERVER_MODULE" \
        --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
        --config-file "$TRAINING_CONFIG" \
        --output-dir "$SCRATCH_DIR/server" \
        --action-stats-path "$ACTION_STATS" \
        --wan2pt2-vae-path "$WAN2PT2_VAE" \
        --qwen-tokenizer-path "$QWEN_TOKENIZER" \
        --host 127.0.0.1 --port "$PORT" \
        --sampler unipc --seed 42 --guidance 1.0 --num-steps 30 \
        --expected-ema-key-count 986 --launch-nonce "$launch_nonce"
) >"$CONTROL_DIR/policy-server.log" 2>&1 &
SERVER_PID=$!
for _ in {1..100}; do
    kill -0 "$SERVER_PID" 2>/dev/null || die "policy server exited before process-group setup"
    SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" | tr -d '[:space:]')"
    [[ "$SERVER_PGID" == "$SERVER_PID" ]] && break
    sleep 0.05
done
[[ "$SERVER_PGID" == "$SERVER_PID" ]] || die "policy server did not enter an isolated process group"

"$COSMOS_PY" "$HELPER" capture-runtime \
    --uri "$POLICY_URI" --output "$CONTROL_DIR/runtime-receipt.json" \
    --timeout-seconds "$SERVER_READY_SECONDS" --launcher-pid "$SERVER_PID" \
    --launch-nonce "$launch_nonce" --checkpoint "$SERVING_CHECKPOINT_PATH" \
    --config "$TRAINING_CONFIG" --action-stats "$ACTION_STATS" \
    --vae "$WAN2PT2_VAE" --tokenizer "$QWEN_TOKENIZER" \
    --source-commit "$SOURCE_COMMIT" --demian-commit "$DEMIAN_COMMIT" \
    >"$CONTROL_DIR/runtime-receipt.log"
write_status RUNNING "checkpoint_loaded_and_runtime_verified"

eval_env=(
    env -u LD_PRELOAD -u LD_AUDIT -u __EGL_VENDOR_LIBRARY_DIRS
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON"
    __GLX_VENDOR_LIBRARY_NAME=nvidia PYTHONPATH="$EVAL_PYTHONPATH"
)
common_eval=(
    --episodes-start 0 --max-steps 400 --open-loop-horizon 12
    --host 127.0.0.1 --port "$PORT" --checkpoint "$SERVING_CHECKPOINT_PATH"
    --no-save-video --headless --camera-height 256 --camera-width 256
    --seed 42 --split atomic
)

timeout --signal=TERM --kill-after=30 3600 \
    "${eval_env[@]}" CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    "$ROBOCASA_PY" "$EVALUATOR" --tasks CloseBlenderLid \
    --task-shard 0 --num-shards 1 --episodes 1 \
    --results-dir "$GATE_ROOT" --run-name "$GATE_RUN_NAME" \
    "${common_eval[@]}" >"$CONTROL_DIR/one-episode-gate.log" 2>&1
"$COSMOS_PY" "$HELPER" verify-gate \
    --results-dir "$GATE_DIR" --checkpoint "$SERVING_CHECKPOINT_PATH" \
    --run-name "$GATE_RUN_NAME" --output "$CONTROL_DIR/gate-receipt.json" \
    >"$CONTROL_DIR/gate-receipt.log"
write_status RUNNING "graphics_gate_passed_formal_340_running"

for shard in {0..7}; do
    kill -0 "$SERVER_PID" 2>/dev/null || die "policy server exited before shard $shard"
    timeout --signal=TERM --kill-after=30 "$EVALUATOR_TIMEOUT_SECONDS" \
        "${eval_env[@]}" CUDA_VISIBLE_DEVICES="$shard" MUJOCO_EGL_DEVICE_ID="$shard" \
        "$ROBOCASA_PY" "$EVALUATOR" --task-shard "$shard" --num-shards 8 \
        --episodes 20 --results-dir "$RESULTS_ROOT" --run-name "$EVAL_RUN_NAME" \
        "${common_eval[@]}" >"$CONTROL_DIR/evaluator-shard${shard}.log" 2>&1
    [[ -s "$RESULTS_DIR/args_shard${shard}.json" ]] || die "missing shard $shard args"
    [[ -s "$RESULTS_DIR/results_shard${shard}.json" ]] || die "missing shard $shard results"
done

"$COSMOS_PY" "$HELPER" shutdown-server \
    --uri "$POLICY_URI" --output "$CONTROL_DIR/shutdown-receipt.json" \
    --timeout-seconds 60 --launch-nonce "$launch_nonce" \
    --runtime-receipt "$CONTROL_DIR/runtime-receipt.json" \
    >"$CONTROL_DIR/shutdown-receipt.log"
deadline=$((SECONDS + SERVER_EXIT_SECONDS))
while kill -0 "$SERVER_PID" 2>/dev/null && ((SECONDS < deadline)); do sleep 1; done
kill -0 "$SERVER_PID" 2>/dev/null && die "torchrun did not exit after controlled shutdown"
set +e
wait "$SERVER_PID"
SERVER_EXIT_CODE=$?
set -e
SERVER_PID=
SERVER_PGID=

"$COSMOS_PY" "$HELPER" summarize \
    --experiment-id "$EVAL_EXPERIMENT_ID" --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --source-commit "$SOURCE_COMMIT" --demian-commit "$DEMIAN_COMMIT" \
    --run-name "$EVAL_RUN_NAME" --results-dir "$RESULTS_DIR" \
    --checkpoint "$SERVING_CHECKPOINT_PATH" \
    --runtime-receipt "$CONTROL_DIR/runtime-receipt.json" \
    --gate-receipt "$CONTROL_DIR/gate-receipt.json" \
    --shutdown-receipt "$CONTROL_DIR/shutdown-receipt.json" \
    --server-log "$CONTROL_DIR/policy-server.log" --server-exit-code "$SERVER_EXIT_CODE" \
    --summary-csv "$EVAL_OUTPUT_ROOT/summary.csv" \
    --task-csv "$EVAL_OUTPUT_ROOT/task_success_rates.csv" \
    --evidence-json "$EVAL_OUTPUT_ROOT/evaluation_evidence.json" \
    --status-json "$EVAL_OUTPUT_ROOT/status.json" \
    >"$CONTROL_DIR/summarize.log"
TERMINAL_WRITTEN=1
printf 'B4_ROBOCASA365_EVAL_COMPLETE evidence=%s\n' "$EVAL_OUTPUT_ROOT/evaluation_evidence.json"
