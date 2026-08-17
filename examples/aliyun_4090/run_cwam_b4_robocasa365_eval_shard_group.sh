#!/usr/bin/env bash
# Run one fixed group of RoboCasa365 logical shards against one policy server.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

required=(
    ROBOCASA_PY EVALUATOR EVAL_PYTHONPATH EVAL_EGL_VENDOR_JSON
    SERVING_CHECKPOINT_PATH RESULTS_ROOT RESULTS_DIR EVAL_RUN_NAME
    EVALUATOR_TIMEOUT_SECONDS EVAL_GROUP_CONTROL_DIR
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done

[[ "$#" -ge 4 ]] || {
    printf 'Usage: %s LABEL PORT SERVER_PID SHARD...\n' "$0" >&2
    exit 2
}

readonly SERVER_LABEL="$1"
readonly PORT="$2"
readonly SERVER_PID="$3"
shift 3
readonly -a SHARDS=("$@")

[[ "$SERVER_LABEL" =~ ^server-[ab]$ ]] || { printf 'ERROR: invalid server label\n' >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid port\n' >&2; exit 2; }
[[ "$SERVER_PID" =~ ^[1-9][0-9]*$ ]] || { printf 'ERROR: invalid server PID\n' >&2; exit 2; }
[[ -f "$ROBOCASA_PY" && ! -L "$ROBOCASA_PY" ]] || { printf 'ERROR: missing RoboCasa Python\n' >&2; exit 2; }
[[ -f "$EVALUATOR" && ! -L "$EVALUATOR" ]] || { printf 'ERROR: missing evaluator\n' >&2; exit 2; }
[[ -d "$EVAL_GROUP_CONTROL_DIR" && ! -L "$EVAL_GROUP_CONTROL_DIR" ]] || {
    printf 'ERROR: missing group control directory\n' >&2
    exit 2
}

declare -A seen=()
for shard in "${SHARDS[@]}"; do
    [[ "$shard" =~ ^[0-7]$ ]] || { printf 'ERROR: invalid shard %s\n' "$shard" >&2; exit 2; }
    [[ -z "${seen[$shard]:-}" ]] || { printf 'ERROR: duplicate shard %s\n' "$shard" >&2; exit 2; }
    seen[$shard]=1
done

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

for shard in "${SHARDS[@]}"; do
    kill -0 "$SERVER_PID" 2>/dev/null || {
        printf 'ERROR: %s policy server exited before shard %s\n' "$SERVER_LABEL" "$shard" >&2
        exit 1
    }
    timeout --signal=TERM --kill-after=30 "$EVALUATOR_TIMEOUT_SECONDS" \
        "${eval_env[@]}" CUDA_VISIBLE_DEVICES="$shard" MUJOCO_EGL_DEVICE_ID="$shard" \
        "$ROBOCASA_PY" "$EVALUATOR" --task-shard "$shard" --num-shards 8 \
        --episodes 20 --results-dir "$RESULTS_ROOT" --run-name "$EVAL_RUN_NAME" \
        "${common_eval[@]}" >"$EVAL_GROUP_CONTROL_DIR/evaluator-shard${shard}.log" 2>&1
    [[ -s "$RESULTS_DIR/args_shard${shard}.json" ]] || {
        printf 'ERROR: missing shard %s args\n' "$shard" >&2
        exit 1
    }
    [[ -s "$RESULTS_DIR/results_shard${shard}.json" ]] || {
        printf 'ERROR: missing shard %s results\n' "$shard" >&2
        exit 1
    }
done

printf 'SHARD_GROUP_COMPLETE label=%s shards=%s\n' "$SERVER_LABEL" "${SHARDS[*]}"
