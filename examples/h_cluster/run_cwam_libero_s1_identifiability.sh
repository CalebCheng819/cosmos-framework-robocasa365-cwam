#!/usr/bin/env bash
# Run the paired seven-condition routed-C protocol on a fixed Cosmos-LIBERO S1 checkpoint.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${PROTOCOL_SOURCE:?Set PROTOCOL_SOURCE to the empirical protocol snapshot.}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${PROTOCOL_COMMIT:?Set PROTOCOL_COMMIT.}"
: "${SOURCE_CHECKPOINT_URI:?Set SOURCE_CHECKPOINT_URI to the fixed DCP checkpoint.}"
: "${OSS_RUN_ROOT:?Set OSS_RUN_ROOT.}"
: "${KUBEBRAIN_RJOB_NAME:=unknown}"
: "${NUM_GPUS:=8}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${PUBLIC_ROOT:=/mnt/shared-storage-gpfs2/gpfs2-shared-public}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${FASTWAM_ROOT:=$PERSONAL_ROOT/FastWAM_yuner}"
: "${LIBERO_PYTHON:=$FASTWAM_ROOT/.conda/fastwam/bin/python}"
: "${LIBERO_REPO:=$FASTWAM_ROOT/third_party/LIBERO}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
: "${WAN_VAE_PATH:=$PUBLIC_ROOT/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=$PUBLIC_ROOT/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${TRIALS_PER_TASK:=2}"
: "${ACTION_HORIZON:=16}"
: "${MAX_STEPS:=0}"
: "${POLICY_SEED:=42}"
: "${SPLIT_SEED:=37}"
: "${JACOBIAN_PROJECTIONS:=4}"
: "${TOPK:=8}"
: "${HTTP_PORT_BASE:=18100}"
: "${NVME_ROOT:=/nvme/cwam-libero-identifiability}"

readonly CONDITIONS=(normal selected complement random_matched zero shuffle cross_episode)
readonly SUITES=(libero_spatial libero_object libero_goal libero_10)
readonly TASK_IDS="0,1,2,3,4,5,6,7,8,9"
readonly EXPECTED_TASKS=40
readonly EXPECTED_EPISODES=$((EXPECTED_TASKS * TRIALS_PER_TASK))
readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"
readonly CHECKPOINT_ROOT="$RUN_ROOT/checkpoint"
readonly RESULTS_ROOT="$RUN_ROOT/results"
readonly TRACE_ROOT="$RUN_ROOT/traces"
readonly SERVER_ROOT="$RUN_ROOT/server"
readonly LOG_ROOT="$RUN_ROOT/logs"
readonly EVIDENCE_ROOT="$RUN_ROOT/evidence"
readonly BANK_PATH="$RUN_ROOT/intervention-bank/routed_c_bank.pt"
readonly BANK_MANIFEST="$RUN_ROOT/intervention-bank/routed_c_bank.json"
readonly HANDOFF_ROOT="$PERSONAL_ROOT/runtime-handoffs/cwam-libero-identifiability/$KUBEBRAIN_RJOB_NAME"
readonly SECRET_ROOT="/dev/shm/cwam-libero-identifiability-$KUBEBRAIN_RJOB_NAME"
readonly RCLONE_CONFIG_PATH="$SECRET_ROOT/rclone.conf"
readonly SERVER_MODULE="cosmos_framework.scripts.action_policy_server_libero"
readonly EVALUATOR="$COSMOS_SOURCE/cosmos_framework/simulation/libero/closed_loop_eval.py"
readonly BANK_BUILDER="$COSMOS_SOURCE/examples/h_cluster/build_libero_intervention_bank.py"
readonly ACTION_STATS="$COSMOS_SOURCE/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
readonly LIBERO_BENCHMARK_ROOT="$LIBERO_REPO/libero/libero"
readonly LIBERO_CONFIG_ROOT="$RUN_ROOT/libero-config"
readonly COSMOS_PYTHON="$COSMOS_ENV/bin/python"

MONITOR_PID=""
CHILD_PIDS=()

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

for path in \
    "$COSMOS_PYTHON" "$LIBERO_PYTHON" "$RCLONE_BIN" "$WAN_VAE_PATH" \
    "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json" "$EVALUATOR" "$BANK_BUILDER" \
    "$ACTION_STATS" "$PROTOCOL_SOURCE/arstate_sim/empirical_protocol.py" \
    "$LIBERO_BENCHMARK_ROOT/bddl_files" "$LIBERO_BENCHMARK_ROOT/init_files"; do
    [[ -e "$path" ]] || fail "required input missing: $path"
done
[[ "$NUM_GPUS" -eq 8 ]] || fail "this protocol requires exactly eight GPUs"
[[ "$(nvidia-smi -L | wc -l)" -eq "$NUM_GPUS" ]] || fail "expected $NUM_GPUS visible GPUs"
[[ "$TRIALS_PER_TASK" -ge 2 ]] || fail "strict same-task shuffle requires at least two trials per task"
[[ "$(git -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || fail "Cosmos source revision mismatch"
[[ "$(git -C "$PROTOCOL_SOURCE" rev-parse HEAD)" == "$PROTOCOL_COMMIT" ]] || fail "protocol source revision mismatch"
[[ "$SOURCE_CHECKPOINT_URI" == eailab-hdd2:eailab/chengjuntao/* ]] || fail "checkpoint URI escapes authorized OSS prefix"
[[ "$OSS_RUN_ROOT" == eailab-hdd2:eailab/chengjuntao/* ]] || fail "run root escapes authorized OSS prefix"

mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$TRACE_ROOT" "$SERVER_ROOT" \
    "$LOG_ROOT" "$EVIDENCE_ROOT" "$(dirname "$BANK_PATH")" "$LIBERO_CONFIG_ROOT"

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
    done
    [[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    done
    [[ -z "$MONITOR_PID" ]] || wait "$MONITOR_PID" 2>/dev/null || true
    if [[ -s "$RCLONE_CONFIG_PATH" ]]; then
        "$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 || true
        "$RCLONE_BIN" copy "$LOG_ROOT" "$OSS_RUN_ROOT/logs" --transfers=8 --checkers=16 || true
    fi
    rm -rf "$SECRET_ROOT"
    rm -f "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json" "$HANDOFF_ROOT/rclone-config.cms.der"
    rmdir "$HANDOFF_ROOT" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p "$SECRET_ROOT" "$HANDOFF_ROOT"
chmod 700 "$SECRET_ROOT" "$HANDOFF_ROOT"
openssl req -x509 -newkey rsa:3072 -nodes -days 1 -subj "/CN=$KUBEBRAIN_RJOB_NAME" \
    -keyout "$SECRET_ROOT/recipient.key" -out "$HANDOFF_ROOT/recipient.pem" >/dev/null 2>&1
export KUBEBRAIN_RJOB_NAME OSS_RUN_ROOT SOURCE_CHECKPOINT_URI
"$COSMOS_PYTHON" - "$HANDOFF_ROOT/request.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "job_id": os.environ["KUBEBRAIN_RJOB_NAME"],
    "oss_run_root": os.environ["OSS_RUN_ROOT"],
    "source_checkpoint_uri": os.environ["SOURCE_CHECKPOINT_URI"],
}, indent=2, sort_keys=True) + "\n")
PY
chmod 444 "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json"
echo "RCLONE_HANDOFF_WAITING root=$HANDOFF_ROOT"
for _ in $(seq 1 240); do
    [[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] && break
    sleep 5
done
[[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] || fail "timed out waiting for rclone handoff"
openssl cms -decrypt -binary -inform DER -in "$HANDOFF_ROOT/rclone-config.cms.der" \
    -recip "$HANDOFF_ROOT/recipient.pem" -inkey "$SECRET_ROOT/recipient.key" -out "$RCLONE_CONFIG_PATH"
chmod 400 "$RCLONE_CONFIG_PATH"
export RCLONE_CONFIG="$RCLONE_CONFIG_PATH"
[[ "$("$RCLONE_BIN" lsf "$SOURCE_CHECKPOINT_URI/model" --files-only --include .metadata)" == ".metadata" ]] || \
    fail "authorized checkpoint has no model/.metadata"

echo "CHECKPOINT_STAGE_START source=$SOURCE_CHECKPOINT_URI"
"$RCLONE_BIN" copy "$SOURCE_CHECKPOINT_URI" "$CHECKPOINT_ROOT" \
    --transfers=16 --checkers=32 --multi-thread-streams=8
[[ -s "$CHECKPOINT_ROOT/model/.metadata" ]] || fail "staged checkpoint has no model/.metadata"
echo "CHECKPOINT_STAGE_COMPLETE local=$CHECKPOINT_ROOT"

"$COSMOS_PYTHON" - "$LIBERO_CONFIG_ROOT/config.yaml" "$LIBERO_BENCHMARK_ROOT" <<'PY'
import sys
from pathlib import Path

output, root = map(Path, sys.argv[1:])
values = {
    "benchmark_root": root,
    "bddl_files": root / "bddl_files",
    "init_states": root / "init_files",
    "datasets": root.parent / "datasets",
    "assets": root / "assets",
}
missing = [str(path) for key, path in values.items() if key != "datasets" and not path.exists()]
if missing:
    raise SystemExit(f"missing LIBERO runtime paths: {missing}")
output.write_text("".join(f"{key}: {value}\n" for key, value in values.items()))
PY

export PATH="$COSMOS_ENV/bin:$(dirname "$RCLONE_BIN"):$PATH"
export WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="$POLICY_SEED" NCCL_NVLS_ENABLE=0
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
export EXPERIMENT_ID KUBEBRAIN_RJOB_NAME SOURCE_COMMIT PROTOCOL_COMMIT SOURCE_CHECKPOINT_URI OSS_RUN_ROOT
export TRIALS_PER_TASK ACTION_HORIZON MAX_STEPS POLICY_SEED SPLIT_SEED JACOBIAN_PROJECTIONS TOPK

"$COSMOS_PYTHON" - "$EVIDENCE_ROOT/run_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = (
    "EXPERIMENT_ID", "KUBEBRAIN_RJOB_NAME", "SOURCE_COMMIT", "PROTOCOL_COMMIT",
    "SOURCE_CHECKPOINT_URI", "OSS_RUN_ROOT", "TRIALS_PER_TASK", "ACTION_HORIZON",
    "MAX_STEPS", "POLICY_SEED", "SPLIT_SEED", "JACOBIAN_PROJECTIONS", "TOPK",
)
Path(sys.argv[1]).write_text(json.dumps({key: os.environ[key] for key in keys}, indent=2, sort_keys=True) + "\n")
PY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
    --format=csv -l 30 >"$EVIDENCE_ROOT/nvidia-smi.csv" 2>&1 &
MONITOR_PID=$!

wait_for_server() {
    local pid="$1" url="$2" log="$3"
    for _ in $(seq 1 180); do
        kill -0 "$pid" 2>/dev/null || { tail -100 "$log" >&2; return 1; }
        if "$COSMOS_PYTHON" - "$url" >/dev/null 2>&1 <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    payload = json.load(response)
if not isinstance(payload, dict):
    raise SystemExit(1)
PY
        then
            return 0
        fi
        sleep 5
    done
    tail -100 "$log" >&2
    return 1
}

run_condition() (
    local condition="$1" gpu="$2" port="$3"
    local condition_root="$RESULTS_ROOT/$condition"
    local trace_dir="$TRACE_ROOT/$condition"
    local server_dir="$SERVER_ROOT/$condition"
    local server_log="$LOG_ROOT/server-$condition.log"
    local server_pid=""
    mkdir -p "$condition_root" "$trace_dir" "$server_dir"

    stop_server() {
        [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true
        [[ -z "$server_pid" ]] || wait "$server_pid" 2>/dev/null || true
    }
    trap stop_server EXIT TERM INT

    server_args=(
        --experiment action_policy_libero_all_s1_nano
        --checkpoint-path "$CHECKPOINT_ROOT"
        --output-dir "$server_dir"
        --sampler unipc
        --seed "$POLICY_SEED"
        --guidance 1.0
        --num-steps 30
        --fps 20
        --raw-action-dim 10
        --action-normalization quantile_rot
        --action-stats-path "$ACTION_STATS"
        --host 127.0.0.1
        --port "$port"
        --action-relevance-intervention "$condition"
        --action-relevance-seed "$POLICY_SEED"
        --action-relevance-trace-dir "$trace_dir"
    )
    if [[ "$condition" == normal ]]; then
        server_args+=(--action-relevance-jacobian-projections "$JACOBIAN_PROJECTIONS")
    elif [[ "$condition" == shuffle || "$condition" == cross_episode ]]; then
        server_args+=(--action-relevance-intervention-bank "$BANK_PATH")
    fi

    CUDA_VISIBLE_DEVICES="$gpu" MASTER_ADDR=127.0.0.1 MASTER_PORT=$((port + 1000)) \
        RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 PYTHONPATH="$COSMOS_SOURCE" \
        "$COSMOS_PYTHON" -m "$SERVER_MODULE" "${server_args[@]}" >"$server_log" 2>&1 &
    server_pid=$!
    wait_for_server "$server_pid" "http://127.0.0.1:$port/info" "$server_log"

    for suite in "${SUITES[@]}"; do
        local suite_root="$condition_root/$suite"
        local suite_log="$LOG_ROOT/eval-$condition-$suite.log"
        mkdir -p "$suite_root"
        CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
            MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0 \
            PYTHONPATH="$COSMOS_SOURCE:$LIBERO_REPO" \
            "$LIBERO_PYTHON" "$EVALUATOR" \
                --server_url "http://127.0.0.1:$port" \
                --task_suite "$suite" \
                --task_ids "$TASK_IDS" \
                --num_trials_per_task "$TRIALS_PER_TASK" \
                --num_envs 1 \
                --camera agentview,wrist \
                --image_size 256 \
                --env_image_size 256 \
                --action_horizon "$ACTION_HORIZON" \
                --max_steps "$MAX_STEPS" \
                --action_space frame_wise_relative \
                --rotation_space 6d \
                --action_dim 10 \
                --gripper_mode zero_one \
                --warmup_steps 10 \
                --seed "$POLICY_SEED" \
                --intervention_condition "$condition" \
                --trace_fixed_horizon \
                --mujoco_gl egl \
                --render_gpu_device_id 0 \
                --wait_timeout 120 \
                --timeout 300 \
                --output_dir "$suite_root" >"$suite_log" 2>&1
        [[ -s "$suite_root/COMPLETE" ]] || { tail -100 "$suite_log" >&2; return 1; }
    done

    stop_server
    server_pid=""
    "$COSMOS_PYTHON" - "$condition_root/COMPLETE" "$condition" "$EXPECTED_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

output, condition, episodes = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
summaries = [json.loads(path.read_text()) for path in sorted(output.parent.glob("*/summary.json"))]
observed = sum(int(item["total_episodes"]) for item in summaries)
if len(summaries) != 4 or observed != episodes or any(item.get("terminal_status") != "COMPLETE" for item in summaries):
    raise SystemExit(f"condition incomplete: summaries={len(summaries)} episodes={observed}")
output.write_text(json.dumps({"status": "COMPLETE", "condition": condition, "episodes": observed}, sort_keys=True) + "\n")
PY
    "$RCLONE_BIN" copy "$condition_root" "$OSS_RUN_ROOT/results/$condition" --transfers=8 --checkers=16
    "$RCLONE_BIN" copy "$trace_dir" "$OSS_RUN_ROOT/traces/$condition" --transfers=8 --checkers=16
)

echo "CONDITION_START condition=normal gpu=0"
run_condition normal 0 "$HTTP_PORT_BASE"
echo "CONDITION_COMPLETE condition=normal"

PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PYTHON" "$BANK_BUILDER" \
    --trace-dir "$TRACE_ROOT/normal" --output "$BANK_PATH" --manifest "$BANK_MANIFEST" \
    --seed "$POLICY_SEED"
"$RCLONE_BIN" copy "$(dirname "$BANK_PATH")" "$OSS_RUN_ROOT/intervention-bank" --transfers=8 --checkers=16

for index in $(seq 1 6); do
    condition="${CONDITIONS[$index]}"
    echo "CONDITION_START condition=$condition gpu=$index"
    (run_condition "$condition" "$index" $((HTTP_PORT_BASE + index))) &
    CHILD_PIDS+=("$!")
done

parallel_status=0
for index in "${!CHILD_PIDS[@]}"; do
    if ! wait "${CHILD_PIDS[$index]}"; then
        echo "ERROR: condition=${CONDITIONS[$((index + 1))]} failed" >&2
        parallel_status=1
    fi
done
CHILD_PIDS=()
[[ "$parallel_status" -eq 0 ]] || fail "one or more intervention conditions failed"

completion_args=()
trace_args=()
closed_loop_args=()
for condition in "${CONDITIONS[@]}"; do
    completion_args+=(--condition "$condition=$RESULTS_ROOT/$condition")
    trace_args+=(--trace "$TRACE_ROOT/$condition")
    closed_loop_args+=("$RESULTS_ROOT/$condition")
done

PYTHONPATH="$PROTOCOL_SOURCE" "$COSMOS_PYTHON" -m arstate_sim.closed_loop_completion \
    "${completion_args[@]}" --expected-episodes "$EXPECTED_EPISODES" \
    --expected-tasks "$EXPECTED_TASKS" --output "$EVIDENCE_ROOT/closed_loop_completion.json"
PYTHONPATH="$PROTOCOL_SOURCE" "$COSMOS_PYTHON" -m arstate_sim.empirical_trace_export \
    "${trace_args[@]}" --output "$EVIDENCE_ROOT/aligned_protocol.npz" \
    --manifest "$EVIDENCE_ROOT/aligned_protocol.manifest.json" --layers all --require-all-conditions
PYTHONPATH="$PROTOCOL_SOURCE" "$COSMOS_PYTHON" -m arstate_sim.empirical_protocol \
    --input "$EVIDENCE_ROOT/aligned_protocol.npz" \
    --output "$EVIDENCE_ROOT/protocol_results.json" \
    --closed-loop "${closed_loop_args[@]}" --eval-fraction 0.25 --seed "$SPLIT_SEED" --topk "$TOPK"

"$COSMOS_PYTHON" - "$EVIDENCE_ROOT/COMPLETE" "$EVIDENCE_ROOT/closed_loop_completion.json" \
    "$EVIDENCE_ROOT/protocol_results.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, completion_path, result_path = map(Path, sys.argv[1:])
completion = json.loads(completion_path.read_text())
results = json.loads(result_path.read_text())
if completion.get("status") != "COMPLETE" or results.get("schema_version") != 1:
    raise SystemExit("terminal protocol evidence is incomplete")
output.write_text(json.dumps({
    "status": "COMPLETE",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "paired_episodes": completion["paired_episode_keys"],
}, sort_keys=True) + "\n")
PY

"$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 \
    --exclude COMPLETE
"$RCLONE_BIN" copy "$LOG_ROOT" "$OSS_RUN_ROOT/logs" --transfers=8 --checkers=16
"$RCLONE_BIN" copyto "$EVIDENCE_ROOT/COMPLETE" "$OSS_RUN_ROOT/evidence/COMPLETE"
echo "PROTOCOL_COMPLETE experiment=$EXPERIMENT_ID paired_episodes=$EXPECTED_EPISODES"
