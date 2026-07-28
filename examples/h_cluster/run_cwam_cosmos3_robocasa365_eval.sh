#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run one immutable CWAM/Cosmos3 RoboCasa365 gate or formal evaluation.
#
# The wrapper launches one policy server and one simulator shard per visible
# H200, captures every live server's metadata before rollouts, creates the
# post-server-start eval contract, and fails closed unless the two required CSV
# artifacts pass completeness/provenance validation.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot}"
: "${COSMOS_ENV:?Set COSMOS_ENV to the pinned Cosmos CUDA environment}"
: "${ROBOCASA_ENV:?Set ROBOCASA_ENV to the pinned RoboCasa evaluation environment}"
: "${DEMIAN_REPO:?Set DEMIAN_REPO to the pinned DeMiAn checkout}"
: "${ROBOCASA_REPO:?Set ROBOCASA_REPO to the pinned RoboCasa checkout}"
: "${ROBOSUITE_REPO:?Set ROBOSUITE_REPO to the pinned robosuite checkout}"
: "${SERVING_CHECKPOINT_PATH:?Set SERVING_CHECKPOINT_PATH to a consolidated checkpoint directory}"
: "${CHECKPOINT_MANIFEST:?Set CHECKPOINT_MANIFEST to the serving export manifest}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG to the exact training config YAML}"
: "${ACTION_STATS:?Set ACTION_STATS to the RoboCasa365 quantile_rot stats JSON}"
: "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to the immutable source manifest}"
: "${EVAL_OUTPUT_ROOT:?Set EVAL_OUTPUT_ROOT to a new per-experiment directory}"
: "${EVAL_EXPERIMENT_ID:?Set EVAL_EXPERIMENT_ID}"
: "${EVAL_LOGICAL_RUN_ID:?Set EVAL_LOGICAL_RUN_ID}"
: "${EVAL_RUN_NAME:?Set EVAL_RUN_NAME}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?Set EXPECTED_SOURCE_MANIFEST_SHA256}"
: "${EXPECTED_CHECKPOINT_MANIFEST_SHA256:?Set EXPECTED_CHECKPOINT_MANIFEST_SHA256}"
: "${EXPECTED_TRAINING_CONFIG_SHA256:?Set EXPECTED_TRAINING_CONFIG_SHA256}"
: "${EXPECTED_ACTION_STATS_SHA256:?Set EXPECTED_ACTION_STATS_SHA256}"
: "${EXPECTED_DEMIAN_COMMIT:?Set EXPECTED_DEMIAN_COMMIT}"
: "${EXPECTED_DEMIAN_EVALUATOR_SHA256:?Set EXPECTED_DEMIAN_EVALUATOR_SHA256}"
: "${EXPECTED_DEMIAN_TASK_REGISTRY_SHA256:?Set EXPECTED_DEMIAN_TASK_REGISTRY_SHA256}"
: "${EXPECTED_ROBOCASA_COMMIT:?Set EXPECTED_ROBOCASA_COMMIT}"
: "${EXPECTED_ROBOSUITE_COMMIT:?Set EXPECTED_ROBOSUITE_COMMIT}"
: "${MINIMUM_MEAN_SUCCESS_RATE:?Pre-register MINIMUM_MEAN_SUCCESS_RATE}"
: "${MINIMUM_SUCCESSFUL_TASKS:?Pre-register MINIMUM_SUCCESSFUL_TASKS}"

: "${EVAL_STAGE:=one_episode_gate}"
: "${EVAL_SPLIT:=atomic}"
: "${EVAL_TASKS_CSV:=}"
: "${EPISODES_PER_TASK:=1}"
: "${NUM_SHARDS:=1}"
: "${BASE_SEED:=42}"
: "${MAX_STEPS:=400}"
: "${OPEN_LOOP_HORIZON:=12}"
: "${POLICY_SAMPLER:=unipc}"
: "${POLICY_SEED:=42}"
: "${POLICY_GUIDANCE:=1.0}"
: "${POLICY_NUM_STEPS:=30}"
: "${BASE_PORT:=16000}"
: "${SERVER_READY_TIMEOUT_SECONDS:=1800}"
: "${ALLOWED_OUTPUT_PARENT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs}"
: "${EVAL_NVME_ROOT:=/nvme/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID}"

readonly DEMIAN_EVALUATOR_REL="demian_wam/eval_utils/run_robocasa_eval.py"
readonly DEMIAN_TASK_REGISTRY_REL="demian_wam/eval_utils/robocasa_task_list.py"
readonly SERVER_MODULE="cosmos_framework.scripts.action_policy_server_robocasa365"
readonly SERVER_MODULE_REL="cosmos_framework/scripts/action_policy_server_robocasa365.py"
readonly RECEIPT_HELPER_REL="examples/h_cluster/capture_robocasa365_server_receipt.py"
readonly EVIDENCE_HELPER_REL="examples/h_cluster/cwam_robocasa365_eval_evidence.py"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_positive_int() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer, got $value"
}

require_nonnegative_int() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$label must be a non-negative integer, got $value"
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

verify_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA256 is malformed"
    [[ -f "$path" ]] || die "$label file does not exist: $path"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] ||
        die "$label SHA256 mismatch: expected=$expected actual=$actual path=$path"
}

canonical_existing() {
    realpath -e -- "$1"
}

require_positive_int NUM_SHARDS "$NUM_SHARDS"
require_positive_int EPISODES_PER_TASK "$EPISODES_PER_TASK"
require_positive_int MAX_STEPS "$MAX_STEPS"
require_positive_int OPEN_LOOP_HORIZON "$OPEN_LOOP_HORIZON"
require_positive_int POLICY_NUM_STEPS "$POLICY_NUM_STEPS"
require_positive_int SERVER_READY_TIMEOUT_SECONDS "$SERVER_READY_TIMEOUT_SECONDS"
require_nonnegative_int BASE_SEED "$BASE_SEED"
require_nonnegative_int POLICY_SEED "$POLICY_SEED"
require_nonnegative_int MINIMUM_SUCCESSFUL_TASKS "$MINIMUM_SUCCESSFUL_TASKS"
[[ "$EVAL_STAGE" == "one_episode_gate" || "$EVAL_STAGE" == "formal" ]] ||
    die "EVAL_STAGE must be one_episode_gate or formal"
[[ "$EVAL_SPLIT" == "atomic" || "$EVAL_SPLIT" == "composite" ]] ||
    die "EVAL_SPLIT must be atomic or composite"
[[ "$POLICY_SAMPLER" == "unipc" || "$POLICY_SAMPLER" == "edm" ]] ||
    die "POLICY_SAMPLER must be unipc or edm"
for commit in \
    "$EXPECTED_SOURCE_COMMIT" \
    "$EXPECTED_DEMIAN_COMMIT" \
    "$EXPECTED_ROBOCASA_COMMIT" \
    "$EXPECTED_ROBOSUITE_COMMIT"; do
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "Expected Git commits must be full lowercase SHA-1 values"
done
if [[ "$EVAL_STAGE" == "one_episode_gate" ]]; then
    [[ "$EPISODES_PER_TASK" == 1 ]] || die "one_episode_gate requires EPISODES_PER_TASK=1"
    [[ -n "$EVAL_TASKS_CSV" ]] || die "one_episode_gate requires EVAL_TASKS_CSV"
else
    [[ "$EPISODES_PER_TASK" == 20 ]] || die "formal evaluation requires EPISODES_PER_TASK=20"
    [[ "$NUM_SHARDS" == 8 ]] || die "formal evaluation requires NUM_SHARDS=8"
    [[ -z "$EVAL_TASKS_CSV" ]] || die "formal evaluation must use the complete canonical split"
fi
[[ "$EVAL_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "EVAL_EXPERIMENT_ID is not path-safe"
[[ "$EVAL_LOGICAL_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "EVAL_LOGICAL_RUN_ID is not path-safe"
[[ "$EVAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "EVAL_RUN_NAME is not path-safe"

COSMOS_SOURCE="$(canonical_existing "$COSMOS_SOURCE")"
COSMOS_ENV="$(canonical_existing "$COSMOS_ENV")"
ROBOCASA_ENV="$(canonical_existing "$ROBOCASA_ENV")"
DEMIAN_REPO="$(canonical_existing "$DEMIAN_REPO")"
ROBOCASA_REPO="$(canonical_existing "$ROBOCASA_REPO")"
ROBOSUITE_REPO="$(canonical_existing "$ROBOSUITE_REPO")"
SERVING_CHECKPOINT_PATH="$(canonical_existing "$SERVING_CHECKPOINT_PATH")"
if [[ -d "$SERVING_CHECKPOINT_PATH/model" ]]; then
    SERVING_CHECKPOINT_PATH="$(canonical_existing "$SERVING_CHECKPOINT_PATH/model")"
fi
CHECKPOINT_MANIFEST="$(canonical_existing "$CHECKPOINT_MANIFEST")"
TRAINING_CONFIG="$(canonical_existing "$TRAINING_CONFIG")"
ACTION_STATS="$(canonical_existing "$ACTION_STATS")"
SOURCE_MANIFEST="$(canonical_existing "$SOURCE_MANIFEST")"
ALLOWED_OUTPUT_PARENT="$(canonical_existing "$ALLOWED_OUTPUT_PARENT")"

[[ -x "$COSMOS_ENV/bin/python" ]] || die "COSMOS_ENV Python is not executable"
[[ -x "$ROBOCASA_ENV/bin/python" ]] || die "ROBOCASA_ENV Python is not executable"
[[ -d "$SERVING_CHECKPOINT_PATH" ]] || die "SERVING_CHECKPOINT_PATH must be a directory"
[[ -f "$COSMOS_SOURCE/$SERVER_MODULE_REL" ]] || die "Missing RoboCasa365 policy server module"
[[ -f "$COSMOS_SOURCE/$RECEIPT_HELPER_REL" ]] || die "Missing server receipt helper"
[[ -f "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" ]] || die "Missing eval evidence helper"
[[ -f "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" ]] || die "Missing DeMiAn evaluator"
[[ -f "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL" ]] || die "Missing DeMiAn task registry"

SOURCE_COMMIT="$(git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse HEAD)"
DEMIAN_COMMIT="$(git -c safe.directory="$DEMIAN_REPO" -C "$DEMIAN_REPO" rev-parse HEAD)"
ROBOCASA_COMMIT="$(git -c safe.directory="$ROBOCASA_REPO" -C "$ROBOCASA_REPO" rev-parse HEAD)"
ROBOSUITE_COMMIT="$(git -c safe.directory="$ROBOSUITE_REPO" -C "$ROBOSUITE_REPO" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" == "$EXPECTED_SOURCE_COMMIT" ]] ||
    die "Cosmos source commit mismatch: expected=$EXPECTED_SOURCE_COMMIT actual=$SOURCE_COMMIT"
[[ "$DEMIAN_COMMIT" == "$EXPECTED_DEMIAN_COMMIT" ]] ||
    die "DeMiAn commit mismatch: expected=$EXPECTED_DEMIAN_COMMIT actual=$DEMIAN_COMMIT"
[[ "$ROBOCASA_COMMIT" == "$EXPECTED_ROBOCASA_COMMIT" ]] ||
    die "RoboCasa commit mismatch: expected=$EXPECTED_ROBOCASA_COMMIT actual=$ROBOCASA_COMMIT"
[[ "$ROBOSUITE_COMMIT" == "$EXPECTED_ROBOSUITE_COMMIT" ]] ||
    die "robosuite commit mismatch: expected=$EXPECTED_ROBOSUITE_COMMIT actual=$ROBOSUITE_COMMIT"
for repo in "$COSMOS_SOURCE" "$DEMIAN_REPO" "$ROBOCASA_REPO" "$ROBOSUITE_REPO"; do
    [[ -z "$(git -c safe.directory="$repo" -C "$repo" status --porcelain --untracked-files=all)" ]] ||
        die "Pinned source checkout is dirty: $repo"
done

verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"
verify_sha256 checkpoint_manifest "$CHECKPOINT_MANIFEST" "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"
verify_sha256 training_config "$TRAINING_CONFIG" "$EXPECTED_TRAINING_CONFIG_SHA256"
verify_sha256 action_stats "$ACTION_STATS" "$EXPECTED_ACTION_STATS_SHA256"
verify_sha256 demian_evaluator \
    "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" \
    "$EXPECTED_DEMIAN_EVALUATOR_SHA256"
verify_sha256 demian_task_registry \
    "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL" \
    "$EXPECTED_DEMIAN_TASK_REGISTRY_SHA256"

EVAL_OUTPUT_ROOT="$(realpath -m -- "$EVAL_OUTPUT_ROOT")"
case "$EVAL_OUTPUT_ROOT/" in
    "$ALLOWED_OUTPUT_PARENT"/*/) ;;
    *) die "EVAL_OUTPUT_ROOT must be a strict child of $ALLOWED_OUTPUT_PARENT" ;;
esac
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || die "Refusing to reuse EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
mkdir -- "$EVAL_OUTPUT_ROOT"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly RESULTS_DIR="$RESULTS_ROOT/$EVAL_RUN_NAME"
readonly CONTRACT_PATH="$CONTROL_DIR/eval-contract.json"
readonly SUMMARY_CSV="$RESULTS_DIR/summary.csv"
readonly TASK_SUCCESS_RATES_CSV="$RESULTS_DIR/task_success_rates.csv"
readonly EVIDENCE_JSON="$RESULTS_DIR/evaluation_evidence.json"
mkdir -- "$CONTROL_DIR" "$RESULTS_ROOT"

EVAL_NVME_ROOT="$(realpath -m -- "$EVAL_NVME_ROOT")"
case "$EVAL_NVME_ROOT/" in
    /nvme/*/) ;;
    *) die "EVAL_NVME_ROOT must be a strict child of /nvme" ;;
esac
[[ "$(findmnt -n -T /nvme -o TARGET)" == /nvme ]] || die "/nvme is not a distinct mount target"
[[ "$(findmnt -n -T /nvme -o SOURCE)" == /dev/nvme1n1 ]] || die "/nvme is not backed by /dev/nvme1n1"
[[ "$(findmnt -n -T /nvme -o FSTYPE)" == xfs ]] || die "/nvme is not xfs"
[[ ! -e "$EVAL_NVME_ROOT" ]] || die "Refusing to reuse EVAL_NVME_ROOT: $EVAL_NVME_ROOT"
mkdir -- "$EVAL_NVME_ROOT"

readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"

"$COSMOS_PY" - "$NUM_SHARDS" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != expected:
    raise SystemExit(f"visible GPU count mismatch: expected={expected}, actual={torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(expected)]
if any("H200" not in name for name in names):
    raise SystemExit(f"all visible GPUs must be H200: {names}")
print(f"EVAL_CUDA_READY count={expected} names={names} torch={torch.__version__} cuda={torch.version.cuda}")
PY

"$COSMOS_PY" - "$BASE_PORT" "$NUM_SHARDS" <<'PY'
import socket
import sys

base_port = int(sys.argv[1])
count = int(sys.argv[2])
sockets = []
try:
    for port in range(base_port, base_port + count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
print(f"EVAL_PORTS_READY base={base_port} count={count}")
PY

SERVER_PIDS=()
EVALUATOR_PIDS=()
LAUNCH_NONCES=()
cleanup() {
    local status="$1"
    local pid
    trap - EXIT INT TERM
    for pid in "${EVALUATOR_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${EVALUATOR_PIDS[@]}" "${SERVER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    port=$((BASE_PORT + shard))
    server_log="$CONTROL_DIR/server-shard${shard}.log"
    server_output="$EVAL_NVME_ROOT/server-output-shard${shard}"
    compile_cache="$EVAL_NVME_ROOT/torchinductor-shard${shard}"
    launch_nonce="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
    LAUNCH_NONCES+=("$launch_nonce")
    mkdir -- "$compile_cache"
    (
        cd "$COSMOS_SOURCE"
        exec env \
            -u WORLD_SIZE \
            -u RANK \
            -u LOCAL_RANK \
            -u MASTER_ADDR \
            -u MASTER_PORT \
            CUDA_VISIBLE_DEVICES="$shard" \
            PYTHONPATH="$POLICY_PYTHONPATH" \
            HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR="$compile_cache" \
            "$COSMOS_PY" -m "$SERVER_MODULE" \
            --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
            --config-file "$TRAINING_CONFIG" \
            --output-dir "$server_output" \
            --action-stats-path "$ACTION_STATS" \
            --host 127.0.0.1 \
            --port "$port" \
            --sampler "$POLICY_SAMPLER" \
            --seed "$POLICY_SEED" \
            --guidance "$POLICY_GUIDANCE" \
            --num-steps "$POLICY_NUM_STEPS" \
            --launch-nonce "$launch_nonce"
    ) >"$server_log" 2>&1 &
    SERVER_PIDS+=("$!")
done

RECEIPTS=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    port=$((BASE_PORT + shard))
    receipt="$CONTROL_DIR/server-runtime-receipt-shard${shard}.json"
    RECEIPTS+=("$receipt")
    PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$RECEIPT_HELPER_REL" \
        --uri "ws://127.0.0.1:$port" \
        --output "$receipt" \
        --timeout-seconds "$SERVER_READY_TIMEOUT_SECONDS" \
        --checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
        --training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
        --action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
        --source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --demian-commit "$EXPECTED_DEMIAN_COMMIT" \
        --shard-index "$shard" \
        --expected-launch-nonce "${LAUNCH_NONCES[$shard]}" \
        --expected-server-pid "${SERVER_PIDS[$shard]}" \
        --watch-pid "${SERVER_PIDS[$shard]}" \
        >"$CONTROL_DIR/server-receipt-shard${shard}.log"
done
echo "ALL_POLICY_SERVERS_READY count=$NUM_SHARDS"

CONTRACT_ARGS=(
    create-contract
    --output "$CONTRACT_PATH"
    --experiment-id "$EVAL_EXPERIMENT_ID"
    --logical-run-id "$EVAL_LOGICAL_RUN_ID"
    --source-commit "$EXPECTED_SOURCE_COMMIT"
    --source-manifest "$SOURCE_MANIFEST"
    --cosmos-server-module "$COSMOS_SOURCE/$SERVER_MODULE_REL"
    --checkpoint-path "$SERVING_CHECKPOINT_PATH"
    --checkpoint-manifest "$CHECKPOINT_MANIFEST"
    --training-config "$TRAINING_CONFIG"
    --action-stats "$ACTION_STATS"
    --demian-repo "$DEMIAN_REPO"
    --demian-commit "$EXPECTED_DEMIAN_COMMIT"
    --demian-evaluator "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL"
    --demian-task-registry "$DEMIAN_REPO/$DEMIAN_TASK_REGISTRY_REL"
    --robocasa-repo "$ROBOCASA_REPO"
    --robocasa-commit "$EXPECTED_ROBOCASA_COMMIT"
    --robosuite-repo "$ROBOSUITE_REPO"
    --robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT"
    --stage "$EVAL_STAGE"
    --split "$EVAL_SPLIT"
    --episodes-per-task "$EPISODES_PER_TASK"
    --base-seed "$BASE_SEED"
    --max-steps "$MAX_STEPS"
    --open-loop-horizon "$OPEN_LOOP_HORIZON"
    --num-shards "$NUM_SHARDS"
    --policy-sampler "$POLICY_SAMPLER"
    --policy-seed "$POLICY_SEED"
    --policy-guidance "$POLICY_GUIDANCE"
    --policy-num-steps "$POLICY_NUM_STEPS"
    --results-dir "$RESULTS_DIR"
    --run-name "$EVAL_RUN_NAME"
    --minimum-mean-success-rate "$MINIMUM_MEAN_SUCCESS_RATE"
    --minimum-successful-tasks "$MINIMUM_SUCCESSFUL_TASKS"
)
for receipt in "${RECEIPTS[@]}"; do
    CONTRACT_ARGS+=(--server-runtime-receipt "$receipt")
done
if [[ "$EVAL_STAGE" == "one_episode_gate" ]]; then
    IFS=',' read -r -a EVAL_TASKS <<<"$EVAL_TASKS_CSV"
    CONTRACT_ARGS+=(--tasks "${EVAL_TASKS[@]}")
fi
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
    "${CONTRACT_ARGS[@]}" >"$CONTROL_DIR/create-contract.log"
echo "EVAL_CONTRACT_READY path=$CONTRACT_PATH sha256=$(sha256_file "$CONTRACT_PATH")"

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    port=$((BASE_PORT + shard))
    evaluator_log="$CONTROL_DIR/evaluator-shard${shard}.log"
    EVALUATOR_ARGS=(
        --task-shard "$shard"
        --num-shards "$NUM_SHARDS"
        --episodes "$EPISODES_PER_TASK"
        --episodes-start 0
        --max-steps "$MAX_STEPS"
        --open-loop-horizon "$OPEN_LOOP_HORIZON"
        --host 127.0.0.1
        --port "$port"
        --checkpoint "$SERVING_CHECKPOINT_PATH"
        --results-dir "$RESULTS_ROOT"
        --run-name "$EVAL_RUN_NAME"
        --no-save-video
        --headless
        --camera-height 256
        --camera-width 256
        --seed "$BASE_SEED"
    )
    if [[ "$EVAL_STAGE" == "one_episode_gate" ]]; then
        EVALUATOR_ARGS+=(--tasks "$EVAL_TASKS_CSV")
    else
        EVALUATOR_ARGS+=(--split "$EVAL_SPLIT")
    fi
    env \
        CUDA_VISIBLE_DEVICES="$shard" \
        MUJOCO_EGL_DEVICE_ID=0 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        PYTHONPATH="$EVAL_PYTHONPATH" \
        "$ROBOCASA_PY" "$DEMIAN_REPO/$DEMIAN_EVALUATOR_REL" \
        "${EVALUATOR_ARGS[@]}" >"$evaluator_log" 2>&1 &
    EVALUATOR_PIDS+=("$!")
done

eval_failed=0
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    if ! wait "${EVALUATOR_PIDS[$shard]}"; then
        echo "ERROR: evaluator shard $shard failed; tail follows" >&2
        tail -200 "$CONTROL_DIR/evaluator-shard${shard}.log" >&2 || true
        eval_failed=1
    fi
done
[[ "$eval_failed" == 0 ]] || die "One or more RoboCasa evaluator shards failed"

SUMMARIZE_ARGS=(
    summarize
    --contract "$CONTRACT_PATH"
    --results-dir "$RESULTS_DIR"
    --summary-csv "$SUMMARY_CSV"
    --task-success-rates-csv "$TASK_SUCCESS_RATES_CSV"
    --evidence-json "$EVIDENCE_JSON"
)
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$COSMOS_SOURCE/$EVIDENCE_HELPER_REL" \
    "${SUMMARIZE_ARGS[@]}" | tee "$CONTROL_DIR/summarize.log"

[[ -s "$SUMMARY_CSV" ]] || die "summary.csv is missing or empty"
[[ -s "$TASK_SUCCESS_RATES_CSV" ]] || die "task_success_rates.csv is missing or empty"
[[ -s "$EVIDENCE_JSON" ]] || die "evaluation_evidence.json is missing or empty"
EFFECT_PASS="$(
    "$COSMOS_PY" -c \
        'import json,sys; value=json.load(open(sys.argv[1]))["effect_pass"]; assert type(value) is bool; print(str(value).lower())' \
        "$EVIDENCE_JSON"
)"
RAW_RESULT_FILES=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    if [[ "$NUM_SHARDS" == 1 ]]; then
        RAW_RESULT_FILES+=("$RESULTS_DIR/args.json")
    else
        RAW_RESULT_FILES+=("$RESULTS_DIR/args_shard${shard}.json")
    fi
    RAW_RESULT_FILES+=("$RESULTS_DIR/results_shard${shard}.json")
done
chmod 0444 \
    "$CONTRACT_PATH" \
    "${RECEIPTS[@]}" \
    "${RAW_RESULT_FILES[@]}" \
    "$SUMMARY_CSV" \
    "$TASK_SUCCESS_RATES_CSV" \
    "$EVIDENCE_JSON"
chmod 0555 "$RESULTS_DIR"
echo "ROBOCASA365_EVAL_EXECUTION_SUCCESS stage=$EVAL_STAGE"
echo "SUMMARY_CSV=$SUMMARY_CSV sha256=$(sha256_file "$SUMMARY_CSV")"
echo "TASK_SUCCESS_RATES_CSV=$TASK_SUCCESS_RATES_CSV sha256=$(sha256_file "$TASK_SUCCESS_RATES_CSV")"
echo "EVIDENCE_JSON=$EVIDENCE_JSON sha256=$(sha256_file "$EVIDENCE_JSON")"
if [[ "$EVAL_STAGE" == "formal" ]]; then
    if [[ "$EFFECT_PASS" == true ]]; then
        echo "ROBOCASA365_EVAL_EFFECT_GATE_PASSED"
    else
        die "Formal evaluation completed and was sealed, but the preregistered effect gate failed"
    fi
fi
