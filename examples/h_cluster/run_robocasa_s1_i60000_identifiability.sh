#!/usr/bin/env bash
# Run the paired seven-condition protocol on the fixed RoboCasa S1 60k policy.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable Cosmos source snapshot.}"
: "${PROTOCOL_SOURCE:?Set PROTOCOL_SOURCE to the immutable protocol snapshot.}"
: "${DEMIAN_SOURCE:?Set DEMIAN_SOURCE to the fixed DeMiAn source.}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${PROTOCOL_COMMIT:?Set PROTOCOL_COMMIT.}"
: "${SOURCE_CHECKPOINT_URI:?Set SOURCE_CHECKPOINT_URI.}"
: "${OSS_RUN_ROOT:?Set OSS_RUN_ROOT.}"
: "${EXPECTED_NODE_TAG:?Set EXPECTED_NODE_TAG.}"
: "${KUBEBRAIN_RJOB_NAME:=unknown}"
: "${NUM_GPUS:=8}"
: "${EPISODES_PER_TASK:=2}"
: "${MAX_STEPS:=400}"
: "${OPEN_LOOP_HORIZON:=12}"
: "${POLICY_SEED:=42}"
: "${SPLIT_SEED:=37}"
: "${JACOBIAN_PROJECTIONS:=4}"
: "${TOPK:=8}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${PUBLIC_ROOT:=/mnt/shared-storage-gpfs2/gpfs2-shared-public}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA_ENV:=$PERSONAL_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared}"
: "${ROBOCASA_REPO:=$PERSONAL_ROOT/code/robocasa-29f7ce8814c1}"
: "${ROBOSUITE_REPO:=$PERSONAL_ROOT/code/robosuite-232ce7d4a6ed}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
: "${NVME_ROOT:=/nvme/cwam-robocasa-identifiability}"
: "${EXTERNAL_BASELINE_RECORD:=$PERSONAL_ROOT/cwam-run-records/CWAM-E35-S1-I60000-EMA-ROB365-1700EP-2H200-S42-R1-20260826/final-summary.json}"

readonly CONDITIONS=(normal selected complement random_matched zero shuffle cross_episode)
readonly EXPECTED_TASKS=17
readonly EXPECTED_EPISODES=$((EXPECTED_TASKS * EPISODES_PER_TASK))
readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"
readonly CHECKPOINT_ROOT="$RUN_ROOT/checkpoint"
readonly SERVING_CHECKPOINT="$CHECKPOINT_ROOT/model"
readonly TRAINING_CONFIG="$CHECKPOINT_ROOT/config.yaml"
readonly RECEIPT="$CHECKPOINT_ROOT/ema-bf16-metadata-receipt.json"
readonly DEMIAN_REPO="$RUN_ROOT/demian"
readonly CONDITIONS_ROOT="$RUN_ROOT/conditions"
readonly TRACE_ROOT="$RUN_ROOT/traces"
readonly SCRATCH_ROOT="$RUN_ROOT/scratch"
readonly EVIDENCE_ROOT="$RUN_ROOT/evidence"
readonly BANK_ROOT="$RUN_ROOT/intervention-bank"
readonly BANK_PATH="$BANK_ROOT/routed_c_bank.pt"
readonly BANK_MANIFEST="$BANK_ROOT/routed_c_bank.json"
readonly HANDOFF_ROOT="$PERSONAL_ROOT/runtime-handoffs/cwam-robocasa-identifiability/$KUBEBRAIN_RJOB_NAME"
readonly SECRET_ROOT="/dev/shm/cwam-robocasa-identifiability-$KUBEBRAIN_RJOB_NAME"
readonly RCLONE_CONFIG_PATH="$SECRET_ROOT/rclone.conf"
readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_robocasa365_eight_gpu_eval.sh"
readonly BANK_BUILDER="$COSMOS_SOURCE/examples/h_cluster/build_robocasa365_cross_episode_bank.py"
readonly DEMIAN_PATCH="$COSMOS_SOURCE/examples/h_cluster/patches/demian_robocasa_eval_cwam_evidence.patch"
readonly ACTION_STATS="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857/stats/robocasa365_native_frame_wise_relative_rot6d.json"
readonly WAN2PT2_VAE="$PUBLIC_ROOT/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly QWEN_TOKENIZER="$PUBLIC_ROOT/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct"
readonly EGL_VENDOR_JSON="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

for path in "$COSMOS_SOURCE" "$PROTOCOL_SOURCE" "$DEMIAN_SOURCE" "$COSMOS_ENV" \
    "$ROBOCASA_ENV" "$ROBOCASA_REPO" "$ROBOSUITE_REPO" "$QWEN_TOKENIZER"; do
    [[ -d "$path" ]] || fail "required directory missing: $path"
done
for path in "$COSMOS_PY" "$RUNNER" "$BANK_BUILDER" "$DEMIAN_PATCH" "$ACTION_STATS" \
    "$WAN2PT2_VAE" "$EGL_VENDOR_JSON" "$RCLONE_BIN"; do
    [[ -e "$path" ]] || fail "required input missing: $path"
done
[[ "$NUM_GPUS" -eq 8 ]] || fail "RoboCasa paired protocol requires exactly eight GPUs"
[[ "$(nvidia-smi -L | wc -l)" -eq "$NUM_GPUS" ]] || fail "expected $NUM_GPUS visible GPUs"
[[ "$EPISODES_PER_TASK" -ge 2 ]] || fail "strict same-task shuffle requires at least two episodes per task"
[[ "$(git -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || fail "Cosmos source revision mismatch"
[[ "$(git -C "$PROTOCOL_SOURCE" rev-parse HEAD)" == "$PROTOCOL_COMMIT" ]] || fail "protocol source revision mismatch"
[[ "$SOURCE_CHECKPOINT_URI" == eailab-hdd2:eailab/chengjuntao/* ]] || fail "checkpoint URI escapes authorized prefix"
[[ "$OSS_RUN_ROOT" == eailab-hdd2:eailab/chengjuntao/* ]] || fail "run URI escapes authorized prefix"

mkdir -p "$CHECKPOINT_ROOT" "$CONDITIONS_ROOT" "$TRACE_ROOT" "$SCRATCH_ROOT" \
    "$EVIDENCE_ROOT" "$BANK_ROOT" "$SECRET_ROOT" "$HANDOFF_ROOT" "$DEMIAN_REPO"
chmod 700 "$SECRET_ROOT" "$HANDOFF_ROOT"

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    if [[ -s "$RCLONE_CONFIG_PATH" ]]; then
        "$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 || true
    fi
    rm -rf "$SECRET_ROOT"
    rm -f "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json" "$HANDOFF_ROOT/rclone-config.cms.der"
    rmdir "$HANDOFF_ROOT" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

openssl req -x509 -newkey rsa:3072 -nodes -days 1 -subj "/CN=$KUBEBRAIN_RJOB_NAME" \
    -keyout "$SECRET_ROOT/recipient.key" -out "$HANDOFF_ROOT/recipient.pem" >/dev/null 2>&1
export KUBEBRAIN_RJOB_NAME OSS_RUN_ROOT SOURCE_CHECKPOINT_URI
"$COSMOS_PY" - "$HANDOFF_ROOT/request.json" <<'PY'
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

"$RCLONE_BIN" copy "$SOURCE_CHECKPOINT_URI" "$CHECKPOINT_ROOT" \
    --transfers=16 --checkers=32 --multi-thread-streams=8
"$COSMOS_PY" - "$RECEIPT" "$SERVING_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text())
model = Path(sys.argv[2])
assert receipt["status"] == "succeeded"
assert receipt["selection"]["key_count"] == 822
assert receipt["selection"]["target_logical_bytes"] == 30363084946
assert receipt["target"]["shard_count"] == 8
assert receipt["validation"]["full_semantic_readback"] is True
assert len(list(model.glob("*.distcp"))) == 8
assert (model / ".metadata").is_file()
PY

cp -a "$DEMIAN_SOURCE/demian_wam" "$DEMIAN_REPO/"
ln -s "$DEMIAN_SOURCE/openpi" "$DEMIAN_REPO/openpi"
(cd "$DEMIAN_REPO" && git apply --check "$DEMIAN_PATCH" && git apply "$DEMIAN_PATCH")

export EXPERIMENT_ID KUBEBRAIN_RJOB_NAME SOURCE_COMMIT PROTOCOL_COMMIT SOURCE_CHECKPOINT_URI OSS_RUN_ROOT
export EPISODES_PER_TASK MAX_STEPS OPEN_LOOP_HORIZON POLICY_SEED SPLIT_SEED JACOBIAN_PROJECTIONS TOPK
export EXTERNAL_BASELINE_RECORD
"$COSMOS_PY" - "$EVIDENCE_ROOT/run_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = (
    "EXPERIMENT_ID", "KUBEBRAIN_RJOB_NAME", "SOURCE_COMMIT", "PROTOCOL_COMMIT",
    "SOURCE_CHECKPOINT_URI", "OSS_RUN_ROOT", "EPISODES_PER_TASK", "MAX_STEPS",
    "OPEN_LOOP_HORIZON", "POLICY_SEED", "SPLIT_SEED", "JACOBIAN_PROJECTIONS", "TOPK",
    "EXTERNAL_BASELINE_RECORD",
)
Path(sys.argv[1]).write_text(json.dumps({key: os.environ[key] for key in keys}, indent=2, sort_keys=True) + "\n")
PY

run_condition() {
    local condition="$1"
    local output_root="$CONDITIONS_ROOT/$condition"
    local trace_dir="$TRACE_ROOT/$condition"
    local condition_scratch="$SCRATCH_ROOT/$condition"
    local bank=""
    local jacobian=0
    [[ "$condition" == normal ]] && jacobian="$JACOBIAN_PROJECTIONS"
    [[ "$condition" == shuffle || "$condition" == cross_episode ]] && bank="$BANK_PATH"

    EVAL_CONTROL_SOURCE="$COSMOS_SOURCE" COSMOS_SOURCE="$COSMOS_SOURCE" \
    COSMOS_ENV="$COSMOS_ENV" ROBOCASA_ENV="$ROBOCASA_ENV" DEMIAN_REPO="$DEMIAN_REPO" \
    ROBOCASA_REPO="$ROBOCASA_REPO" ROBOSUITE_REPO="$ROBOSUITE_REPO" \
    SERVING_CHECKPOINT_PATH="$SERVING_CHECKPOINT" TRAINING_CONFIG="$TRAINING_CONFIG" \
    ACTION_STATS="$ACTION_STATS" WAN2PT2_VAE="$WAN2PT2_VAE" QWEN_TOKENIZER="$QWEN_TOKENIZER" \
    EVAL_OUTPUT_ROOT="$output_root" EVAL_OUTPUT_ALLOWED_PREFIX="$CONDITIONS_ROOT" \
    EVAL_SCRATCH_ROOT="$condition_scratch" EVAL_EXPERIMENT_ID="$EXPERIMENT_ID-$condition" \
    EVAL_LOGICAL_RUN_ID="$EXPERIMENT_ID-$condition" EVAL_RUN_NAME="s1-i60k-$condition" \
    MODEL_LABEL="S1-E35-iter60000-EMA" MODEL_SOURCE_REVISION="$SOURCE_COMMIT" \
    CONTROL_SOURCE_REVISION="$SOURCE_COMMIT" EXPECTED_EMA_KEY_COUNT=822 \
    EVAL_EGL_VENDOR_JSON="$EGL_VENDOR_JSON" EXPECTED_NODE_TAG="$EXPECTED_NODE_TAG" \
    EVAL_GPU_COUNT="$NUM_GPUS" RJOB_NAME="$KUBEBRAIN_RJOB_NAME" \
    PROTOCOL_EPISODES_PER_TASK="$EPISODES_PER_TASK" PROTOCOL_MAX_STEPS="$MAX_STEPS" \
    PROTOCOL_OPEN_LOOP_HORIZON="$OPEN_LOOP_HORIZON" PROTOCOL_TASK_LIMIT="$EXPECTED_TASKS" \
    PROTOCOL_SEED="$POLICY_SEED" PROTOCOL_TRACE_FIXED_HORIZON=1 \
    ACTION_RELEVANCE_INTERVENTION="$condition" ACTION_RELEVANCE_TRACE_DIR="$trace_dir" \
    ACTION_RELEVANCE_INTERVENTION_BANK="$bank" \
    ACTION_RELEVANCE_JACOBIAN_PROJECTIONS="$jacobian" \
        bash "$RUNNER"
    [[ "$("$COSMOS_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output_root/status.json")" == COMPLETE ]] || \
        fail "condition=$condition did not produce a terminal COMPLETE status"
    "$RCLONE_BIN" copy "$output_root" "$OSS_RUN_ROOT/conditions/$condition" --transfers=8 --checkers=16
    "$RCLONE_BIN" copy "$trace_dir" "$OSS_RUN_ROOT/traces/$condition" --transfers=8 --checkers=16
}

run_condition normal
PYTHONPATH="$COSMOS_SOURCE" "$COSMOS_PY" "$BANK_BUILDER" \
    --trace-dir "$TRACE_ROOT/normal" --output "$BANK_PATH" --manifest "$BANK_MANIFEST" \
    --seed "$POLICY_SEED"
"$RCLONE_BIN" copy "$BANK_ROOT" "$OSS_RUN_ROOT/intervention-bank" --transfers=8 --checkers=16

for condition in "${CONDITIONS[@]:1}"; do
    run_condition "$condition"
done

completion_args=()
trace_args=()
closed_loop_args=()
for condition in "${CONDITIONS[@]}"; do
    completion_args+=(--condition "$condition=$CONDITIONS_ROOT/$condition")
    trace_args+=(--trace "$TRACE_ROOT/$condition")
    closed_loop_args+=("$CONDITIONS_ROOT/$condition")
done

PYTHONPATH="$PROTOCOL_SOURCE" "$COSMOS_PY" -m arstate_sim.closed_loop_completion \
    "${completion_args[@]}" --expected-episodes "$EXPECTED_EPISODES" \
    --expected-tasks "$EXPECTED_TASKS" --output "$EVIDENCE_ROOT/closed_loop_completion.json"
PYTHONPATH="$PROTOCOL_SOURCE" "$COSMOS_PY" -m arstate_sim.empirical_trace_export \
    "${trace_args[@]}" --output "$EVIDENCE_ROOT/aligned_protocol.npz" \
    --manifest "$EVIDENCE_ROOT/aligned_protocol.manifest.json" --layers all --require-all-conditions
PYTHONPATH="$PROTOCOL_SOURCE" "$COSMOS_PY" -m arstate_sim.empirical_protocol \
    --input "$EVIDENCE_ROOT/aligned_protocol.npz" --output "$EVIDENCE_ROOT/protocol_results.json" \
    --closed-loop "${closed_loop_args[@]}" --eval-fraction 0.25 --seed "$SPLIT_SEED" --topk "$TOPK"

"$COSMOS_PY" - "$EVIDENCE_ROOT/COMPLETE" "$EVIDENCE_ROOT/closed_loop_completion.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, completion_path = map(Path, sys.argv[1:])
completion = json.loads(completion_path.read_text())
if completion.get("status") != "COMPLETE":
    raise SystemExit("paired protocol is incomplete")
output.write_text(json.dumps({
    "status": "COMPLETE",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "paired_episodes": completion["paired_episode_keys"],
}, sort_keys=True) + "\n")
PY
"$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 --exclude COMPLETE
"$RCLONE_BIN" copyto "$EVIDENCE_ROOT/COMPLETE" "$OSS_RUN_ROOT/evidence/COMPLETE"
echo "ROBOCASA_IDENTIFIABILITY_COMPLETE experiment=$EXPERIMENT_ID paired_episodes=$EXPECTED_EPISODES"
