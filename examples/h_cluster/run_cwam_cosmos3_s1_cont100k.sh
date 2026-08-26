#!/usr/bin/env bash
# Continue E35-S1 from a durable checkpoint to 100,000 on one 8xH200 node.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable code snapshot.}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${LOG_FILENAME:=${RUN_NAME}.train.log}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${OSS_RUN_ROOT:?Set OSS_RUN_ROOT below an authorized chengjuntao prefix.}"
: "${SOURCE_CHECKPOINT_URI:?Set SOURCE_CHECKPOINT_URI to the durable checkpoint on OSS.}"
: "${RESUME_STEP:=10000}"
: "${CHECKPOINT_KEYS_NOT_TO_RESUME:=[scheduler,dataloader]}"
: "${GPU_COUNT:=8}"
: "${GRAD_ACCUM_ITER:=2}"
: "${KUBEBRAIN_RJOB_NAME:=unknown}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
: "${RCLONE_HANDOFF_BASE:=$PERSONAL_ROOT/runtime-handoffs/cwam-s1-cont100k}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${GPFS_LATENT_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/robocasa365-wan22-ti2v5b-postcrop-bf16-cwam-v1}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${NVME_ROOT:=/nvme/cwam-s1-cont100k}"

[[ "$RESUME_STEP" =~ ^[0-9]+$ ]] || { echo "ERROR: RESUME_STEP must be a non-negative integer" >&2; exit 1; }
[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: GPU_COUNT must be a positive integer" >&2; exit 1; }
[[ "$GRAD_ACCUM_ITER" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: GRAD_ACCUM_ITER must be a positive integer" >&2; exit 1; }
printf -v RESUME_TAG 'iter_%09d' "$RESUME_STEP"

readonly CONFIG="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_robocasa365_s1_cont100k_nano.toml"
readonly LAUNCHER="$COSMOS_SOURCE/examples/launch_sft_action_policy_robocasa365_nano.sh"
readonly ARCHIVER="$COSMOS_SOURCE/examples/h_cluster/archive_cwam_s1_cont100k.py"
readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"
readonly OUTPUT_ROOT="$RUN_ROOT/output"
readonly RESUME_ROOT="$RUN_ROOT/resume/$RESUME_TAG"
readonly CACHE_ROOT="$RUN_ROOT/latent-cache"
readonly EVIDENCE_ROOT="$RUN_ROOT/evidence"
readonly RCLONE_HANDOFF_ROOT="$RCLONE_HANDOFF_BASE/$KUBEBRAIN_RJOB_NAME"
readonly RCLONE_SECRET_ROOT="/dev/shm/cwam-s1-cont100k-rclone"
readonly RCLONE_CONFIG_PATH="$RCLONE_SECRET_ROOT/rclone.conf"
readonly RCLONE_PRIVATE_KEY="$RCLONE_SECRET_ROOT/recipient.key"
readonly RCLONE_CERTIFICATE="$RCLONE_HANDOFF_ROOT/recipient.pem"
readonly RCLONE_REQUEST="$RCLONE_HANDOFF_ROOT/request.json"
readonly RCLONE_ENCRYPTED_CONFIG="$RCLONE_HANDOFF_ROOT/rclone-config.cms.der"
readonly CHECKPOINTS_DIR="$OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft/$RUN_NAME/checkpoints"
MONITOR_PID=""
ARCHIVER_PID=""

for path in "$CONFIG" "$LAUNCHER" "$ARCHIVER" "$COSMOS_ENV/bin/python" "$BASE_CHECKPOINT_PATH/model/.metadata" \
    "$ROBOCASA365_ROOT/_SUCCESS.json" "$GPFS_LATENT_CACHE_ROOT/_SUCCESS.json" \
    "$WAN_VAE_PATH" "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json"; do
    [[ -e "$path" ]] || { echo "ERROR: required input missing: $path" >&2; exit 1; }
done
[[ "$(nvidia-smi -L | wc -l)" -eq "$GPU_COUNT" ]] || { echo "ERROR: visible GPU count does not match GPU_COUNT=$GPU_COUNT" >&2; exit 1; }
[[ -x "$RCLONE_BIN" ]] || { echo "ERROR: rclone binary missing: $RCLONE_BIN" >&2; exit 1; }
[[ -x /usr/bin/openssl ]] || { echo "ERROR: openssl is unavailable" >&2; exit 1; }
command -v rsync >/dev/null

mkdir -p "$RESUME_ROOT" "$CACHE_ROOT" "$EVIDENCE_ROOT" "$RCLONE_SECRET_ROOT" "$RCLONE_HANDOFF_ROOT"
chmod 700 "$RCLONE_SECRET_ROOT" "$RCLONE_HANDOFF_ROOT"
export PATH="$(dirname "$RCLONE_BIN"):$COSMOS_ENV/bin:$PATH"
export RCLONE_CONFIG="$RCLONE_CONFIG_PATH"

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    [[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true
    if [[ "$status" -ne 0 && -n "$ARCHIVER_PID" ]]; then
        kill "$ARCHIVER_PID" 2>/dev/null || true
    fi
    [[ -z "$MONITOR_PID" ]] || wait "$MONITOR_PID" 2>/dev/null || true
    if [[ -s "$RCLONE_CONFIG_PATH" ]]; then
        "$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 || true
        local launch_log="$OUTPUT_ROOT/logs/$LOG_FILENAME"
        [[ ! -f "$launch_log" ]] || "$RCLONE_BIN" copyto "$launch_log" "$OSS_RUN_ROOT/logs/$LOG_FILENAME" || true
    fi
    rm -f "$RCLONE_CONFIG_PATH" "$RCLONE_PRIVATE_KEY"
    rm -f "$RCLONE_CERTIFICATE" "$RCLONE_REQUEST" "$RCLONE_ENCRYPTED_CONFIG"
    rm -f "$RCLONE_CERTIFICATE.tmp" "$RCLONE_REQUEST.tmp"
    rmdir "$RCLONE_SECRET_ROOT" "$RCLONE_HANDOFF_ROOT" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for path in "$RCLONE_CERTIFICATE" "$RCLONE_REQUEST" "$RCLONE_ENCRYPTED_CONFIG"; do
    [[ ! -e "$path" ]] || { echo "ERROR: refusing to reuse rclone handoff artifact: $path" >&2; exit 1; }
done
/usr/bin/openssl req -x509 -newkey rsa:3072 -nodes -days 1 \
    -subj "/CN=$KUBEBRAIN_RJOB_NAME" \
    -keyout "$RCLONE_PRIVATE_KEY" \
    -out "$RCLONE_CERTIFICATE.tmp" >/dev/null 2>&1
chmod 400 "$RCLONE_PRIVATE_KEY"
chmod 444 "$RCLONE_CERTIFICATE.tmp"
mv "$RCLONE_CERTIFICATE.tmp" "$RCLONE_CERTIFICATE"
export EXPERIMENT_ID KUBEBRAIN_RJOB_NAME OSS_RUN_ROOT SOURCE_CHECKPOINT_URI
"$COSMOS_ENV/bin/python" - "$RCLONE_REQUEST.tmp" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema_version": 1,
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "job_id": os.environ["KUBEBRAIN_RJOB_NAME"],
    "oss_run_root": os.environ["OSS_RUN_ROOT"],
    "source_checkpoint_uri": os.environ["SOURCE_CHECKPOINT_URI"],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
chmod 444 "$RCLONE_REQUEST.tmp"
mv "$RCLONE_REQUEST.tmp" "$RCLONE_REQUEST"
echo "RCLONE_HANDOFF_WAITING root=$RCLONE_HANDOFF_ROOT"
for _ in $(seq 1 240); do
    [[ -s "$RCLONE_ENCRYPTED_CONFIG" ]] && break
    sleep 5
done
[[ -s "$RCLONE_ENCRYPTED_CONFIG" ]] || { echo "ERROR: timed out waiting for rclone handoff" >&2; exit 1; }
/usr/bin/openssl cms -decrypt -binary -inform DER \
    -in "$RCLONE_ENCRYPTED_CONFIG" \
    -recip "$RCLONE_CERTIFICATE" \
    -inkey "$RCLONE_PRIVATE_KEY" \
    -out "$RCLONE_CONFIG_PATH"
chmod 400 "$RCLONE_CONFIG_PATH"
rm -f "$RCLONE_PRIVATE_KEY" "$RCLONE_CERTIFICATE" "$RCLONE_REQUEST" "$RCLONE_ENCRYPTED_CONFIG"
rmdir "$RCLONE_HANDOFF_ROOT" 2>/dev/null || true
"$RCLONE_BIN" listremotes | grep -Fx 'eailab-hdd2:' >/dev/null
"$RCLONE_BIN" lsf "$SOURCE_CHECKPOINT_URI" --max-depth 1 >/dev/null
echo "RCLONE_HANDOFF_READY remote=eailab-hdd2"
echo "STAGE_START source_checkpoint=$SOURCE_CHECKPOINT_URI"
"$RCLONE_BIN" copy "$SOURCE_CHECKPOINT_URI" "$RESUME_ROOT" --transfers=16 --checkers=32 --multi-thread-streams=8
for component in model optim scheduler trainer; do
    [[ -f "$RESUME_ROOT/$component/.metadata" ]] || { echo "ERROR: incomplete resume checkpoint: $component" >&2; exit 1; }
done

echo "CACHE_STAGE_START source=$GPFS_LATENT_CACHE_ROOT"
rsync -a --delete --info=stats2 "$GPFS_LATENT_CACHE_ROOT/" "$CACHE_ROOT/"
[[ -f "$CACHE_ROOT/_SUCCESS.json" ]] || { echo "ERROR: staged latent cache is incomplete" >&2; exit 1; }

source "$COSMOS_ENV/bin/activate"
export PATH="$(dirname "$RCLONE_BIN"):$COSMOS_ENV/bin:$PATH" PYTHONPATH="$COSMOS_SOURCE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42 NCCL_NVLS_ENABLE=0 NPROC_PER_NODE="$GPU_COUNT"
export ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export ROBOCASA365_LATENT_CACHE_ROOT="$CACHE_ROOT"
export ROBOCASA365_LATENT_CACHE_MODE=required CWAM_METADATA_ONLY_VIDEO_TOKENIZER=1
export ROBOCASA365_TRAIN_NUM_WORKERS=4 ROBOCASA365_VAL_NUM_WORKERS=2
export CWAM_ACTIVATION_CHECKPOINTING_MODE=selective
export TOML_FILE="$CONFIG" RESUME_CHECKPOINT_PATH="$RESUME_ROOT"
export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" RUN_NAME
export RESUME_EXISTING_RUN=0 ALLOW_DATA_STREAM_REPLAY=0
export LOG_FILENAME
export EXTRA_TAIL_OVERRIDES="checkpoint.load_training_state=true checkpoint.strict_resume=true checkpoint.keys_not_to_resume=$CHECKPOINT_KEYS_NOT_TO_RESUME checkpoint.keys_to_skip_loading=[] checkpoint.dcp_async_mode_enabled=false trainer.run_validation_on_start=false trainer.validation_iter=10000 trainer.max_val_iter=8 trainer.grad_accum_iter=$GRAD_ACCUM_ITER"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
    --format=csv -l 30 >"$EVIDENCE_ROOT/nvidia-smi.csv" 2>&1 &
MONITOR_PID=$!
(
    for _ in $(seq 1 600); do
        [[ -d "$CHECKPOINTS_DIR" ]] && break
        sleep 1
    done
    [[ -d "$CHECKPOINTS_DIR" ]] || {
        echo "ERROR: timed out waiting for launcher to create checkpoint directory: $CHECKPOINTS_DIR" >&2
        exit 1
    }
    exec "$COSMOS_ENV/bin/python" "$ARCHIVER" \
        --checkpoints-dir "$CHECKPOINTS_DIR" \
        --remote-root "$OSS_RUN_ROOT" \
        --receipts-dir "$EVIDENCE_ROOT/checkpoint-receipts" \
        --experiment-id "$EXPERIMENT_ID" \
        --job-id "$KUBEBRAIN_RJOB_NAME" \
        --source-commit "$SOURCE_COMMIT"
) >"$EVIDENCE_ROOT/archive.log" 2>&1 &
ARCHIVER_PID=$!

echo "TRAIN_START experiment=$EXPERIMENT_ID resume_step=$RESUME_STEP target_step=100000 gpu_count=$GPU_COUNT grad_accum_iter=$GRAD_ACCUM_ITER keys_not_to_resume=$CHECKPOINT_KEYS_NOT_TO_RESUME"
"$LAUNCHER"
wait "$ARCHIVER_PID"
echo "TRAIN_AND_ARCHIVE_COMPLETE target_step=100000"
