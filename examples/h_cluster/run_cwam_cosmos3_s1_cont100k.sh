#!/usr/bin/env bash
# Continue E35-S1 from iter 10,000 to 100,000 on one 8xH200 node.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable code snapshot.}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${OSS_RUN_ROOT:?Set OSS_RUN_ROOT below an authorized chengjuntao prefix.}"
: "${SOURCE_CHECKPOINT_URI:?Set SOURCE_CHECKPOINT_URI to iter_000010000 on OSS.}"
: "${KUBEBRAIN_RJOB_NAME:=unknown}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${GPFS_LATENT_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/robocasa365-wan22-ti2v5b-postcrop-bf16-cwam-v1}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${NVME_ROOT:=/nvme/cwam-s1-cont100k}"

readonly CONFIG="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_robocasa365_s1_cont100k_nano.toml"
readonly LAUNCHER="$COSMOS_SOURCE/examples/launch_sft_action_policy_robocasa365_nano.sh"
readonly ARCHIVER="$COSMOS_SOURCE/examples/h_cluster/archive_cwam_s1_cont100k.py"
readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID"
readonly OUTPUT_ROOT="$RUN_ROOT/output"
readonly RESUME_ROOT="$RUN_ROOT/resume/iter_000010000"
readonly CACHE_ROOT="$RUN_ROOT/latent-cache"
readonly EVIDENCE_ROOT="$RUN_ROOT/evidence"
readonly CHECKPOINTS_DIR="$OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft/$RUN_NAME/checkpoints"

for path in "$CONFIG" "$LAUNCHER" "$ARCHIVER" "$COSMOS_ENV/bin/python" "$BASE_CHECKPOINT_PATH/model/.metadata" \
    "$ROBOCASA365_ROOT/_SUCCESS.json" "$GPFS_LATENT_CACHE_ROOT/_SUCCESS.json" \
    "$WAN_VAE_PATH" "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json"; do
    [[ -e "$path" ]] || { echo "ERROR: required input missing: $path" >&2; exit 1; }
done
[[ "$(nvidia-smi -L | wc -l)" -eq 8 ]] || { echo "ERROR: expected exactly eight GPUs" >&2; exit 1; }
command -v rclone >/dev/null
command -v rsync >/dev/null

mkdir -p "$RESUME_ROOT" "$CACHE_ROOT" "$EVIDENCE_ROOT"
echo "STAGE_START source_checkpoint=$SOURCE_CHECKPOINT_URI"
rclone copy "$SOURCE_CHECKPOINT_URI" "$RESUME_ROOT" --transfers=16 --checkers=32 --multi-thread-streams=8
for component in model optim scheduler trainer; do
    [[ -f "$RESUME_ROOT/$component/.metadata" ]] || { echo "ERROR: incomplete resume checkpoint: $component" >&2; exit 1; }
done

echo "CACHE_STAGE_START source=$GPFS_LATENT_CACHE_ROOT"
rsync -a --delete --info=stats2 "$GPFS_LATENT_CACHE_ROOT/" "$CACHE_ROOT/"
[[ -f "$CACHE_ROOT/_SUCCESS.json" ]] || { echo "ERROR: staged latent cache is incomplete" >&2; exit 1; }

source "$COSMOS_ENV/bin/activate"
export PATH="$COSMOS_ENV/bin:$PATH" PYTHONPATH="$COSMOS_SOURCE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42 NCCL_NVLS_ENABLE=0 NPROC_PER_NODE=8
export ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export ROBOCASA365_LATENT_CACHE_ROOT="$CACHE_ROOT"
export ROBOCASA365_LATENT_CACHE_MODE=required CWAM_METADATA_ONLY_VIDEO_TOKENIZER=1
export ROBOCASA365_TRAIN_NUM_WORKERS=4 ROBOCASA365_VAL_NUM_WORKERS=2
export CWAM_ACTIVATION_CHECKPOINTING_MODE=selective
export TOML_FILE="$CONFIG" RESUME_CHECKPOINT_PATH="$RESUME_ROOT"
export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" RUN_NAME
export RESUME_EXISTING_RUN=0 ALLOW_DATA_STREAM_REPLAY=0
export LOG_FILENAME="${RUN_NAME}.train.log"
export EXTRA_TAIL_OVERRIDES="checkpoint.load_training_state=true checkpoint.strict_resume=true checkpoint.keys_not_to_resume=[scheduler,dataloader] checkpoint.keys_to_skip_loading=[] checkpoint.dcp_async_mode_enabled=false trainer.run_validation_on_start=false trainer.validation_iter=10000 trainer.max_val_iter=8"

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINTS_DIR"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
    --format=csv -l 30 >"$EVIDENCE_ROOT/nvidia-smi.csv" 2>&1 &
MONITOR_PID=$!
"$COSMOS_ENV/bin/python" "$ARCHIVER" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --remote-root "$OSS_RUN_ROOT" \
    --receipts-dir "$EVIDENCE_ROOT/checkpoint-receipts" \
    --experiment-id "$EXPERIMENT_ID" \
    --job-id "$KUBEBRAIN_RJOB_NAME" \
    --source-commit "$SOURCE_COMMIT" \
    >"$EVIDENCE_ROOT/archive.log" 2>&1 &
ARCHIVER_PID=$!

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    kill "$MONITOR_PID" 2>/dev/null || true
    if [[ "$status" -ne 0 ]]; then
        kill "$ARCHIVER_PID" 2>/dev/null || true
    fi
    wait "$MONITOR_PID" 2>/dev/null || true
    rclone copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 || true
    local launch_log="$OUTPUT_ROOT/logs/$LOG_FILENAME"
    [[ ! -f "$launch_log" ]] || rclone copyto "$launch_log" "$OSS_RUN_ROOT/logs/$LOG_FILENAME" || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "TRAIN_START experiment=$EXPERIMENT_ID resume_step=10000 target_step=100000"
"$LAUNCHER"
wait "$ARCHIVER_PID"
echo "TRAIN_AND_ARCHIVE_COMPLETE target_step=100000"
