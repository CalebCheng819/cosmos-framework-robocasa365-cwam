#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data0/wam/cwam-b5/source/CWAM-COSMOS3-B5-VOM-R4-GIT-20260817
COMMIT=0169f11f719b4abb5b4b3580e0131406872d5b04
RUN=CWAM-COSMOS3-B5-VOM-PROBE-R4-B4WARM-500S-20260817
JOB=cwam-b5-vom-probe-r4
MONITOR=${JOB}-gpu-monitor
IMAGE=cwam-b4-runtime:ngc25.09-ffmpeg6
EVIDENCE=/data0/wam/cwam-b5/evidence/${RUN}

test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)" -eq 0
! sudo -n docker ps -a --format '{{.Names}}' | grep -Fxq "$JOB"
! sudo -n docker ps -a --format '{{.Names}}' | grep -Fxq "$MONITOR"

mkdir -p "$EVIDENCE"
mkdir -p "/data0/wam/cwam-b5/cache/home/$RUN"
mkdir -p "/data0/wam/cwam-b5/cache/tmp/$RUN"
mkdir -p "/data0/wam/cwam-b5/cache/compile/$RUN"

sudo -n docker run -d --rm \
  --name "$MONITOR" \
  --gpus all \
  --network host \
  -v /data0:/data0 \
  "$IMAGE" \
  bash -lc "exec nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 30 > '$EVIDENCE/gpu.csv'"

sudo -n docker run -d --rm \
  --name "$JOB" \
  --gpus all \
  --ipc host \
  --network host \
  -v /data0:/data0 \
  -v /data1:/data1 \
  -w "$SOURCE" \
  -e NCCL_NVLS_ENABLE=0 \
  -e PYTHONPATH="$SOURCE" \
  -e TOML_FILE="$SOURCE/examples/toml/sft_config/action_policy_robocasa365_b5_visual_object_memory_nano.toml" \
  -e HOME="/data0/wam/cwam-b5/cache/home/$RUN" \
  -e OMP_NUM_THREADS=12 \
  -e BASE_CHECKPOINT_PATH=/data1/wam/cwam-b4/runs/cosmos3_action_robocasa365/action_sft/CWAM-COSMOS3-B4-ADAPTIVE-DSELECT-R1-20260814/checkpoints/iter_000020000 \
  -e OUTPUT_ROOT=/data0/wam/cwam-b5/runs \
  -e ROBOCASA365_SOURCE_ROOT=/data0/wam/cwam-b4/inputs/source-data \
  -e ROBOCASA365_CONTROL_CACHE_ROOT=/data0/wam/cwam-b4/inputs/control-cache \
  -e ROBOCASA365_STRONG_LATENT_CACHE_MODE=required \
  -e CWAM_B5_RUN_CHECKPOINT_MODE=b4 \
  -e CWAM_B5_PROBE_TRAIN_SCOPE=strong_action_expert \
  -e EXTRA_TAIL_OVERRIDES='dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=1 trainer.callbacks.production_batch_memory_gate.expected_samples_per_microbatch=64 trainer.callbacks.production_batch_memory_gate.expected_grad_accum_iter=1 model.config.max_num_tokens_after_packing=148000 trainer.max_iter=500 trainer.max_val_iter=8 trainer.validation_iter=500 trainer.logging_iter=1 trainer.save_final_checkpoint=false checkpoint.save_iter=500 scheduler.warm_up_steps=[50] scheduler.cycle_lengths=[500]' \
  -e TMPDIR="/data0/wam/cwam-b5/cache/tmp/$RUN" \
  -e ROBOCASA365_ROOT=/data0/wam/cwam-b4/inputs/processed \
  -e CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER=1 \
  -e RUN_NAME="$RUN" \
  -e RESUME_EXISTING_RUN=0 \
  -e ALLOW_DATA_STREAM_REPLAY=0 \
  -e CWAM_VISUAL_MEMORY_INTERVENTION=none \
  -e CWAM_RANK_CACHE_ROOT="/data0/wam/cwam-b5/cache/compile/$RUN" \
  -e HF_HOME=/data0/wam/cwam-b5/cache/huggingface \
  -e ROBOCASA365_STRONG_LATENT_CACHE_ROOT=/data0/wam/cwam-b4/inputs/latent-cache \
  -e IMAGINAIRE_OUTPUT_ROOT=/data0/wam/cwam-b5/runs \
  -e NPROC_PER_NODE=8 \
  -e USER=wam \
  -e LOGNAME=wam \
  -e NCCL_DEBUG=WARN \
  -e CWAM_ACTIVATION_CHECKPOINTING_MODE=selective \
  -e LOG_FILENAME="$RUN.log" \
  -e WAN_VAE_PATH=/data0/wam/cwam-b4/inputs/wan22-vae/Wan2.2_VAE.pth \
  -e ROBOCASA365_VISUAL_HISTORY_CACHE_ROOT=/data0/wam/cwam-b5/inputs/visual-history-cache \
  -e QWEN3_VL_TOKENIZER_PATH=/data0/wam/cwam-b4/inputs/qwen3-vl-tokenizer \
  "$IMAGE" \
  bash -lc "export PATH='/data0/wam/cwam-b4/env/cosmos-cu130/bin':\"\$PATH\"; export PYTHONPATH='$SOURCE':\"\${PYTHONPATH:-}\"; exec examples/launch_sft_action_policy_robocasa365_nano.sh"

sudo -n docker inspect "$JOB" > "$EVIDENCE/docker-inspect-start.json"
printf 'job=%s\nmonitor=%s\nrun=%s\ncommit=%s\ntrain_scope=strong_action_expert\n' \
  "$JOB" "$MONITOR" "$RUN" "$COMMIT"
