#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data0/wam/cwam-b5/source/CWAM-COSMOS3-B5-VOM-FORMAL-R7-GIT-20260817
COMMIT=88f3c95c18a3f3195b66e35b2cd8f6ecfe595b81
BASE_CHECKPOINT_PATH=/data0/wam/cwam-b5/runs/cosmos3_action_robocasa365/action_sft/CWAM-COSMOS3-B5-VOM-PROBE-R4-B4WARM-500S-20260817/checkpoints/iter_000000500
RUN=CWAM-COSMOS3-B5-VOM-FORMAL-R7-POLICY-RANK-20K-20260817
JOB=cwam-b5-vom-formal-r7-20k
MONITOR=${JOB}-gpu-monitor
IMAGE=cwam-b4-runtime:ngc25.09-ffmpeg6
EVIDENCE=/data0/wam/cwam-b5/evidence/${RUN}
HOME_DIR=/data0/wam/cwam-b5/cache/home/${RUN}
TMP_DIR=/data0/wam/cwam-b5/cache/tmp/${RUN}
COMPILE_DIR=/data0/wam/cwam-b5/cache/compile/${RUN}
RUN_DIR=/data0/wam/cwam-b5/runs/cosmos3_action_robocasa365/action_sft/${RUN}

test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
test -f "$BASE_CHECKPOINT_PATH/model/.metadata"
test "$(find "$BASE_CHECKPOINT_PATH/model" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 8
test ! -e "$RUN_DIR"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)" -eq 0
! sudo -n docker ps -a --format '{{.Names}}' | grep -Fxq "$JOB"
! sudo -n docker ps -a --format '{{.Names}}' | grep -Fxq "$MONITOR"

mkdir -p "$EVIDENCE" "$HOME_DIR" "$TMP_DIR" "$COMPILE_DIR"

sudo -n docker run -d --rm \
  --name "$MONITOR" --gpus all --network host -v /data0:/data0 "$IMAGE" \
  bash -lc "exec nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 30 > '$EVIDENCE/gpu.csv'"

sudo -n docker run -d \
  --name "$JOB" --gpus all --ipc host --network host \
  -v /data0:/data0 -v /data1:/data1 -w "$SOURCE" \
  -e NCCL_NVLS_ENABLE=0 -e PYTHONPATH="$SOURCE" \
  -e TOML_FILE="$SOURCE/examples/toml/sft_config/action_policy_robocasa365_b5_visual_object_memory_nano.toml" \
  -e HOME="$HOME_DIR" -e OMP_NUM_THREADS=12 \
  -e BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" -e OUTPUT_ROOT=/data0/wam/cwam-b5/runs \
  -e ROBOCASA365_SOURCE_ROOT=/data0/wam/cwam-b4/inputs/source-data \
  -e ROBOCASA365_CONTROL_CACHE_ROOT=/data0/wam/cwam-b4/inputs/control-cache \
  -e ROBOCASA365_STRONG_LATENT_CACHE_MODE=required \
  -e CWAM_B5_RUN_CHECKPOINT_MODE=r4 -e CWAM_B5_PROBE_TRAIN_SCOPE=strong_action_expert \
  -e CWAM_B5_ALLOW_FORMAL_R4_WARM_START=1 \
  -e EXTRA_TAIL_OVERRIDES='dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=1 trainer.callbacks.production_batch_memory_gate.expected_samples_per_microbatch=64 trainer.callbacks.production_batch_memory_gate.expected_grad_accum_iter=1 model.config.max_num_tokens_after_packing=148000 trainer.max_iter=20000 trainer.max_val_iter=8 trainer.validation_iter=500 trainer.logging_iter=1 trainer.run_validation=true trainer.run_validation_on_start=true trainer.save_final_checkpoint=false checkpoint.save_iter=1000 scheduler.warm_up_steps=[500] scheduler.cycle_lengths=[20000]' \
  -e TMPDIR="$TMP_DIR" -e ROBOCASA365_ROOT=/data0/wam/cwam-b4/inputs/processed \
  -e CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER=1 -e RUN_NAME="$RUN" -e RESUME_EXISTING_RUN=0 \
  -e ALLOW_DATA_STREAM_REPLAY=0 -e CWAM_VISUAL_MEMORY_INTERVENTION=none \
  -e CWAM_VISUAL_MEMORY_DELTA_COMPONENT=full -e CWAM_VISUAL_MEMORY_EXCLUSIVE_DYNAMIC=1 \
  -e CWAM_VISUAL_MEMORY_DELTA_CONDITIONING_LAYERS=0 \
  -e CWAM_VISUAL_MEMORY_POLICY_RANK_ENABLED=1 \
  -e CWAM_VISUAL_MEMORY_POLICY_RANK_WEIGHT=1.0 \
  -e CWAM_VISUAL_MEMORY_POLICY_RANK_MARGIN=0.01 \
  -e CWAM_VISUAL_MEMORY_POLICY_RANK_WARMUP_ITERS=50 \
  -e CWAM_RANK_CACHE_ROOT="$COMPILE_DIR" \
  -e HF_HOME=/data0/wam/cwam-b5/cache/huggingface \
  -e ROBOCASA365_STRONG_LATENT_CACHE_ROOT=/data0/wam/cwam-b4/inputs/latent-cache \
  -e IMAGINAIRE_OUTPUT_ROOT=/data0/wam/cwam-b5/runs -e NPROC_PER_NODE=8 \
  -e USER=wam -e LOGNAME=wam -e NCCL_DEBUG=WARN -e CWAM_ACTIVATION_CHECKPOINTING_MODE=selective \
  -e LOG_FILENAME="$RUN.log" -e WAN_VAE_PATH=/data0/wam/cwam-b4/inputs/wan22-vae/Wan2.2_VAE.pth \
  -e ROBOCASA365_VISUAL_HISTORY_CACHE_ROOT=/data0/wam/cwam-b5/inputs/visual-history-cache \
  -e QWEN3_VL_TOKENIZER_PATH=/data0/wam/cwam-b4/inputs/qwen3-vl-tokenizer \
  "$IMAGE" bash -lc "export PATH='/data0/wam/cwam-b4/env/cosmos-cu130/bin':\"\$PATH\"; export PYTHONPATH='$SOURCE':\"\${PYTHONPATH:-}\"; exec examples/launch_sft_action_policy_robocasa365_nano.sh"

sudo -n docker inspect "$JOB" > "$EVIDENCE/docker-inspect-start.json"
printf 'job=%s\nmonitor=%s\nrun=%s\ncommit=%s\nbase_checkpoint=%s\nsteps=%s\nglobal_batch=%s\nrank_weight=%s\nrank_margin=%s\nrank_warmup_iters=%s\n' \
  "$JOB" "$MONITOR" "$RUN" "$COMMIT" "$BASE_CHECKPOINT_PATH" 20000 512 1.0 0.01 50
