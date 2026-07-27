#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run the CPU contract checks, real Wan2.2 VAE encode/decode, strict
# production-trainer Cosmos3-Nano DCP load, and an EMA-off/action-head-only
# forward-backward-update-save smoke as separate processes in one H200
# allocation. The action smoke is disposable evidence; it is not a formal
# full-optimizer, EMA, multi-GPU FSDP, or resumability claim.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${GATE_OUTPUT_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-gates}"
: "${GATE_LOG_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/logs/cwam-cosmos3-gates}"
: "${RUN_CPU_GATE:=1}"
: "${RUN_VAE_GATE:=1}"
: "${RUN_TRAINER_LOAD_GATE:=1}"
: "${RUN_ACTION_BACKWARD_GATE:=1}"

for flag in RUN_CPU_GATE RUN_VAE_GATE RUN_TRAINER_LOAD_GATE RUN_ACTION_BACKWARD_GATE; do
    [[ "${!flag}" =~ ^[01]$ ]] ||
        { echo "ERROR: $flag must be 0 or 1" >&2; exit 1; }
done
if [[ "$RUN_CPU_GATE" == 0 && "$RUN_VAE_GATE" == 0 && "$RUN_TRAINER_LOAD_GATE" == 0 &&
    "$RUN_ACTION_BACKWARD_GATE" == 0 ]]; then
    echo "ERROR: at least one gate must be enabled" >&2
    exit 1
fi

test -d "$COSMOS_SOURCE"
case "$COSMOS_SOURCE" in
    /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots/*) ;;
    *)
        echo "ERROR: COSMOS_SOURCE must be a personal immutable run snapshot: $COSMOS_SOURCE" >&2
        exit 1
        ;;
esac
CWAM_WRITABLE_ENTRY="$(find "$COSMOS_SOURCE" -xdev -perm /222 -print -quit)"
if [[ -n "$CWAM_WRITABLE_ENTRY" ]]; then
    echo "ERROR: immutable source contains a write-enabled entry: $CWAM_WRITABLE_ENTRY" >&2
    exit 1
fi
test -x "$COSMOS_ENV/bin/python"
test -f "$COSMOS_ENV/.cwam_env_ready.json"
# The marker and post-training contracts use Python assertions; never inherit -O.
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
source "$COSMOS_SOURCE/examples/h_cluster/configure_torchcodec_runtime.sh"
PYTHONNOUSERSITE=1 "$COSMOS_ENV/bin/python" \
    "$COSMOS_SOURCE/examples/h_cluster/validate_cu128_train_env_marker.py" \
    --environment "$COSMOS_ENV"

GATE_RUN_ID="${GATE_RUN_ID:-cwam-c3-1g-gates-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$GATE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: GATE_RUN_ID contains unsupported characters" >&2; exit 1; }
mkdir -p "$GATE_LOG_ROOT" "$GATE_OUTPUT_ROOT"
GATE_LOG="$GATE_LOG_ROOT/${GATE_RUN_ID}.log"
exec > >(tee -a "$GATE_LOG") 2>&1

echo "GATE_RUN_ID=$GATE_RUN_ID"
echo "GATE_LOG=$GATE_LOG"
echo "COSMOS_SOURCE=$COSMOS_SOURCE"
date -Is

export PATH="$COSMOS_ENV/bin:$PATH"
export PYTHONPATH="$COSMOS_SOURCE"
export PYTHONNOUSERSITE=1
export COSMOS_DEVICE=cuda
export COSMOS_INTERNAL=0
export COSMOS_TRAINING=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export COSMOS_ENV ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT
export BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH

cd "$COSMOS_SOURCE"

if [[ "$RUN_CPU_GATE" == 1 ]]; then
    env \
        RUN_CPU_GATE=1 \
        RUN_GPU_GATE=0 \
        VALIDATION_RUN_ID="${GATE_RUN_ID}-cpu" \
        bash examples/h_cluster/run_robocasa365_cosmos3_nano_validation.sh
fi

if [[ "$RUN_VAE_GATE" == 1 ]]; then
    env \
        RUN_CPU_GATE=0 \
        RUN_GPU_GATE=1 \
        VALIDATION_RUN_ID="${GATE_RUN_ID}-vae" \
        bash examples/h_cluster/run_robocasa365_cosmos3_nano_validation.sh
fi

if [[ "$RUN_TRAINER_LOAD_GATE" == 1 ]]; then
    export NPROC_PER_NODE=1
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export RUN_NAME="${GATE_RUN_ID}-load-only"
    export OUTPUT_ROOT="$GATE_OUTPUT_ROOT"
    export IMAGINAIRE_OUTPUT_ROOT="$GATE_OUTPUT_ROOT"
    export EXTRA_TAIL_OVERRIDES="
trainer.max_iter=0
trainer.run_validation=false
trainer.run_validation_on_start=false
trainer.save_zero_checkpoint=false
checkpoint.strict_resume=true
model.config.compile.enabled=false
trainer.callbacks.compile_tokenizer.enabled=false
trainer.callbacks.production_batch_memory_gate.required=false
dataloader_train.max_samples_per_batch=1
dataloader_train.dataloader.num_workers=0
dataloader_train.dataloader.persistent_workers=false
dataloader_train.dataloader.prefetch_factor=null
dataloader_val.dataloader.num_workers=0
dataloader_val.dataloader.persistent_workers=false
dataloader_val.dataloader.prefetch_factor=null
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh

    RUN_OUTPUT_DIR="$GATE_OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft/$RUN_NAME"
    test -f "$RUN_OUTPUT_DIR/config.yaml"
    if [[ -e "$RUN_OUTPUT_DIR/checkpoints/latest_checkpoint.txt" ]]; then
        echo "ERROR: load-only gate unexpectedly wrote a training checkpoint" >&2
        exit 1
    fi
fi

if [[ "$RUN_ACTION_BACKWARD_GATE" == 1 ]]; then
    export NPROC_PER_NODE=1
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export RUN_NAME="${GATE_RUN_ID}-action1"
    export OUTPUT_ROOT="$GATE_OUTPUT_ROOT"
    export IMAGINAIRE_OUTPUT_ROOT="$GATE_OUTPUT_ROOT"
    export EXTRA_TAIL_OVERRIDES="
trainer.max_iter=1
trainer.grad_accum_iter=2
trainer.run_validation=false
trainer.run_validation_on_start=false
trainer.save_zero_checkpoint=false
checkpoint.strict_resume=true
checkpoint.save_iter=1
model.config.ema.enabled=false
model.config.compile.enabled=false
trainer.callbacks.compile_tokenizer.enabled=false
trainer.callbacks.action_gradient_evidence.every_n=1
trainer.callbacks.production_batch_memory_gate.required=false
trainer.callbacks.device_monitor.every_n=1
optimizer.keys_to_select=[action2llm,llm2action,action_modality_embed]
checkpoint.keys_to_skip_loading=[action2llm,llm2action,action_modality_embed,action_pos_embed]
dataloader_train.max_samples_per_batch=1
dataloader_train.dataloader.num_workers=0
dataloader_train.dataloader.persistent_workers=false
dataloader_train.dataloader.prefetch_factor=null
dataloader_val.dataloader.num_workers=0
dataloader_val.dataloader.persistent_workers=false
dataloader_val.dataloader.prefetch_factor=null
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh

    ACTION_OUTPUT_DIR="$GATE_OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft/$RUN_NAME"
    "$COSMOS_ENV/bin/python" - "$ACTION_OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

output_dir = Path(sys.argv[1])
config_path = output_dir / "config.yaml"
latest_path = output_dir / "checkpoints" / "latest_checkpoint.txt"
evidence_path = output_dir / "action_gradient_evidence" / "iteration_000000001.json"

if not config_path.is_file():
    raise SystemExit(f"missing resolved config: {config_path}")
if latest_path.read_text().strip() != "iter_000000001":
    raise SystemExit(f"unexpected latest checkpoint pointer: {latest_path.read_text()!r}")
checkpoint_dir = latest_path.parent / "iter_000000001"
for component in ("model", "optim", "scheduler", "trainer"):
    metadata = checkpoint_dir / component / ".metadata"
    if not metadata.is_file():
        raise SystemExit(f"missing DCP component metadata: {metadata}")

config = yaml.safe_load(config_path.read_text())
trainer = config["trainer"]
checkpoint = config["checkpoint"]
model_config = config["model"]["config"]
optimizer = config["optimizer"]
train_loader = config["dataloader_train"]
val_loader = config["dataloader_val"]
expected_action_keys = ["action2llm", "llm2action", "action_modality_embed"]
expected_skip_keys = expected_action_keys + ["action_pos_embed"]
assert trainer["max_iter"] == 1
assert trainer["grad_accum_iter"] == 2
assert trainer["run_validation"] is False
assert trainer["run_validation_on_start"] is False
assert checkpoint["save_iter"] == 1
assert checkpoint["strict_resume"] is True
assert model_config["ema"]["enabled"] is False
assert model_config["compile"]["enabled"] is False
assert model_config["parallelism"]["data_parallel_shard_degree"] == 1
assert optimizer["keys_to_select"] == expected_action_keys
assert checkpoint["keys_to_skip_loading"] == expected_skip_keys
assert train_loader["max_samples_per_batch"] == 1
assert train_loader["dataloader"]["num_workers"] == 0
assert val_loader["dataloader"]["num_workers"] == 0

payload = json.loads(evidence_path.read_text())
if payload.get("status") != "passed":
    raise SystemExit(f"action-gradient evidence did not pass: {payload}")
losses = payload.get("losses", {})
for key in (
    "final_microbatch_total_loss_rank_mean",
    "final_microbatch_flow_matching_loss_action_rank_mean",
):
    value = float(losses[key])
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"invalid {key}: {value}")
for family in expected_action_keys:
    stats = payload["families"][family]
    parameter_elements = int(stats["parameter_elements"])
    if parameter_elements <= 0:
        raise SystemExit(f"{family}: no parameters")
    for key in ("trainable_elements", "optimizer_elements", "gradient_elements"):
        if int(stats[key]) != parameter_elements:
            raise SystemExit(f"{family}: {key}={stats[key]} != {parameter_elements}")
    if int(stats["nonfinite_gradient_elements"]) != 0:
        raise SystemExit(f"{family}: non-finite gradients")
    if int(stats["nonzero_gradient_elements"]) <= 0:
        raise SystemExit(f"{family}: zero gradients")
print(
    json.dumps(
        {
            "status": "passed",
            "checkpoint": str(checkpoint_dir),
            "gradient_evidence": str(evidence_path),
            "scope": "single_gpu_ema_off_action_head_only_disposable_smoke",
        },
        sort_keys=True,
    )
)
PY
fi

echo "CWAM_COSMOS3_1GPU_GATES_PASSED cpu=$RUN_CPU_GATE vae=$RUN_VAE_GATE trainer_load=$RUN_TRAINER_LOAD_GATE action_backward=$RUN_ACTION_BACKWARD_GATE"
date -Is
