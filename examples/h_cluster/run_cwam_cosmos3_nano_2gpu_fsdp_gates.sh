#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Single-node, two-H200 CWAM/Cosmos3-Nano training gates:
#   A. compile-off FSDP step 1 + synchronous DCP save;
#   B. a new torchrun process strictly restores model/optimizer/scheduler/
#      trainer/RNG state, explicitly acknowledges iterable-data replay, advances
#      one optimizer step, and saves step 2;
#   C. a fresh compile-on run with the same gate configuration (apart from the
#      run name and compile switch), one optimizer step, and a DCP save.
#
# This is a structural/runtime gate. It does NOT establish an exact dataloader
# cursor resume, eager/compile numerical equivalence, production-batch memory
# safety, convergence, or closed-loop control success.

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
: "${NVIDIA_SMI_INTERVAL_SECONDS:=2}"
: "${NVIDIA_SMI_HEADROOM_FRACTION:=0.94}"
: "${NVIDIA_SMI_HARD_MAX_FRACTION:=0.98}"

GATE_RUN_ID="${GATE_RUN_ID:-cwam-c3-2g-fsdp-gates-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$GATE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: GATE_RUN_ID contains unsupported characters" >&2; exit 1; }
[[ "$NVIDIA_SMI_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: NVIDIA_SMI_INTERVAL_SECONDS must be a positive integer" >&2; exit 1; }

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
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
source "$COSMOS_SOURCE/examples/h_cluster/configure_torchcodec_runtime.sh"
PYTHONNOUSERSITE=1 "$COSMOS_ENV/bin/python" \
    "$COSMOS_SOURCE/examples/h_cluster/validate_cu128_train_env_marker.py" \
    --environment "$COSMOS_ENV"
command -v nvidia-smi >/dev/null

EAGER_RUN_NAME="${GATE_RUN_ID}-eager"
COMPILE_RUN_NAME="${GATE_RUN_ID}-compile"
VALIDATION_RUN_NAME="${GATE_RUN_ID}-validation"
CANONICAL_RUN_ROOT="$GATE_OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft"
EAGER_RUN_DIR="$CANONICAL_RUN_ROOT/$EAGER_RUN_NAME"
COMPILE_RUN_DIR="$CANONICAL_RUN_ROOT/$COMPILE_RUN_NAME"
VALIDATION_RUN_DIR="$CANONICAL_RUN_ROOT/$VALIDATION_RUN_NAME"
EVIDENCE_DIR="$GATE_OUTPUT_ROOT/gate_evidence/$GATE_RUN_ID"
PHASE_LOG_DIR="$GATE_LOG_ROOT/$GATE_RUN_ID"
SUMMARY_PATH="$EVIDENCE_DIR/summary.json"
INDUCTOR_CACHE_DIR="/tmp/${GATE_RUN_ID}-inductor"

for fresh_path in \
    "$EAGER_RUN_DIR" \
    "$COMPILE_RUN_DIR" \
    "$VALIDATION_RUN_DIR" \
    "$EVIDENCE_DIR" \
    "$PHASE_LOG_DIR" \
    "$INDUCTOR_CACHE_DIR"; do
    if [[ -e "$fresh_path" ]]; then
        echo "ERROR: refusing to reuse a gate path: $fresh_path" >&2
        exit 1
    fi
done
mkdir -p "$EVIDENCE_DIR/assertions" "$EVIDENCE_DIR/configs" "$EVIDENCE_DIR/gpu_samples" "$PHASE_LOG_DIR"

SAMPLER_PID=""
SAMPLER_FILE=""
SAMPLER_STDERR=""
CURRENT_PHASE="preflight"
FINAL_STATUS="failed"

cleanup_sampler() {
    local pid="${SAMPLER_PID:-}"
    if [[ -z "$pid" ]]; then
        return 0
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
    SAMPLER_PID=""
}

write_failure_summary() {
    local exit_code="$1"
    set +e
    if [[ "$FINAL_STATUS" != "passed" && -d "$EVIDENCE_DIR" && -x "$COSMOS_ENV/bin/python" ]]; then
        "$COSMOS_ENV/bin/python" - "$SUMMARY_PATH" "$GATE_RUN_ID" "$CURRENT_PHASE" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "failed",
    "gate_run_id": sys.argv[2],
    "failed_phase": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "resume_scope": "functional_state_resume_with_acknowledged_iterable_data_replay",
    "exact_data_resume": False,
}
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY
    fi
}

on_exit() {
    local exit_code="$1"
    set +e
    cleanup_sampler
    write_failure_summary "$exit_code"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

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
export NPROC_PER_NODE=2
export OUTPUT_ROOT="$GATE_OUTPUT_ROOT"
export IMAGINAIRE_OUTPUT_ROOT="$GATE_OUTPUT_ROOT"
export COSMOS_ENV ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT
export BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH

cd "$COSMOS_SOURCE"

"$COSMOS_ENV/bin/python" - "$EVIDENCE_DIR/preflight.json" "$COSMOS_SOURCE" "$COSMOS_ENV" \
    "$GATE_RUN_ID" "$NVIDIA_SMI_HEADROOM_FRACTION" "$NVIDIA_SMI_HARD_MAX_FRACTION" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

target = Path(sys.argv[1])
source = Path(sys.argv[2])
environment = Path(sys.argv[3])
headroom_fraction = float(sys.argv[5])
hard_max_fraction = float(sys.argv[6])
if not (0.0 < headroom_fraction < hard_max_fraction < 1.0):
    raise SystemExit(
        f"invalid GPU memory thresholds: headroom={headroom_fraction}, hard_max={hard_max_fraction}"
    )
if torch.cuda.device_count() != 2:
    raise SystemExit(f"2-GPU gate requires exactly two visible CUDA devices, got {torch.cuda.device_count()}")
devices = []
for index in range(2):
    properties = torch.cuda.get_device_properties(index)
    if "H200" not in properties.name:
        raise SystemExit(f"CUDA device {index} is not an H200: {properties.name}")
    devices.append(
        {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
        }
    )
payload = {
    "schema_version": 1,
    "status": "passed",
    "gate_run_id": sys.argv[4],
    "source": str(source),
    "environment": str(environment),
    "devices": devices,
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    "nccl_ib_disable": os.environ["NCCL_IB_DISABLE"],
    "nccl_nvls_enable": os.environ["NCCL_NVLS_ENABLE"],
    "resume_scope": "functional_state_resume_with_acknowledged_iterable_data_replay",
    "exact_data_resume": False,
    "memory_thresholds": {
        "headroom_fraction": headroom_fraction,
        "hard_max_fraction": hard_max_fraction,
    },
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY

COMMON_OVERRIDES='
trainer.distributed_parallelism=fsdp
trainer.grad_accum_iter=1
trainer.logging_iter=1
trainer.run_validation=false
trainer.run_validation_on_start=false
trainer.save_zero_checkpoint=false
trainer.callbacks.compile_tokenizer.enabled=false
trainer.callbacks.action_gradient_evidence.every_n=1
trainer.callbacks.action_gradient_evidence.fail_on_first_step=true
trainer.callbacks.production_batch_memory_gate.required=false
trainer.callbacks.device_monitor.every_n=1
trainer.callbacks.training_stats.log_freq=1
checkpoint.strict_resume=true
checkpoint.save_iter=1
checkpoint.dcp_async_mode_enabled=false
checkpoint.load_training_state=false
checkpoint.load_from_object_store.enabled=false
checkpoint.save_to_object_store.enabled=false
checkpoint.hf_export.enabled=false
model.config.ema.enabled=true
model.config.compile.compile_dynamic=true
model.config.compile.compiled_region=language
model.config.parallelism.data_parallel_replicate_degree=1
model.config.parallelism.context_parallel_shard_degree=1
model.config.parallelism.cfg_parallel_shard_degree=1
dataloader_train.max_samples_per_batch=1
dataloader_train.dataloader.num_workers=0
dataloader_train.dataloader.persistent_workers=false
dataloader_train.dataloader.prefetch_factor=null
dataloader_val.max_samples_per_batch=1
dataloader_val.dataloader.num_workers=0
dataloader_val.dataloader.persistent_workers=false
dataloader_val.dataloader.prefetch_factor=null
'

start_sampler() {
    local phase="$1"
    SAMPLER_FILE="$EVIDENCE_DIR/gpu_samples/${phase}.csv"
    SAMPLER_STDERR="$EVIDENCE_DIR/gpu_samples/${phase}.stderr"
    nvidia-smi \
        --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits \
        --loop="$NVIDIA_SMI_INTERVAL_SECONDS" \
        >"$SAMPLER_FILE" 2>"$SAMPLER_STDERR" &
    SAMPLER_PID=$!

    for _ in {1..50}; do
        if [[ -s "$SAMPLER_FILE" ]]; then
            return 0
        fi
        if ! kill -0 "$SAMPLER_PID" 2>/dev/null; then
            wait "$SAMPLER_PID" 2>/dev/null || true
            echo "ERROR: nvidia-smi sampler exited before producing data for $phase" >&2
            return 1
        fi
        sleep 0.2
    done
    echo "ERROR: nvidia-smi sampler produced no data for $phase" >&2
    return 1
}

stop_sampler() {
    local require_alive="${1:-0}"
    local was_alive=0
    if [[ -z "${SAMPLER_PID:-}" ]]; then
        [[ "$require_alive" == 0 ]] && return 0
        echo "ERROR: sampler PID is absent" >&2
        return 1
    fi
    if kill -0 "$SAMPLER_PID" 2>/dev/null; then
        was_alive=1
    fi
    cleanup_sampler
    if [[ "$require_alive" == 1 && "$was_alive" != 1 ]]; then
        echo "ERROR: nvidia-smi sampler died during the phase" >&2
        return 1
    fi
}

validate_sampler() {
    local phase="$1"
    local csv_path="$2"
    local output_path="$EVIDENCE_DIR/assertions/${phase}-gpu-memory.json"
    "$COSMOS_ENV/bin/python" - "$phase" "$csv_path" "$output_path" \
        "$NVIDIA_SMI_HEADROOM_FRACTION" "$NVIDIA_SMI_HARD_MAX_FRACTION" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

phase, csv_name, output_name = sys.argv[1:4]
headroom_fraction = float(sys.argv[4])
hard_max_fraction = float(sys.argv[5])
records = []
with Path(csv_name).open(newline="") as handle:
    for row_number, row in enumerate(csv.reader(handle), start=1):
        row = [value.strip() for value in row]
        if not row or not any(row):
            continue
        if len(row) != 9:
            raise SystemExit(f"{phase}: malformed nvidia-smi row {row_number}: {row!r}")
        timestamp, index, uuid, name, total, used, free, utilization, power = row
        try:
            parsed = {
                "timestamp": timestamp,
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": float(total),
                "memory_used_mib": float(used),
                "memory_free_mib": float(free),
                "utilization_gpu_percent": float(utilization),
                "power_draw_w": (
                    None
                    if power.casefold() in {"n/a", "[n/a]", "not supported"}
                    else float(power)
                ),
            }
        except ValueError as error:
            raise SystemExit(f"{phase}: invalid nvidia-smi row {row_number}: {row!r}: {error}") from error
        for key in ("memory_total_mib", "memory_used_mib", "memory_free_mib"):
            if not math.isfinite(parsed[key]):
                raise SystemExit(f"{phase}: non-finite {key} in row {row_number}")
        if "H200" not in name or not (140000.0 <= parsed["memory_total_mib"] <= 150000.0):
            raise SystemExit(f"{phase}: unexpected GPU identity/capacity: {parsed}")
        parsed["memory_used_fraction"] = parsed["memory_used_mib"] / parsed["memory_total_mib"]
        records.append(parsed)

by_uuid = defaultdict(list)
for record in records:
    by_uuid[record["uuid"]].append(record)
if len(by_uuid) != 2:
    raise SystemExit(f"{phase}: expected samples from exactly two GPU UUIDs, got {sorted(by_uuid)}")
if any(not samples for samples in by_uuid.values()):
    raise SystemExit(f"{phase}: one or more GPUs have no samples")

per_gpu = {}
global_max = 0.0
for uuid, samples in sorted(by_uuid.items()):
    max_used = max(item["memory_used_fraction"] for item in samples)
    global_max = max(global_max, max_used)
    per_gpu[uuid] = {
        "name": samples[0]["name"],
        "sample_count": len(samples),
        "max_memory_used_fraction": max_used,
        "max_memory_used_mib": max(item["memory_used_mib"] for item in samples),
        "min_memory_free_mib": min(item["memory_free_mib"] for item in samples),
        "max_utilization_gpu_percent": max(item["utilization_gpu_percent"] for item in samples),
    }
if global_max >= hard_max_fraction:
    raise SystemExit(
        f"{phase}: peak sampled GPU-memory fraction {global_max:.6f} >= hard limit {hard_max_fraction:.6f}"
    )

payload = {
    "schema_version": 1,
    "status": "passed",
    "phase": phase,
    "sample_file": str(Path(csv_name)),
    "gpu_count": len(by_uuid),
    "sample_count": len(records),
    "max_memory_used_fraction": global_max,
    "memory_classification": "headroom" if global_max <= headroom_fraction else "memory_tight",
    "headroom_fraction": headroom_fraction,
    "hard_max_fraction": hard_max_fraction,
    "per_gpu": per_gpu,
}
target = Path(output_name)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY
}

run_logged_phase() {
    local phase="$1"
    local outer_log="$2"
    shift 2
    CURRENT_PHASE="$phase"
    start_sampler "$phase"

    set +e
    {
        echo "PHASE_START=$phase"
        date -Is
        "$@"
    } 2>&1 | tee "$outer_log"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e

    local sampler_status=0
    stop_sampler 1 || sampler_status=$?
    if [[ "${pipeline_status[0]}" -ne 0 ]]; then
        echo "ERROR: phase $phase failed with exit ${pipeline_status[0]}" >&2
        return "${pipeline_status[0]}"
    fi
    if [[ "${pipeline_status[1]}" -ne 0 ]]; then
        echo "ERROR: tee failed for phase $phase with exit ${pipeline_status[1]}" >&2
        return "${pipeline_status[1]}"
    fi
    if [[ "$sampler_status" -ne 0 ]]; then
        return "$sampler_status"
    fi
    validate_sampler "$phase" "$EVIDENCE_DIR/gpu_samples/${phase}.csv"
    echo "PHASE_FINISH=$phase"
}

archive_config() {
    local phase="$1"
    local run_dir="$2"
    local archive_root="$EVIDENCE_DIR/configs"
    local files=(config.yaml config.pkl launch_info.yaml job_env.yaml)
    for name in "${files[@]}"; do
        test -s "$run_dir/$name"
        cp -p -- "$run_dir/$name" "$archive_root/${phase}-${name}"
    done
    (
        cd "$archive_root"
        sha256sum "${phase}-config.yaml" "${phase}-config.pkl" \
            "${phase}-launch_info.yaml" "${phase}-job_env.yaml"
    ) >"$archive_root/${phase}-sha256.txt"
}

assert_phase_artifacts() {
    local phase="$1"
    local run_dir="$2"
    local step="$3"
    local max_iter="$4"
    local compile_enabled="$5"
    local archived_config="$EVIDENCE_DIR/configs/${phase}-config.yaml"
    local sampler_assertion="$EVIDENCE_DIR/assertions/${phase}-gpu-memory.json"
    local output_path="$EVIDENCE_DIR/assertions/${phase}-artifacts.json"

    "$COSMOS_ENV/bin/python" - "$phase" "$run_dir" "$step" "$max_iter" \
        "$compile_enabled" "$archived_config" "$sampler_assertion" "$output_path" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

phase = sys.argv[1]
run_dir = Path(sys.argv[2])
step = int(sys.argv[3])
max_iter = int(sys.argv[4])
compile_enabled = sys.argv[5].lower() == "true"
config_path = Path(sys.argv[6])
sampler_assertion = Path(sys.argv[7])
output_path = Path(sys.argv[8])
config = yaml.safe_load(config_path.read_text())

expected_optimizer_keys = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
expected_skip_keys = [
    "net_ema.",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "action_pos_embed",
]
trainer = config["trainer"]
checkpoint = config["checkpoint"]
model_config = config["model"]["config"]
optimizer = config["optimizer"]
parallelism = model_config["parallelism"]
compile_config = model_config["compile"]

assert trainer["distributed_parallelism"] == "fsdp"
assert trainer["max_iter"] == max_iter
assert trainer["grad_accum_iter"] == 1
assert trainer["logging_iter"] == 1
assert trainer["run_validation"] is False
assert trainer["run_validation_on_start"] is False
assert trainer["save_zero_checkpoint"] is False
assert checkpoint["strict_resume"] is True
assert checkpoint["save_iter"] == 1
assert checkpoint["dcp_async_mode_enabled"] is False
assert checkpoint["load_training_state"] is False
assert checkpoint["load_from_object_store"]["enabled"] is False
assert checkpoint["save_to_object_store"]["enabled"] is False
assert checkpoint["hf_export"]["enabled"] is False
assert checkpoint["keys_to_skip_loading"] == expected_skip_keys
assert optimizer["keys_to_select"] == expected_optimizer_keys
assert model_config["ema"]["enabled"] is True
assert compile_config["enabled"] is compile_enabled
assert compile_config["compile_dynamic"] is True
assert compile_config["compiled_region"] == "language"
assert parallelism["data_parallel_shard_degree"] == 2
assert parallelism["data_parallel_replicate_degree"] == 1
assert parallelism["context_parallel_shard_degree"] == 1
assert parallelism["cfg_parallel_shard_degree"] == 1
for loader_name in ("dataloader_train", "dataloader_val"):
    loader = config[loader_name]
    assert loader["max_samples_per_batch"] == 1
    inner = loader["dataloader"]
    assert inner["num_workers"] == 0
    assert inner["persistent_workers"] is False
    assert inner["prefetch_factor"] is None

latest = run_dir / "checkpoints" / "latest_checkpoint.txt"
expected_pointer = f"iter_{step:09d}"
if latest.read_text().strip() != expected_pointer:
    raise SystemExit(f"{phase}: unexpected checkpoint pointer: {latest.read_text()!r}")
checkpoint_dir = latest.parent / expected_pointer
component_files = {}
for component in ("model", "optim", "scheduler", "trainer"):
    component_dir = checkpoint_dir / component
    metadata_file = component_dir / ".metadata"
    if not metadata_file.is_file() or metadata_file.stat().st_size <= 0:
        raise SystemExit(f"{phase}: missing/empty DCP metadata: {metadata_file}")
    distcp_files = [path for path in component_dir.glob("*.distcp") if path.stat().st_size > 0]
    if not distcp_files:
        raise SystemExit(f"{phase}: no non-empty DCP shards in {component_dir}")
    component_files[component] = {
        "metadata_bytes": metadata_file.stat().st_size,
        "nonempty_distcp_file_count": len(distcp_files),
        "distcp_bytes": sum(path.stat().st_size for path in distcp_files),
    }

if (checkpoint_dir / "dataloader").exists():
    raise SystemExit(f"{phase}: unexpected dataloader cursor checkpoint; replay contract must be re-audited")

evidence_path = run_dir / "action_gradient_evidence" / f"iteration_{step:09d}.json"
evidence = json.loads(evidence_path.read_text())
assert evidence["status"] == "passed"
assert evidence["iteration"] == step
assert evidence["world_size"] == 2
losses = evidence["losses"]
for key in (
    "final_microbatch_total_loss_rank_mean",
    "final_microbatch_flow_matching_loss_action_rank_mean",
):
    value = float(losses[key])
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"{phase}: invalid {key}: {value}")
for family in ("action2llm", "llm2action", "action_modality_embed"):
    stats = evidence["families"][family]
    parameter_elements = int(stats["parameter_elements"])
    if parameter_elements <= 0:
        raise SystemExit(f"{phase}: {family} has no parameters")
    for key in ("trainable_elements", "optimizer_elements", "gradient_elements"):
        if int(stats[key]) != parameter_elements:
            raise SystemExit(f"{phase}: {family} {key} does not cover all parameters")
    if int(stats["nonfinite_gradient_elements"]) != 0:
        raise SystemExit(f"{phase}: {family} has non-finite gradients")
    if int(stats["nonzero_gradient_elements"]) <= 0:
        raise SystemExit(f"{phase}: {family} has zero gradients")

memory = json.loads(sampler_assertion.read_text())
assert memory["status"] == "passed"
assert memory["gpu_count"] == 2

payload = {
    "schema_version": 1,
    "status": "passed",
    "phase": phase,
    "run_dir": str(run_dir),
    "checkpoint": str(checkpoint_dir),
    "checkpoint_pointer": expected_pointer,
    "dcp_component_files": component_files,
    "config": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "action_gradient_evidence": str(evidence_path),
    "action_loss": losses["final_microbatch_flow_matching_loss_action_rank_mean"],
    "gpu_memory": memory,
    "resume_scope": "functional_state_resume_with_acknowledged_iterable_data_replay",
    "exact_data_resume": False,
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
}

assert_phase_log() {
    local phase="$1"
    local log_path="$2"
    local run_dir="$3"
    local output_path="$EVIDENCE_DIR/assertions/${phase}-log.json"
    "$COSMOS_ENV/bin/python" - "$phase" "$log_path" "$run_dir" "$output_path" <<'PY'
import json
import re
import sys
from pathlib import Path

phase = sys.argv[1]
log_path = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
output_path = Path(sys.argv[4])
text = log_path.read_text(errors="replace")
required = [
    "Distributed parallelism mode: fsdp",
    "Done with training.",
]
regex_required = [r"Building 2-D device mesh.*\[1, 2\]"]
if phase in ("a-eager", "c-compile"):
    required.extend(
        [
            "Resetting EMA weights from regular weights after regular-only warm start.",
            f"Saved checkpoint to {run_dir}/checkpoints/iter_000000001",
        ]
    )
elif phase == "b-resume":
    checkpoint_dir = run_dir / "checkpoints" / "iter_000000001"
    required.extend(
        [
            "WARNING: resuming model/optim/scheduler/trainer with a restarted data stream.",
            f"Resuming ckpt {checkpoint_dir} with keys:",
            f"Start loading checkpoint from {checkpoint_dir}/model",
            f"Start loading checkpoint from {checkpoint_dir}/optim",
            f"Start loading checkpoint from {checkpoint_dir}/scheduler",
            f"Start loading checkpoint from {checkpoint_dir}/trainer",
            f"Checkpoint {checkpoint_dir}/dataloader does not exist, skip loading dataloader.",
            f"Loaded checkpoint from {checkpoint_dir} in iteration 1",
            f"Saved checkpoint to {run_dir}/checkpoints/iter_000000002",
        ]
    )
else:
    raise SystemExit(f"unsupported phase: {phase}")

missing = [needle for needle in required if needle not in text]
for pattern in regex_required:
    if re.search(pattern, text) is None:
        missing.append(f"regex:{pattern}")
forbidden = [
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ncclUnhandledCudaError",
    '"status": "failed"',
]
if phase == "b-resume":
    forbidden.extend(["Training from scratch.", "[DCP-LOAD-SKIP-KEYS]"])
present_forbidden = [needle for needle in forbidden if needle in text]
if missing or present_forbidden:
    raise SystemExit(
        f"{phase}: log contract failed; missing={missing}, forbidden_present={present_forbidden}"
    )

payload = {
    "schema_version": 1,
    "status": "passed",
    "phase": phase,
    "log": str(log_path),
    "required_substrings": required,
    "required_regexes": regex_required,
    "forbidden_substrings": forbidden,
    "resume_scope": "functional_state_resume_with_acknowledged_iterable_data_replay",
    "exact_data_resume": False,
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
}

assert_validation_phase() {
    local phase="$1"
    local run_dir="$2"
    local log_path="$3"
    local archived_config="$EVIDENCE_DIR/configs/${phase}-config.yaml"
    local sampler_assertion="$EVIDENCE_DIR/assertions/${phase}-gpu-memory.json"
    local output_path="$EVIDENCE_DIR/assertions/${phase}-validation.json"

    "$COSMOS_ENV/bin/python" - "$phase" "$run_dir" "$log_path" "$archived_config" \
        "$sampler_assertion" "$output_path" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml

phase = sys.argv[1]
run_dir = Path(sys.argv[2])
log_path = Path(sys.argv[3])
config_path = Path(sys.argv[4])
sampler_assertion = Path(sys.argv[5])
output_path = Path(sys.argv[6])
config = yaml.safe_load(config_path.read_text())
trainer = config["trainer"]
checkpoint = config["checkpoint"]
model_config = config["model"]["config"]
parallelism = model_config["parallelism"]

assert trainer["distributed_parallelism"] == "fsdp"
assert trainer["max_iter"] == 0
assert trainer["max_val_iter"] == 1
assert trainer["run_validation"] is True
assert trainer["run_validation_on_start"] is True
assert trainer["save_zero_checkpoint"] is False
assert checkpoint["strict_resume"] is True
assert checkpoint["load_training_state"] is False
assert checkpoint["dcp_async_mode_enabled"] is False
assert model_config["ema"]["enabled"] is True
assert model_config["compile"]["enabled"] is False
assert parallelism["data_parallel_shard_degree"] == 2
assert parallelism["data_parallel_replicate_degree"] == 1
assert config["dataloader_train"]["dataloader"]["num_workers"] == 0
val_loader = config["dataloader_val"]
assert val_loader["max_samples_per_batch"] == 1
assert val_loader["restart_on_iter"] is True
assert val_loader["dataloader"]["num_workers"] == 2
assert val_loader["dataloader"]["persistent_workers"] is True
assert val_loader["dataloader"]["prefetch_factor"] == 4

latest = run_dir / "checkpoints" / "latest_checkpoint.txt"
if latest.exists():
    raise SystemExit(f"{phase}: max_iter=0 validation gate unexpectedly wrote {latest}")

text = log_path.read_text(errors="replace")
required = [
    "Distributed parallelism mode: fsdp",
    "Resetting EMA weights from regular weights after regular-only warm start.",
    "validation: Switched to EMA weights",
    "validation: Restored training weights",
    "Done with training.",
]
missing = [needle for needle in required if needle not in text]
forbidden = [
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ncclUnhandledCudaError",
    "Unstable val loss",
    '"status": "failed"',
]
present_forbidden = [needle for needle in forbidden if needle in text]
matches = re.findall(
    r"\[val\] number of unique url and key: (\d+) / (\d+); avg_final_loss: "
    r"([-+0-9.eE]+)",
    text,
)
if len(matches) != 1:
    raise SystemExit(f"{phase}: expected one rank-0 validation summary, found {matches!r}")
unique_count, total_count, loss_text = matches[0]
unique_count = int(unique_count)
total_count = int(total_count)
loss = float(loss_text)
if unique_count != total_count or total_count != 2:
    raise SystemExit(
        f"{phase}: expected two distinct rank-partitioned validation samples, "
        f"got unique={unique_count}, total={total_count}"
    )
if not math.isfinite(loss) or loss <= 0.0:
    raise SystemExit(f"{phase}: invalid validation loss: {loss}")
if missing or present_forbidden:
    raise SystemExit(
        f"{phase}: validation log contract failed; "
        f"missing={missing}, forbidden_present={present_forbidden}"
    )

memory = json.loads(sampler_assertion.read_text())
assert memory["status"] == "passed"
assert memory["gpu_count"] == 2
payload = {
    "schema_version": 1,
    "status": "passed",
    "phase": phase,
    "run_dir": str(run_dir),
    "config": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "validation_loss": loss,
    "validation_unique_url_keys": unique_count,
    "validation_total_url_keys": total_count,
    "ema_scope_observed": True,
    "persistent_workers": 2,
    "gpu_memory": memory,
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
}

assert_config_parity() {
    local mode="$1"
    local left="$2"
    local right="$3"
    local output_path="$EVIDENCE_DIR/assertions/${mode}-config-parity.json"
    "$COSMOS_ENV/bin/python" - "$mode" "$left" "$right" "$output_path" "$EAGER_RUN_NAME" <<'PY'
import copy
import json
import sys
from pathlib import Path

import yaml

mode = sys.argv[1]
left_path = Path(sys.argv[2])
right_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
eager_run_name = sys.argv[5]
left = yaml.safe_load(left_path.read_text())
right = yaml.safe_load(right_path.read_text())
normalized_left = copy.deepcopy(left)
normalized_right = copy.deepcopy(right)

if mode == "a-vs-b":
    normalized_right["trainer"]["max_iter"] = normalized_left["trainer"]["max_iter"]
    allowed_differences = ["trainer.max_iter"]
elif mode == "a-vs-c":
    eager_sampled_media_uri = normalized_left["trainer"]["callbacks"]["sampled_media"]["output_uri"]
    compile_run_name = normalized_right["job"]["name"]
    compile_sampled_media_uri = normalized_right["trainer"]["callbacks"]["sampled_media"]["output_uri"]
    expected_compile_sampled_media_uri = eager_sampled_media_uri.replace(
        f"/{eager_run_name}/sampled_media.lance",
        f"/{compile_run_name}/sampled_media.lance",
    )
    if expected_compile_sampled_media_uri == eager_sampled_media_uri:
        raise SystemExit(
            "a-vs-c: eager sampled-media URI does not contain the expected run-name-derived suffix"
        )
    if compile_sampled_media_uri != expected_compile_sampled_media_uri:
        raise SystemExit(
            "a-vs-c: compile sampled-media URI is not the expected run-name-derived path"
        )
    normalized_right["job"]["name"] = eager_run_name
    normalized_right["model"]["config"]["compile"]["enabled"] = normalized_left["model"]["config"]["compile"][
        "enabled"
    ]
    normalized_right["trainer"]["callbacks"]["sampled_media"]["output_uri"] = eager_sampled_media_uri
    allowed_differences = [
        "job.name",
        "model.config.compile.enabled",
        "trainer.callbacks.sampled_media.output_uri",
    ]
else:
    raise SystemExit(f"unsupported parity mode: {mode}")

if normalized_left != normalized_right:
    raise SystemExit(f"{mode}: configs differ outside the declared fields")
payload = {
    "schema_version": 1,
    "status": "passed",
    "mode": mode,
    "left": str(left_path),
    "right": str(right_path),
    "allowed_differences": allowed_differences,
    "numeric_parity_claim": False,
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
}

phase_a() {
    export RUN_NAME="$EAGER_RUN_NAME"
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export LOG_FILENAME="${GATE_RUN_ID}.a-eager.torchrun.log"
    export EXTRA_TAIL_OVERRIDES="$COMMON_OVERRIDES
trainer.max_iter=1
model.config.compile.enabled=false
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

phase_b() {
    export RUN_NAME="$EAGER_RUN_NAME"
    export RESUME_EXISTING_RUN=1
    export ALLOW_DATA_STREAM_REPLAY=1
    export LOG_FILENAME="${GATE_RUN_ID}.b-resume.torchrun.log"
    export EXTRA_TAIL_OVERRIDES="$COMMON_OVERRIDES
trainer.max_iter=2
model.config.compile.enabled=false
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

phase_c() {
    export RUN_NAME="$COMPILE_RUN_NAME"
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export LOG_FILENAME="${GATE_RUN_ID}.c-compile.torchrun.log"
    export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR"
    export TORCHDYNAMO_DISABLE=0
    export TORCH_LOGS=recompiles
    export EXTRA_TAIL_OVERRIDES="$COMMON_OVERRIDES
trainer.max_iter=1
model.config.compile.enabled=true
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

phase_d() {
    export RUN_NAME="$VALIDATION_RUN_NAME"
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export LOG_FILENAME="${GATE_RUN_ID}.d-validation.torchrun.log"
    export EXTRA_TAIL_OVERRIDES="$COMMON_OVERRIDES
trainer.max_iter=0
trainer.max_val_iter=1
trainer.run_validation=true
trainer.run_validation_on_start=true
model.config.compile.enabled=false
dataloader_val.dataloader.num_workers=2
dataloader_val.dataloader.persistent_workers=true
dataloader_val.dataloader.prefetch_factor=4
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

echo "GATE_RUN_ID=$GATE_RUN_ID"
echo "COSMOS_SOURCE=$COSMOS_SOURCE"
echo "EVIDENCE_DIR=$EVIDENCE_DIR"
echo "PHASE_LOG_DIR=$PHASE_LOG_DIR"
echo "RESUME_SCOPE=functional_state_resume_with_acknowledged_iterable_data_replay"
echo "EXACT_DATA_RESUME=false"
date -Is

A_OUTER_LOG="$PHASE_LOG_DIR/a-eager.outer.log"
run_logged_phase "a-eager" "$A_OUTER_LOG" phase_a
archive_config "a-eager" "$EAGER_RUN_DIR"
assert_phase_artifacts "a-eager" "$EAGER_RUN_DIR" 1 1 false
assert_phase_log "a-eager" "$A_OUTER_LOG" "$EAGER_RUN_DIR"

D_OUTER_LOG="$PHASE_LOG_DIR/d-validation.outer.log"
run_logged_phase "d-validation" "$D_OUTER_LOG" phase_d
archive_config "d-validation" "$VALIDATION_RUN_DIR"
assert_validation_phase "d-validation" "$VALIDATION_RUN_DIR" "$D_OUTER_LOG"

# Preserve A's max_iter=1 config before B reuses the canonical run directory
# and overwrites config.yaml/config.pkl with its max_iter=2 resolved config.
test "$(cat "$EAGER_RUN_DIR/checkpoints/latest_checkpoint.txt")" = "iter_000000001"

B_OUTER_LOG="$PHASE_LOG_DIR/b-resume.outer.log"
run_logged_phase "b-resume" "$B_OUTER_LOG" phase_b
archive_config "b-resume" "$EAGER_RUN_DIR"
assert_phase_artifacts "b-resume" "$EAGER_RUN_DIR" 2 2 false
assert_phase_log "b-resume" "$B_OUTER_LOG" "$EAGER_RUN_DIR"
test -s "$EAGER_RUN_DIR/checkpoints/iter_000000001/model/.metadata"
test -s "$EAGER_RUN_DIR/action_gradient_evidence/iteration_000000001.json"
assert_config_parity "a-vs-b" \
    "$EVIDENCE_DIR/configs/a-eager-config.yaml" \
    "$EVIDENCE_DIR/configs/b-resume-config.yaml"

mkdir -p "$INDUCTOR_CACHE_DIR"
C_OUTER_LOG="$PHASE_LOG_DIR/c-compile.outer.log"
run_logged_phase "c-compile" "$C_OUTER_LOG" phase_c
archive_config "c-compile" "$COMPILE_RUN_DIR"
assert_phase_artifacts "c-compile" "$COMPILE_RUN_DIR" 1 1 true
assert_phase_log "c-compile" "$C_OUTER_LOG" "$COMPILE_RUN_DIR"
assert_config_parity "a-vs-c" \
    "$EVIDENCE_DIR/configs/a-eager-config.yaml" \
    "$EVIDENCE_DIR/configs/c-compile-config.yaml"

"$COSMOS_ENV/bin/python" - "$INDUCTOR_CACHE_DIR" "$EVIDENCE_DIR/assertions/c-compile-inductor.json" <<'PY'
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
output = Path(sys.argv[2])
files = [path for path in cache.rglob("*") if path.is_file() and path.stat().st_size > 0]
if not files:
    raise SystemExit(f"compile-on gate produced no non-empty Inductor cache files under {cache}")
payload = {
    "schema_version": 1,
    "status": "passed",
    "cache": str(cache),
    "nonempty_file_count": len(files),
    "total_bytes": sum(path.stat().st_size for path in files),
    "numeric_parity_claim": False,
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY

CURRENT_PHASE="final_assertions"
"$COSMOS_ENV/bin/python" - "$EVIDENCE_DIR" "$SUMMARY_PATH" "$GATE_RUN_ID" \
    "$EAGER_RUN_DIR" "$COMPILE_RUN_DIR" "$VALIDATION_RUN_DIR" \
    "$A_OUTER_LOG" "$B_OUTER_LOG" "$C_OUTER_LOG" "$D_OUTER_LOG" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

evidence_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
required_assertions = [
    "a-eager-gpu-memory.json",
    "a-eager-artifacts.json",
    "a-eager-log.json",
    "d-validation-gpu-memory.json",
    "d-validation-validation.json",
    "b-resume-gpu-memory.json",
    "b-resume-artifacts.json",
    "b-resume-log.json",
    "a-vs-b-config-parity.json",
    "c-compile-gpu-memory.json",
    "c-compile-artifacts.json",
    "c-compile-log.json",
    "a-vs-c-config-parity.json",
    "c-compile-inductor.json",
]
assertion_payloads = {}
for name in required_assertions:
    path = evidence_dir / "assertions" / name
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed":
        raise SystemExit(f"assertion did not pass: {path}: {payload}")
    assertion_payloads[name] = payload

summary = {
    "schema_version": 1,
    "status": "passed",
    "gate_run_id": sys.argv[3],
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "eager_run_dir": sys.argv[4],
    "compile_run_dir": sys.argv[5],
    "validation_run_dir": sys.argv[6],
    "outer_logs": {
        "a_eager": sys.argv[7],
        "b_resume": sys.argv[8],
        "c_compile": sys.argv[9],
        "d_validation": sys.argv[10],
    },
    "assertions": {name: str(evidence_dir / "assertions" / name) for name in required_assertions},
    "resume_scope": "functional_model_optimizer_scheduler_trainer_rng_resume_with_iterable_data_replay",
    "exact_data_resume": False,
    "compile_parity_scope": "structural_runtime_parity_not_numeric_parity",
    "claims_supported": [
        "single_node_two_rank_fsdp_runtime",
        "production_recipe_optimizer_selection_and_ema_instantiated",
        "action_head_finite_nonzero_gradients_and_optimizer_membership",
        "model_optim_scheduler_trainer_dcp_save",
        "functional_state_resume_then_one_additional_optimizer_step",
        "compile_on_one_step_and_dcp_save",
        "real_two_rank_fsdp_ema_validation_with_distinct_sample_coverage",
    ],
    "claims_not_supported": [
        "exact_iterable_dataloader_cursor_resume",
        "eager_compile_bitwise_or_numeric_equivalence",
        "production_max_samples_per_batch_memory_safety",
        "eight_gpu_or_multinode_stability",
        "convergence_or_closed_loop_control_success",
    ],
}
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(summary_path)
PY

FINAL_STATUS="passed"
CURRENT_PHASE="complete"
echo "CWAM_COSMOS3_2GPU_FSDP_GATES_PASSED summary=$SUMMARY_PATH exact_data_resume=false"
date -Is
