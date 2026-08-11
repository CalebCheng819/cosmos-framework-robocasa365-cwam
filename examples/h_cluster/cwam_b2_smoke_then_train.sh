#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Execute the B2 experiment inside an already allocated eight-H200 worker.
# A two-step, production-batch smoke must pass before the 20k-step run starts.
# Runtime outputs stay on worker NVMe; checkpoints and evidence are copied to
# OSS and verified by path and byte size before older local checkpoints are
# removed.

set -Eeuo pipefail
umask 077

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable B2 source snapshot.}"
: "${FORMAL_RUN_NAME:=CWAM-B2-20260811-R1}"
: "${EXPECTED_HOST:=gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"
: "${CONTROL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-b2-20260811/control}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${ROBOCASA365_CONTROL_CACHE_ROOT:=/nvme/cwam-b2/inputs/control-cache}"
: "${ROBOCASA365_STRONG_LATENT_CACHE_ROOT:=/nvme/cwam-b2/inputs/latent-cache}"
: "${RUN_ROOT:=/nvme/cwam-b2/runs/$FORMAL_RUN_NAME}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
: "${RCLONE_CONFIG:=/dev/shm/cwam-b2-rclone/rclone.conf}"
: "${OSS_ROOT:=eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/B2/$FORMAL_RUN_NAME}"
: "${GIT_REVISION:=unknown}"
: "${NVIDIA_SMI_INTERVAL_SECONDS:=2}"
: "${UPLOAD_INTERVAL_SECONDS:=120}"

[[ "$FORMAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: unsupported FORMAL_RUN_NAME: $FORMAL_RUN_NAME" >&2
    exit 1
}
[[ "$NVIDIA_SMI_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: NVIDIA_SMI_INTERVAL_SECONDS must be a positive integer" >&2
    exit 1
}
[[ "$UPLOAD_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: UPLOAD_INTERVAL_SECONDS must be a positive integer" >&2
    exit 1
}

readonly TOML_FILE="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_robocasa365_b2_nano.toml"
readonly LAUNCHER="$COSMOS_SOURCE/examples/launch_sft_action_policy_robocasa365_nano.sh"
readonly SMOKE_RUN_NAME="${FORMAL_RUN_NAME}-smoke"
readonly SMOKE_OUTPUT_ROOT="$RUN_ROOT/smoke"
readonly FORMAL_OUTPUT_ROOT="$RUN_ROOT/formal"
readonly CANONICAL_RELATIVE_ROOT="cosmos3_action_robocasa365/action_sft"
readonly SMOKE_RUN_DIR="$SMOKE_OUTPUT_ROOT/$CANONICAL_RELATIVE_ROOT/$SMOKE_RUN_NAME"
readonly FORMAL_RUN_DIR="$FORMAL_OUTPUT_ROOT/$CANONICAL_RELATIVE_ROOT/$FORMAL_RUN_NAME"
readonly EVIDENCE_DIR="$RUN_ROOT/evidence"
readonly STATUS_PATH="$CONTROL_ROOT/results/${FORMAL_RUN_NAME}.status.json"
readonly LOCAL_STATUS_PATH="$EVIDENCE_DIR/status.json"
readonly SMOKE_TELEMETRY="$EVIDENCE_DIR/smoke-gpu.csv"
readonly FORMAL_TELEMETRY="$EVIDENCE_DIR/formal-gpu.csv"
readonly SMOKE_LOG_NAME="${SMOKE_RUN_NAME}.torchrun.log"
readonly FORMAL_LOG_NAME="${FORMAL_RUN_NAME}.torchrun.log"
readonly SMOKE_LOG="$SMOKE_OUTPUT_ROOT/logs/$SMOKE_LOG_NAME"
readonly FORMAL_LOG="$FORMAL_OUTPUT_ROOT/logs/$FORMAL_LOG_NAME"
readonly UPLOADED_CHECKPOINTS="$EVIDENCE_DIR/uploaded-checkpoints.txt"
readonly UPLOADER_STOP="$EVIDENCE_DIR/uploader.stop"

SAMPLER_PID=""
TRAIN_PID=""
UPLOADER_PID=""
CURRENT_PHASE="preflight"
FINAL_STATE="failed"
TRAPS_ACTIVE=0

rclone_cmd() {
    RCLONE_CONFIG="$RCLONE_CONFIG" "$RCLONE_BIN" "$@"
}

write_status() {
    local state="$1"
    local phase="$2"
    local exit_code="${3:-}"
    mkdir -p "$EVIDENCE_DIR" "$(dirname "$STATUS_PATH")"
    "$COSMOS_ENV/bin/python" - "$LOCAL_STATUS_PATH" "$STATUS_PATH" \
        "$FORMAL_RUN_NAME" "$state" "$phase" "$exit_code" "$EXPECTED_HOST" \
        "$GIT_REVISION" "$OSS_ROOT" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

local_path, shared_path = map(Path, sys.argv[1:3])
payload = {
    "schema_version": 1,
    "run_id": sys.argv[3],
    "state": sys.argv[4],
    "phase": sys.argv[5],
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": os.uname().nodename,
    "expected_hostname": sys.argv[7],
    "git_revision": sys.argv[8],
    "oss_root": sys.argv[9],
}
if sys.argv[6]:
    payload["exit_code"] = int(sys.argv[6])
for target in (local_path, shared_path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
PY
}

stop_process() {
    local pid="$1"
    local process_group="${2:-0}"
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        if [[ "$process_group" == 1 ]]; then
            kill -TERM -- "-$pid" 2>/dev/null || true
        else
            kill -TERM "$pid" 2>/dev/null || true
        fi
        for _ in {1..50}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        if kill -0 "$pid" 2>/dev/null; then
            if [[ "$process_group" == 1 ]]; then
                kill -KILL -- "-$pid" 2>/dev/null || true
            else
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
    fi
    wait "$pid" 2>/dev/null || true
}

stop_sampler() {
    stop_process "${SAMPLER_PID:-}" 0
    SAMPLER_PID=""
}

stop_uploader() {
    [[ -n "${UPLOADER_PID:-}" ]] || return 0
    touch "$UPLOADER_STOP"
    stop_process "$UPLOADER_PID" 0
    UPLOADER_PID=""
}

best_effort_upload_evidence() {
    [[ -x "$RCLONE_BIN" && -s "$RCLONE_CONFIG" && -d "$EVIDENCE_DIR" ]] || return 0
    rclone_cmd copy "$EVIDENCE_DIR" "$OSS_ROOT/evidence" \
        --transfers 8 --checkers 16 --contimeout 10s --timeout 2m \
        >/dev/null 2>&1 || true
}

on_exit() {
    local exit_code="$1"
    set +e
    stop_sampler
    stop_uploader
    stop_process "${TRAIN_PID:-}" 1
    if [[ "$TRAPS_ACTIVE" == 1 && "$FINAL_STATE" != "completed" && "$FINAL_STATE" != "running" ]]; then
        write_status failed "$CURRENT_PHASE" "$exit_code"
    fi
    if [[ "$TRAPS_ACTIVE" == 1 ]]; then
        best_effort_upload_evidence
    fi
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

start_sampler() {
    local target="$1"
    rm -f "$target" "$target.stderr"
    nvidia-smi \
        --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits \
        --loop="$NVIDIA_SMI_INTERVAL_SECONDS" \
        >"$target" 2>"$target.stderr" &
    SAMPLER_PID=$!
    for _ in {1..50}; do
        [[ -s "$target" ]] && return 0
        kill -0 "$SAMPLER_PID" 2>/dev/null || {
            echo "ERROR: GPU sampler exited before producing data" >&2
            return 1
        }
        sleep 0.2
    done
    echo "ERROR: GPU sampler produced no data" >&2
    return 1
}

run_training_phase() {
    local phase="$1"
    local run_name="$2"
    local output_root="$3"
    local telemetry="$4"
    local log_name="$5"
    local overrides="$6"
    local outer_log="$EVIDENCE_DIR/${phase}.outer.log"

    start_sampler "$telemetry"
    setsid env \
        TOML_FILE="$TOML_FILE" \
        RUN_NAME="$run_name" \
        OUTPUT_ROOT="$output_root" \
        IMAGINAIRE_OUTPUT_ROOT="$output_root" \
        LOG_FILENAME="$log_name" \
        EXTRA_TAIL_OVERRIDES="$overrides" \
        NPROC_PER_NODE=8 \
        RESUME_EXISTING_RUN=0 \
        ROBOCASA365_ROOT="$ROBOCASA365_ROOT" \
        ROBOCASA365_SOURCE_ROOT="$ROBOCASA365_SOURCE_ROOT" \
        BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
        WAN_VAE_PATH="$WAN_VAE_PATH" \
        QWEN3_VL_TOKENIZER_PATH="$QWEN3_VL_TOKENIZER_PATH" \
        ROBOCASA365_CONTROL_CACHE_ROOT="$ROBOCASA365_CONTROL_CACHE_ROOT" \
        ROBOCASA365_STRONG_LATENT_CACHE_ROOT="$ROBOCASA365_STRONG_LATENT_CACHE_ROOT" \
        ROBOCASA365_STRONG_LATENT_CACHE_MODE=required \
        CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER=1 \
        bash "$LAUNCHER" > >(tee "$outer_log") 2>&1 &
    TRAIN_PID=$!
    set +e
    wait "$TRAIN_PID"
    local train_rc=$?
    set -e
    TRAIN_PID=""
    stop_sampler
    return "$train_rc"
}

validate_training_log() {
    local log_path="$1"
    local expected_iteration="$2"
    "$COSMOS_ENV/bin/python" - "$log_path" "$expected_iteration" <<'PY'
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
text = path.read_text(errors="replace")
matches = re.findall(
    r"Iteration:\s*(\d+),.*?total loss\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    text,
)
if not matches:
    raise SystemExit(f"no iteration/loss records found in {path}")
iterations = [int(item[0]) for item in matches]
losses = [float(item[1]) for item in matches]
if expected not in iterations:
    raise SystemExit(f"iteration {expected} absent; observed tail={iterations[-10:]}")
if not all(math.isfinite(value) for value in losses):
    raise SystemExit(f"non-finite loss observed: {losses}")
print(
    "TRAINING_LOG_VALID",
    f"expected_iteration={expected}",
    f"last_iteration={iterations[-1]}",
    f"last_loss={losses[-1]:.8g}",
)
PY
}

validate_gpu_telemetry() {
    local telemetry="$1"
    "$COSMOS_ENV/bin/python" - "$telemetry" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
max_utilization = defaultdict(float)
max_memory_mib = defaultdict(float)
with path.open(newline="") as handle:
    for row in csv.reader(handle):
        if len(row) != 9:
            continue
        uuid = row[2].strip()
        max_memory_mib[uuid] = max(max_memory_mib[uuid], float(row[5].strip()))
        max_utilization[uuid] = max(max_utilization[uuid], float(row[7].strip()))
if len(max_utilization) != 8:
    raise SystemExit(f"expected telemetry for 8 GPUs, found {len(max_utilization)}")
inactive = [uuid for uuid, value in max_utilization.items() if value <= 0.0]
if inactive:
    raise SystemExit(f"GPUs without observed utilization: {inactive}")
print(
    "GPU_TELEMETRY_VALID",
    "gpu_count=8",
    f"min_peak_utilization={min(max_utilization.values()):.1f}",
    f"max_memory_mib={max(max_memory_mib.values()):.1f}",
)
PY
}

verify_checkpoint() {
    local run_dir="$1"
    local expected="$2"
    local checkpoint_root="$run_dir/checkpoints"
    local latest
    [[ -s "$checkpoint_root/latest_checkpoint.txt" ]] || {
        echo "ERROR: checkpoint marker is absent: $checkpoint_root/latest_checkpoint.txt" >&2
        return 1
    }
    latest="$(tr -d '[:space:]' < "$checkpoint_root/latest_checkpoint.txt")"
    [[ "$latest" == "$expected" ]] || {
        echo "ERROR: expected checkpoint $expected, got $latest" >&2
        return 1
    }
    for component in model optim scheduler trainer; do
        [[ -d "$checkpoint_root/$latest/$component" ]] || {
            echo "ERROR: checkpoint component is absent: $component" >&2
            return 1
        }
        find "$checkpoint_root/$latest/$component" -type f -print -quit | grep -q . || {
            echo "ERROR: checkpoint component is empty: $component" >&2
            return 1
        }
    done
    echo "CHECKPOINT_VALID latest=$latest"
}

rclone_copy_and_check() {
    local source="$1"
    local destination="$2"
    rclone_cmd copy "$source" "$destination" \
        --transfers 16 --checkers 32 --multi-thread-cutoff 128M \
        --multi-thread-streams 8 --buffer-size 32M --retries 5 \
        --low-level-retries 20 --stats 30s --stats-one-line
    rclone_cmd check "$source" "$destination" --size-only --one-way \
        --checkers 32 --retries 5 --low-level-retries 20
}

upload_run_metadata() {
    if [[ -d "$FORMAL_OUTPUT_ROOT/logs" ]]; then
        rclone_cmd copy "$FORMAL_OUTPUT_ROOT/logs" "$OSS_ROOT/formal/logs" \
            --transfers 8 --checkers 16 --retries 3 --low-level-retries 10
    fi
    if [[ -d "$FORMAL_RUN_DIR" ]]; then
        rclone_cmd copy "$FORMAL_RUN_DIR" "$OSS_ROOT/formal/run" \
            --exclude '/checkpoints/**' --transfers 8 --checkers 16 \
            --retries 3 --low-level-retries 10
    fi
    rclone_cmd copy "$EVIDENCE_DIR" "$OSS_ROOT/evidence" \
        --transfers 8 --checkers 16 --retries 3 --low-level-retries 10
}

upload_latest_checkpoint() {
    local checkpoint_root="$FORMAL_RUN_DIR/checkpoints"
    local marker="$checkpoint_root/latest_checkpoint.txt"
    local latest candidate name
    [[ -s "$marker" ]] || return 0
    latest="$(tr -d '[:space:]' < "$marker")"
    [[ "$latest" =~ ^iter_[0-9]{9}$ && -d "$checkpoint_root/$latest" ]] || {
        echo "ERROR: malformed or incomplete latest checkpoint: $latest" >&2
        return 1
    }
    if grep -Fxq "$latest" "$UPLOADED_CHECKPOINTS" 2>/dev/null; then
        return 0
    fi

    rclone_copy_and_check \
        "$checkpoint_root/$latest" \
        "$OSS_ROOT/formal/run/checkpoints/$latest"
    rclone_cmd copyto "$marker" "$OSS_ROOT/formal/run/checkpoints/latest_checkpoint.txt"
    printf '%s\n' "$latest" >> "$UPLOADED_CHECKPOINTS"
    upload_run_metadata

    shopt -s nullglob
    for candidate in "$checkpoint_root"/iter_[0-9]*; do
        name="$(basename "$candidate")"
        if [[ "$name" < "$latest" ]]; then
            rm -rf --one-file-system "$candidate"
            echo "LOCAL_CHECKPOINT_PRUNED checkpoint=$name retained=$latest"
        fi
    done
    shopt -u nullglob
    echo "CHECKPOINT_PUBLISHED checkpoint=$latest destination=$OSS_ROOT"
}

formal_uploader_loop() {
    while [[ ! -e "$UPLOADER_STOP" ]]; do
        if ! upload_latest_checkpoint; then
            echo "UPLOAD_RETRY checkpoint recorded_at=$(date -Is)" >&2
        fi
        if ! upload_run_metadata; then
            echo "UPLOAD_RETRY metadata recorded_at=$(date -Is)" >&2
        fi
        if [[ -s "$FORMAL_LOG" ]] && grep -q 'Iteration:' "$FORMAL_LOG"; then
            if [[ ! -e "$EVIDENCE_DIR/formal-first-iteration.ready" ]]; then
                date -Is > "$EVIDENCE_DIR/formal-first-iteration.ready"
                write_status running formal_training
            fi
        fi
        for _ in $(seq 1 "$UPLOAD_INTERVAL_SECONDS"); do
            [[ -e "$UPLOADER_STOP" ]] && break
            sleep 1
        done
    done
}

CURRENT_PHASE="preflight"
[[ ! -e "$RUN_ROOT" ]] || {
    echo "ERROR: refusing to reuse B2 run root: $RUN_ROOT" >&2
    exit 1
}
mkdir -p "$EVIDENCE_DIR"
rm -f "$UPLOADER_STOP"
TRAPS_ACTIVE=1
[[ "$(hostname -f)" == "$EXPECTED_HOST" ]] || {
    echo "ERROR: expected $EXPECTED_HOST, got $(hostname -f)" >&2
    exit 1
}
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8 ]] || {
    echo "ERROR: B2 requires exactly eight visible GPUs" >&2
    exit 1
}
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fvx 'NVIDIA H200' >/dev/null; then
    echo "ERROR: every visible GPU must be an NVIDIA H200" >&2
    exit 1
fi
[[ "$COSMOS_SOURCE" == "$PERSONAL_ROOT/run_snapshots/$FORMAL_RUN_NAME/source" ]] || {
    echo "ERROR: COSMOS_SOURCE must be the B2 run snapshot" >&2
    exit 1
}
[[ -f "$COSMOS_SOURCE/GIT_REVISION" && -f "$COSMOS_SOURCE/RUN_ID" ]] || {
    echo "ERROR: snapshot identity files are absent" >&2
    exit 1
}
[[ "$(cat "$COSMOS_SOURCE/RUN_ID")" == "$FORMAL_RUN_NAME" ]] || {
    echo "ERROR: snapshot RUN_ID does not match FORMAL_RUN_NAME" >&2
    exit 1
}
[[ "$(cat "$COSMOS_SOURCE/GIT_REVISION")" == "$GIT_REVISION" ]] || {
    echo "ERROR: snapshot Git revision does not match the launch record" >&2
    exit 1
}
writable_snapshot_entry="$(
    find "$COSMOS_SOURCE" -xdev \( -type f -o -type d \) -perm /222 -print -quit
)"
[[ -z "$writable_snapshot_entry" ]] || {
    echo "ERROR: source snapshot is writable: $writable_snapshot_entry" >&2
    exit 1
}
for executable in "$COSMOS_ENV/bin/python" "$RCLONE_BIN" "$LAUNCHER"; do
    [[ -x "$executable" ]] || { echo "ERROR: missing executable: $executable" >&2; exit 1; }
done
command -v setsid >/dev/null || { echo "ERROR: setsid is required for process-group cleanup" >&2; exit 1; }
for terminal_file in \
    "$ROBOCASA365_ROOT/_SUCCESS.json" \
    "$ROBOCASA365_ROOT/dataset_manifest.json" \
    "$ROBOCASA365_CONTROL_CACHE_ROOT/_SUCCESS.json" \
    "$ROBOCASA365_CONTROL_CACHE_ROOT/manifest.json" \
    "$ROBOCASA365_STRONG_LATENT_CACHE_ROOT/_SUCCESS.json" \
    "$ROBOCASA365_STRONG_LATENT_CACHE_ROOT/manifest.json" \
    "$BASE_CHECKPOINT_PATH/checkpoint.json" \
    "$BASE_CHECKPOINT_PATH/model/.metadata" \
    "$WAN_VAE_PATH" \
    "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json" \
    "$TOML_FILE"; do
    [[ -f "$terminal_file" ]] || { echo "ERROR: required input is absent: $terminal_file" >&2; exit 1; }
done
[[ -s "$RCLONE_CONFIG" ]] || { echo "ERROR: in-memory rclone config is absent" >&2; exit 1; }
rclone_cmd listremotes | grep -Fx 'eailab-hdd2:' >/dev/null
available_nvme_bytes="$(df --output=avail -B1 /nvme | tail -1 | tr -d '[:space:]')"
[[ "$available_nvme_bytes" -ge 1099511627776 ]] || {
    echo "ERROR: B2 requires at least 1 TiB free on /nvme" >&2
    exit 1
}
export PATH="$COSMOS_ENV/bin:$PATH"
export PYTHONPATH="$COSMOS_SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export COSMOS_DEVICE=cuda
export COSMOS_INTERNAL=0
export COSMOS_TRAINING=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
source "$COSMOS_SOURCE/examples/h_cluster/configure_torchcodec_runtime.sh"

"$COSMOS_ENV/bin/python" \
    "$COSMOS_SOURCE/examples/h_cluster/validate_cu128_train_env_marker.py" \
    --environment "$COSMOS_ENV"

"$COSMOS_ENV/bin/python" - "$EVIDENCE_DIR/run-metadata.json" <<PY
import datetime
import json
import os
from pathlib import Path

payload = {
    "schema_version": 1,
    "run_id": "$FORMAL_RUN_NAME",
    "git_revision": "$GIT_REVISION",
    "hostname": os.uname().nodename,
    "gpu_count": 8,
    "recipe": "action_policy_robocasa365_b2_nano",
    "max_iter": 20000,
    "oss_root": "$OSS_ROOT",
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
Path("$EVIDENCE_DIR/run-metadata.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
write_status running preflight

CURRENT_PHASE="smoke"
write_status running smoke
SMOKE_OVERRIDES="trainer.max_iter=2 trainer.run_validation=false trainer.run_validation_on_start=false trainer.validation_iter=1000 trainer.logging_iter=1 checkpoint.save_iter=2 checkpoint.dcp_async_mode_enabled=false scheduler.cycle_lengths=[2] scheduler.warm_up_steps=[0]"
run_training_phase smoke "$SMOKE_RUN_NAME" "$SMOKE_OUTPUT_ROOT" \
    "$SMOKE_TELEMETRY" "$SMOKE_LOG_NAME" "$SMOKE_OVERRIDES"
validate_training_log "$SMOKE_LOG" 2
validate_gpu_telemetry "$SMOKE_TELEMETRY"
verify_checkpoint "$SMOKE_RUN_DIR" iter_000000002
rclone_copy_and_check "$SMOKE_OUTPUT_ROOT" "$OSS_ROOT/smoke"
rclone_copy_and_check "$EVIDENCE_DIR" "$OSS_ROOT/evidence"
date -Is > "$EVIDENCE_DIR/smoke-passed.ready"
write_status running smoke_passed

CURRENT_PHASE="formal_training"
write_status running formal_starting
formal_uploader_loop > >(tee "$EVIDENCE_DIR/uploader.log") 2>&1 &
UPLOADER_PID=$!
set +e
run_training_phase formal "$FORMAL_RUN_NAME" "$FORMAL_OUTPUT_ROOT" \
    "$FORMAL_TELEMETRY" "$FORMAL_LOG_NAME" ""
formal_rc=$?
set -e
touch "$UPLOADER_STOP"
wait "$UPLOADER_PID" || uploader_rc=$?
UPLOADER_PID=""
[[ "${uploader_rc:-0}" -eq 0 ]] || {
    echo "ERROR: checkpoint uploader failed with rc=${uploader_rc}" >&2
    exit "${uploader_rc}"
}
upload_latest_checkpoint
upload_run_metadata
[[ "$formal_rc" -eq 0 ]] || {
    echo "ERROR: formal B2 training failed with rc=$formal_rc" >&2
    exit "$formal_rc"
}

CURRENT_PHASE="formal_validation"
validate_training_log "$FORMAL_LOG" 20000
validate_gpu_telemetry "$FORMAL_TELEMETRY"
verify_checkpoint "$FORMAL_RUN_DIR" iter_000020000
rclone_copy_and_check "$FORMAL_OUTPUT_ROOT/logs" "$OSS_ROOT/formal/logs"
rclone_copy_and_check "$FORMAL_RUN_DIR/checkpoints/iter_000020000" \
    "$OSS_ROOT/formal/run/checkpoints/iter_000020000"
rclone_copy_and_check "$EVIDENCE_DIR" "$OSS_ROOT/evidence"
FINAL_STATE="completed"
write_status completed completed 0
best_effort_upload_evidence
echo "B2_TRAINING_COMPLETE run=$FORMAL_RUN_NAME destination=$OSS_ROOT"
