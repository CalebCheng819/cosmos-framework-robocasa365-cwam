#!/usr/bin/env bash
set -Eeuo pipefail

: "${EXPERIMENT_ID:?Set the experiment identifier}"
: "${ATTEMPT_ID:?Set the immutable attempt identifier}"
: "${COSMOS_SOURCE:?Set the staged Cosmos3 evaluation source}"
: "${OUTPUT_REMOTE:?Set the exact OSS attempt destination}"

EXPECTED_NODE_TAG="node/gpu-l-lg-cmc-h-h200-0252.host.h.pjlab.org.cn"
GPFS_USER="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
COSMOS_ENV="${COSMOS_ENV:-${GPFS_USER}/envs/cosmos-framework-09f23119-cu128-train}"
SIM_ENV="${SIM_ENV:-${GPFS_USER}/envs/fastwam-b4-h254-py310-20260820}"
SIM_OVERLAY="${SIM_OVERLAY:-${GPFS_USER}/imagewam-eval/IMAGEWAM-N234-VG1H1-50K-RF-CLOSEDLOOP-H299-S42-R1-20260826/env/robofactory-py310-overlay}"
EVAL_ASSET_ROOT="${EVAL_ASSET_ROOT:-${GPFS_USER}/imagewam-eval/IMAGEWAM-TABLE11-STATEFULSAFE-R3-CLOSEDEVAL-S40K-H298-S42-R1-20260828}"
INPUT_ROOT="${INPUT_ROOT:-${EVAL_ASSET_ROOT}/inputs}"
DATASET_ROOT="${DATASET_ROOT:-${INPUT_ROOT}/dataset}"
PANEL_PATH="${PANEL_PATH:-${INPUT_ROOT}/table11-fixed-val100-panel.json}"
STATS_PATH="${STATS_PATH:-${INPUT_ROOT}/checkpoint/dataset_stats.json}"
ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT:-${EVAL_ASSET_ROOT}/source/robofactory}"
POLICY_LIGHTNING_ROOT="${POLICY_LIGHTNING_ROOT:-${GPFS_USER}/imagewam-formal/source/policy-lightning-c944b498-clean}"
RCLONE_BIN="${RCLONE_BIN:-${GPFS_USER}/tools/rclone}"
RCLONE_CONFIG_SOURCE="${RCLONE_CONFIG_SOURCE:-${GPFS_USER}/.fastwam_runtime/rclone.conf}"
CHECKPOINT_REMOTE="${CHECKPOINT_REMOTE:-eailab-hdd2:eailab/chengjuntao/cwam/cosmos3-robofactory-mr/ITER-2026-08-30-COSMOS3-ROBOFACTORY-MR/checkpoints/iter_000060000/model}"
NOPOSPLAT_REMOTE="${NOPOSPLAT_REMOTE:-eailab-hdd2:fkp-migrate/ailab-eailabagent-gpfs/chengjuntao/placefood_wam/Policy-Lightning/weights/noposplat.ckpt}"
WAN_VAE_REMOTE="${WAN_VAE_REMOTE:-eailab-hdd2:fkp-migrate/gpfs-migrations/weights-results-20260727T164026+0800/external-link-targets/public-wan/Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
QWEN3_VL_TOKENIZER_REMOTE="${QWEN3_VL_TOKENIZER_REMOTE:-eailab-hdd2:eailab/chengjuntao/cwam/cosmos3-robofactory-mr/eval-assets/qwen3-vl-8b-instruct-0c351d-processor-v1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-100}"
EXEC_HORIZON="${EXEC_HORIZON:-1}"
EVAL_CODE_VERSION="${EVAL_CODE_VERSION:-ITER-2026-09-01-COSMOS3-RF-MR-EVAL2-H252-code-r11}"

[[ "${EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EPISODES_PER_TASK must be positive" >&2
  exit 11
}
[[ "${EXEC_HORIZON}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXEC_HORIZON must be positive" >&2
  exit 12
}
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne 2 ]]; then
  echo "the formal job must see exactly two GPUs" >&2
  exit 14
fi
echo "{\"event\":\"scheduler_contract\",\"expected_node_tag\":\"${EXPECTED_NODE_TAG}\",\"pod_hostname\":\"$(hostname -s)\"}"

TRAIN_PY="${COSMOS_ENV}/bin/python"
SIM_PY="${SIM_ENV}/bin/python"
EVALUATOR="${COSMOS_SOURCE}/experiments/robofactory/eval_robofactory_closed_loop.py"
AGGREGATOR="${COSMOS_SOURCE}/experiments/robofactory/aggregate_cosmos3_robofactory_eval.py"
RUN_ROOT="${GPFS_USER}/evals/${EXPERIMENT_ID}/attempts/${ATTEMPT_ID}"
CACHE_ROOT="${GPFS_USER}/eval-cache/cosmos3-robofactory-mr/iter_000060000"
RUNTIME_ROOT="${GPFS_USER}/eval-runtime/${EXPERIMENT_ID}/${ATTEMPT_ID}"
CHECKPOINT_ROOT="${CACHE_ROOT}/checkpoint/iter_000060000"
NOPOSPLAT_CHECKPOINT="${CACHE_ROOT}/noposplat.ckpt"
FOUNDATION_ASSET_ROOT="${CACHE_ROOT}/foundation-assets"
WAN_VAE_PATH="${WAN_VAE_PATH:-${FOUNDATION_ASSET_ROOT}/Wan2.2_VAE.pth}"
QWEN3_VL_TOKENIZER_PATH="${QWEN3_VL_TOKENIZER_PATH:-${FOUNDATION_ASSET_ROOT}/Qwen3-VL-8B-Instruct}"
LOG_ROOT="${RUN_ROOT}/logs"
RESULTS_ROOT="${RUN_ROOT}/results"
RCLONE_CONFIG="/dev/shm/${EXPERIMENT_ID}-${ATTEMPT_ID}-rclone.conf"

for required in \
  "${TRAIN_PY}" "${SIM_PY}" "${EVALUATOR}" "${AGGREGATOR}" \
  "${PANEL_PATH}" "${STATS_PATH}" "${RCLONE_BIN}" \
  "${RCLONE_CONFIG_SOURCE}"; do
  [[ -e "${required}" ]] || { echo "required path missing: ${required}" >&2; exit 15; }
done
for required_dir in \
  "${COSMOS_SOURCE}" "${DATASET_ROOT}" "${ROBOFACTORY_ROOT}" \
  "${POLICY_LIGHTNING_ROOT}" "${SIM_OVERLAY}"; do
  [[ -d "${required_dir}" ]] || { echo "required directory missing: ${required_dir}" >&2; exit 16; }
done

mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}/cosmos3" "${RUNTIME_ROOT}"
install -m 600 "${RCLONE_CONFIG_SOURCE}" "${RCLONE_CONFIG}"

declare -a CHILD_PIDS=()
ARCHIVER_PID=""
archive_once() {
  "${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy "${RUN_ROOT}" "${OUTPUT_REMOTE}" \
    --transfers 8 --checkers 16 --retries 5 --low-level-retries 20
}
archive_loop() {
  while true; do
    archive_once || true
    sleep 120
  done
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${CHILD_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  if [[ -n "${ARCHIVER_PID}" ]]; then
    kill "${ARCHIVER_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  archive_once || true
  rm -f "${RCLONE_CONFIG}"
  exit "${code}"
}
trap cleanup EXIT INT TERM

echo "{\"event\":\"asset_staging_started\",\"checkpoint\":\"iter_000060000\",\"storage\":\"gpfs_cache\"}"
mkdir -p "${CHECKPOINT_ROOT}/model" "${FOUNDATION_ASSET_ROOT}" "${QWEN3_VL_TOKENIZER_PATH}"
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copyto \
  "${WAN_VAE_REMOTE}" "${WAN_VAE_PATH}" \
  --retries 8 --low-level-retries 30
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy \
  "${QWEN3_VL_TOKENIZER_REMOTE}" "${QWEN3_VL_TOKENIZER_PATH}" \
  --transfers 8 --checkers 16 --retries 8 --low-level-retries 30
[[ "$(stat -c %s "${WAN_VAE_PATH}")" -eq 2818839170 ]] || {
  echo "Wan2.2 VAE size mismatch" >&2
  exit 19
}
for required in \
  config.json chat_template.json merges.txt preprocessor_config.json \
  tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do
  [[ -s "${QWEN3_VL_TOKENIZER_PATH}/${required}" ]] || {
    echo "staged Qwen processor file missing: ${required}" >&2
    exit 20
  }
done
qwen_file_count=0
qwen_total_bytes=0
while IFS= read -r -d '' qwen_file; do
  qwen_file_count=$((qwen_file_count + 1))
  qwen_total_bytes=$((qwen_total_bytes + $(stat -c %s "${qwen_file}")))
done < <(find "${QWEN3_VL_TOKENIZER_PATH}" -maxdepth 1 -type f -print0)
[[ "${qwen_file_count}" -eq 9 && "${qwen_total_bytes}" -eq 11499960 ]] || {
  echo "Qwen processor inventory mismatch: files=${qwen_file_count} bytes=${qwen_total_bytes}" >&2
  exit 21
}
echo "{\"event\":\"foundation_assets_staged\",\"wan_vae_bytes\":2818839170,\"qwen_files\":${qwen_file_count},\"qwen_bytes\":${qwen_total_bytes}}"

"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy \
  "${CHECKPOINT_REMOTE}" "${CHECKPOINT_ROOT}/model" \
  --transfers 16 --checkers 32 --retries 8 --low-level-retries 30
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copyto \
  "${NOPOSPLAT_REMOTE}" "${NOPOSPLAT_CHECKPOINT}" \
  --retries 8 --low-level-retries 30
[[ -s "${CHECKPOINT_ROOT}/model/.metadata" ]] || {
  echo "staged DCP model metadata is missing" >&2
  exit 17
}
[[ "$(stat -c %s "${NOPOSPLAT_CHECKPOINT}")" -eq 2448469295 ]] || {
  echo "NoPoSplat checkpoint size mismatch" >&2
  exit 18
}

export BASE_CHECKPOINT_PATH="${CHECKPOINT_ROOT}"
export ROBOFACTORY_ROOT ROBOFACTORY_ACTION_STATS_PATH="${STATS_PATH}"
export ROBOFACTORY_GAUSSIAN_CACHE_ROOT="${DATASET_ROOT}"
export WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export IMAGINAIRE_OUTPUT_ROOT="${RUNTIME_ROOT}/cosmos-output"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1

"${TRAIN_PY}" - <<'PY'
import json
import os
from pathlib import Path

import torch

required = {
    "checkpoint_metadata": Path(os.environ["BASE_CHECKPOINT_PATH"]) / "model/.metadata",
    "stats": Path(os.environ["ROBOFACTORY_ACTION_STATS_PATH"]),
    "vae": Path(os.environ["WAN_VAE_PATH"]),
    "tokenizer": Path(os.environ["QWEN3_VL_TOKENIZER_PATH"]) / "tokenizer_config.json",
}
missing = {name: str(path) for name, path in required.items() if not path.is_file()}
if missing:
    raise RuntimeError(f"Cosmos3 asset preflight failed: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("Cosmos3 preflight requires exactly two visible GPUs")
print(json.dumps({"event": "cosmos_preflight_passed", "gpu_count": 2}, sort_keys=True), flush=True)
PY

archive_loop >"${LOG_ROOT}/archiver.log" 2>&1 &
ARCHIVER_PID=$!

start_server() {
  local gpu="$1"
  local port="$2"
  local process_group_port="$3"
  local lane="$4"
  (
    cd "${COSMOS_SOURCE}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${process_group_port}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \
    PYTHONPATH="${COSMOS_SOURCE}" \
      "${TRAIN_PY}" -u -m cosmos_framework.scripts.action_policy_server_robofactory \
        --checkpoint-path "${CHECKPOINT_ROOT}" \
        --config-file cosmos_framework/configs/base/config.py \
        --experiment action_policy_robofactory_multi_robot_nano \
        --output-dir "${RUNTIME_ROOT}/server-${lane}" \
        --seed 10000 \
        --guidance 1.0 \
        --num-steps 30 \
        --fps 10 \
        --action-chunk-size 32 \
        --raw-action-dim 8 \
        --action-stats-path "${STATS_PATH}" \
        --host 127.0.0.1 \
        --port "${port}"
  ) >"${LOG_ROOT}/server-${lane}.log" 2>&1 &
  CHILD_PIDS+=("$!")
}

start_server 0 18080 29580 shared

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local lane="$3"
  local waited=0
  until "${SIM_PY}" - "${port}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/info", timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("adapter") != "RoboFactoryActionModelService":
    raise RuntimeError(payload)
PY
  do
    kill -0 "${server_pid}" 2>/dev/null || {
      echo "server exited during startup: lane=${lane}" >&2
      return 1
    }
    sleep 10
    waited=$((waited + 10))
    if (( waited >= 3600 )); then
      echo "server readiness timeout: lane=${lane}" >&2
      return 1
    fi
  done
  echo "{\"event\":\"server_ready\",\"lane\":\"${lane}\",\"waited_seconds\":${waited}}"
}

wait_for_server 18080 "${CHILD_PIDS[0]}" shared

export NVIDIA_DRIVER_CAPABILITIES=all PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
SIM_PYTHONPATH="${COSMOS_SOURCE}/experiments/robofactory/compat:${SIM_OVERLAY}:${COSMOS_SOURCE}:${ROBOFACTORY_ROOT}:${POLICY_LIGHTNING_ROOT}"

run_task() {
  local gpu="$1"
  local port="$2"
  local task="$3"
  local lane="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SIM_PYTHONPATH}" \
    "${SIM_PY}" -u "${EVALUATOR}" \
      --mode cosmos3 \
      --task "${task}" \
      --panel "${PANEL_PATH}" \
      --dataset-root "${DATASET_ROOT}" \
      --robofactory-root "${ROBOFACTORY_ROOT}" \
      --output-dir "${RESULTS_ROOT}/cosmos3/${task}" \
      --eval-code-commit "${EVAL_CODE_VERSION}" \
      --episode-start 0 \
      --num-episodes "${EPISODES_PER_TASK}" \
      --exec-horizon "${EXEC_HORIZON}" \
      --policy-seed 10000 \
      --policy-lightning-repo "${POLICY_LIGHTNING_ROOT}" \
      --noposplat-checkpoint "${NOPOSPLAT_CHECKPOINT}" \
      --policy-server-url "http://127.0.0.1:${port}" \
      --request-timeout 900 \
      --device cuda:0 \
      --action-horizon 32 \
      --continue-on-error \
      --resume \
      >"${LOG_ROOT}/eval-${lane}.log" 2>&1 &
  CHILD_PIDS+=("$!")
}

echo '{"event":"formal_two_task_eval_started","execution":"sequential","tasks":["PlaceFood-rf","TwoRobotsStackCube-rf"]}'
run_task 1 18080 PlaceFood-rf placefood
eval_failed=0
if ! wait "${CHILD_PIDS[1]}"; then
  eval_failed=1
fi
run_task 1 18080 TwoRobotsStackCube-rf stackcube
if ! wait "${CHILD_PIDS[2]}"; then
  eval_failed=1
fi
if [[ "${eval_failed}" -ne 0 ]]; then
  echo "one or more task evaluators failed" >&2
  exit 19
fi

"${SIM_PY}" -u "${AGGREGATOR}" \
  --results-root "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/aggregate.json" \
  --episodes-per-task "${EPISODES_PER_TASK}" \
  --exec-horizon "${EXEC_HORIZON}" \
  --task PlaceFood-rf \
  --task TwoRobotsStackCube-rf

archive_once
echo '{"event":"formal_two_task_eval_completed"}'
