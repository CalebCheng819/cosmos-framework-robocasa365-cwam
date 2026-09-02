#!/usr/bin/env bash
set -Eeuo pipefail

: "${EXPERIMENT_ID:?Set the experiment identifier}"
: "${ATTEMPT_ID:?Set the immutable attempt identifier}"
: "${COSMOS_SOURCE:?Set the staged Cosmos3 evaluation source}"
: "${OUTPUT_REMOTE:?Set the exact OSS attempt destination}"

EXPECTED_NODE_TAG="${EXPECTED_NODE_TAG:-node/gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
[[ "${EXPECTED_GPU_COUNT}" == "4" || "${EXPECTED_GPU_COUNT}" == "8" ]] || {
  echo "Stage-1 supports exactly four or eight GPUs: ${EXPECTED_GPU_COUNT}" >&2
  exit 14
}
[[ "${OUTPUT_REMOTE}" == eailab-hdd2:eailab/chengjuntao/* ]] || {
  echo "Stage-1 output must stay under eailab-hdd2:eailab/chengjuntao/: ${OUTPUT_REMOTE}" >&2
  exit 13
}
GPFS_USER="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
COSMOS_ENV="${COSMOS_ENV:-${GPFS_USER}/envs/cosmos-framework-09f23119-cu128-train}"
SIM_ENV="${SIM_ENV:-${GPFS_USER}/envs/fastwam-b4-h254-py310-20260820}"
SIM_BOOTSTRAP_LIB="${SIM_BOOTSTRAP_LIB:-${GPFS_USER}/data-cache-migration/h-eailabagent-20260819/fastwam_env_cache/bootstrap-py310/lib}"
SIM_OVERLAY="${SIM_OVERLAY:-${GPFS_USER}/imagewam-eval/IMAGEWAM-N234-VG1H1-50K-RF-CLOSEDLOOP-H299-S42-R1-20260826/env/robofactory-py310-overlay}"
SIM_DEP_OVERLAY="${SIM_DEP_OVERLAY:-${GPFS_USER}/eval-env-overlays/cosmos3-robofactory-eval-r12-py310}"
SIM_FIX_OVERLAY="${SIM_FIX_OVERLAY:-${GPFS_USER}/eval-env-overlays/cosmos3-robofactory-eval-r16-py310}"
EVAL_ASSET_ROOT="${EVAL_ASSET_ROOT:-${GPFS_USER}/imagewam-eval/IMAGEWAM-TABLE11-STATEFULSAFE-R3-CLOSEDEVAL-S40K-H298-S42-R1-20260828}"
INPUT_ROOT="${INPUT_ROOT:-${EVAL_ASSET_ROOT}/inputs}"
DATASET_ROOT="${DATASET_ROOT:-${INPUT_ROOT}/dataset}"
PANEL_PATH="${PANEL_PATH:-${INPUT_ROOT}/table11-fixed-val100-panel.json}"
STATS_PATH="${STATS_PATH:-${INPUT_ROOT}/checkpoint/dataset_stats.json}"
ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT:-${EVAL_ASSET_ROOT}/source/robofactory}"
ROBOFACTORY_RUNTIME_ROOT="${ROBOFACTORY_RUNTIME_ROOT:-${GPFS_USER}/eval-env-overlays/robofactory-assets-official-20260901}"
POLICY_LIGHTNING_ROOT="${POLICY_LIGHTNING_ROOT:-${GPFS_USER}/imagewam-formal/source/policy-lightning-c944b498-clean}"
RCLONE_BIN="${RCLONE_BIN:-${GPFS_USER}/tools/rclone}"
RCLONE_CONFIG_SOURCE="${RCLONE_CONFIG_SOURCE:-${GPFS_USER}/.fastwam_runtime/rclone.conf}"
CHECKPOINT_REMOTE="${CHECKPOINT_REMOTE:-eailab-hdd2:eailab/chengjuntao/cwam/cosmos3-robofactory-mr/ITER-2026-08-30-COSMOS3-ROBOFACTORY-MR/checkpoints/iter_000060000/model}"
NOPOSPLAT_REMOTE="${NOPOSPLAT_REMOTE:-eailab-hdd2:fkp-migrate/ailab-eailabagent-gpfs/chengjuntao/placefood_wam/Policy-Lightning/weights/noposplat.ckpt}"
WAN_VAE_REMOTE="${WAN_VAE_REMOTE:-eailab-hdd2:fkp-migrate/gpfs-migrations/weights-results-20260727T164026+0800/external-link-targets/public-wan/Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
QWEN3_VL_TOKENIZER_REMOTE="${QWEN3_VL_TOKENIZER_REMOTE:-eailab-hdd2:eailab/chengjuntao/cwam/cosmos3-robofactory-mr/eval-assets/qwen3-vl-8b-instruct-0c351d-processor-v1}"
BASELINE_EPISODES_REMOTE="${BASELINE_EPISODES_REMOTE:-eailab-hdd2:eailab/chengjuntao/cwam/cosmos3-robofactory-mr/evals/COSMOS3-ROBOFACTORY-MR-CLOSEDEVAL2-N2-S60K-S42-R1-20260901/attempts/attempt-r16-h252-2g/results/cosmos3/PlaceFood-rf/episodes.jsonl}"
EVAL_CODE_VERSION="${EVAL_CODE_VERSION:-stage1-unset}"
REPAIR_MODE="${REPAIR_MODE:-full}"
PARENT_OUTPUT_REMOTE="${PARENT_OUTPUT_REMOTE:-}"

if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne "${EXPECTED_GPU_COUNT}" ]]; then
  echo "the formal Stage-1 job must see exactly ${EXPECTED_GPU_COUNT} GPUs" >&2
  exit 14
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA H200')" -ne "${EXPECTED_GPU_COUNT}" ]]; then
  echo "the formal Stage-1 job must run on ${EXPECTED_GPU_COUNT} NVIDIA H200 GPUs" >&2
  exit 14
fi
echo "{\"event\":\"scheduler_contract\",\"expected_node_tag\":\"${EXPECTED_NODE_TAG}\",\"pod_hostname\":\"$(hostname -s)\",\"gpus\":${EXPECTED_GPU_COUNT}}"

TRAIN_PY="${COSMOS_ENV}/bin/python"
SIM_PY="${SIM_ENV}/bin/python"
EVALUATOR="${COSMOS_SOURCE}/experiments/robofactory/eval_robofactory_closed_loop.py"
AGGREGATOR="${COSMOS_SOURCE}/experiments/robofactory/aggregate_cosmos3_stage1.py"
RUN_ROOT="/nvme/chengjuntao/evals/${EXPERIMENT_ID}/attempts/${ATTEMPT_ID}"
CACHE_ROOT="/nvme/chengjuntao/eval-cache/cosmos3-robofactory-mr/iter_000060000"
RUNTIME_ROOT="/nvme/chengjuntao/eval-runtime/${EXPERIMENT_ID}/${ATTEMPT_ID}"
CHECKPOINT_ROOT="${CACHE_ROOT}/checkpoint/iter_000060000"
NOPOSPLAT_CHECKPOINT="${CACHE_ROOT}/noposplat.ckpt"
FOUNDATION_ASSET_ROOT="${CACHE_ROOT}/foundation-assets"
WAN_VAE_PATH="${FOUNDATION_ASSET_ROOT}/Wan2.2_VAE.pth"
QWEN3_VL_TOKENIZER_PATH="${FOUNDATION_ASSET_ROOT}/Qwen3-VL-8B-Instruct"
LOG_ROOT="${RUN_ROOT}/logs"
RESULTS_ROOT="${RUN_ROOT}/results/stage1"
BASELINE_ROOT="${RUN_ROOT}/baseline/a0-eh1-rolling"
RCLONE_CONFIG="/dev/shm/${EXPERIMENT_ID}-${ATTEMPT_ID}-rclone.conf"

for required in \
  "${TRAIN_PY}" "${SIM_PY}" "${EVALUATOR}" "${AGGREGATOR}" \
  "${PANEL_PATH}" "${STATS_PATH}" "${RCLONE_BIN}" "${RCLONE_CONFIG_SOURCE}"; do
  [[ -e "${required}" ]] || { echo "required path missing: ${required}" >&2; exit 15; }
done
for required_dir in \
  "${COSMOS_SOURCE}" "${DATASET_ROOT}" "${ROBOFACTORY_ROOT}" \
  "${ROBOFACTORY_RUNTIME_ROOT}" "${POLICY_LIGHTNING_ROOT}" \
  "${SIM_OVERLAY}" "${SIM_DEP_OVERLAY}" "${SIM_FIX_OVERLAY}" \
  "${SIM_BOOTSTRAP_LIB}"; do
  [[ -d "${required_dir}" ]] || { echo "required directory missing: ${required_dir}" >&2; exit 16; }
done

ROBOFACTORY_ASSET_DIR="${ROBOFACTORY_RUNTIME_ROOT}/assets"
for required_asset in \
  scenes/table/table.glb \
  objects/pot_annotated/mobility.urdf \
  objects/pot_annotated/model_data.json \
  objects/meat_annotated/textured.obj \
  objects/meat_annotated/model_data.json; do
  [[ -s "${ROBOFACTORY_ASSET_DIR}/${required_asset}" ]] || {
    echo "required RoboFactory runtime asset missing: ${required_asset}" >&2
    exit 22
  }
done
robofactory_asset_files="$(find "${ROBOFACTORY_ASSET_DIR}" -type f | wc -l)"
[[ "${robofactory_asset_files}" -ge 1168 ]] || {
  echo "RoboFactory runtime asset inventory is incomplete: files=${robofactory_asset_files}" >&2
  exit 23
}

nvme_free_bytes="$(df -B1 --output=avail /nvme | tail -n 1 | tr -d ' ')"
[[ "${nvme_free_bytes}" -ge 160000000000 ]] || {
  echo "node NVMe free space is below the 160 GB Stage-1 floor: ${nvme_free_bytes}" >&2
  exit 17
}
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}" "${BASELINE_ROOT}" "${RUNTIME_ROOT}"
install -m 600 "${RCLONE_CONFIG_SOURCE}" "${RCLONE_CONFIG}"

declare -a SERVER_PIDS=()
declare -a LANE_PIDS=()
ARCHIVER_PID=""
archive_once() {
  "${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy "${RUN_ROOT}" "${OUTPUT_REMOTE}" \
    --transfers 12 --checkers 24 --retries 5 --low-level-retries 20
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
  for pid in "${LANE_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  [[ -n "${ARCHIVER_PID}" ]] && kill "${ARCHIVER_PID}" 2>/dev/null || true
  wait 2>/dev/null || true
  archive_once || true
  rm -f "${RCLONE_CONFIG}"
  exit "${code}"
}
trap cleanup EXIT INT TERM

mkdir -p "${CHECKPOINT_ROOT}/model" "${FOUNDATION_ASSET_ROOT}" "${QWEN3_VL_TOKENIZER_PATH}"
echo '{"event":"asset_staging_started","checkpoint":"iter_000060000","storage":"node_nvme"}'
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copyto "${WAN_VAE_REMOTE}" "${WAN_VAE_PATH}" --retries 8 --low-level-retries 30
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy "${QWEN3_VL_TOKENIZER_REMOTE}" "${QWEN3_VL_TOKENIZER_PATH}" --transfers 8 --checkers 16 --retries 8 --low-level-retries 30
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy "${CHECKPOINT_REMOTE}" "${CHECKPOINT_ROOT}/model" --transfers 16 --checkers 32 --retries 8 --low-level-retries 30
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copyto "${NOPOSPLAT_REMOTE}" "${NOPOSPLAT_CHECKPOINT}" --retries 8 --low-level-retries 30
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copyto "${BASELINE_EPISODES_REMOTE}" "${BASELINE_ROOT}/episodes.jsonl" --retries 8 --low-level-retries 30
[[ -s "${CHECKPOINT_ROOT}/model/.metadata" ]]
[[ "$(stat -c %s "${NOPOSPLAT_CHECKPOINT}")" -eq 2448469295 ]]
[[ "$(stat -c %s "${WAN_VAE_PATH}")" -eq 2818839170 ]]
[[ "$(wc -l < "${BASELINE_ROOT}/episodes.jsonl")" -eq 12 ]]
qwen_file_count="$(find "${QWEN3_VL_TOKENIZER_PATH}" -maxdepth 1 -type f | wc -l)"
qwen_total_bytes="$(find "${QWEN3_VL_TOKENIZER_PATH}" -maxdepth 1 -type f -printf '%s\n' | awk '{total += $1} END {print total + 0}')"
[[ "${qwen_file_count}" -eq 9 && "${qwen_total_bytes}" -eq 11499960 ]] || {
  echo "Qwen processor inventory mismatch: files=${qwen_file_count} bytes=${qwen_total_bytes}" >&2
  exit 21
}

export BASE_CHECKPOINT_PATH="${CHECKPOINT_ROOT}"
export ROBOFACTORY_ROOT ROBOFACTORY_ACTION_STATS_PATH="${STATS_PATH}"
export ROBOFACTORY_GAUSSIAN_CACHE_ROOT="${DATASET_ROOT}"
export WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export IMAGINAIRE_OUTPUT_ROOT="${RUNTIME_ROOT}/cosmos-output"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_IB_DISABLE=1

SIM_PYTHONPATH="${SIM_FIX_OVERLAY}:${SIM_DEP_OVERLAY}:${COSMOS_SOURCE}/experiments/robofactory/compat:${SIM_OVERLAY}:${COSMOS_SOURCE}:${ROBOFACTORY_ROOT}:${POLICY_LIGHTNING_ROOT}"
SIM_LD_LIBRARY_PATH="${SIM_BOOTSTRAP_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LD_LIBRARY_PATH="${SIM_LD_LIBRARY_PATH}" PYTHONPATH="${SIM_PYTHONPATH}" "${SIM_PY}" - <<'PY'
import imageio
import imageio_ffmpeg
import jaxtyping
import exceptiongroup
from model.noposplat.encoder import get_encoder

assert callable(get_encoder)
assert imageio_ffmpeg.get_ffmpeg_exe()
print('{"event":"simulator_dependency_preflight_passed","video":"mp4"}', flush=True)
PY
LD_LIBRARY_PATH="${SIM_LD_LIBRARY_PATH}" PYTHONPATH="${SIM_PYTHONPATH}" "${SIM_PY}" - <<'PY'
from experiments.robofactory import eval_robofactory_closed_loop_test as tests

for name in sorted(vars(tests)):
    if name.startswith("test_"):
        vars(tests)[name]()
print('{"event":"stage1_helper_tests_passed","tests":3}', flush=True)
PY
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT}" "${TRAIN_PY}" - <<'PY'
import json
import os
import torch
expected = int(os.environ["EXPECTED_GPU_COUNT"])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise RuntimeError(f"Cosmos3 Stage-1 preflight requires exactly {expected} visible GPUs")
print(json.dumps({"event": "cosmos_preflight_passed", "gpu_count": expected}), flush=True)
PY

archive_loop >"${LOG_ROOT}/archiver.log" 2>&1 &
ARCHIVER_PID=$!

start_server() {
  local gpu="$1" port="$2" process_group_port="$3" lane="$4"
  (
    cd "${COSMOS_SOURCE}"
    CUDA_VISIBLE_DEVICES="${gpu}" MASTER_ADDR=127.0.0.1 MASTER_PORT="${process_group_port}" \
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 PYTHONPATH="${COSMOS_SOURCE}" \
      "${TRAIN_PY}" -u -m cosmos_framework.scripts.action_policy_server_robofactory \
        --checkpoint-path "${CHECKPOINT_ROOT}" \
        --config-file cosmos_framework/configs/base/config.py \
        --experiment action_policy_robofactory_multi_robot_nano \
        --output-dir "${RUNTIME_ROOT}/server-${lane}" --seed 10000 \
        --guidance 1.0 --num-steps 30 --fps 10 --action-chunk-size 32 \
        --raw-action-dim 8 --action-stats-path "${STATS_PATH}" \
        --host 127.0.0.1 --port "${port}"
  ) >"${LOG_ROOT}/server-${lane}.log" 2>&1 &
  SERVER_PIDS+=("$!")
}

wait_for_server() {
  local port="$1" server_pid="$2" lane="$3" waited=0
  until "${SIM_PY}" - "${port}" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/info", timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("adapter") != "RoboFactoryActionModelService":
    raise RuntimeError(payload)
PY
  do
    kill -0 "${server_pid}" 2>/dev/null || { echo "server exited: ${lane}" >&2; return 1; }
    sleep 10
    waited=$((waited + 10))
    (( waited < 3600 )) || { echo "server readiness timeout: ${lane}" >&2; return 1; }
  done
  echo "{\"event\":\"server_ready\",\"lane\":\"${lane}\",\"waited_seconds\":${waited}}"
}

start_server 0 18080 29580 lane0
start_server 2 18081 29581 lane1
lane_count=2
if [[ "${EXPECTED_GPU_COUNT}" -eq 8 ]]; then
  start_server 4 18082 29582 lane2
  start_server 6 18083 29583 lane3
  lane_count=4
fi
for ((lane = 0; lane < lane_count; lane++)); do
  wait_for_server "$((18080 + lane))" "${SERVER_PIDS[lane]}" "lane${lane}"
done

export NVIDIA_DRIVER_CAPABILITIES=all PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
export OMP_NUM_THREADS=12
run_arm() {
  local gpu="$1" port="$2" arm="$3" episodes="$4" exec_horizon="$5" noise_mode="$6" oracle="$7"
  echo "{\"event\":\"arm_started\",\"arm\":\"${arm}\",\"gpu\":${gpu},\"episodes\":${episodes},\"exec_horizon\":${exec_horizon},\"noise_mode\":\"${noise_mode}\",\"oracle\":\"${oracle}\"}"
  CUDA_VISIBLE_DEVICES="${gpu}" LD_LIBRARY_PATH="${SIM_LD_LIBRARY_PATH}" PYTHONPATH="${SIM_PYTHONPATH}" \
    "${SIM_PY}" -u "${EVALUATOR}" --mode cosmos3 --task PlaceFood-rf \
      --panel "${PANEL_PATH}" --dataset-root "${DATASET_ROOT}" \
      --robofactory-root "${ROBOFACTORY_ROOT}" \
      --robofactory-runtime-root "${ROBOFACTORY_RUNTIME_ROOT}" \
      --output-dir "${RESULTS_ROOT}/${arm}" --eval-code-commit "${EVAL_CODE_VERSION}" \
      --episode-start 0 --num-episodes "${episodes}" --exec-horizon "${exec_horizon}" \
      --noise-mode "${noise_mode}" --oracle-intervention "${oracle}" \
      --policy-seed 10000 --policy-lightning-repo "${POLICY_LIGHTNING_ROOT}" \
      --noposplat-checkpoint "${NOPOSPLAT_CHECKPOINT}" \
      --policy-server-url "http://127.0.0.1:${port}" --request-timeout 900 \
      --device cuda:0 --action-horizon 32 --record-video --resume \
      >"${LOG_ROOT}/eval-${arm}.log" 2>&1
  echo "{\"event\":\"arm_completed\",\"arm\":\"${arm}\"}"
}

# Repair-1 is a continuation of a scheduler-stopped Stage-1 attempt.  The
# parent result tree is copied into the new attempt, then only the incomplete
# robot0-pose prefix and the missing robot1-action oracle are evaluated.  The
# copied prefix is deliberately kept in resume_progress.json so the evaluator
# validates the exact panel prefix before doing any new episode.
if [[ "${REPAIR_MODE}" == "repair1" ]]; then
  [[ "${EXPECTED_GPU_COUNT}" -eq 4 ]] || {
    echo "repair1 requires exactly four GPUs" >&2
    exit 26
  }
  [[ -n "${PARENT_OUTPUT_REMOTE}" ]] || {
    echo "repair1 requires PARENT_OUTPUT_REMOTE" >&2
    exit 27
  }
  echo "{\"event\":\"repair1_parent_copy_started\",\"source\":\"${PARENT_OUTPUT_REMOTE}\"}"
  "${RCLONE_BIN}" --config "${RCLONE_CONFIG}" copy \
    "${PARENT_OUTPUT_REMOTE}/results/stage1" "${RESULTS_ROOT}" \
    --transfers 12 --checkers 24 --retries 5 --low-level-retries 20
  [[ -s "${RESULTS_ROOT}/o1-r0-pose/resume_progress.json" ]] || {
    echo "repair1 parent has no resumable robot0-pose prefix" >&2
    exit 28
  }
  [[ "$(jq -r '.episodes_completed' "${RESULTS_ROOT}/o1-r0-pose/resume_progress.json")" == "3" ]] || {
    echo "repair1 requires exactly the validated three-episode robot0-pose prefix" >&2
    exit 29
  }
  [[ -s "${RESULTS_ROOT}/o3-r1-action/episodes.jsonl" ]] && {
    echo "repair1 parent unexpectedly contains robot1-action results" >&2
    exit 30
  }
  # Remove only stale control/trace/video files from the interrupted episode;
  # completed O1 videos and all other parent-arm evidence remain untouched.
  rm -f "${RESULTS_ROOT}/o1-r0-pose/action_trace.jsonl" \
    "${RESULTS_ROOT}/o1-r0-pose/run_manifest.json" \
    "${RESULTS_ROOT}/o1-r0-pose/videos/task-003-episode-006.mp4"
  mkdir -p "${RESULTS_ROOT}/o3-r1-action"
  (
    run_arm 1 18080 o1-r0-pose 8 1 rolling-query robot0-pose
  ) & LANE_PIDS+=("$!")
  (
    run_arm 3 18081 o3-r1-action 8 1 rolling-query robot1-action
  ) & LANE_PIDS+=("$!")
  lane_failed=0
  for pid in "${LANE_PIDS[@]}"; do
    wait "${pid}" || lane_failed=1
  done
  [[ "${lane_failed}" -eq 0 ]] || {
    echo "one or more repair1 lanes failed" >&2
    exit 31
  }
  "${SIM_PY}" -u "${AGGREGATOR}" \
    --results-root "${RESULTS_ROOT}" \
    --baseline-episodes "${BASELINE_ROOT}/episodes.jsonl" \
    --output "${RUN_ROOT}/stage1-summary.json"
  video_count="$(find "${RESULTS_ROOT}" -type f -path '*/videos/*.mp4' -size +0c | wc -l)"
  [[ "${video_count}" -eq 60 ]] || {
    echo "repair1 video inventory mismatch: expected=60 actual=${video_count}" >&2
    exit 32
  }
  archive_once
  "${RCLONE_BIN}" --config "${RCLONE_CONFIG}" check "${RUN_ROOT}" "${OUTPUT_REMOTE}" \
    --size-only --one-way --checkers 24 --retries 5 --low-level-retries 20
  echo '{"event":"formal_stage1_repair1_completed","status":"PASS"}'
  exit 0
fi

if [[ "${EXPECTED_GPU_COUNT}" -eq 4 ]]; then
  (
    run_arm 1 18080 a1-eh5-rolling 12 5 rolling-query none
    run_arm 1 18080 a3-eh5-fixed 12 5 fixed-episode none
    run_arm 1 18080 o2-r0-gripper 8 1 rolling-query robot0-gripper
  ) & LANE_PIDS+=("$!")
  (
    run_arm 3 18081 a2-eh1-fixed 12 1 fixed-episode none
    run_arm 3 18081 o1-r0-pose 8 1 rolling-query robot0-pose
    run_arm 3 18081 o3-r1-action 8 1 rolling-query robot1-action
  ) & LANE_PIDS+=("$!")
else
  (
    run_arm 1 18080 a1-eh5-rolling 12 5 rolling-query none
    run_arm 1 18080 o2-r0-gripper 8 1 rolling-query robot0-gripper
  ) & LANE_PIDS+=("$!")
  (
    run_arm 3 18081 a2-eh1-fixed 12 1 fixed-episode none
  ) & LANE_PIDS+=("$!")
  (
    run_arm 5 18082 a3-eh5-fixed 12 5 fixed-episode none
    run_arm 5 18082 o3-r1-action 8 1 rolling-query robot1-action
  ) & LANE_PIDS+=("$!")
  (
    run_arm 7 18083 o1-r0-pose 8 1 rolling-query robot0-pose
  ) & LANE_PIDS+=("$!")
fi

lane_failed=0
for pid in "${LANE_PIDS[@]}"; do
  wait "${pid}" || lane_failed=1
done
[[ "${lane_failed}" -eq 0 ]] || { echo "one or more Stage-1 lanes failed" >&2; exit 24; }

"${SIM_PY}" -u "${AGGREGATOR}" \
  --results-root "${RESULTS_ROOT}" \
  --baseline-episodes "${BASELINE_ROOT}/episodes.jsonl" \
  --output "${RUN_ROOT}/stage1-summary.json"
video_count="$(find "${RESULTS_ROOT}" -type f -path '*/videos/*.mp4' -size +0c | wc -l)"
[[ "${video_count}" -eq 60 ]] || {
  echo "Stage-1 video inventory mismatch: expected=60 actual=${video_count}" >&2
  exit 25
}
archive_once
"${RCLONE_BIN}" --config "${RCLONE_CONFIG}" check "${RUN_ROOT}" "${OUTPUT_REMOTE}" \
  --size-only --one-way --checkers 24 --retries 5 --low-level-retries 20
echo '{"event":"formal_stage1_completed","status":"PASS"}'
