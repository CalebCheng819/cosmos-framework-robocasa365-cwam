#!/usr/bin/env bash
# Train CWAM S1 on local LIBERO data; transient output stays on node NVMe.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${KUBEBRAIN_RJOB_NAME:=unknown}"
: "${NUM_GPUS:=4}"
: "${ENABLE_OSS:=1}"
: "${RUN_PREFLIGHT_FIRST:=1}"
: "${OSS_RUN_ROOT:=}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${LIBERO_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/libero/fastwam/LIBERO-fastwam}"
: "${LIBERO_SUITE_DIR_SUFFIX:=_no_noops_lerobot}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
: "${NVME_ROOT:=/nvme/cwam-libero-s1}"
: "${ARCHIVE_START_STEP:=500}"
: "${ARCHIVE_END_STEP:=5000}"
: "${ARCHIVE_INTERVAL:=500}"

readonly CONFIG="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_libero_all_s1_nano.toml"
readonly LAUNCHER="$COSMOS_SOURCE/examples/launch_sft_action_policy_libero_all_s1_nano.sh"
readonly ARCHIVER="$COSMOS_SOURCE/examples/h_cluster/archive_cwam_s1_cont100k.py"
readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"
readonly OUTPUT_ROOT="$RUN_ROOT/output"
readonly PREFLIGHT_OUTPUT_ROOT="$RUN_ROOT/preflight-output"
readonly EVIDENCE_ROOT="$RUN_ROOT/evidence"
readonly CHECKPOINTS_DIR="$OUTPUT_ROOT/cosmos3_action_libero/action_sft/$RUN_NAME/checkpoints"
readonly HANDOFF_ROOT="$PERSONAL_ROOT/runtime-handoffs/cwam-libero-s1/$KUBEBRAIN_RJOB_NAME"
readonly SECRET_ROOT="/dev/shm/cwam-libero-s1-rclone"
readonly RCLONE_CONFIG_PATH="$SECRET_ROOT/rclone.conf"

MONITOR_PID=""
ARCHIVER_PID=""
for path in "$CONFIG" "$LAUNCHER" "$ARCHIVER" "$COSMOS_ENV/bin/python" \
    "$BASE_CHECKPOINT_PATH/model/.metadata" "$WAN_VAE_PATH" "$QWEN3_VL_TOKENIZER_PATH/tokenizer.json"; do
    [[ -e "$path" ]] || { echo "ERROR: required input missing: $path" >&2; exit 1; }
done
for suite in libero_spatial libero_object libero_goal libero_10; do
    [[ -f "$LIBERO_ROOT/${suite}${LIBERO_SUITE_DIR_SUFFIX}/meta/info.json" ]] || {
        echo "ERROR: LIBERO suite missing: ${suite}${LIBERO_SUITE_DIR_SUFFIX}" >&2
        exit 1
    }
done
[[ "$(nvidia-smi -L | wc -l)" -eq "$NUM_GPUS" ]] || { echo "ERROR: expected $NUM_GPUS GPUs" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT" "$PREFLIGHT_OUTPUT_ROOT" "$EVIDENCE_ROOT"
export PATH="$COSMOS_ENV/bin:$PATH"

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    [[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true
    [[ -z "$ARCHIVER_PID" || "$status" -eq 0 ]] || kill "$ARCHIVER_PID" 2>/dev/null || true
    [[ -z "$MONITOR_PID" ]] || wait "$MONITOR_PID" 2>/dev/null || true
    if [[ "$ENABLE_OSS" == 1 && -s "$RCLONE_CONFIG_PATH" ]]; then
        "$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence" --transfers=8 --checkers=16 || true
        local launch_log="$OUTPUT_ROOT/logs/${RUN_NAME}.train.log"
        [[ ! -f "$launch_log" ]] || "$RCLONE_BIN" copyto "$launch_log" "$OSS_RUN_ROOT/logs/${RUN_NAME}.train.log" || true
    fi
    rm -rf "$SECRET_ROOT"
    rm -f "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json" "$HANDOFF_ROOT/rclone-config.cms.der"
    rmdir "$HANDOFF_ROOT" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "$ENABLE_OSS" == 1 ]]; then
    [[ "$OSS_RUN_ROOT" == eailab-hdd2:eailab/chengjuntao/* ]] || { echo "ERROR: invalid OSS_RUN_ROOT" >&2; exit 1; }
    [[ -x "$RCLONE_BIN" ]] || { echo "ERROR: missing rclone: $RCLONE_BIN" >&2; exit 1; }
    mkdir -p "$SECRET_ROOT" "$HANDOFF_ROOT"
    chmod 700 "$SECRET_ROOT" "$HANDOFF_ROOT"
    openssl req -x509 -newkey rsa:3072 -nodes -days 1 -subj "/CN=$KUBEBRAIN_RJOB_NAME" \
        -keyout "$SECRET_ROOT/recipient.key" -out "$HANDOFF_ROOT/recipient.pem" >/dev/null 2>&1
    export KUBEBRAIN_RJOB_NAME OSS_RUN_ROOT
    "$COSMOS_ENV/bin/python" - "$HANDOFF_ROOT/request.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "job_id": os.environ["KUBEBRAIN_RJOB_NAME"],
    "oss_run_root": os.environ["OSS_RUN_ROOT"],
}, indent=2) + "\n")
PY
    chmod 444 "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json"
    echo "RCLONE_HANDOFF_WAITING root=$HANDOFF_ROOT"
    for _ in $(seq 1 240); do
        [[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] && break
        sleep 5
    done
    [[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] || { echo "ERROR: timed out waiting for rclone handoff" >&2; exit 1; }
    openssl cms -decrypt -binary -inform DER -in "$HANDOFF_ROOT/rclone-config.cms.der" \
        -recip "$HANDOFF_ROOT/recipient.pem" -inkey "$SECRET_ROOT/recipient.key" -out "$RCLONE_CONFIG_PATH"
    chmod 400 "$RCLONE_CONFIG_PATH"
    export RCLONE_CONFIG="$RCLONE_CONFIG_PATH"
    "$RCLONE_BIN" lsd eailab-hdd2:eailab/chengjuntao >/dev/null
    echo "RCLONE_HANDOFF_READY remote=eailab-hdd2"
fi

source "$COSMOS_ENV/bin/activate"
export PATH="$(dirname "$RCLONE_BIN"):$COSMOS_ENV/bin:$PATH" PYTHONPATH="$COSMOS_SOURCE"
# TorchCodec needs standard FFmpeg sonames, while PyAV's wheel contains the same
# FFmpeg 8 ABI under auditwheel-renamed filenames. Expose only compatibility
# symlinks and leave the packaged libraries unchanged.
AV_WHEEL_LIB_DIR="$(find "$COSMOS_ENV/lib" -type d -path '*/site-packages/av.libs' -print -quit)"
[[ -n "$AV_WHEEL_LIB_DIR" ]] || { echo "ERROR: PyAV wheel libraries not found" >&2; exit 1; }
TORCHCODEC_FFMPEG_LIB_DIR="$COSMOS_ENV/torchcodec_ffmpeg8_compat"
mkdir -p "$TORCHCODEC_FFMPEG_LIB_DIR"
link_ffmpeg_library() {
    local pattern="$1" soname="$2" target
    target="$(find "$AV_WHEEL_LIB_DIR" -maxdepth 1 -type f -name "$pattern" -print -quit)"
    [[ -n "$target" ]] || { echo "ERROR: FFmpeg library not found: $pattern" >&2; exit 1; }
    ln -sfn "$target" "$TORCHCODEC_FFMPEG_LIB_DIR/$soname"
}
link_ffmpeg_library 'libavcodec-*.so.62.*' libavcodec.so.62
link_ffmpeg_library 'libavdevice-*.so.62.*' libavdevice.so.62
link_ffmpeg_library 'libavfilter-*.so.11.*' libavfilter.so.11
link_ffmpeg_library 'libavformat-*.so.62.*' libavformat.so.62
link_ffmpeg_library 'libavutil-*.so.60.*' libavutil.so.60
link_ffmpeg_library 'libswresample-*.so.6.*' libswresample.so.6
link_ffmpeg_library 'libswscale-*.so.9.*' libswscale.so.9

# TorchCodec also links against CUDA wheel libraries, not only the driver
# libraries injected by the H worker.
CUDA_WHEEL_LIB_DIRS="$(find "$COSMOS_ENV/lib" -type d -path '*/site-packages/nvidia/*/lib' -print | paste -sd: -)"
[[ -n "$CUDA_WHEEL_LIB_DIRS" ]] || { echo "ERROR: CUDA wheel library directories not found" >&2; exit 1; }
export LD_LIBRARY_PATH="$TORCHCODEC_FFMPEG_LIB_DIR:$AV_WHEEL_LIB_DIR:$CUDA_WHEEL_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42 NCCL_NVLS_ENABLE=0 NPROC_PER_NODE="$NUM_GPUS"
export LIBERO_ROOT LIBERO_SUITE_DIR_SUFFIX BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export EXPERIMENT_ID SOURCE_COMMIT
export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" RUN_NAME
export LOG_FILENAME="${RUN_NAME}.train.log"

"$COSMOS_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

from torchcodec.decoders import VideoDecoder

video = (
    Path(os.environ["LIBERO_ROOT"])
    / f"libero_spatial{os.environ['LIBERO_SUITE_DIR_SUFFIX']}"
    / "videos/chunk-000/observation.images.image/episode_000000.mp4"
)
frame = VideoDecoder(str(video), device="cpu")[0]
if tuple(frame.shape) != (3, 512, 512):
    raise RuntimeError(f"unexpected TorchCodec frame shape: {tuple(frame.shape)}")
print(f"TORCHCODEC_DECODE_READY video={video} shape={tuple(frame.shape)} dtype={frame.dtype}")
PY

"$COSMOS_ENV/bin/python" - "$EVIDENCE_ROOT/run_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = ("EXPERIMENT_ID", "RUN_NAME", "SOURCE_COMMIT", "LIBERO_ROOT", "LIBERO_SUITE_DIR_SUFFIX", "BASE_CHECKPOINT_PATH", "OSS_RUN_ROOT")
Path(sys.argv[1]).write_text(json.dumps({key: os.environ.get(key, "") for key in keys}, indent=2) + "\n")
PY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
    --format=csv -l 30 >"$EVIDENCE_ROOT/nvidia-smi.csv" 2>&1 &
MONITOR_PID=$!

if [[ "$RUN_PREFLIGHT_FIRST" == 1 ]]; then
    readonly PREFLIGHT_RUN_NAME="${RUN_NAME}-preflight"
    readonly PREFLIGHT_LOG="$PREFLIGHT_OUTPUT_ROOT/logs/${PREFLIGHT_RUN_NAME}.train.log"
    echo "PREFLIGHT_START experiment=$EXPERIMENT_ID gpus=$NUM_GPUS steps=2"
    env \
        OUTPUT_ROOT="$PREFLIGHT_OUTPUT_ROOT" \
        IMAGINAIRE_OUTPUT_ROOT="$PREFLIGHT_OUTPUT_ROOT" \
        RUN_NAME="$PREFLIGHT_RUN_NAME" \
        LOG_FILENAME="${PREFLIGHT_RUN_NAME}.train.log" \
        EXTRA_TAIL_OVERRIDES="trainer.max_iter=2 trainer.logging_iter=1 trainer.grad_accum_iter=1 checkpoint.save_iter=1000 dataloader_train.max_samples_per_batch=4" \
        "$LAUNCHER"
    [[ -s "$PREFLIGHT_LOG" ]] || { echo "ERROR: preflight log missing: $PREFLIGHT_LOG" >&2; exit 1; }
    "$COSMOS_ENV/bin/python" "$COSMOS_SOURCE/examples/h_cluster/validate_training_log_finite.py" \
        "$PREFLIGHT_LOG" --min-iterations 2
    echo "PREFLIGHT_COMPLETE experiment=$EXPERIMENT_ID steps=2"
fi

if [[ "$ENABLE_OSS" == 1 ]]; then
    (
        for _ in $(seq 1 1200); do [[ -d "$CHECKPOINTS_DIR" ]] && break; sleep 1; done
        [[ -d "$CHECKPOINTS_DIR" ]]
        exec "$COSMOS_ENV/bin/python" "$ARCHIVER" \
            --checkpoints-dir "$CHECKPOINTS_DIR" --remote-root "$OSS_RUN_ROOT" \
            --receipts-dir "$EVIDENCE_ROOT/checkpoint-receipts" --experiment-id "$EXPERIMENT_ID" \
            --job-id "$KUBEBRAIN_RJOB_NAME" --source-commit "$SOURCE_COMMIT" \
            --start-step "$ARCHIVE_START_STEP" --end-step "$ARCHIVE_END_STEP" --interval "$ARCHIVE_INTERVAL"
    ) >"$EVIDENCE_ROOT/archive.log" 2>&1 &
    ARCHIVER_PID=$!
fi

echo "TRAIN_START experiment=$EXPERIMENT_ID gpus=$NUM_GPUS"
"$LAUNCHER"
[[ -z "$ARCHIVER_PID" ]] || wait "$ARCHIVER_PID"
echo "TRAIN_AND_ARCHIVE_COMPLETE target_step=$ARCHIVE_END_STEP"
