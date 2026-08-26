#!/usr/bin/env bash
# Build the four-suite LIBERO Wan VAE cache on worker-local NVMe and OSS.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the committed Cosmos source.}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${KUBEBRAIN_RJOB_NAME:?Set KUBEBRAIN_RJOB_NAME.}"
: "${NUM_GPUS:=3}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${LIBERO_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/libero/fastwam/LIBERO-fastwam}"
: "${LIBERO_SUITE_DIR_SUFFIX:=_no_noops_lerobot}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${CACHE_REMOTE:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826/wan-latent-cache-v1}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"
: "${NVME_ROOT:=/nvme/libero-wan-latent-cache}"
: "${VAE_BATCH_SIZE:=8}"
: "${RECORDS_PER_SHARD:=1024}"

readonly BUILDER="$COSMOS_SOURCE/examples/h_cluster/build_libero_wan_latent_cache.py"
readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"
readonly STAGING_ROOT="$RUN_ROOT/staging"
readonly EVIDENCE_ROOT="$RUN_ROOT/evidence"
readonly BUILD_LOG="$EVIDENCE_ROOT/build.log"
readonly HANDOFF_ROOT="$PERSONAL_ROOT/runtime-handoffs/libero-wan-cache/$KUBEBRAIN_RJOB_NAME"
readonly SECRET_ROOT="/dev/shm/libero-wan-cache-rclone-$KUBEBRAIN_RJOB_NAME"
readonly RCLONE_CONFIG_PATH="$SECRET_ROOT/rclone.conf"
readonly EVIDENCE_REMOTE="$CACHE_REMOTE/_build_evidence/$EXPERIMENT_ID"

MONITOR_PID=""
for path in "$BUILDER" "$COSMOS_ENV/bin/python" "$WAN_VAE_PATH" "$RCLONE_BIN"; do
    [[ -e "$path" ]] || { echo "ERROR: required input missing: $path" >&2; exit 1; }
done
[[ "$CACHE_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || {
    echo "ERROR: CACHE_REMOTE must stay under eailab-hdd2:eailab/chengjuntao/" >&2
    exit 1
}
[[ "$(nvidia-smi -L | wc -l)" -eq "$NUM_GPUS" ]] || {
    echo "ERROR: expected exactly $NUM_GPUS visible GPUs" >&2
    exit 1
}
for suite in libero_spatial libero_object libero_goal libero_10; do
    [[ -f "$LIBERO_ROOT/${suite}${LIBERO_SUITE_DIR_SUFFIX}/meta/info.json" ]] || {
        echo "ERROR: LIBERO suite missing: ${suite}${LIBERO_SUITE_DIR_SUFFIX}" >&2
        exit 1
    }
done

mkdir -p "$STAGING_ROOT" "$EVIDENCE_ROOT" "$SECRET_ROOT" "$HANDOFF_ROOT"
chmod 700 "$RUN_ROOT" "$EVIDENCE_ROOT" "$SECRET_ROOT" "$HANDOFF_ROOT"

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    [[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true
    [[ -z "$MONITOR_PID" ]] || wait "$MONITOR_PID" 2>/dev/null || true
    if [[ -s "$RCLONE_CONFIG_PATH" ]]; then
        "$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$EVIDENCE_REMOTE" --transfers=8 --checkers=16 || true
    fi
    rm -rf "$SECRET_ROOT"
    rm -f "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json" "$HANDOFF_ROOT/rclone-config.cms.der"
    rmdir "$HANDOFF_ROOT" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

openssl req -x509 -newkey rsa:3072 -nodes -days 1 -subj "/CN=$KUBEBRAIN_RJOB_NAME" \
    -keyout "$SECRET_ROOT/recipient.key" -out "$HANDOFF_ROOT/recipient.pem" >/dev/null 2>&1
export KUBEBRAIN_RJOB_NAME CACHE_REMOTE EXPERIMENT_ID
"$COSMOS_ENV/bin/python" - "$HANDOFF_ROOT/request.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "job_id": os.environ["KUBEBRAIN_RJOB_NAME"],
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "cache_remote": os.environ["CACHE_REMOTE"],
}, indent=2) + "\n")
PY
chmod 444 "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json"
echo "RCLONE_HANDOFF_WAITING root=$HANDOFF_ROOT"
for _ in $(seq 1 360); do
    [[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] && break
    sleep 5
done
[[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] || {
    echo "ERROR: timed out waiting for encrypted rclone config" >&2
    exit 1
}
openssl cms -decrypt -binary -inform DER -in "$HANDOFF_ROOT/rclone-config.cms.der" \
    -recip "$HANDOFF_ROOT/recipient.pem" -inkey "$SECRET_ROOT/recipient.key" -out "$RCLONE_CONFIG_PATH"
chmod 400 "$RCLONE_CONFIG_PATH"
export RCLONE_CONFIG="$RCLONE_CONFIG_PATH"
"$RCLONE_BIN" lsd eailab-hdd2:eailab/chengjuntao >/dev/null
echo "RCLONE_HANDOFF_READY remote=eailab-hdd2"

source "$COSMOS_ENV/bin/activate"
export PATH="$(dirname "$RCLONE_BIN"):$COSMOS_ENV/bin:$PATH" PYTHONPATH="$COSMOS_SOURCE"

# TorchCodec needs the standard FFmpeg 8 sonames provided inside PyAV's wheel.
AV_WHEEL_LIB_DIR="$(find "$COSMOS_ENV/lib" -type d -path '*/site-packages/av.libs' -print -quit)"
[[ -n "$AV_WHEEL_LIB_DIR" ]] || { echo "ERROR: PyAV wheel libraries not found" >&2; exit 1; }
TORCHCODEC_FFMPEG_LIB_DIR="$RUN_ROOT/torchcodec-ffmpeg8"
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
CUDA_WHEEL_LIB_DIRS="$(find "$COSMOS_ENV/lib" -type d -path '*/site-packages/nvidia/*/lib' -print | paste -sd: -)"
[[ -n "$CUDA_WHEEL_LIB_DIRS" ]] || { echo "ERROR: CUDA wheel libraries not found" >&2; exit 1; }
export LD_LIBRARY_PATH="$TORCHCODEC_FFMPEG_LIB_DIR:$AV_WHEEL_LIB_DIR:$CUDA_WHEEL_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42 NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export EXPERIMENT_ID KUBEBRAIN_RJOB_NAME SOURCE_COMMIT CACHE_REMOTE NUM_GPUS
export LIBERO_ROOT LIBERO_SUITE_DIR_SUFFIX WAN_VAE_PATH VAE_BATCH_SIZE RECORDS_PER_SHARD

"$COSMOS_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
from torchcodec.decoders import VideoDecoder

video = Path(os.environ["LIBERO_ROOT"]) / (
    "libero_spatial" + os.environ["LIBERO_SUITE_DIR_SUFFIX"]
) / "videos/chunk-000/observation.images.image/episode_000000.mp4"
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

keys = (
    "EXPERIMENT_ID", "KUBEBRAIN_RJOB_NAME", "SOURCE_COMMIT", "LIBERO_ROOT",
    "LIBERO_SUITE_DIR_SUFFIX", "WAN_VAE_PATH", "CACHE_REMOTE", "NUM_GPUS",
    "VAE_BATCH_SIZE", "RECORDS_PER_SHARD",
)
Path(sys.argv[1]).write_text(json.dumps({key: os.environ.get(key, "") for key in keys}, indent=2) + "\n")
PY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
    --format=csv -l 30 >"$EVIDENCE_ROOT/nvidia-smi.csv" 2>&1 &
MONITOR_PID=$!

echo "CACHE_BUILD_START experiment=$EXPERIMENT_ID gpus=$NUM_GPUS remote=$CACHE_REMOTE"
torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" "$BUILDER" \
    --dataset-root "$LIBERO_ROOT" \
    --suite-suffix "$LIBERO_SUITE_DIR_SUFFIX" \
    --wan-vae "$WAN_VAE_PATH" \
    --output-remote "$CACHE_REMOTE" \
    --staging-root "$STAGING_ROOT" \
    --rclone "$RCLONE_BIN" \
    --experiment-id "$EXPERIMENT_ID" \
    --source-commit "$SOURCE_COMMIT" \
    --vae-batch-size "$VAE_BATCH_SIZE" \
    --records-per-shard "$RECORDS_PER_SHARD" 2>&1 | tee "$BUILD_LOG"
echo "CACHE_BUILD_AND_UPLOAD_COMPLETE experiment=$EXPERIMENT_ID remote=$CACHE_REMOTE"
