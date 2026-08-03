#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly QUOTA_GROUP="eailabagent_gpu"
readonly EXCLUDED_NODE="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly PERSONAL_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly OUTPUT_ROOT="$PERSONAL_ROOT/outputs/robocasa365-wan-fullpad-oss"
readonly SNAPSHOT_ROOT="$PERSONAL_ROOT/run_snapshots"
readonly PERSONAL_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_MOUNT="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly OFFICIAL_VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
readonly RJOB_BIN="/usr/local/bin/rjob"
readonly PYTHON3_BIN="/usr/bin/python3"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly REALPATH_BIN="/usr/bin/realpath"
readonly BASH_BIN="/usr/bin/bash"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${LATENT_CACHE_RUN_ID:?Set LATENT_CACHE_RUN_ID.}"
: "${RJOB_NAME:?Set RJOB_NAME.}"
: "${CACHE_JOB_MODE:?Set CACHE_JOB_MODE to preflight or full.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${RCLONE_BIN:?Set the pinned GPFS rclone binary.}"
: "${EXPECTED_RCLONE_SHA256:?Set EXPECTED_RCLONE_SHA256.}"
: "${RJOB_PREDICT_ONLY:=1}"
: "${ROBOCASA365_OSS_REMOTE:=}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${VAE_BATCH_SIZE:=8}"
: "${LOADER_WORKERS:=8}"

[[ "$#" -eq 0 ]] || { echo "ERROR: command-line arguments are forbidden" >&2; exit 1; }
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] || { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$CACHE_JOB_MODE" == preflight || "$CACHE_JOB_MODE" == full ]] ||
  { echo "ERROR: CACHE_JOB_MODE must be preflight or full" >&2; exit 1; }
[[ "$LATENT_CACHE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
  { echo "ERROR: invalid rjob name" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid source commit" >&2; exit 1; }
[[ "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  { echo "ERROR: invalid snapshot manifest SHA-256" >&2; exit 1; }
[[ "$EXPECTED_RCLONE_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: invalid rclone SHA-256" >&2; exit 1; }
[[ "$VAE_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$LOADER_WORKERS" =~ ^[0-9]+$ ]] ||
  { echo "ERROR: invalid VAE batch or loader-worker count" >&2; exit 1; }
if [[ "$CACHE_JOB_MODE" == full ]]; then
  [[ "$ROBOCASA365_OSS_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] ||
    { echo "ERROR: full mode requires a guarded eailab/chengjuntao OSS target" >&2; exit 1; }
else
  [[ -z "$ROBOCASA365_OSS_REMOTE" ]] ||
    { echo "ERROR: preflight mode must not carry an OSS target" >&2; exit 1; }
fi

for tool in "$RJOB_BIN" "$PYTHON3_BIN" "$SHA256SUM_BIN" "$REALPATH_BIN" "$BASH_BIN"; do
  [[ -x "$tool" ]] || { echo "ERROR: required tool is unavailable: $tool" >&2; exit 1; }
done

sha256_file() {
  local digest remainder
  read -r digest remainder < <("$SHA256SUM_BIN" "$1")
  printf '%s\n' "$digest"
}

COSMOS_SOURCE_REAL="$($REALPATH_BIN -e -- "$COSMOS_SOURCE")"
COSMOS_ENV_REAL="$($REALPATH_BIN -e -- "$COSMOS_ENV")"
DATASET_REAL="$($REALPATH_BIN -e -- "$ROBOCASA365_ROOT")"
SOURCE_REAL="$($REALPATH_BIN -e -- "$ROBOCASA365_SOURCE_ROOT")"
VAE_REAL="$($REALPATH_BIN -e -- "$WAN_VAE_PATH")"
RCLONE_REAL="$($REALPATH_BIN -e -- "$RCLONE_BIN")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REAL" && "$COSMOS_SOURCE_REAL" == "$SNAPSHOT_ROOT"/* ]] ||
  { echo "ERROR: COSMOS_SOURCE must be one immutable personal-GPFS snapshot" >&2; exit 1; }
[[ "$COSMOS_ENV" == "$COSMOS_ENV_REAL" && "$ROBOCASA365_ROOT" == "$DATASET_REAL" ]] ||
  { echo "ERROR: environment/dataset paths must be canonical" >&2; exit 1; }
[[ "$ROBOCASA365_SOURCE_ROOT" == "$SOURCE_REAL" && "$WAN_VAE_PATH" == "$VAE_REAL" ]] ||
  { echo "ERROR: source/VAE paths must be canonical" >&2; exit 1; }
[[ "$RCLONE_BIN" == "$RCLONE_REAL" && "$(sha256_file "$RCLONE_REAL")" == "$EXPECTED_RCLONE_SHA256" ]] ||
  { echo "ERROR: pinned rclone path or checksum mismatch" >&2; exit 1; }
[[ "$(sha256_file "$VAE_REAL")" == "$OFFICIAL_VAE_SHA256" ]] ||
  { echo "ERROR: Wan2.2 VAE checksum mismatch" >&2; exit 1; }

SNAPSHOT_MANIFEST="$COSMOS_SOURCE_REAL/SNAPSHOT_MANIFEST.json"
PROVENANCE_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_formal_provenance.py"
BUILD_WRAPPER="$COSMOS_SOURCE_REAL/examples/h_cluster/build_robocasa365_wan_latent_cache.sh"
BUILD_PYTHON="$COSMOS_SOURCE_REAL/examples/h_cluster/build_robocasa365_wan_latent_cache.py"
FULLPAD_RUNNER="$COSMOS_SOURCE_REAL/examples/h_cluster/build_and_publish_robocasa365_wan_fullpad_oss.sh"
OSS_PUBLISHER="$COSMOS_SOURCE_REAL/examples/h_cluster/publish_robocasa365_wan_cache_to_oss.py"
TORCHCODEC_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/configure_torchcodec_runtime.sh"
for file in \
  "$SNAPSHOT_MANIFEST" \
  "$PROVENANCE_HELPER" \
  "$BUILD_WRAPPER" \
  "$BUILD_PYTHON" \
  "$FULLPAD_RUNNER" \
  "$OSS_PUBLISHER" \
  "$TORCHCODEC_HELPER"; do
  [[ -s "$file" && ! -L "$file" ]] || { echo "ERROR: snapshot file missing or unsafe: $file" >&2; exit 1; }
done
[[ "$(sha256_file "$SNAPSHOT_MANIFEST")" == "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" ]] ||
  { echo "ERROR: snapshot manifest checksum mismatch" >&2; exit 1; }
"$PYTHON3_BIN" "$PROVENANCE_HELPER" verify-snapshot \
  --snapshot-root "$COSMOS_SOURCE_REAL" \
  --allowed-root "$SNAPSHOT_ROOT" \
  --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null

"$PYTHON3_BIN" - "$SNAPSHOT_MANIFEST" "$EXPECTED_SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("source_head") != sys.argv[2] or manifest.get("source_dirty") is not False:
    raise SystemExit("snapshot does not bind the expected clean source commit")
PY

ENV_MARKER="$COSMOS_ENV_REAL/.cwam_env_ready.json"
RUNTIME_PYTHON="$COSMOS_ENV_REAL/bin/python"
[[ -s "$ENV_MARKER" && -x "$RUNTIME_PYTHON" ]] || { echo "ERROR: audited Cosmos environment is incomplete" >&2; exit 1; }
EXPECTED_ENV_MARKER_SHA256="$(sha256_file "$ENV_MARKER")"
EXPECTED_PYTHON_REALPATH="$($REALPATH_BIN -e -- "$RUNTIME_PYTHON")"
EXPECTED_BUILD_WRAPPER_SHA256="$(sha256_file "$BUILD_WRAPPER")"
EXPECTED_BUILD_PYTHON_SHA256="$(sha256_file "$BUILD_PYTHON")"
EXPECTED_PROVENANCE_HELPER_SHA256="$(sha256_file "$PROVENANCE_HELPER")"
EXPECTED_TORCHCODEC_HELPER_SHA256="$(sha256_file "$TORCHCODEC_HELPER")"
EXPECTED_FULLPAD_RUNNER_SHA256="$(sha256_file "$FULLPAD_RUNNER")"
EXPECTED_OSS_PUBLISHER_SHA256="$(sha256_file "$OSS_PUBLISHER")"

WORK_ROOT="$OUTPUT_ROOT/$LATENT_CACHE_RUN_ID"
PREFLIGHT_RECEIPT="$WORK_ROOT/eight-gpu-preflight.json"
NVME_OUTPUT="/nvme/cwam-robocasa365-wan-fullpad/$LATENT_CACHE_RUN_ID/cache"

printf -v Q_SOURCE %q "$COSMOS_SOURCE_REAL"
printf -v Q_ENV %q "$COSMOS_ENV_REAL"
printf -v Q_DATASET %q "$DATASET_REAL"
printf -v Q_RAW_SOURCE %q "$SOURCE_REAL"
printf -v Q_VAE %q "$VAE_REAL"
printf -v Q_RUN_ID %q "$LATENT_CACHE_RUN_ID"
printf -v Q_WORK_ROOT %q "$WORK_ROOT"
printf -v Q_OUTPUT %q "$NVME_OUTPUT"
printf -v Q_REMOTE %q "$ROBOCASA365_OSS_REMOTE"
printf -v Q_RCLONE %q "$RCLONE_REAL"
printf -v Q_BUILD_WRAPPER %q "$BUILD_WRAPPER"
printf -v Q_FULL_RUNNER %q "$FULLPAD_RUNNER"
printf -v Q_PREFLIGHT_RECEIPT %q "$PREFLIGHT_RECEIPT"
printf -v Q_RUNTIME_PYTHON %q "$RUNTIME_PYTHON"
CAPACITY_PROBE_CODE='import sys; sys.path.insert(0, sys.argv[1]); from examples.h_cluster.publish_robocasa365_wan_cache_to_oss import _capacity_gate; result = _capacity_gate(0); print(f"OSS_CAPACITY_PREFLIGHT current_gb={result['\''current_used_gb'\'']:.2f} remaining_gb={result['\''remaining_to_threshold_gb'\'']:.2f}")'
printf -v Q_CAPACITY_PROBE_CODE %q "$CAPACITY_PROBE_CODE"

COMMAND_LINES=(
  "set -Eeuo pipefail"
  "unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE"
  "export PYTHONDONTWRITEBYTECODE=1"
  "export PYTHONNOUSERSITE=1"
  "export COSMOS_SOURCE=$Q_SOURCE"
  "export COSMOS_ENV=$Q_ENV"
  "export ROBOCASA365_ROOT=$Q_DATASET"
  "export ROBOCASA365_SOURCE_ROOT=$Q_RAW_SOURCE"
  "export WAN_VAE_PATH=$Q_VAE"
  "export LATENT_CACHE_RUN_ID=$Q_RUN_ID"
  "export FORMAL_EXPERIMENT_ID=$Q_RUN_ID"
  "export FORMAL_RUN_NAME=$Q_RUN_ID"
  "export ROBOCASA365_NVME_SOURCE_ROOT=/nvme/cwam-robocasa365-d5eac857"
  "export ROBOCASA365_CACHE_SPLITS=train,val,test"
  "export ROBOCASA365_CACHE_CAMERA_MODE=triple_view"
  "export ROBOCASA365_CACHE_RESOLUTION=480"
  "export ROBOCASA365_CACHE_STAGE=full_padding"
  "export NUM_GPUS=8"
  "export LOADER_WORKERS=$LOADER_WORKERS"
  "export VAE_BATCH_SIZE=$VAE_BATCH_SIZE"
  "export EQUIVALENCE_SAMPLES_PER_RANK=8"
  "export EXPECTED_SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
  "export EXPECTED_SNAPSHOT_MANIFEST_SHA256=$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
  "export EXPECTED_BUILD_WRAPPER_SHA256=$EXPECTED_BUILD_WRAPPER_SHA256"
  "export EXPECTED_BUILD_PYTHON_SHA256=$EXPECTED_BUILD_PYTHON_SHA256"
  "export EXPECTED_PROVENANCE_HELPER_SHA256=$EXPECTED_PROVENANCE_HELPER_SHA256"
  "export EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256=$EXPECTED_TORCHCODEC_HELPER_SHA256"
  "export EXPECTED_ENV_REALPATH=$Q_ENV"
  "export EXPECTED_ENV_MARKER_SHA256=$EXPECTED_ENV_MARKER_SHA256"
  "export EXPECTED_PYTHON_REALPATH=$EXPECTED_PYTHON_REALPATH"
  "export EXPECTED_WAN_VAE_SHA256=$OFFICIAL_VAE_SHA256"
  "/usr/bin/test -x /usr/bin/openssl"
  "/usr/bin/openssl cms -help >/dev/null 2>&1"
  "[[ \$(/usr/bin/sha256sum $Q_RCLONE | /usr/bin/cut -d' ' -f1) == $EXPECTED_RCLONE_SHA256 ]]"
  "$Q_RCLONE version"
  "$Q_RCLONE help flags | /usr/bin/grep -q -- --s3-upload-concurrency"
  "$Q_RUNTIME_PYTHON -B -c $Q_CAPACITY_PROBE_CODE $Q_SOURCE"
)
if [[ "$CACHE_JOB_MODE" == preflight ]]; then
  COMMAND_LINES+=(
    "export ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT=$Q_OUTPUT"
    "export ROBOCASA365_CACHE_PREFLIGHT_ONLY=1"
    "export ROBOCASA365_CACHE_PREFLIGHT_RECEIPT=$Q_PREFLIGHT_RECEIPT"
    "/usr/bin/mkdir -p -- /nvme/cwam-robocasa365-wan-fullpad/$Q_RUN_ID"
    "exec $BASH_BIN $Q_BUILD_WRAPPER"
  )
else
  COMMAND_LINES+=(
    "export ROBOCASA365_OSS_REMOTE=$Q_REMOTE"
    "export ROBOCASA365_OSS_WORK_ROOT=$Q_WORK_ROOT"
    "export RCLONE_BIN=$Q_RCLONE"
    "export EXPECTED_RCLONE_SHA256=$EXPECTED_RCLONE_SHA256"
    "export EXPECTED_FULLPAD_RUNNER_SHA256=$EXPECTED_FULLPAD_RUNNER_SHA256"
    "export EXPECTED_OSS_PUBLISHER_SHA256=$EXPECTED_OSS_PUBLISHER_SHA256"
    "exec $BASH_BIN $Q_FULL_RUNNER"
  )
fi
WORKER_COMMAND="$(printf '%s\n' "${COMMAND_LINES[@]}")"
WORKER_COMMAND_SHA256="$(printf '%s' "$WORKER_COMMAND" | "$SHA256SUM_BIN" | /usr/bin/cut -d' ' -f1)"

RJOB_ARGS=(
  submit
  --name "$RJOB_NAME"
  --group "$QUOTA_GROUP"
  --charged-group "$QUOTA_GROUP"
  --private-machine group
  --priority 9
  --task-type normal
  --replicas 1
  --restart-policy never
  --termination-grace-period-seconds 60
  --backoff_limit 1
  --enable-sshd
  --image "$PINNED_IMAGE"
  --cpu 128
  --gpu 8
  --memory 1500000
  --negative-tags "$EXCLUDED_NODE"
  --store-host-nvme
  --mount "$PERSONAL_MOUNT" "$PUBLIC_MOUNT"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
  RJOB_ARGS+=(--predict-only true)
fi

echo "CACHE_JOB_MODE=$CACHE_JOB_MODE"
echo "LATENT_CACHE_RUN_ID=$LATENT_CACHE_RUN_ID"
echo "RJOB_NAME=$RJOB_NAME"
echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
echo "SNAPSHOT_MANIFEST_SHA256=$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
echo "WORKER_COMMAND_SHA256=$WORKER_COMMAND_SHA256"
echo "LATENT_CONTRACT=stage=full_padding shape=48x5x30x52 image_size=480x832x256x768"
if [[ "$CACHE_JOB_MODE" == full ]]; then
  echo "OSS_REMOTE=$ROBOCASA365_OSS_REMOTE"
fi

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
  "$RJOB_BIN" "${RJOB_ARGS[@]}" -- /usr/bin/env -u BASH_ENV -u ENV "$BASH_BIN" --noprofile --norc -c "$WORKER_COMMAND"
  echo "RJOB_PREDICTION_COMPLETE reservation_created=false"
  exit 0
fi

/usr/bin/mkdir -p -- "$OUTPUT_ROOT"
if ! /usr/bin/mkdir --mode=0700 -- "$WORK_ROOT"; then
  echo "ERROR: duplicate run reservation exists: $WORK_ROOT" >&2
  exit 1
fi
INTENT="$WORK_ROOT/submission-intent.json"
"$PYTHON3_BIN" - "$INTENT" \
  "$LATENT_CACHE_RUN_ID" "$CACHE_JOB_MODE" "$RJOB_NAME" "$EXPECTED_SOURCE_COMMIT" \
  "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" "$WORKER_COMMAND_SHA256" "$ROBOCASA365_OSS_REMOTE" <<'PY'
import datetime
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "robocasa365_wan_full_padding_submission_intent",
    "status": "submitting",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run_id": sys.argv[2],
    "mode": sys.argv[3],
    "showname": sys.argv[4],
    "source_commit": sys.argv[5],
    "snapshot_manifest_sha256": sys.argv[6],
    "worker_command_sha256": sys.argv[7],
    "oss_remote": sys.argv[8] or None,
    "resources": {"replicas": 1, "gpu": 8, "cpu": 128, "memory_mib": 1500000},
    "latent_contract": {
        "stage": "full_padding",
        "shape": [48, 5, 30, 52],
        "image_size": [480, 832, 256, 768],
        "storage_dtype": "bfloat16",
    },
}
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(target, 0o444)
PY

SUBMIT_LOG="$WORK_ROOT/rjob-submit.log"
set +e
"$RJOB_BIN" "${RJOB_ARGS[@]}" -- /usr/bin/env -u BASH_ENV -u ENV "$BASH_BIN" --noprofile --norc -c "$WORKER_COMMAND" \
  > >( /usr/bin/tee "$SUBMIT_LOG" ) 2>&1
SUBMIT_STATUS=$?
set -e
[[ "$SUBMIT_STATUS" -eq 0 ]] || { echo "ERROR: rjob submission failed; reservation retained" >&2; exit 1; }
/usr/bin/chmod 0444 "$SUBMIT_LOG"
SCHEDULER_NAME="$($PYTHON3_BIN - "$SUBMIT_LOG" "$RJOB_NAME" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
matches = re.findall(r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)", text)
if len(matches) != 1 or not matches[0].startswith(sys.argv[2] + "-"):
    raise SystemExit(f"cannot bind one scheduler name: {matches!r}")
print(matches[0])
PY
)"
SCHEDULER_READBACK="$WORK_ROOT/rjob-get.log"
"$RJOB_BIN" get "$SCHEDULER_NAME" >"$SCHEDULER_READBACK" 2>&1
/usr/bin/chmod 0444 "$SCHEDULER_READBACK"
printf '%s\n' "$SCHEDULER_NAME" >"$WORK_ROOT/scheduler-rjob-name.txt"
/usr/bin/chmod 0444 "$WORK_ROOT/scheduler-rjob-name.txt"
echo "RJOB_SUBMISSION_ACCEPTED scheduler_name=$SCHEDULER_NAME work_root=$WORK_ROOT"
