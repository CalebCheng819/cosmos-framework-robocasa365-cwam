#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Submit exactly one single-node, eight-H200 RoboCasa365 Wan latent-cache build.
# The default is prediction-only. Set RJOB_PREDICT_ONLY=0 deliberately for the
# actual submission; that path creates an exclusive provenance reservation and
# retains it on every ambiguous or failed outcome.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
readonly PINNED_NVME_SOURCE_ROOT="/nvme/cwam-robocasa365-d5eac857"
readonly PINNED_SUBMISSION_OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/robocasa365-wan-latent-cache-build"
readonly SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
readonly CACHE_OUTPUT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data"
readonly OFFICIAL_WAN22_VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
readonly PERSONAL_GPFS_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS_MOUNT="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly RJOB_REPLICAS=1
readonly NPROC_REQUIRED=8
# H200 nodes expose 186 allocatable CPUs to Kubernetes. 128 CPUs leave
# headroom for the node services while supplying two cores per loader worker
# at the fixed 8 ranks x 8 workers/rank cache-build setting.
readonly RJOB_CPU=128
readonly RJOB_MEMORY_MIB=1500000
readonly SCHEDULER_RECEIPT_ATTEMPTS=86400
readonly SCHEDULER_RECEIPT_RETRY_SECONDS=1
readonly SCHEDULER_RECEIPT_TIMEOUT_SECONDS=90000
readonly REALPATH_BIN="/usr/bin/realpath"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly PYTHON3_BIN="/usr/bin/python3"
readonly BASH_BIN="/usr/bin/bash"
readonly TEE_BIN="/usr/bin/tee"
readonly SLEEP_BIN="/usr/bin/sleep"
readonly CHMOD_BIN="/usr/bin/chmod"
readonly STAT_BIN="/usr/bin/stat"
readonly TIMEOUT_BIN="/usr/bin/timeout"
readonly ENV_BIN="/usr/bin/env"
readonly RJOB_BIN="/usr/local/bin/rjob"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable Cosmos source snapshot.}"
: "${LATENT_CACHE_RUN_ID:?Set a unique LATENT_CACHE_RUN_ID.}"
: "${RJOB_NAME:?Set a unique lowercase H-cluster rjob show name.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT:?Set the final shared cache output root.}"
: "${RJOB_PREDICT_ONLY:=1}"
: "${RJOB_IMAGE:=$PINNED_CONTAINER_IMAGE}"
: "${RJOB_QUOTA_GROUP:=$PINNED_QUOTA_GROUP}"
: "${RJOB_EXCLUDED_NODE_TAG:=$PINNED_EXCLUDED_NODE_TAG}"
: "${ROBOCASA365_NVME_SOURCE_ROOT:=$PINNED_NVME_SOURCE_ROOT}"
: "${ROBOCASA365_CACHE_SPLITS:=train,val,test}"
: "${NUM_GPUS:=8}"
: "${LOADER_WORKERS:=8}"
: "${VAE_BATCH_SIZE:=8}"
: "${EQUIVALENCE_SAMPLES_PER_RANK:=8}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] ||
  { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
  { echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2; exit 1; }
[[ "$LATENT_CACHE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
  { echo "ERROR: LATENT_CACHE_RUN_ID contains unsupported characters" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase 40-character Git SHA" >&2; exit 1; }
[[ "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  { echo "ERROR: EXPECTED_SNAPSHOT_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$RJOB_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
  { echo "ERROR: RJOB_IMAGE differs from the audited image digest" >&2; exit 1; }
[[ "$RJOB_QUOTA_GROUP" == "$PINNED_QUOTA_GROUP" ]] ||
  { echo "ERROR: RJOB_QUOTA_GROUP differs from eailabagent_gpu" >&2; exit 1; }
[[ "$RJOB_EXCLUDED_NODE_TAG" == "$PINNED_EXCLUDED_NODE_TAG" ]] ||
  { echo "ERROR: RJOB_EXCLUDED_NODE_TAG differs from the audited node0685 exclusion" >&2; exit 1; }
[[ "$ROBOCASA365_NVME_SOURCE_ROOT" == "$PINNED_NVME_SOURCE_ROOT" ]] ||
  { echo "ERROR: ROBOCASA365_NVME_SOURCE_ROOT must be exactly $PINNED_NVME_SOURCE_ROOT" >&2; exit 1; }
[[ "$ROBOCASA365_CACHE_SPLITS" == "train,val,test" ]] ||
  { echo "ERROR: the synchronized full cache requires splits=train,val,test" >&2; exit 1; }
[[ "$NUM_GPUS" == "$NPROC_REQUIRED" ]] ||
  { echo "ERROR: NUM_GPUS must be exactly 8" >&2; exit 1; }
[[ "$LOADER_WORKERS" =~ ^[0-9]+$ ]] ||
  { echo "ERROR: LOADER_WORKERS must be a non-negative integer" >&2; exit 1; }
[[ "$VAE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
  { echo "ERROR: VAE_BATCH_SIZE must be a positive integer" >&2; exit 1; }
[[ "$EQUIVALENCE_SAMPLES_PER_RANK" =~ ^[1-9][0-9]*$ ]] ||
  { echo "ERROR: EQUIVALENCE_SAMPLES_PER_RANK must be a positive integer" >&2; exit 1; }

for required_tool in \
  "$REALPATH_BIN" \
  "$SHA256SUM_BIN" \
  "$PYTHON3_BIN" \
  "$BASH_BIN" \
  "$TEE_BIN" \
  "$SLEEP_BIN" \
  "$CHMOD_BIN" \
  "$STAT_BIN" \
  "$TIMEOUT_BIN" \
  "$ENV_BIN" \
  "$RJOB_BIN"; do
  [[ -x "$required_tool" ]] ||
    { echo "ERROR: required absolute tool is missing or not executable: $required_tool" >&2; exit 1; }
done
[[ "$("$REALPATH_BIN" -e -- "$RJOB_BIN")" == "$RJOB_BIN" ]] ||
  { echo "ERROR: pinned rjob client must be the canonical /usr/local/bin/rjob" >&2; exit 1; }
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

sha256_file() {
  local digest remainder
  read -r digest remainder < <("$SHA256SUM_BIN" "$1")
  printf '%s\n' "$digest"
}

path_is_within() {
  local candidate="$1"
  local parent="$2"
  [[ "$candidate" == "$parent" || "$candidate" == "$parent/"* ]]
}

BASH_REALPATH="$("$REALPATH_BIN" -e -- /proc/"$$"/exe)"
[[ "$BASH_REALPATH" == "$BASH_BIN" ]] ||
  { echo "ERROR: submitter must run with the audited Bash interpreter: $BASH_BIN" >&2; exit 1; }

COSMOS_SOURCE_REALPATH="$("$REALPATH_BIN" -e -- "$COSMOS_SOURCE")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REALPATH" && -d "$COSMOS_SOURCE_REALPATH" ]] ||
  { echo "ERROR: COSMOS_SOURCE must be an exact canonical directory" >&2; exit 1; }
SUBMITTER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/submit_robocasa365_wan_latent_cache_rjob.sh"
BUILD_WRAPPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/build_robocasa365_wan_latent_cache.sh"
BUILD_PYTHON="$COSMOS_SOURCE_REALPATH/examples/h_cluster/build_robocasa365_wan_latent_cache.py"
PROVENANCE_HELPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/cwam_formal_provenance.py"
TORCHCODEC_RUNTIME_HELPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/configure_torchcodec_runtime.sh"
SCHEDULER_CAPTURE_HELPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/capture_cwam_formal_rjob_receipt.py"
SNAPSHOT_MANIFEST="$COSMOS_SOURCE_REALPATH/SNAPSHOT_MANIFEST.json"
THIS_SUBMITTER_REALPATH="$("$REALPATH_BIN" -e -- "${BASH_SOURCE[0]}")"
[[ "$THIS_SUBMITTER_REALPATH" == "$SUBMITTER" ]] ||
  { echo "ERROR: submitter must execute from the exact immutable COSMOS_SOURCE snapshot" >&2; exit 1; }
for required_file in \
  "$SUBMITTER" \
  "$BUILD_WRAPPER" \
  "$BUILD_PYTHON" \
  "$PROVENANCE_HELPER" \
  "$TORCHCODEC_RUNTIME_HELPER" \
  "$SCHEDULER_CAPTURE_HELPER" \
  "$SNAPSHOT_MANIFEST"; do
  [[ -s "$required_file" && ! -L "$required_file" ]] ||
    { echo "ERROR: required immutable snapshot file is missing, empty, or a symlink: $required_file" >&2; exit 1; }
done

ACTUAL_SNAPSHOT_MANIFEST_SHA256="$(sha256_file "$SNAPSHOT_MANIFEST")"
[[ "$ACTUAL_SNAPSHOT_MANIFEST_SHA256" == "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" ]] ||
  { echo "ERROR: snapshot manifest SHA-256 mismatch before provenance helper execution" >&2; exit 1; }
EXPECTED_SUBMITTER_SHA256="$(sha256_file "$SUBMITTER")"
EXPECTED_BUILD_WRAPPER_SHA256="$(sha256_file "$BUILD_WRAPPER")"
EXPECTED_BUILD_PYTHON_SHA256="$(sha256_file "$BUILD_PYTHON")"
EXPECTED_PROVENANCE_HELPER_SHA256="$(sha256_file "$PROVENANCE_HELPER")"
EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256="$(sha256_file "$TORCHCODEC_RUNTIME_HELPER")"
EXPECTED_SCHEDULER_CAPTURE_HELPER_SHA256="$(sha256_file "$SCHEDULER_CAPTURE_HELPER")"
RJOB_BIN_SHA256="$(sha256_file "$RJOB_BIN")"

# Bootstrap trust without executing any snapshot code: the externally supplied
# manifest digest authenticates the exact helper bytes that perform the full
# immutable-tree verification below.
"$PYTHON3_BIN" -c '
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("schema_version") != 1:
    raise SystemExit(f"unsupported snapshot schema: {manifest.get('"'"'schema_version'"'"')!r}")
if manifest.get("source_head") != sys.argv[2]:
    raise SystemExit(
        f"snapshot source_head mismatch: expected={sys.argv[2]!r}, actual={manifest.get('"'"'source_head'"'"')!r}"
    )
if manifest.get("source_dirty") is not False:
    raise SystemExit("snapshot source_dirty must be false")
entries = manifest.get("files")
if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
    raise SystemExit("snapshot manifest file list/count is malformed")
indexed = {}
for entry in entries:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise SystemExit(f"malformed snapshot file entry: {entry!r}")
    if entry["path"] in indexed:
        raise SystemExit(f"duplicate snapshot file entry: {entry['"'"'path'"'"']}")
    indexed[entry["path"]] = entry
arguments = sys.argv[3:]
if len(arguments) % 3:
    raise SystemExit("internal bootstrap contract argument mismatch")
for offset in range(0, len(arguments), 3):
    relative, actual_sha256, actual_bytes_text = arguments[offset : offset + 3]
    entry = indexed.get(relative)
    if entry is None:
        raise SystemExit(f"snapshot manifest does not authenticate {relative}")
    expected = (entry.get("sha256"), entry.get("bytes"))
    actual = (actual_sha256, int(actual_bytes_text))
    if expected != actual:
        raise SystemExit(f"snapshot bootstrap mismatch for {relative}: expected={expected!r}, actual={actual!r}")
' \
  "$SNAPSHOT_MANIFEST" \
  "$EXPECTED_SOURCE_COMMIT" \
  "examples/h_cluster/submit_robocasa365_wan_latent_cache_rjob.sh" \
  "$EXPECTED_SUBMITTER_SHA256" \
  "$("$STAT_BIN" -c %s -- "$SUBMITTER")" \
  "examples/h_cluster/build_robocasa365_wan_latent_cache.sh" \
  "$EXPECTED_BUILD_WRAPPER_SHA256" \
  "$("$STAT_BIN" -c %s -- "$BUILD_WRAPPER")" \
  "examples/h_cluster/build_robocasa365_wan_latent_cache.py" \
  "$EXPECTED_BUILD_PYTHON_SHA256" \
  "$("$STAT_BIN" -c %s -- "$BUILD_PYTHON")" \
  "examples/h_cluster/cwam_formal_provenance.py" \
  "$EXPECTED_PROVENANCE_HELPER_SHA256" \
  "$("$STAT_BIN" -c %s -- "$PROVENANCE_HELPER")" \
  "examples/h_cluster/configure_torchcodec_runtime.sh" \
  "$EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256" \
  "$("$STAT_BIN" -c %s -- "$TORCHCODEC_RUNTIME_HELPER")" \
  "examples/h_cluster/capture_cwam_formal_rjob_receipt.py" \
  "$EXPECTED_SCHEDULER_CAPTURE_HELPER_SHA256" \
  "$("$STAT_BIN" -c %s -- "$SCHEDULER_CAPTURE_HELPER")"

"$PYTHON3_BIN" "$PROVENANCE_HELPER" verify-snapshot \
  --snapshot-root "$COSMOS_SOURCE_REALPATH" \
  --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
  --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null

COSMOS_ENV_REALPATH="$("$REALPATH_BIN" -e -- "$COSMOS_ENV")"
[[ "$COSMOS_ENV" == "$COSMOS_ENV_REALPATH" && -d "$COSMOS_ENV_REALPATH" ]] ||
  { echo "ERROR: COSMOS_ENV must be an exact canonical directory" >&2; exit 1; }
ENV_MARKER="$COSMOS_ENV_REALPATH/.cwam_env_ready.json"
[[ -s "$ENV_MARKER" && ! -L "$ENV_MARKER" ]] ||
  { echo "ERROR: pinned environment readiness marker is missing or unsafe" >&2; exit 1; }
EXPECTED_ENV_MARKER_SHA256="$(sha256_file "$ENV_MARKER")"
RUNTIME_PYTHON="$COSMOS_ENV_REALPATH/bin/python"
RUNTIME_PYTHON_REALPATH="$("$REALPATH_BIN" -e -- "$RUNTIME_PYTHON")"
[[ -x "$RUNTIME_PYTHON" ]] ||
  { echo "ERROR: COSMOS_ENV Python is not executable" >&2; exit 1; }

ROBOCASA365_ROOT_REALPATH="$("$REALPATH_BIN" -e -- "$ROBOCASA365_ROOT")"
ROBOCASA365_SOURCE_ROOT_REALPATH="$("$REALPATH_BIN" -e -- "$ROBOCASA365_SOURCE_ROOT")"
ROBOCASA365_OUTPUT_REALPATH="$("$REALPATH_BIN" -m -- "$ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT")"
CACHE_OUTPUT_ALLOWED_REALPATH="$("$REALPATH_BIN" -e -- "$CACHE_OUTPUT_ALLOWED_ROOT")"
WAN_VAE_REALPATH="$("$REALPATH_BIN" -e -- "$WAN_VAE_PATH")"
[[ "$ROBOCASA365_ROOT" == "$ROBOCASA365_ROOT_REALPATH" && -d "$ROBOCASA365_ROOT_REALPATH" ]] ||
  { echo "ERROR: ROBOCASA365_ROOT must be an exact canonical directory" >&2; exit 1; }
[[ "$ROBOCASA365_SOURCE_ROOT" == "$ROBOCASA365_SOURCE_ROOT_REALPATH" && -d "$ROBOCASA365_SOURCE_ROOT_REALPATH" ]] ||
  { echo "ERROR: ROBOCASA365_SOURCE_ROOT must be an exact canonical directory" >&2; exit 1; }
[[ "$ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT" == "$ROBOCASA365_OUTPUT_REALPATH" ]] ||
  { echo "ERROR: ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT must be an exact canonical path" >&2; exit 1; }
if ! path_is_within "$ROBOCASA365_OUTPUT_REALPATH" "$CACHE_OUTPUT_ALLOWED_REALPATH" ||
   [[ "$ROBOCASA365_OUTPUT_REALPATH" == "$CACHE_OUTPUT_ALLOWED_REALPATH" ]]; then
  echo "ERROR: cache output must be strictly below the shared-data root" >&2
  exit 1
fi
for input_root in "$ROBOCASA365_ROOT_REALPATH" "$ROBOCASA365_SOURCE_ROOT_REALPATH"; do
  if path_is_within "$ROBOCASA365_OUTPUT_REALPATH" "$input_root" ||
     path_is_within "$input_root" "$ROBOCASA365_OUTPUT_REALPATH"; then
    echo "ERROR: cache output and dataset input roots must not overlap: $input_root" >&2
    exit 1
  fi
done
ROBOCASA365_OUTPUT_PARENT="$("$REALPATH_BIN" -e -- "$("$REALPATH_BIN" -m -- "$ROBOCASA365_OUTPUT_REALPATH/..")")"
[[ -d "$ROBOCASA365_OUTPUT_PARENT" && -w "$ROBOCASA365_OUTPUT_PARENT" ]] ||
  { echo "ERROR: cache output parent must already exist and be writable: $ROBOCASA365_OUTPUT_PARENT" >&2; exit 1; }
[[ -f "$WAN_VAE_REALPATH" ]] ||
  { echo "ERROR: WAN_VAE_PATH must resolve to a regular file" >&2; exit 1; }
ACTUAL_WAN_VAE_SHA256="$(sha256_file "$WAN_VAE_REALPATH")"
[[ "$ACTUAL_WAN_VAE_SHA256" == "$OFFICIAL_WAN22_VAE_SHA256" ]] ||
  {
    echo "ERROR: official Wan2.2 VAE submit-side SHA-256 mismatch: expected=$OFFICIAL_WAN22_VAE_SHA256 actual=$ACTUAL_WAN_VAE_SHA256" >&2
    exit 1
  }

printf -v Q_COSMOS_SOURCE '%q' "$COSMOS_SOURCE_REALPATH"
printf -v Q_COSMOS_ENV '%q' "$COSMOS_ENV_REALPATH"
printf -v Q_ROBOCASA365_ROOT '%q' "$ROBOCASA365_ROOT_REALPATH"
printf -v Q_ROBOCASA365_SOURCE_ROOT '%q' "$ROBOCASA365_SOURCE_ROOT_REALPATH"
printf -v Q_ROBOCASA365_OUTPUT_ROOT '%q' "$ROBOCASA365_OUTPUT_REALPATH"
printf -v Q_WAN_VAE_PATH '%q' "$WAN_VAE_REALPATH"
printf -v Q_NVME_SOURCE_ROOT '%q' "$PINNED_NVME_SOURCE_ROOT"
printf -v Q_CACHE_SPLITS '%q' "$ROBOCASA365_CACHE_SPLITS"
printf -v Q_NUM_GPUS '%q' "$NUM_GPUS"
printf -v Q_LOADER_WORKERS '%q' "$LOADER_WORKERS"
printf -v Q_VAE_BATCH_SIZE '%q' "$VAE_BATCH_SIZE"
printf -v Q_EQUIVALENCE_SAMPLES '%q' "$EQUIVALENCE_SAMPLES_PER_RANK"
printf -v Q_EXPECTED_SOURCE_COMMIT '%q' "$EXPECTED_SOURCE_COMMIT"
printf -v Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256 '%q' "$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
printf -v Q_EXPECTED_BUILD_WRAPPER_SHA256 '%q' "$EXPECTED_BUILD_WRAPPER_SHA256"
printf -v Q_EXPECTED_BUILD_PYTHON_SHA256 '%q' "$EXPECTED_BUILD_PYTHON_SHA256"
printf -v Q_EXPECTED_PROVENANCE_HELPER_SHA256 '%q' "$EXPECTED_PROVENANCE_HELPER_SHA256"
printf -v Q_EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256 '%q' "$EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256"
printf -v Q_EXPECTED_ENV_REALPATH '%q' "$COSMOS_ENV_REALPATH"
printf -v Q_EXPECTED_ENV_MARKER_SHA256 '%q' "$EXPECTED_ENV_MARKER_SHA256"
printf -v Q_EXPECTED_PYTHON_REALPATH '%q' "$RUNTIME_PYTHON_REALPATH"
printf -v Q_EXPECTED_WAN_VAE_SHA256 '%q' "$OFFICIAL_WAN22_VAE_SHA256"
printf -v Q_LATENT_CACHE_RUN_ID '%q' "$LATENT_CACHE_RUN_ID"
printf -v Q_BUILD_WRAPPER '%q' "$BUILD_WRAPPER"

CACHE_COMMAND="$(
  printf '%s\n' \
    "set -Eeuo pipefail" \
    "unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE" \
    "cd $Q_COSMOS_SOURCE" \
    "export COSMOS_SOURCE=$Q_COSMOS_SOURCE" \
    "export LATENT_CACHE_RUN_ID=$Q_LATENT_CACHE_RUN_ID" \
    "export FORMAL_EXPERIMENT_ID=$Q_LATENT_CACHE_RUN_ID" \
    "export FORMAL_RUN_NAME=$Q_LATENT_CACHE_RUN_ID" \
    "export COSMOS_ENV=$Q_COSMOS_ENV" \
    "export ROBOCASA365_ROOT=$Q_ROBOCASA365_ROOT" \
    "export ROBOCASA365_SOURCE_ROOT=$Q_ROBOCASA365_SOURCE_ROOT" \
    "export ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT=$Q_ROBOCASA365_OUTPUT_ROOT" \
    "export WAN_VAE_PATH=$Q_WAN_VAE_PATH" \
    "export ROBOCASA365_NVME_SOURCE_ROOT=$Q_NVME_SOURCE_ROOT" \
    "export ROBOCASA365_CACHE_SPLITS=$Q_CACHE_SPLITS" \
    "export NUM_GPUS=$Q_NUM_GPUS" \
    "export LOADER_WORKERS=$Q_LOADER_WORKERS" \
    "export VAE_BATCH_SIZE=$Q_VAE_BATCH_SIZE" \
    "export EQUIVALENCE_SAMPLES_PER_RANK=$Q_EQUIVALENCE_SAMPLES" \
    "export EXPECTED_SOURCE_COMMIT=$Q_EXPECTED_SOURCE_COMMIT" \
    "export EXPECTED_SNAPSHOT_MANIFEST_SHA256=$Q_EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "export EXPECTED_BUILD_WRAPPER_SHA256=$Q_EXPECTED_BUILD_WRAPPER_SHA256" \
    "export EXPECTED_BUILD_PYTHON_SHA256=$Q_EXPECTED_BUILD_PYTHON_SHA256" \
    "export EXPECTED_PROVENANCE_HELPER_SHA256=$Q_EXPECTED_PROVENANCE_HELPER_SHA256" \
    "export EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256=$Q_EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256" \
    "export EXPECTED_ENV_REALPATH=$Q_EXPECTED_ENV_REALPATH" \
    "export EXPECTED_ENV_MARKER_SHA256=$Q_EXPECTED_ENV_MARKER_SHA256" \
    "export EXPECTED_PYTHON_REALPATH=$Q_EXPECTED_PYTHON_REALPATH" \
    "export EXPECTED_WAN_VAE_SHA256=$Q_EXPECTED_WAN_VAE_SHA256" \
    "exec $BASH_BIN $Q_BUILD_WRAPPER"
)"
CACHE_COMMAND_SHA256="$(printf '%s' "$CACHE_COMMAND" | "$SHA256SUM_BIN" | {
  read -r digest remainder
  printf '%s\n' "$digest"
})"

SUBMIT_ARGS=(
  submit
  --name "$RJOB_NAME"
  --group "$RJOB_QUOTA_GROUP"
  --charged-group "$RJOB_QUOTA_GROUP"
  --private-machine group
  --priority 9
  --task-type normal
  --replicas "$RJOB_REPLICAS"
  --restart-policy never
  --termination-grace-period-seconds 60
  --backoff_limit 1
  --image "$RJOB_IMAGE"
  --cpu "$RJOB_CPU"
  --gpu "$NPROC_REQUIRED"
  --memory "$RJOB_MEMORY_MIB"
  --negative-tags "$RJOB_EXCLUDED_NODE_TAG"
  --store-host-nvme true
  --mount
  "$PERSONAL_GPFS_MOUNT"
  "$PUBLIC_GPFS_MOUNT"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
  SUBMIT_ARGS+=(--predict-only true)
fi
WORKER_COMMAND=(
  "$ENV_BIN"
  -u BASH_ENV
  -u ENV
  "$BASH_BIN"
  --noprofile
  --norc
  -c
  "$CACHE_COMMAND"
)

echo "RJOB_NAME=$RJOB_NAME"
echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "RJOB_IMAGE=$RJOB_IMAGE"
echo "RJOB_QUOTA_GROUP=$RJOB_QUOTA_GROUP"
echo "RJOB_RESOURCES=single-node cpu=$RJOB_CPU memory_mib=$RJOB_MEMORY_MIB gpu=$NPROC_REQUIRED"
echo "RJOB_SCHEDULING=task_type=normal priority=9 restart_policy=never excluded_node=node0685"
echo "RJOB_STORE_HOST_NVME=true"
echo "ROBOCASA365_NVME_SOURCE_ROOT=$PINNED_NVME_SOURCE_ROOT"
echo "ROBOCASA365_CACHE_SPLITS=$ROBOCASA365_CACHE_SPLITS"
echo "LOADER_WORKERS=$LOADER_WORKERS"
echo "VAE_BATCH_SIZE=$VAE_BATCH_SIZE"
echo "SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
echo "SNAPSHOT_MANIFEST_SHA256=$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
echo "SUBMITTER_SHA256=$EXPECTED_SUBMITTER_SHA256"
echo "BUILD_WRAPPER_SHA256=$EXPECTED_BUILD_WRAPPER_SHA256"
echo "BUILD_PYTHON_SHA256=$EXPECTED_BUILD_PYTHON_SHA256"
echo "PROVENANCE_HELPER_SHA256=$EXPECTED_PROVENANCE_HELPER_SHA256"
echo "TORCHCODEC_RUNTIME_HELPER_SHA256=$EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256"
echo "SCHEDULER_CAPTURE_HELPER_SHA256=$EXPECTED_SCHEDULER_CAPTURE_HELPER_SHA256"
echo "RJOB_BIN_SHA256=$RJOB_BIN_SHA256"
echo "WAN_VAE_SHA256=$ACTUAL_WAN_VAE_SHA256"
echo "ENV_MARKER_SHA256=$EXPECTED_ENV_MARKER_SHA256"
echo "RUNTIME_PYTHON_REALPATH=$RUNTIME_PYTHON_REALPATH"
echo "CACHE_COMMAND_SHA256=$CACHE_COMMAND_SHA256"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
  "$RJOB_BIN" "${SUBMIT_ARGS[@]}" -- "${WORKER_COMMAND[@]}"
  echo "LATENT_CACHE_SUBMISSION_PREDICT_ONLY reservation_created=false"
  exit 0
fi

SUBMISSION_ROOT="$PINNED_SUBMISSION_OUTPUT_ROOT/submission-reservations"
/usr/bin/mkdir -p -- "$SUBMISSION_ROOT"
SUBMISSION_ROOT_REALPATH="$("$REALPATH_BIN" -e -- "$SUBMISSION_ROOT")"
[[ "$SUBMISSION_ROOT_REALPATH" == "$SUBMISSION_ROOT" && -d "$SUBMISSION_ROOT_REALPATH" ]] ||
  { echo "ERROR: submission root is not the exact canonical pinned path" >&2; exit 1; }
SUBMISSION_DIR="$SUBMISSION_ROOT_REALPATH/$LATENT_CACHE_RUN_ID"
if ! /usr/bin/mkdir --mode=0700 -- "$SUBMISSION_DIR"; then
  echo "ERROR: duplicate submission reservation exists: $SUBMISSION_DIR" >&2
  exit 1
fi
SUBMISSION_PENDING="$SUBMISSION_DIR/submission-pending.json"
SUBMISSION_LOG="$SUBMISSION_DIR/rjob-submit.log"
SCHEDULER_READBACK="$SUBMISSION_DIR/rjob-get.log"
SCHEDULER_EXPECTED_CONTRACT="$SUBMISSION_DIR/scheduler-expected-contract.json"
SCHEDULER_CAPTURE_LOG="$SUBMISSION_DIR/scheduler-capture.log"
SCHEDULER_STRUCTURED_RECEIPT="$SUBMISSION_DIR/scheduler-structured-receipt.json"
SCHEDULER_READBACK_RECEIPT="$SUBMISSION_DIR/scheduler-readback-receipt.json"
SUBMISSION_UNKNOWN="$SUBMISSION_DIR/submission-unknown.json"

"$PYTHON3_BIN" - "$SUBMISSION_PENDING" \
  "$LATENT_CACHE_RUN_ID" \
  "$RJOB_NAME" \
  "$RJOB_IMAGE" \
  "$EXPECTED_SOURCE_COMMIT" \
  "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
  "$EXPECTED_SUBMITTER_SHA256" \
  "$EXPECTED_BUILD_WRAPPER_SHA256" \
  "$EXPECTED_BUILD_PYTHON_SHA256" \
  "$EXPECTED_PROVENANCE_HELPER_SHA256" \
  "$EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_SCHEDULER_CAPTURE_HELPER_SHA256" \
  "$CACHE_COMMAND_SHA256" \
  "$ROBOCASA365_ROOT_REALPATH" \
  "$ROBOCASA365_SOURCE_ROOT_REALPATH" \
  "$ROBOCASA365_OUTPUT_REALPATH" \
  "$WAN_VAE_REALPATH" \
  "$ACTUAL_WAN_VAE_SHA256" \
  "$COSMOS_ENV_REALPATH" \
  "$EXPECTED_ENV_MARKER_SHA256" \
  "$RUNTIME_PYTHON_REALPATH" \
  "$LOADER_WORKERS" \
  "$VAE_BATCH_SIZE" \
  "$EQUIVALENCE_SAMPLES_PER_RANK" \
  "$RJOB_BIN" \
  "$RJOB_BIN_SHA256" <<'PY'
import datetime
import json
import os
import pathlib
import sys

(
    target,
    run_id,
    showname,
    image,
    source_commit,
    snapshot_manifest_sha256,
    submitter_sha256,
    wrapper_sha256,
    builder_sha256,
    provenance_helper_sha256,
    torchcodec_runtime_helper_sha256,
    scheduler_capture_helper_sha256,
    command_sha256,
    dataset_root,
    source_root,
    output_root,
    wan_vae_path,
    wan_vae_sha256,
    environment,
    environment_marker_sha256,
    python_realpath,
    loader_workers,
    vae_batch_size,
    equivalence_samples,
    rjob_realpath,
    rjob_sha256,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "kind": "robocasa365_wan_latent_cache_submission_reservation",
    "status": "submitting",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run_id": run_id,
    "scheduler_expected": {
        "showname": showname,
        "quota_group": "eailabagent_gpu",
        "private_machine": "group",
        "task_type": "normal",
        "priority": 9,
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "backoff_limit": 1,
        "replicas": 1,
        "image": image,
        "cpu": 128,
        "memory_mib": 1500000,
        "gpu": 8,
        "negative_tags": ["node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"],
        "store_host_nvme": True,
        "mounts": [
            "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
            "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
        ],
    },
    "source": {
        "commit": source_commit,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "submitter_sha256": submitter_sha256,
        "build_wrapper_sha256": wrapper_sha256,
        "build_python_sha256": builder_sha256,
        "provenance_helper_sha256": provenance_helper_sha256,
        "torchcodec_runtime_helper_sha256": torchcodec_runtime_helper_sha256,
        "scheduler_capture_helper_sha256": scheduler_capture_helper_sha256,
    },
    "runtime": {
        "environment": environment,
        "environment_marker_sha256": environment_marker_sha256,
        "python_realpath": python_realpath,
        "command_sha256": command_sha256,
        "rjob_realpath": rjob_realpath,
        "rjob_sha256": rjob_sha256,
        "worker_command_prefix": [
            "/usr/bin/env",
            "-u",
            "BASH_ENV",
            "-u",
            "ENV",
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
        ],
    },
    "cache": {
        "dataset_root": dataset_root,
        "source_root": source_root,
        "output_root": output_root,
        "wan_vae_path": wan_vae_path,
        "wan_vae_sha256": wan_vae_sha256,
        "nvme_source_root": "/nvme/cwam-robocasa365-d5eac857",
        "splits": ["train", "val", "test"],
        "loader_workers": int(loader_workers),
        "vae_batch_size": int(vae_batch_size),
        "equivalence_samples_per_rank": int(equivalence_samples),
    },
}
path = pathlib.Path(target)
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
"$CHMOD_BIN" 0444 "$SUBMISSION_PENDING"

mark_submission_unknown() {
  local reason="$1"
  local scheduler_name="${2:-}"
  local writer_status
  set +e
  "$PYTHON3_BIN" - \
    "$SUBMISSION_PENDING" \
    "$SUBMISSION_UNKNOWN" \
    "$reason" \
    "$scheduler_name" \
    "$SUBMISSION_LOG" \
    "$SCHEDULER_EXPECTED_CONTRACT" \
    "$SCHEDULER_CAPTURE_LOG" \
    "$SCHEDULER_STRUCTURED_RECEIPT" \
    "$SCHEDULER_READBACK" \
    "$SCHEDULER_READBACK_RECEIPT" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys

pending = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
payload = json.loads(pending.read_text())
payload.update(
    {
        "status": "unknown",
        "unknown_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": sys.argv[3],
        "scheduler_rjob_name": sys.argv[4] or None,
    }
)
artifacts = {}
for literal in sys.argv[5:]:
    path = pathlib.Path(literal)
    if path.is_file() and not path.is_symlink():
        artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
payload["available_artifact_sha256"] = artifacts
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  writer_status=$?
  for artifact in \
    "$SUBMISSION_PENDING" \
    "$SUBMISSION_LOG" \
    "$SCHEDULER_EXPECTED_CONTRACT" \
    "$SCHEDULER_CAPTURE_LOG" \
    "$SCHEDULER_STRUCTURED_RECEIPT" \
    "$SCHEDULER_READBACK" \
    "$SCHEDULER_READBACK_RECEIPT" \
    "$SUBMISSION_UNKNOWN"; do
    if [[ -f "$artifact" && ! -L "$artifact" ]]; then
      "$CHMOD_BIN" 0444 "$artifact" >/dev/null 2>&1 || true
    fi
  done
  "$CHMOD_BIN" 0555 "$SUBMISSION_DIR" >/dev/null 2>&1 || true
  set -e
  if [[ "$writer_status" -ne 0 ]]; then
    echo "ERROR: failed to seal submission-unknown receipt in $SUBMISSION_DIR" >&2
  fi
}

set +e
"$TIMEOUT_BIN" --foreground --kill-after=10s 180s \
  "$RJOB_BIN" "${SUBMIT_ARGS[@]}" -- "${WORKER_COMMAND[@]}" 2>&1 |
  "$TEE_BIN" "$SUBMISSION_LOG"
SUBMIT_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
if [[ "${SUBMIT_PIPE_STATUS[0]}" -ne 0 || "${SUBMIT_PIPE_STATUS[1]}" -ne 0 ]]; then
  mark_submission_unknown "rjob_submit_failed_or_ambiguous"
  echo "ERROR: rjob submission failed or became ambiguous; reservation retained at $SUBMISSION_DIR" >&2
  exit 1
fi
"$CHMOD_BIN" 0444 "$SUBMISSION_LOG"

set +e
SCHEDULER_RJOB_NAME="$(
  "$PYTHON3_BIN" -c '
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
matches = re.findall(r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)", text)
if len(matches) != 1 or not matches[0].startswith(sys.argv[2] + "-"):
    raise SystemExit(f"cannot bind exactly one scheduler rjob name: {matches!r}")
print(matches[0])
' "$SUBMISSION_LOG" "$RJOB_NAME"
)"
SCHEDULER_NAME_PARSE_STATUS=$?
set -e
if [[ "$SCHEDULER_NAME_PARSE_STATUS" -ne 0 || -z "$SCHEDULER_RJOB_NAME" ]]; then
  mark_submission_unknown "accepted_cli_result_without_unique_scheduler_name"
  echo "ERROR: cannot bind the accepted submission to exactly one scheduler RJob" >&2
  exit 1
fi

set +e
"$PYTHON3_BIN" - \
  "$SUBMISSION_PENDING" \
  "$SCHEDULER_EXPECTED_CONTRACT" \
  "$SCHEDULER_RJOB_NAME" \
  "$COSMOS_SOURCE_REALPATH" \
  "$BUILD_WRAPPER" \
  "$LATENT_CACHE_RUN_ID" <<'PY'
import json
import os
import pathlib
import sys

pending = json.loads(pathlib.Path(sys.argv[1]).read_text())
scheduler = pending["scheduler_expected"]
payload = {
    "schema_version": 1,
    "scheduler_rjob_name": sys.argv[3],
    "showname": scheduler["showname"],
    "quota_group": scheduler["quota_group"],
    "task_type": scheduler["task_type"],
    "priority": scheduler["priority"],
    "backoff_limit": scheduler["backoff_limit"],
    "task_name": "t0",
    "replicas": scheduler["replicas"],
    "image": scheduler["image"],
    "cpu": scheduler["cpu"],
    "memory_mib": scheduler["memory_mib"],
    "gpu": scheduler["gpu"],
    "negative_tags": scheduler["negative_tags"],
    "private_machine": scheduler["private_machine"],
    "restart_policy": scheduler["restart_policy"],
    "termination_grace_period_seconds": scheduler["termination_grace_period_seconds"],
    "mounts": scheduler["mounts"],
    "snapshot_root": sys.argv[4],
    "wrapper_path": sys.argv[5],
    "experiment_id": sys.argv[6],
    "command_sha256": pending["runtime"]["command_sha256"],
}
target = pathlib.Path(sys.argv[2])
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
SCHEDULER_EXPECTED_STATUS=$?
set -e
if [[ "$SCHEDULER_EXPECTED_STATUS" -ne 0 || ! -s "$SCHEDULER_EXPECTED_CONTRACT" ]]; then
  mark_submission_unknown "accepted_cli_result_without_expected_scheduler_contract" "$SCHEDULER_RJOB_NAME"
  echo "ERROR: failed to publish the expected scheduler contract" >&2
  exit 1
fi
"$CHMOD_BIN" 0444 "$SCHEDULER_EXPECTED_CONTRACT"

# Strictly validate the authoritative RJob and Replica CRDs. The existing
# authenticated helper owns all scheduler/resource normalization; this adapter
# replaces only its formal-training command grammar with this cache worker's
# clean-env command grammar.
set +e
"$TIMEOUT_BIN" --foreground --kill-after=10s "${SCHEDULER_RECEIPT_TIMEOUT_SECONDS}s" \
  "$PYTHON3_BIN" - \
    "$SCHEDULER_CAPTURE_HELPER" \
    "$SCHEDULER_RJOB_NAME" \
    "$SCHEDULER_EXPECTED_CONTRACT" \
    "$SCHEDULER_STRUCTURED_RECEIPT" \
    "$SCHEDULER_RECEIPT_ATTEMPTS" \
    "$SCHEDULER_RECEIPT_RETRY_SECONDS" >"$SCHEDULER_CAPTURE_LOG" 2>&1 <<'PY'
import hashlib
import importlib.util
import pathlib
import shlex
import sys

helper_path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("cwam_cache_scheduler_receipt", helper_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load authenticated scheduler helper: {helper_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

module._FORMAL_WRAPPER_RELATIVE_PATH = pathlib.PurePosixPath(
    "examples/h_cluster/build_robocasa365_wan_latent_cache.sh"
)
module._RECEIPT_KIND = "robocasa365_wan_latent_cache_rjob_scheduler_receipt"
host_nvme_annotation = "kubebrain.brainpp.cn/writeread-hostpath-mounts"
host_nvme_contract = "/nvme:/mnt/nvme1n1,/cfs:/mnt/cfs"
clean_prefix = [
    "/usr/bin/env",
    "-u",
    "BASH_ENV",
    "-u",
    "ENV",
    "/usr/bin/bash",
    "--noprofile",
    "--norc",
    "-c",
]


def validate_cache_command(command_value, expected):
    command = module._require_list(command_value, "scheduler worker command")
    if (
        len(command) != len(clean_prefix) + 1
        or command[: len(clean_prefix)] != clean_prefix
        or not isinstance(command[-1], str)
    ):
        raise module.ReceiptValidationError(
            f"scheduler worker command is not the audited clean-env command: {command!r}"
        )
    shell_script = command[-1]
    actual_sha256 = hashlib.sha256(shell_script.encode()).hexdigest()
    module._require_exact(actual_sha256, expected["command_sha256"], "scheduler command SHA-256")
    exports = module._parse_exports(shell_script)
    module._require_one_export(exports, "COSMOS_SOURCE", expected["snapshot_root"])
    module._require_one_export(exports, "LATENT_CACHE_RUN_ID", expected["experiment_id"])
    module._require_one_export(exports, "FORMAL_EXPERIMENT_ID", expected["experiment_id"])
    module._require_one_export(exports, "FORMAL_RUN_NAME", expected["experiment_id"])
    wrapper_invocations = []
    for line_number, line in enumerate(shell_script.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as error:
            raise module.ReceiptValidationError(
                f"scheduler command line {line_number} is not valid shell syntax"
            ) from error
        if tokens == ["exec", "/usr/bin/bash", expected["wrapper_path"]]:
            wrapper_invocations.append(tokens)
    if len(wrapper_invocations) != 1:
        raise module.ReceiptValidationError(
            "scheduler command must exec the exact immutable cache wrapper exactly once"
        )
    return [str(item) for item in command]


module._validate_command = validate_cache_command
original_validate_receipt = module.validate_and_normalize_receipt


def validate_cache_receipt(raw_rjob, raw_replicas, expected_value, *, captured_at=None):
    rjob = module._require_mapping(raw_rjob, "rjob")
    spec_value = module._require_mapping(rjob.get("spec"), "rjob.spec")
    task_specs = module._require_mapping(spec_value.get("taskSpecs"), "rjob.spec.taskSpecs")
    task = module._require_mapping(task_specs.get("t0"), "rjob.spec.taskSpecs.t0")
    template = module._require_mapping(task.get("template"), "rjob.spec.taskSpecs.t0.template")
    metadata = module._require_mapping(
        template.get("metadata"),
        "rjob.spec.taskSpecs.t0.template.metadata",
    )
    annotations = module._require_mapping(
        metadata.get("annotations"),
        "rjob.spec.taskSpecs.t0.template.metadata.annotations",
    )
    module._require_exact(
        annotations.get(host_nvme_annotation),
        host_nvme_contract,
        host_nvme_annotation,
    )
    receipt = original_validate_receipt(
        raw_rjob,
        raw_replicas,
        expected_value,
        captured_at=captured_at,
    )
    receipt["spec"]["task"]["store_host_nvme"] = {
        "annotation": host_nvme_annotation,
        "value": host_nvme_contract,
        "container_target": "/nvme",
        "host_source": "/mnt/nvme1n1",
    }
    return receipt


module.validate_and_normalize_receipt = validate_cache_receipt
receipt, digest = module.run_capture(
    scheduler_rjob_name=sys.argv[2],
    expected_json=pathlib.Path(sys.argv[3]),
    output=pathlib.Path(sys.argv[4]),
    attempts=int(sys.argv[5]),
    retry_seconds=float(sys.argv[6]),
)
print(
    "SCHEDULER_STRUCTURED_RECEIPT_VALID",
    f"rjob={receipt['metadata']['name']}",
    f"sha256={digest}",
)
PY
SCHEDULER_CAPTURE_STATUS=$?
set -e
"$CHMOD_BIN" 0444 "$SCHEDULER_CAPTURE_LOG" 2>/dev/null || true
if [[ "$SCHEDULER_CAPTURE_STATUS" -ne 0 || ! -s "$SCHEDULER_STRUCTURED_RECEIPT" ]]; then
  mark_submission_unknown "accepted_cli_result_without_validated_scheduler_contract" "$SCHEDULER_RJOB_NAME"
  echo "ERROR: scheduler CRD validation failed; reservation retained at $SUBMISSION_DIR" >&2
  exit 1
fi

READBACK_OK=0
READBACK_CONTENT=""
for _ in {1..10}; do
  set +e
  READBACK_CONTENT="$(
    "$TIMEOUT_BIN" --foreground --kill-after=5s 20s \
      "$RJOB_BIN" get "$SCHEDULER_RJOB_NAME" 2>&1
  )"
  READBACK_STATUS=$?
  set -e
  if [[ "$READBACK_STATUS" -eq 0 &&
        "$READBACK_CONTENT" == *"rjob $SCHEDULER_RJOB_NAME (showname=$RJOB_NAME):"* ]]; then
    printf '%s\n' "$READBACK_CONTENT" >"$SCHEDULER_READBACK"
    READBACK_OK=1
    break
  fi
  "$SLEEP_BIN" 1
done
if [[ "$READBACK_OK" != 1 ]]; then
  printf '%s\n' "$READBACK_CONTENT" >"$SCHEDULER_READBACK"
  mark_submission_unknown "accepted_scheduler_contract_without_text_readback" "$SCHEDULER_RJOB_NAME"
  echo "ERROR: accepted rjob has no scheduler readback; reservation retained at $SUBMISSION_DIR" >&2
  exit 1
fi
"$CHMOD_BIN" 0444 "$SCHEDULER_READBACK"

set +e
"$PYTHON3_BIN" - \
  "$SUBMISSION_PENDING" \
  "$SCHEDULER_READBACK_RECEIPT" \
  "$SCHEDULER_RJOB_NAME" \
  "$SUBMISSION_LOG" \
  "$SCHEDULER_READBACK" \
  "$SCHEDULER_EXPECTED_CONTRACT" \
  "$SCHEDULER_CAPTURE_LOG" \
  "$SCHEDULER_STRUCTURED_RECEIPT" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys

pending_path = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
submission_log = pathlib.Path(sys.argv[4])
readback = pathlib.Path(sys.argv[5])
expected_contract = pathlib.Path(sys.argv[6])
capture_log = pathlib.Path(sys.argv[7])
structured_receipt = pathlib.Path(sys.argv[8])
pending = json.loads(pending_path.read_text())
payload = {
    "schema_version": 1,
    "kind": "robocasa365_wan_latent_cache_scheduler_readback_receipt",
    "status": "accepted",
    "accepted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run_id": pending["run_id"],
    "scheduler_rjob_name": sys.argv[3],
    "scheduler_expected": pending["scheduler_expected"],
    "source": pending["source"],
    "runtime": pending["runtime"],
    "cache": pending["cache"],
    "submission_reservation": str(pending_path),
    "submission_reservation_sha256": hashlib.sha256(pending_path.read_bytes()).hexdigest(),
    "submission_log": str(submission_log),
    "submission_log_sha256": hashlib.sha256(submission_log.read_bytes()).hexdigest(),
    "scheduler_readback": str(readback),
    "scheduler_readback_sha256": hashlib.sha256(readback.read_bytes()).hexdigest(),
    "scheduler_expected_contract": str(expected_contract),
    "scheduler_expected_contract_sha256": hashlib.sha256(expected_contract.read_bytes()).hexdigest(),
    "scheduler_capture_log": str(capture_log),
    "scheduler_capture_log_sha256": hashlib.sha256(capture_log.read_bytes()).hexdigest(),
    "scheduler_structured_receipt": str(structured_receipt),
    "scheduler_structured_receipt_sha256": hashlib.sha256(structured_receipt.read_bytes()).hexdigest(),
}
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
FINAL_RECEIPT_STATUS=$?
set -e
if [[ "$FINAL_RECEIPT_STATUS" -ne 0 || ! -s "$SCHEDULER_READBACK_RECEIPT" ]]; then
  mark_submission_unknown "validated_scheduler_without_final_composite_receipt" "$SCHEDULER_RJOB_NAME"
  echo "ERROR: failed to publish the final scheduler readback receipt" >&2
  exit 1
fi
"$CHMOD_BIN" 0444 \
  "$SUBMISSION_PENDING" \
  "$SUBMISSION_LOG" \
  "$SCHEDULER_EXPECTED_CONTRACT" \
  "$SCHEDULER_CAPTURE_LOG" \
  "$SCHEDULER_STRUCTURED_RECEIPT" \
  "$SCHEDULER_READBACK" \
  "$SCHEDULER_READBACK_RECEIPT"
"$CHMOD_BIN" 0555 "$SUBMISSION_DIR"
echo "LATENT_CACHE_SUBMISSION_ACCEPTED scheduler_rjob=$SCHEDULER_RJOB_NAME receipt=$SCHEDULER_READBACK_RECEIPT"
