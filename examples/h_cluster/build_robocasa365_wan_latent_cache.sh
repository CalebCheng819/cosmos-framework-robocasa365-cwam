#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Run this *inside* an allocated H job. When NVMe staging is enabled, the rjob
# submission must include the platform's bare `--store-host-nvme` switch.
# The live help renders a value metavar, but the executable parser rejects a
# trailing `true`; the prediction-only gate verifies the bare-switch contract.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly OFFICIAL_WAN22_VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
readonly PINNED_NVME_SOURCE_ROOT="/nvme/cwam-robocasa365-d5eac857"
readonly SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
readonly CACHE_OUTPUT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data"
readonly REALPATH_BIN="/usr/bin/realpath"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly BASH_BIN="/usr/bin/bash"
readonly FINDMNT_BIN="/usr/bin/findmnt"
readonly H_NVIDIA_DRIVER_BIN="/usr/local/nvidia/bin"
readonly H_NVIDIA_DRIVER_LIB="/usr/local/nvidia/lib"
readonly H_NVIDIA_DRIVER_LIB64="/usr/local/nvidia/lib64"
# Capture worker-init's value before sanitizing the process environment.  The H
# runtime currently injects these allowlisted entries even when one of the
# conventional directories is not materialized in the container namespace.
readonly H_WORKER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable Cosmos source snapshot.}"
: "${COSMOS_ENV:?Set COSMOS_ENV to the exact runtime environment.}"
: "${ROBOCASA365_ROOT:?Set the converted RoboCasa365 dataset root.}"
: "${ROBOCASA365_SOURCE_ROOT:?Set the original raw RoboCasa365 source root.}"
: "${ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT:?Set the final shared cache root.}"
: "${WAN_VAE_PATH:?Set the exact Wan2.2 VAE weights file.}"
: "${ROBOCASA365_NVME_SOURCE_ROOT:?Set the fixed host-NVMe source staging root.}"
: "${EXPECTED_SOURCE_COMMIT:?Set the immutable snapshot source commit.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set the immutable snapshot manifest SHA-256.}"
: "${EXPECTED_BUILD_WRAPPER_SHA256:?Set the audited build wrapper SHA-256.}"
: "${EXPECTED_BUILD_PYTHON_SHA256:?Set the audited Python builder SHA-256.}"
: "${EXPECTED_PROVENANCE_HELPER_SHA256:?Set the audited provenance helper SHA-256.}"
: "${EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256:?Set the audited TorchCodec runtime helper SHA-256.}"
: "${EXPECTED_ENV_REALPATH:?Set the audited runtime environment realpath.}"
: "${EXPECTED_ENV_MARKER_SHA256:?Set the audited environment readiness marker SHA-256.}"
: "${EXPECTED_PYTHON_REALPATH:?Set the audited runtime Python realpath.}"
: "${EXPECTED_WAN_VAE_SHA256:?Set the audited Wan2.2 VAE SHA-256.}"

NUM_GPUS="${NUM_GPUS:-8}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-8}"
EQUIVALENCE_SAMPLES_PER_RANK="${EQUIVALENCE_SAMPLES_PER_RANK:-8}"
ROBOCASA365_CACHE_SPLITS="${ROBOCASA365_CACHE_SPLITS:-train,val,test}"

[[ "$#" -eq 0 ]] || {
  echo "ERROR: unbound command-line arguments are forbidden; use the audited environment contract" >&2
  exit 1
}
[[ "$NUM_GPUS" == 8 ]] ||
  { echo "ERROR: NUM_GPUS must be exactly 8" >&2; exit 1; }
[[ "$LOADER_WORKERS" =~ ^[0-9]+$ ]] ||
  { echo "ERROR: LOADER_WORKERS must be a non-negative integer" >&2; exit 1; }
[[ "$VAE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
  { echo "ERROR: VAE_BATCH_SIZE must be a positive integer" >&2; exit 1; }
[[ "$EQUIVALENCE_SAMPLES_PER_RANK" =~ ^[1-9][0-9]*$ ]] ||
  { echo "ERROR: EQUIVALENCE_SAMPLES_PER_RANK must be a positive integer" >&2; exit 1; }
[[ "$ROBOCASA365_CACHE_SPLITS" == "train,val,test" ]] ||
  { echo "ERROR: the synchronized full cache requires splits=train,val,test" >&2; exit 1; }
[[ "$ROBOCASA365_NVME_SOURCE_ROOT" == "$PINNED_NVME_SOURCE_ROOT" ]] ||
  { echo "ERROR: ROBOCASA365_NVME_SOURCE_ROOT differs from the audited /nvme destination" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase 40-character Git SHA" >&2; exit 1; }
for expected_sha in \
  "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
  "$EXPECTED_BUILD_WRAPPER_SHA256" \
  "$EXPECTED_BUILD_PYTHON_SHA256" \
  "$EXPECTED_PROVENANCE_HELPER_SHA256" \
  "$EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256" \
  "$EXPECTED_ENV_MARKER_SHA256" \
  "$EXPECTED_WAN_VAE_SHA256"; do
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: an expected SHA-256 is malformed: $expected_sha" >&2; exit 1; }
done
[[ "$EXPECTED_WAN_VAE_SHA256" == "$OFFICIAL_WAN22_VAE_SHA256" ]] ||
  { echo "ERROR: EXPECTED_WAN_VAE_SHA256 is not the official audited Wan2.2 VAE digest" >&2; exit 1; }

for required_tool in "$REALPATH_BIN" "$SHA256SUM_BIN" "$BASH_BIN" "$FINDMNT_BIN"; do
  [[ -x "$required_tool" ]] ||
    { echo "ERROR: required absolute tool is missing or not executable: $required_tool" >&2; exit 1; }
done

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
  { echo "ERROR: wrapper must run with the audited Bash interpreter: $BASH_BIN" >&2; exit 1; }

COSMOS_SOURCE_REALPATH="$("$REALPATH_BIN" -e -- "$COSMOS_SOURCE")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REALPATH" && -d "$COSMOS_SOURCE_REALPATH" ]] ||
  { echo "ERROR: COSMOS_SOURCE must be an exact canonical directory" >&2; exit 1; }
BUILD_WRAPPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/build_robocasa365_wan_latent_cache.sh"
BUILD_PYTHON="$COSMOS_SOURCE_REALPATH/examples/h_cluster/build_robocasa365_wan_latent_cache.py"
PROVENANCE_HELPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/cwam_formal_provenance.py"
TORCHCODEC_RUNTIME_HELPER="$COSMOS_SOURCE_REALPATH/examples/h_cluster/configure_torchcodec_runtime.sh"
SNAPSHOT_MANIFEST="$COSMOS_SOURCE_REALPATH/SNAPSHOT_MANIFEST.json"
THIS_WRAPPER_REALPATH="$("$REALPATH_BIN" -e -- "${BASH_SOURCE[0]}")"
[[ "$THIS_WRAPPER_REALPATH" == "$BUILD_WRAPPER" ]] ||
  { echo "ERROR: wrapper must execute from the exact immutable COSMOS_SOURCE snapshot" >&2; exit 1; }
for required_file in \
  "$BUILD_WRAPPER" \
  "$BUILD_PYTHON" \
  "$PROVENANCE_HELPER" \
  "$TORCHCODEC_RUNTIME_HELPER" \
  "$SNAPSHOT_MANIFEST"; do
  [[ -s "$required_file" && ! -L "$required_file" ]] ||
    { echo "ERROR: required immutable snapshot file is missing, empty, or a symlink: $required_file" >&2; exit 1; }
done
[[ "$(sha256_file "$BUILD_WRAPPER")" == "$EXPECTED_BUILD_WRAPPER_SHA256" ]] ||
  { echo "ERROR: build wrapper SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256_file "$BUILD_PYTHON")" == "$EXPECTED_BUILD_PYTHON_SHA256" ]] ||
  { echo "ERROR: Python builder SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256_file "$PROVENANCE_HELPER")" == "$EXPECTED_PROVENANCE_HELPER_SHA256" ]] ||
  { echo "ERROR: provenance helper SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256_file "$TORCHCODEC_RUNTIME_HELPER")" == "$EXPECTED_TORCHCODEC_RUNTIME_HELPER_SHA256" ]] ||
  { echo "ERROR: TorchCodec runtime helper SHA-256 mismatch" >&2; exit 1; }

COSMOS_ENV_REALPATH="$("$REALPATH_BIN" -e -- "$COSMOS_ENV")"
[[ "$COSMOS_ENV" == "$COSMOS_ENV_REALPATH" && "$COSMOS_ENV_REALPATH" == "$EXPECTED_ENV_REALPATH" ]] ||
  { echo "ERROR: COSMOS_ENV does not match the audited canonical environment" >&2; exit 1; }
ENV_MARKER="$COSMOS_ENV_REALPATH/.cwam_env_ready.json"
[[ -s "$ENV_MARKER" && ! -L "$ENV_MARKER" ]] ||
  { echo "ERROR: pinned environment readiness marker is missing or unsafe" >&2; exit 1; }
[[ "$(sha256_file "$ENV_MARKER")" == "$EXPECTED_ENV_MARKER_SHA256" ]] ||
  { echo "ERROR: environment readiness marker SHA-256 differs from the submission contract" >&2; exit 1; }
RUNTIME_PYTHON="$COSMOS_ENV_REALPATH/bin/python"
RUNTIME_PYTHON_REALPATH="$("$REALPATH_BIN" -e -- "$RUNTIME_PYTHON")"
[[ -x "$RUNTIME_PYTHON" && "$RUNTIME_PYTHON_REALPATH" == "$EXPECTED_PYTHON_REALPATH" ]] ||
  { echo "ERROR: runtime Python realpath differs from the submission contract" >&2; exit 1; }

export PATH="$COSMOS_ENV_REALPATH/bin:$H_NVIDIA_DRIVER_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$COSMOS_SOURCE_REALPATH"
# Do not inherit arbitrary shell library paths.  Validate worker-init's input
# against the two audited H driver locations, then rebuild a canonical value.
# Do not require either path to have a particular directory/file layout:
# NVIDIA container injection may expose the driver through the dynamic loader
# while still advertising these conventional paths.  The nvidia-smi, libcuda,
# Torch/CUDA, device-count, and H200 checks below are the authoritative
# fail-closed capability gates.
[[ -n "$H_WORKER_LD_LIBRARY_PATH" ]] ||
  { echo "ERROR: H worker-init did not provide LD_LIBRARY_PATH" >&2; exit 1; }
IFS=: read -r -a H_WORKER_DRIVER_LIBRARY_ENTRIES <<<"$H_WORKER_LD_LIBRARY_PATH"
[[ "${#H_WORKER_DRIVER_LIBRARY_ENTRIES[@]}" -gt 0 ]] ||
  { echo "ERROR: H worker-init LD_LIBRARY_PATH is empty" >&2; exit 1; }
for driver_library_entry in "${H_WORKER_DRIVER_LIBRARY_ENTRIES[@]}"; do
  case "$driver_library_entry" in
    "$H_NVIDIA_DRIVER_LIB"|"$H_NVIDIA_DRIVER_LIB64") ;;
    *)
      echo "ERROR: H worker-init LD_LIBRARY_PATH contains an unaudited entry" >&2
      exit 1
      ;;
  esac
done
export LD_LIBRARY_PATH="$H_NVIDIA_DRIVER_LIB:$H_NVIDIA_DRIVER_LIB64"
unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE
PATH_PYTHON="$(type -P python || true)"
[[ "$PATH_PYTHON" == "$RUNTIME_PYTHON" &&
   "$("$REALPATH_BIN" -e -- "$PATH_PYTHON")" == "$EXPECTED_PYTHON_REALPATH" ]] ||
  { echo "ERROR: sanitized PATH python does not resolve to the audited COSMOS_ENV interpreter" >&2; exit 1; }

"$RUNTIME_PYTHON" "$PROVENANCE_HELPER" verify-snapshot \
  --snapshot-root "$COSMOS_SOURCE_REALPATH" \
  --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
  --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null
"$RUNTIME_PYTHON" -c '
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("source_head") != sys.argv[2]:
    raise SystemExit(
        f"snapshot source_head mismatch: expected={sys.argv[2]!r}, actual={manifest.get('"'"'source_head'"'"')!r}"
    )
if manifest.get("source_dirty") is not False:
    raise SystemExit("snapshot source_dirty must be false")
' "$SNAPSHOT_MANIFEST" "$EXPECTED_SOURCE_COMMIT"

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
[[ "$WAN_VAE_PATH" == "$WAN_VAE_REALPATH" && -f "$WAN_VAE_REALPATH" ]] ||
  { echo "ERROR: WAN_VAE_PATH must be an exact canonical regular file" >&2; exit 1; }
ACTUAL_WAN_VAE_SHA256="$(sha256_file "$WAN_VAE_REALPATH")"
[[ "$ACTUAL_WAN_VAE_SHA256" == "$OFFICIAL_WAN22_VAE_SHA256" ]] ||
  {
    echo "ERROR: official Wan2.2 VAE runtime SHA-256 mismatch: expected=$OFFICIAL_WAN22_VAE_SHA256 actual=$ACTUAL_WAN_VAE_SHA256" >&2
    exit 1
  }

NVME_MOUNT_ROOT="$("$REALPATH_BIN" -e -- /nvme)"
[[ "$NVME_MOUNT_ROOT" == /nvme && -d "$NVME_MOUNT_ROOT" && -w "$NVME_MOUNT_ROOT" ]] ||
  { echo "ERROR: --store-host-nvme must expose writable /nvme" >&2; exit 1; }
NVME_MOUNT_TARGET="$("$FINDMNT_BIN" -n -T "$NVME_MOUNT_ROOT" -o TARGET)"
NVME_MOUNT_SOURCE="$("$FINDMNT_BIN" -n -T "$NVME_MOUNT_ROOT" -o SOURCE)"
NVME_MOUNT_FSTYPE="$("$FINDMNT_BIN" -n -T "$NVME_MOUNT_ROOT" -o FSTYPE)"
NVME_MOUNT_TARGET="$("$REALPATH_BIN" -e -- "$NVME_MOUNT_TARGET")"
[[ "$NVME_MOUNT_TARGET" == /nvme ]] ||
  { echo "ERROR: H host-NVMe mount target must resolve exactly to /nvme" >&2; exit 1; }
[[ "$NVME_MOUNT_SOURCE" == /dev/nvme1n1 ]] ||
  { echo "ERROR: H host-NVMe mount source must be /dev/nvme1n1" >&2; exit 1; }
[[ "$NVME_MOUNT_FSTYPE" == xfs ]] ||
  { echo "ERROR: H host-NVMe filesystem must be xfs" >&2; exit 1; }

# This authenticated snapshot helper creates a process-local FFmpeg-soname
# directory and exports the exact LD_LIBRARY_PATH required by TorchCodec.
source "$TORCHCODEC_RUNTIME_HELPER"

NVIDIA_SMI_BIN="$(type -P nvidia-smi || true)"
[[ -n "$NVIDIA_SMI_BIN" && -x "$NVIDIA_SMI_BIN" ]] ||
  { echo "ERROR: nvidia-smi is unavailable on the sanitized PATH" >&2; exit 1; }
NVIDIA_SMI_REALPATH="$("$REALPATH_BIN" -e -- "$NVIDIA_SMI_BIN")"
echo "NVIDIA_SMI_READY realpath=$NVIDIA_SMI_REALPATH"
"$NVIDIA_SMI_BIN" -L
"$RUNTIME_PYTHON" - "$NUM_GPUS" <<'PY'
import ctypes
import os
import sys

import torch

expected_device_count = int(sys.argv[1])
expected_torch_version = "2.10.0+cu128"
expected_torch_cuda = "12.8"
ctypes.CDLL("libcuda.so.1")
if torch.__version__ != expected_torch_version or torch.version.cuda != expected_torch_cuda:
    raise SystemExit(
        "CUDA runtime preflight failed: unexpected pinned Torch runtime "
        f"torch={torch.__version__!r}, torch_cuda={torch.version.cuda!r}"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA runtime preflight failed: torch.cuda.is_available() is false")
device_count = torch.cuda.device_count()
if device_count != expected_device_count:
    raise SystemExit(
        "CUDA runtime preflight failed: "
        f"expected {expected_device_count} visible devices, got {device_count}"
)
device_properties = [torch.cuda.get_device_properties(index) for index in range(device_count)]
device_names = [properties.name for properties in device_properties]
if any("H200" not in name for name in device_names):
    raise SystemExit(
        "CUDA runtime preflight failed: expected only H200 devices, "
        f"got {device_names!r}"
    )
print(
    "CUDA_RUNTIME_READY",
    f"torch={torch.__version__}",
    f"torch_cuda={torch.version.cuda}",
    f"device_count={device_count}",
    f"devices={device_names!r}",
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
)
PY

cd "$COSMOS_SOURCE_REALPATH"
exec "$RUNTIME_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NUM_GPUS}" \
  "$BUILD_PYTHON" \
  --dataset-root "${ROBOCASA365_ROOT_REALPATH}" \
  --source-root "${ROBOCASA365_SOURCE_ROOT_REALPATH}" \
  --output-root "${ROBOCASA365_OUTPUT_REALPATH}" \
  --wan-vae "${WAN_VAE_REALPATH}" \
  --nvme-source-root "${PINNED_NVME_SOURCE_ROOT}" \
  --splits "${ROBOCASA365_CACHE_SPLITS}" \
  --loader-workers "${LOADER_WORKERS}" \
  --vae-batch-size "${VAE_BATCH_SIZE}" \
  --equivalence-samples-per-rank "${EQUIVALENCE_SAMPLES_PER_RANK}"
