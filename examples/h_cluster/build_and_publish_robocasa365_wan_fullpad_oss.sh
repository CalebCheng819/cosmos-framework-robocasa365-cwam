#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PERSONAL_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly OUTPUT_ROOT="$PERSONAL_GPFS_ROOT/outputs/robocasa365-wan-fullpad-oss"
readonly NVME_ROOT="/nvme/cwam-robocasa365-wan-fullpad"
readonly REALPATH_BIN="/usr/bin/realpath"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly OPENSSL_BIN="/usr/bin/openssl"
readonly CHMOD_BIN="/usr/bin/chmod"
readonly RM_BIN="/usr/bin/rm"
readonly SLEEP_BIN="/usr/bin/sleep"
readonly TEE_BIN="/usr/bin/tee"
readonly TIMEOUT_SECONDS=3600

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${COSMOS_ENV:?Set COSMOS_ENV to the audited runtime environment.}"
: "${LATENT_CACHE_RUN_ID:?Set LATENT_CACHE_RUN_ID.}"
: "${ROBOCASA365_OSS_REMOTE:?Set the exact guarded HDD2 target prefix.}"
: "${ROBOCASA365_OSS_WORK_ROOT:?Set the run-specific personal-GPFS handoff root.}"
: "${RCLONE_BIN:?Set the pinned rclone binary.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_FULLPAD_RUNNER_SHA256:?Set EXPECTED_FULLPAD_RUNNER_SHA256.}"
: "${EXPECTED_OSS_PUBLISHER_SHA256:?Set EXPECTED_OSS_PUBLISHER_SHA256.}"
: "${EXPECTED_RCLONE_SHA256:?Set EXPECTED_RCLONE_SHA256.}"

[[ "$#" -eq 0 ]] || { echo "ERROR: command-line arguments are forbidden" >&2; exit 1; }
[[ "$LATENT_CACHE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
  { echo "ERROR: LATENT_CACHE_RUN_ID contains unsupported characters" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase Git SHA" >&2; exit 1; }
for digest in \
  "$EXPECTED_FULLPAD_RUNNER_SHA256" \
  "$EXPECTED_OSS_PUBLISHER_SHA256" \
  "$EXPECTED_RCLONE_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: one required SHA-256 is malformed" >&2; exit 1; }
done
[[ "$ROBOCASA365_OSS_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] ||
  { echo "ERROR: OSS target must stay below eailab/chengjuntao" >&2; exit 1; }

for tool in \
  "$REALPATH_BIN" \
  "$SHA256SUM_BIN" \
  "$OPENSSL_BIN" \
  "$CHMOD_BIN" \
  "$RM_BIN" \
  "$SLEEP_BIN" \
  "$TEE_BIN"; do
  [[ -x "$tool" ]] || { echo "ERROR: required tool is unavailable: $tool" >&2; exit 1; }
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

COSMOS_SOURCE_REAL="$($REALPATH_BIN -e -- "$COSMOS_SOURCE")"
COSMOS_ENV_REAL="$($REALPATH_BIN -e -- "$COSMOS_ENV")"
WORK_ROOT_REAL="$($REALPATH_BIN -e -- "$ROBOCASA365_OSS_WORK_ROOT")"
OUTPUT_ROOT_REAL="$($REALPATH_BIN -e -- "$OUTPUT_ROOT")"
RCLONE_REAL="$($REALPATH_BIN -e -- "$RCLONE_BIN")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REAL" && "$COSMOS_ENV" == "$COSMOS_ENV_REAL" ]] ||
  { echo "ERROR: source/environment paths must be canonical" >&2; exit 1; }
[[ "$ROBOCASA365_OSS_WORK_ROOT" == "$WORK_ROOT_REAL" ]] ||
  { echo "ERROR: handoff root must be canonical and pre-created" >&2; exit 1; }
path_is_within "$WORK_ROOT_REAL" "$OUTPUT_ROOT_REAL" ||
  { echo "ERROR: handoff root escapes the pinned personal-GPFS output root" >&2; exit 1; }
[[ "$WORK_ROOT_REAL" == "$OUTPUT_ROOT_REAL/$LATENT_CACHE_RUN_ID" ]] ||
  { echo "ERROR: handoff root must be the exact run-id directory" >&2; exit 1; }

FULLPAD_RUNNER="$COSMOS_SOURCE_REAL/examples/h_cluster/build_and_publish_robocasa365_wan_fullpad_oss.sh"
BUILD_WRAPPER="$COSMOS_SOURCE_REAL/examples/h_cluster/build_robocasa365_wan_latent_cache.sh"
OSS_PUBLISHER="$COSMOS_SOURCE_REAL/examples/h_cluster/publish_robocasa365_wan_cache_to_oss.py"
[[ "$($REALPATH_BIN -e -- "${BASH_SOURCE[0]}")" == "$FULLPAD_RUNNER" ]] ||
  { echo "ERROR: full-padding runner must execute from the immutable snapshot" >&2; exit 1; }
[[ "$(sha256_file "$FULLPAD_RUNNER")" == "$EXPECTED_FULLPAD_RUNNER_SHA256" ]] ||
  { echo "ERROR: full-padding runner SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256_file "$OSS_PUBLISHER")" == "$EXPECTED_OSS_PUBLISHER_SHA256" ]] ||
  { echo "ERROR: OSS publisher SHA-256 mismatch" >&2; exit 1; }
[[ "$RCLONE_BIN" == "$RCLONE_REAL" && "$(sha256_file "$RCLONE_REAL")" == "$EXPECTED_RCLONE_SHA256" ]] ||
  { echo "ERROR: pinned rclone path or SHA-256 mismatch" >&2; exit 1; }

/usr/bin/mkdir -p -- "$NVME_ROOT/$LATENT_CACHE_RUN_ID"
NVME_RUN_ROOT="$($REALPATH_BIN -e -- "$NVME_ROOT/$LATENT_CACHE_RUN_ID")"
[[ "$NVME_RUN_ROOT" == "$NVME_ROOT/$LATENT_CACHE_RUN_ID" ]] ||
  { echo "ERROR: NVMe run root is not canonical" >&2; exit 1; }
CACHE_ROOT="$NVME_RUN_ROOT/cache"

export ROBOCASA365_LATENT_CACHE_OUTPUT_ROOT="$CACHE_ROOT"
export ROBOCASA365_CACHE_CAMERA_MODE="triple_view"
export ROBOCASA365_CACHE_RESOLUTION="480"
export ROBOCASA365_CACHE_STAGE="full_padding"
export ROBOCASA365_CACHE_PREFLIGHT_ONLY="0"
unset ROBOCASA365_CACHE_PREFLIGHT_RECEIPT

NVIDIA_SMI_BIN="$(type -P nvidia-smi || true)"
[[ -n "$NVIDIA_SMI_BIN" && -x "$NVIDIA_SMI_BIN" ]] ||
  { echo "ERROR: nvidia-smi is unavailable for the required GPU monitor" >&2; exit 1; }
GPU_MONITOR_PARTIAL="$WORK_ROOT_REAL/gpu-monitor.csv.partial"
GPU_MONITOR_FINAL="$WORK_ROOT_REAL/gpu-monitor.csv"
BUILD_LOG_PARTIAL="$WORK_ROOT_REAL/build.log.partial"
BUILD_LOG_FINAL="$WORK_ROOT_REAL/build.log"
for path in "$GPU_MONITOR_PARTIAL" "$GPU_MONITOR_FINAL" "$BUILD_LOG_PARTIAL" "$BUILD_LOG_FINAL"; do
  [[ ! -e "$path" ]] || { echo "ERROR: refusing to replace an existing build artifact: $path" >&2; exit 1; }
done
"$NVIDIA_SMI_BIN" \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
  --format=csv -l 30 >"$GPU_MONITOR_PARTIAL" 2>&1 &
GPU_MONITOR_PID=$!
set +e
"/usr/bin/bash" "$BUILD_WRAPPER" 2>&1 | "$TEE_BIN" "$BUILD_LOG_PARTIAL"
BUILD_STATUS="${PIPESTATUS[0]}"
/usr/bin/kill "$GPU_MONITOR_PID" >/dev/null 2>&1
wait "$GPU_MONITOR_PID" >/dev/null 2>&1
set -e
/usr/bin/chmod 0444 "$GPU_MONITOR_PARTIAL"
/usr/bin/mv -- "$GPU_MONITOR_PARTIAL" "$GPU_MONITOR_FINAL"
/usr/bin/chmod 0444 "$BUILD_LOG_PARTIAL"
/usr/bin/mv -- "$BUILD_LOG_PARTIAL" "$BUILD_LOG_FINAL"
[[ "$BUILD_STATUS" -eq 0 ]] || { echo "ERROR: full-padding cache build failed" >&2; exit "$BUILD_STATUS"; }

[[ -s "$CACHE_ROOT/_SUCCESS.json" && -s "$CACHE_ROOT/manifest.json" ]] ||
  { echo "ERROR: full-padding builder returned without a published local cache" >&2; exit 1; }

SECRET_ROOT="/dev/shm/cwam-rclone-$LATENT_CACHE_RUN_ID-$$"
/usr/bin/mkdir --mode=0700 -- "$SECRET_ROOT"
PRIVATE_KEY="$SECRET_ROOT/private-key.pem"
CERTIFICATE="$SECRET_ROOT/certificate.pem"
DECRYPTED_CONFIG="$SECRET_ROOT/rclone.conf"
cleanup_secrets() {
  "$RM_BIN" -rf -- "$SECRET_ROOT"
}
trap cleanup_secrets EXIT INT TERM

"$OPENSSL_BIN" req -x509 -newkey rsa:3072 -sha256 -nodes -days 1 \
  -subj "/CN=$LATENT_CACHE_RUN_ID" \
  -keyout "$PRIVATE_KEY" \
  -out "$CERTIFICATE" >/dev/null 2>&1
"$CHMOD_BIN" 0400 "$PRIVATE_KEY"

PUBLIC_CERT="$WORK_ROOT_REAL/rclone-config-recipient.pem"
if [[ -e "$PUBLIC_CERT" ]]; then
  [[ "$(sha256_file "$PUBLIC_CERT")" == "$(sha256_file "$CERTIFICATE")" ]] ||
    { echo "ERROR: existing handoff certificate differs" >&2; exit 1; }
else
  /usr/bin/install -m 0444 -- "$CERTIFICATE" "$PUBLIC_CERT"
fi
HANDOFF_REQUEST="$WORK_ROOT_REAL/rclone-config-request.json"
"$COSMOS_ENV_REAL/bin/python" - \
  "$HANDOFF_REQUEST" \
  "$LATENT_CACHE_RUN_ID" \
  "$ROBOCASA365_OSS_REMOTE" \
  "$CACHE_ROOT" \
  "$PUBLIC_CERT" \
  "$(sha256_file "$PUBLIC_CERT")" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "rclone_config_encryption_request",
    "status": "waiting",
    "run_id": sys.argv[2],
    "remote": sys.argv[3],
    "cache_root": sys.argv[4],
    "recipient_certificate": sys.argv[5],
    "recipient_certificate_sha256": sys.argv[6],
    "envelope": "openssl-cms-enveloped-data-aes-256-cbc-der",
}
if target.exists():
    if json.loads(target.read_text()) != payload:
        raise SystemExit("existing rclone handoff request differs")
else:
    temporary = target.with_name("." + target.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, target)
PY

ENCRYPTED_CONFIG="$WORK_ROOT_REAL/rclone-config.cms.der"
ENCRYPTED_CONFIG_SHA="$WORK_ROOT_REAL/rclone-config.cms.der.sha256"
echo "RCLONE_CONFIG_HANDOFF_WAIT request=$HANDOFF_REQUEST timeout_seconds=$TIMEOUT_SECONDS"
for ((attempt = 0; attempt < TIMEOUT_SECONDS; attempt++)); do
  if [[ -s "$ENCRYPTED_CONFIG" && -s "$ENCRYPTED_CONFIG_SHA" ]]; then
    break
  fi
  "$SLEEP_BIN" 1
done
[[ -s "$ENCRYPTED_CONFIG" && -s "$ENCRYPTED_CONFIG_SHA" ]] ||
  { echo "ERROR: timed out waiting for encrypted rclone config" >&2; exit 1; }
read -r EXPECTED_ENCRYPTED_SHA remainder <"$ENCRYPTED_CONFIG_SHA"
[[ "$EXPECTED_ENCRYPTED_SHA" =~ ^[0-9a-f]{64}$ && -z "${remainder:-}" ]] ||
  { echo "ERROR: encrypted-config SHA sidecar is malformed" >&2; exit 1; }
[[ "$(sha256_file "$ENCRYPTED_CONFIG")" == "$EXPECTED_ENCRYPTED_SHA" ]] ||
  { echo "ERROR: encrypted rclone config SHA-256 mismatch" >&2; exit 1; }

"$OPENSSL_BIN" cms -decrypt -binary -inform DER \
  -in "$ENCRYPTED_CONFIG" \
  -recip "$CERTIFICATE" \
  -inkey "$PRIVATE_KEY" \
  -out "$DECRYPTED_CONFIG"
"$CHMOD_BIN" 0600 "$DECRYPTED_CONFIG"

PUBLISH_LOG_PARTIAL="$WORK_ROOT_REAL/publish.log.partial"
PUBLISH_LOG_FINAL="$WORK_ROOT_REAL/publish.log"
[[ ! -e "$PUBLISH_LOG_PARTIAL" && ! -e "$PUBLISH_LOG_FINAL" ]] ||
  { echo "ERROR: refusing to replace an existing publisher log" >&2; exit 1; }
set +e
"$COSMOS_ENV_REAL/bin/python" "$OSS_PUBLISHER" \
  --cache-root "$CACHE_ROOT" \
  --work-root "$WORK_ROOT_REAL" \
  --remote "$ROBOCASA365_OSS_REMOTE" \
  --run-id "$LATENT_CACHE_RUN_ID" \
  --source-commit "$EXPECTED_SOURCE_COMMIT" \
  --rclone "$RCLONE_REAL" \
  --rclone-config "$DECRYPTED_CONFIG" \
  --expected-rclone-sha256 "$EXPECTED_RCLONE_SHA256" \
  --receipt-path "$WORK_ROOT_REAL/terminal-receipt.json" \
  2>&1 | "$TEE_BIN" "$PUBLISH_LOG_PARTIAL"
PUBLISH_STATUS="${PIPESTATUS[0]}"
set -e
/usr/bin/chmod 0444 "$PUBLISH_LOG_PARTIAL"
/usr/bin/mv -- "$PUBLISH_LOG_PARTIAL" "$PUBLISH_LOG_FINAL"
[[ "$PUBLISH_STATUS" -eq 0 ]] || { echo "ERROR: OSS publication failed" >&2; exit "$PUBLISH_STATUS"; }

echo "FULL_PADDING_CACHE_TERMINAL_COMPLETE run_id=$LATENT_CACHE_RUN_ID remote=$ROBOCASA365_OSS_REMOTE"
