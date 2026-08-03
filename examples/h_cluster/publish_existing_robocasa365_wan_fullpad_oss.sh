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

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable publisher snapshot.}"
: "${COSMOS_ENV:?Set COSMOS_ENV to the audited runtime environment.}"
: "${PUBLISH_RUN_ID:?Set PUBLISH_RUN_ID.}"
: "${SOURCE_LATENT_CACHE_RUN_ID:?Set SOURCE_LATENT_CACHE_RUN_ID.}"
: "${ROBOCASA365_OSS_REMOTE:?Set the exact guarded HDD2 target prefix.}"
: "${ROBOCASA365_OSS_WORK_ROOT:?Set the run-specific personal-GPFS handoff root.}"
: "${RCLONE_BIN:?Set the pinned rclone binary.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_CACHE_SOURCE_COMMIT:?Set EXPECTED_CACHE_SOURCE_COMMIT.}"
: "${EXPECTED_RECOVERY_RUNNER_SHA256:?Set EXPECTED_RECOVERY_RUNNER_SHA256.}"
: "${EXPECTED_OSS_PUBLISHER_SHA256:?Set EXPECTED_OSS_PUBLISHER_SHA256.}"
: "${EXPECTED_RCLONE_SHA256:?Set EXPECTED_RCLONE_SHA256.}"
: "${EXPECTED_CACHE_ID:?Set EXPECTED_CACHE_ID.}"
: "${EXPECTED_CACHE_MANIFEST_SHA256:?Set EXPECTED_CACHE_MANIFEST_SHA256.}"

[[ "$#" -eq 0 ]] || { echo "ERROR: command-line arguments are forbidden" >&2; exit 1; }
for run_id in "$PUBLISH_RUN_ID" "$SOURCE_LATENT_CACHE_RUN_ID"; do
  [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: one run ID is invalid" >&2; exit 1; }
done
[[ "$PUBLISH_RUN_ID" != "$SOURCE_LATENT_CACHE_RUN_ID" ]] ||
  { echo "ERROR: recovery publication must use a fresh run ID" >&2; exit 1; }
for commit in "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_CACHE_SOURCE_COMMIT"; do
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: one source commit is malformed" >&2; exit 1; }
done
for digest in \
  "$EXPECTED_RECOVERY_RUNNER_SHA256" \
  "$EXPECTED_OSS_PUBLISHER_SHA256" \
  "$EXPECTED_RCLONE_SHA256" \
  "$EXPECTED_CACHE_MANIFEST_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: one required SHA-256 is malformed" >&2; exit 1; }
done
[[ "$EXPECTED_CACHE_ID" =~ ^robocasa365-wan-[0-9a-f]{24}$ ]] ||
  { echo "ERROR: EXPECTED_CACHE_ID is malformed" >&2; exit 1; }
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

COSMOS_SOURCE_REAL="$("$REALPATH_BIN" -e -- "$COSMOS_SOURCE")"
COSMOS_ENV_REAL="$("$REALPATH_BIN" -e -- "$COSMOS_ENV")"
WORK_ROOT_REAL="$("$REALPATH_BIN" -e -- "$ROBOCASA365_OSS_WORK_ROOT")"
OUTPUT_ROOT_REAL="$("$REALPATH_BIN" -e -- "$OUTPUT_ROOT")"
RCLONE_REAL="$("$REALPATH_BIN" -e -- "$RCLONE_BIN")"
CACHE_ROOT="$NVME_ROOT/$SOURCE_LATENT_CACHE_RUN_ID/cache"
CACHE_ROOT_REAL="$("$REALPATH_BIN" -e -- "$CACHE_ROOT")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REAL" && "$COSMOS_ENV" == "$COSMOS_ENV_REAL" ]] ||
  { echo "ERROR: source/environment paths must be canonical" >&2; exit 1; }
[[ "$ROBOCASA365_OSS_WORK_ROOT" == "$WORK_ROOT_REAL" ]] ||
  { echo "ERROR: handoff root must be canonical and pre-created" >&2; exit 1; }
path_is_within "$WORK_ROOT_REAL" "$OUTPUT_ROOT_REAL" ||
  { echo "ERROR: handoff root escapes the pinned personal-GPFS output root" >&2; exit 1; }
[[ "$WORK_ROOT_REAL" == "$OUTPUT_ROOT_REAL/$PUBLISH_RUN_ID" ]] ||
  { echo "ERROR: handoff root must be the exact publish run directory" >&2; exit 1; }
[[ "$CACHE_ROOT" == "$CACHE_ROOT_REAL" && -d "$CACHE_ROOT_REAL" && ! -L "$CACHE_ROOT_REAL" ]] ||
  { echo "ERROR: source cache is absent, non-canonical, or a symlink" >&2; exit 1; }

RECOVERY_RUNNER="$COSMOS_SOURCE_REAL/examples/h_cluster/publish_existing_robocasa365_wan_fullpad_oss.sh"
OSS_PUBLISHER="$COSMOS_SOURCE_REAL/examples/h_cluster/publish_robocasa365_wan_cache_to_oss.py"
[[ "$("$REALPATH_BIN" -e -- "${BASH_SOURCE[0]}")" == "$RECOVERY_RUNNER" ]] ||
  { echo "ERROR: recovery runner must execute from the immutable snapshot" >&2; exit 1; }
[[ "$(sha256_file "$RECOVERY_RUNNER")" == "$EXPECTED_RECOVERY_RUNNER_SHA256" ]] ||
  { echo "ERROR: recovery runner SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256_file "$OSS_PUBLISHER")" == "$EXPECTED_OSS_PUBLISHER_SHA256" ]] ||
  { echo "ERROR: OSS publisher SHA-256 mismatch" >&2; exit 1; }
[[ "$RCLONE_BIN" == "$RCLONE_REAL" && "$(sha256_file "$RCLONE_REAL")" == "$EXPECTED_RCLONE_SHA256" ]] ||
  { echo "ERROR: pinned rclone path or SHA-256 mismatch" >&2; exit 1; }

CACHE_VALIDATION="$WORK_ROOT_REAL/source-cache-validation.json"
[[ ! -e "$CACHE_VALIDATION" ]] || { echo "ERROR: source-cache validation receipt already exists" >&2; exit 1; }
PYTHONPATH="$COSMOS_SOURCE_REAL" "$COSMOS_ENV_REAL/bin/python" - \
  "$CACHE_ROOT_REAL" \
  "$CACHE_VALIDATION" \
  "$PUBLISH_RUN_ID" \
  "$SOURCE_LATENT_CACHE_RUN_ID" \
  "$EXPECTED_CACHE_ID" \
  "$EXPECTED_CACHE_MANIFEST_SHA256" \
  "$EXPECTED_CACHE_SOURCE_COMMIT" <<'PY'
import datetime
import json
import os
import pathlib
import sys

from cosmos_framework.data.generator.action.datasets.robocasa365_latent_cache import (
    CACHE_STAGE_FULL_PADDING,
    validate_published_cache_files,
)

cache_root = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
published = validate_published_cache_files(cache_root, verify_shard_checksums=False)
manifest = published["manifest"]
latent = manifest.get("latent", {})
if published["cache_id"] != sys.argv[5]:
    raise SystemExit("source cache ID differs from the registered R3 cache")
if published["manifest_sha256"] != sys.argv[6]:
    raise SystemExit("source cache manifest checksum differs from the registered R3 cache")
expected_latent = {
    "stage": CACHE_STAGE_FULL_PADDING,
    "shape": [48, 5, 30, 52],
    "image_size": [480, 832, 256, 768],
    "storage_dtype": "bfloat16",
}
if {key: latent.get(key) for key in expected_latent} != expected_latent:
    raise SystemExit(f"source cache latent contract mismatch: {latent!r}")
payload = {
    "schema_version": 1,
    "kind": "robocasa365_wan_full_padding_recovery_source_validation",
    "status": "passed",
    "validated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "publish_run_id": sys.argv[3],
    "source_cache_run_id": sys.argv[4],
    "source_cache_root": str(cache_root),
    "cache_id": published["cache_id"],
    "manifest_sha256": published["manifest_sha256"],
    "cache_source_commit": sys.argv[7],
    "latent_contract": expected_latent,
    "verify_shard_checksums": False,
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o444)
os.replace(temporary, target)
print(
    "RECOVERY_SOURCE_CACHE_VALIDATED "
    f"cache_id={published['cache_id']} manifest_sha256={published['manifest_sha256']}",
    flush=True,
)
PY

SECRET_ROOT="/dev/shm/cwam-rclone-$PUBLISH_RUN_ID-$$"
/usr/bin/mkdir --mode=0700 -- "$SECRET_ROOT"
PRIVATE_KEY="$SECRET_ROOT/private-key.pem"
CERTIFICATE="$SECRET_ROOT/certificate.pem"
DECRYPTED_CONFIG="$SECRET_ROOT/rclone.conf"
cleanup_secrets() {
  "$RM_BIN" -rf -- "$SECRET_ROOT"
}
trap cleanup_secrets EXIT INT TERM

"$OPENSSL_BIN" req -x509 -newkey rsa:3072 -sha256 -nodes -days 1 \
  -subj "/CN=$PUBLISH_RUN_ID" \
  -keyout "$PRIVATE_KEY" \
  -out "$CERTIFICATE" >/dev/null 2>&1
"$CHMOD_BIN" 0400 "$PRIVATE_KEY"

PUBLIC_CERT="$WORK_ROOT_REAL/rclone-config-recipient.pem"
[[ ! -e "$PUBLIC_CERT" ]] || { echo "ERROR: refusing to reuse an existing handoff certificate" >&2; exit 1; }
/usr/bin/install -m 0444 -- "$CERTIFICATE" "$PUBLIC_CERT"
HANDOFF_REQUEST="$WORK_ROOT_REAL/rclone-config-request.json"
PYTHONPATH="$COSMOS_SOURCE_REAL" "$COSMOS_ENV_REAL/bin/python" - \
  "$HANDOFF_REQUEST" \
  "$PUBLISH_RUN_ID" \
  "$SOURCE_LATENT_CACHE_RUN_ID" \
  "$ROBOCASA365_OSS_REMOTE" \
  "$CACHE_ROOT_REAL" \
  "$EXPECTED_CACHE_ID" \
  "$EXPECTED_CACHE_MANIFEST_SHA256" \
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
    "source_cache_run_id": sys.argv[3],
    "remote": sys.argv[4],
    "cache_root": sys.argv[5],
    "cache_id": sys.argv[6],
    "manifest_sha256": sys.argv[7],
    "recipient_certificate": sys.argv[8],
    "recipient_certificate_sha256": sys.argv[9],
    "envelope": "openssl-cms-enveloped-data-aes-256-cbc-der",
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
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
PYTHONPATH="$COSMOS_SOURCE_REAL" "$COSMOS_ENV_REAL/bin/python" "$OSS_PUBLISHER" \
  --cache-root "$CACHE_ROOT_REAL" \
  --work-root "$WORK_ROOT_REAL" \
  --remote "$ROBOCASA365_OSS_REMOTE" \
  --run-id "$PUBLISH_RUN_ID" \
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

echo "EXISTING_FULL_PADDING_CACHE_TERMINAL_COMPLETE publish_run_id=$PUBLISH_RUN_ID source_cache_run_id=$SOURCE_LATENT_CACHE_RUN_ID remote=$ROBOCASA365_OSS_REMOTE"
