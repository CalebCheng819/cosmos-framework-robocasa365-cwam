#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail
umask 077

readonly OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/robocasa365-wan-fullpad-oss"
readonly RCLONE_CONFIG_FILE="/root/.config/rclone/rclone.conf"
readonly OPENSSL_BIN="/usr/bin/openssl"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly REALPATH_BIN="/usr/bin/realpath"

[[ "$#" -eq 1 ]] || { echo "usage: $0 <run-id>" >&2; exit 2; }
RUN_ID="$1"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 1; }

for tool in "$OPENSSL_BIN" "$SHA256SUM_BIN" "$REALPATH_BIN" /usr/bin/python3; do
  [[ -x "$tool" ]] || { echo "ERROR: required tool is unavailable: $tool" >&2; exit 1; }
done

WORK_ROOT="$OUTPUT_ROOT/$RUN_ID"
WORK_ROOT_REAL="$($REALPATH_BIN -e -- "$WORK_ROOT")"
[[ "$WORK_ROOT_REAL" == "$WORK_ROOT" ]] || { echo "ERROR: work root is not canonical" >&2; exit 1; }
REQUEST="$WORK_ROOT_REAL/rclone-config-request.json"
CERTIFICATE="$WORK_ROOT_REAL/rclone-config-recipient.pem"
[[ -s "$REQUEST" && -s "$CERTIFICATE" ]] ||
  { echo "ERROR: worker has not published a complete encryption request" >&2; exit 1; }
[[ -f "$RCLONE_CONFIG_FILE" && ! -L "$RCLONE_CONFIG_FILE" ]] ||
  { echo "ERROR: local rclone config is missing or unsafe" >&2; exit 1; }
[[ "$($REALPATH_BIN -e -- "$RCLONE_CONFIG_FILE")" == "$RCLONE_CONFIG_FILE" ]] ||
  { echo "ERROR: local rclone config path is not canonical" >&2; exit 1; }

"/usr/bin/python3" - "$REQUEST" "$CERTIFICATE" "$RCLONE_CONFIG_FILE" "$RUN_ID" <<'PY'
import configparser
import hashlib
import json
import os
import pathlib
import stat
import sys

request_path = pathlib.Path(sys.argv[1])
certificate = pathlib.Path(sys.argv[2])
config_path = pathlib.Path(sys.argv[3])
run_id = sys.argv[4]
request = json.loads(request_path.read_text())
required = {
    "schema_version": 1,
    "kind": "rclone_config_encryption_request",
    "status": "waiting",
    "run_id": run_id,
    "recipient_certificate": str(certificate),
    "envelope": "openssl-cms-enveloped-data-aes-256-cbc-der",
}
for key, expected in required.items():
    if request.get(key) != expected:
        raise SystemExit(f"handoff request mismatch for {key}")
if not str(request.get("remote", "")).startswith("eailab-hdd2:eailab/chengjuntao/"):
    raise SystemExit("handoff request remote escapes the guarded prefix")
actual_cert_sha = hashlib.sha256(certificate.read_bytes()).hexdigest()
if request.get("recipient_certificate_sha256") != actual_cert_sha:
    raise SystemExit("handoff certificate checksum mismatch")
metadata = config_path.stat()
if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
    raise SystemExit("local rclone config must be an owner-only regular file")
parser = configparser.RawConfigParser()
parser.read(config_path)
if not parser.has_section("eailab-hdd2"):
    raise SystemExit("local rclone config has no eailab-hdd2 remote")
section = parser["eailab-hdd2"]
expected = {
    "type": "s3",
    "provider": "Ceph",
    "endpoint": "http://hdd2.h.pjlab.org.cn:8060",
    "acl": "private",
}
if {key: section.get(key) for key in expected} != expected:
    raise SystemExit("local eailab-hdd2 non-secret configuration differs from the audited contract")
if not section.get("access_key_id") or not section.get("secret_access_key"):
    raise SystemExit("local eailab-hdd2 credentials are absent")
PY

ENCRYPTED="$WORK_ROOT_REAL/rclone-config.cms.der"
SIDECAR="$WORK_ROOT_REAL/rclone-config.cms.der.sha256"
[[ ! -e "$ENCRYPTED" && ! -e "$SIDECAR" ]] ||
  { echo "ERROR: refusing to replace an existing encrypted credential handoff" >&2; exit 1; }
TEMP_ENCRYPTED="$WORK_ROOT_REAL/.rclone-config.cms.der.tmp-$$"
TEMP_SIDECAR="$WORK_ROOT_REAL/.rclone-config.cms.der.sha256.tmp-$$"
trap '/usr/bin/rm -f -- "$TEMP_ENCRYPTED" "$TEMP_SIDECAR"' EXIT INT TERM

"$OPENSSL_BIN" cms -encrypt -binary -aes-256-cbc -outform DER \
  -in "$RCLONE_CONFIG_FILE" \
  -out "$TEMP_ENCRYPTED" \
  "$CERTIFICATE"
/usr/bin/chmod 0400 "$TEMP_ENCRYPTED"
read -r ENCRYPTED_SHA remainder < <("$SHA256SUM_BIN" "$TEMP_ENCRYPTED")
printf '%s\n' "$ENCRYPTED_SHA" >"$TEMP_SIDECAR"
/usr/bin/chmod 0444 "$TEMP_SIDECAR"
/usr/bin/mv -- "$TEMP_ENCRYPTED" "$ENCRYPTED"
/usr/bin/mv -- "$TEMP_SIDECAR" "$SIDECAR"
trap - EXIT INT TERM

echo "RCLONE_CONFIG_HANDOFF_COMPLETE run_id=$RUN_ID encrypted_sha256=$ENCRYPTED_SHA"
