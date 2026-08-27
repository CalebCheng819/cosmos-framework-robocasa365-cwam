#!/usr/bin/env bash
# Deliver one encrypted eailab-hdd2 config to an authorized RoboCasa worker.

set -Eeuo pipefail
umask 077

: "${HANDOFF_ROOT:?Set HANDOFF_ROOT to the path printed by the worker.}"
: "${EXPECTED_JOB_ID:?Set EXPECTED_JOB_ID.}"
: "${EXPECTED_SOURCE_CHECKPOINT_URI:?Set EXPECTED_SOURCE_CHECKPOINT_URI.}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${LOCAL_RCLONE_CONFIG:=/root/.config/rclone/rclone.conf}"

readonly ALLOWED_BASE="$PERSONAL_ROOT/runtime-handoffs/cwam-robocasa-identifiability"
readonly REQUEST="$HANDOFF_ROOT/request.json"
readonly CERTIFICATE="$HANDOFF_ROOT/recipient.pem"
readonly DESTINATION="$HANDOFF_ROOT/rclone-config.cms.der"
readonly TEMP_CONFIG="/dev/shm/cwam-robocasa-identifiability-rclone.$$.conf"
readonly TEMP_ENCRYPTED="$HANDOFF_ROOT/.rclone-config.cms.der.tmp-$$"

[[ "$(realpath -m -- "$HANDOFF_ROOT")" == "$ALLOWED_BASE/$EXPECTED_JOB_ID" ]]
[[ -s "$REQUEST" && -s "$CERTIFICATE" ]]
[[ -f "$LOCAL_RCLONE_CONFIG" && ! -L "$LOCAL_RCLONE_CONFIG" ]]
[[ ! -e "$DESTINATION" ]]
trap 'rm -f "$TEMP_CONFIG" "$TEMP_ENCRYPTED"' EXIT INT TERM

/usr/bin/python3 - "$REQUEST" "$LOCAL_RCLONE_CONFIG" "$TEMP_CONFIG" \
    "$EXPECTED_JOB_ID" "$EXPECTED_SOURCE_CHECKPOINT_URI" <<'PY'
import configparser
import json
import sys
from pathlib import Path

request_path, source_path, destination_path = map(Path, sys.argv[1:4])
expected_job, expected_checkpoint = sys.argv[4:6]
request = json.loads(request_path.read_text())
if request.get("schema_version") != 1 or request.get("job_id") != expected_job:
    raise SystemExit("worker request identity mismatch")
if request.get("source_checkpoint_uri") != expected_checkpoint:
    raise SystemExit("worker checkpoint request mismatch")
if not str(request.get("oss_run_root", "")).startswith("eailab-hdd2:eailab/chengjuntao/"):
    raise SystemExit("worker run root escapes authorized prefix")
source = configparser.ConfigParser(interpolation=None)
source.read(source_path)
section = source["eailab-hdd2"]
if section.get("endpoint") != "http://hdd2.h.pjlab.org.cn:8060":
    raise SystemExit("unexpected eailab-hdd2 endpoint")
if not section.get("access_key_id") or not section.get("secret_access_key"):
    raise SystemExit("eailab-hdd2 credentials are absent")
target = configparser.ConfigParser(interpolation=None)
target["eailab-hdd2"] = dict(section)
with destination_path.open("w") as stream:
    target.write(stream)
PY
chmod 600 "$TEMP_CONFIG"
/usr/bin/openssl cms -encrypt -binary -aes-256-cbc -outform DER \
    -in "$TEMP_CONFIG" -out "$TEMP_ENCRYPTED" "$CERTIFICATE"
chmod 400 "$TEMP_ENCRYPTED"
mv "$TEMP_ENCRYPTED" "$DESTINATION"
rm -f "$TEMP_CONFIG"
trap - EXIT INT TERM
echo "RCLONE_HANDOFF_COMPLETE job_id=$EXPECTED_JOB_ID"
