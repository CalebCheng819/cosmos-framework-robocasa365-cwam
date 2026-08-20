#!/usr/bin/env bash
# Encrypt the audited workspace rclone remote for one S1 continuation worker.

set -Eeuo pipefail
umask 077

: "${HANDOFF_ROOT:?Set HANDOFF_ROOT to the path printed by the worker.}"
: "${EXPECTED_JOB_ID:?Set EXPECTED_JOB_ID to the formal rjob metadata name.}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${LOCAL_RCLONE_CONFIG:=/root/.config/rclone/rclone.conf}"

readonly ALLOWED_BASE="$PERSONAL_ROOT/runtime-handoffs/cwam-s1-cont100k"
readonly REQUEST="$HANDOFF_ROOT/request.json"
readonly CERTIFICATE="$HANDOFF_ROOT/recipient.pem"
readonly DESTINATION="$HANDOFF_ROOT/rclone-config.cms.der"
readonly TEMP_CONFIG="/dev/shm/cwam-s1-rclone-config.$$.conf"
readonly TEMP_ENCRYPTED="$HANDOFF_ROOT/.rclone-config.cms.der.tmp-$$"

[[ "$(realpath -m -- "$HANDOFF_ROOT")" == "$ALLOWED_BASE/$EXPECTED_JOB_ID" ]] || {
    echo "ERROR: handoff root does not match the expected job" >&2
    exit 1
}
[[ -s "$REQUEST" && -s "$CERTIFICATE" ]] || { echo "ERROR: worker handoff request is incomplete" >&2; exit 1; }
[[ -f "$LOCAL_RCLONE_CONFIG" && ! -L "$LOCAL_RCLONE_CONFIG" ]] || {
    echo "ERROR: audited local rclone config is missing or unsafe" >&2
    exit 1
}
[[ ! -e "$DESTINATION" ]] || { echo "ERROR: refusing to replace an existing handoff" >&2; exit 1; }
trap 'rm -f "$TEMP_CONFIG" "$TEMP_ENCRYPTED"' EXIT INT TERM

/usr/bin/python3 - "$REQUEST" "$LOCAL_RCLONE_CONFIG" "$TEMP_CONFIG" "$EXPECTED_JOB_ID" <<'PY'
import configparser
import json
import sys
from pathlib import Path

request_path, source_path, destination_path = map(Path, sys.argv[1:4])
expected_job_id = sys.argv[4]
request = json.loads(request_path.read_text())
if request.get("schema_version") != 1 or request.get("job_id") != expected_job_id:
    raise SystemExit("worker request identity mismatch")
allowed = "eailab-hdd2:eailab/chengjuntao/"
for key in ("source_checkpoint_uri", "oss_run_root"):
    if not str(request.get(key, "")).startswith(allowed):
        raise SystemExit(f"worker request escapes the authorized prefix: {key}")

source = configparser.ConfigParser(interpolation=None)
source.read(source_path)
if "eailab-hdd2" not in source:
    raise SystemExit("local eailab-hdd2 remote is absent")
section = source["eailab-hdd2"]
if not section.get("access_key_id") or not section.get("secret_access_key"):
    raise SystemExit("local eailab-hdd2 credentials are absent")
if section.get("endpoint") != "http://hdd2.h.pjlab.org.cn:8060":
    raise SystemExit("local eailab-hdd2 endpoint differs from the audited contract")

target = configparser.ConfigParser(interpolation=None)
target["eailab-hdd2"] = dict(section)
with destination_path.open("w") as stream:
    target.write(stream)
PY
chmod 600 "$TEMP_CONFIG"
/usr/bin/openssl cms -encrypt -binary -aes-256-cbc -outform DER \
    -in "$TEMP_CONFIG" \
    -out "$TEMP_ENCRYPTED" \
    "$CERTIFICATE"
chmod 400 "$TEMP_ENCRYPTED"
mv "$TEMP_ENCRYPTED" "$DESTINATION"
rm -f "$TEMP_CONFIG"
trap - EXIT INT TERM
echo "RCLONE_HANDOFF_COMPLETE job_id=$EXPECTED_JOB_ID"
