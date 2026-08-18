#!/usr/bin/env bash
# Relay one encrypted rclone config from an rjob package to a waiting export worker.

set -Eeuo pipefail
umask 077

required=(EXPORT_ID HANDOFF_ROOT ENCRYPTED_CONFIG_SOURCE)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done

readonly PERSONAL_HANDOFF_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly REQUEST="$HANDOFF_ROOT/request.json"
readonly STATUS_JSON="$HANDOFF_ROOT/evidence/status.json"
readonly OUTPUT="$HANDOFF_ROOT/rclone-config.cms.der"

[[ "$EXPORT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: unsafe EXPORT_ID" >&2; exit 1; }
[[ "$HANDOFF_ROOT" == "$PERSONAL_HANDOFF_ROOT/"* ]] || {
    echo "ERROR: handoff root escapes personal GPFS" >&2
    exit 1
}
[[ -f "$REQUEST" && ! -L "$REQUEST" && -f "$STATUS_JSON" && ! -L "$STATUS_JSON" ]] || {
    echo "ERROR: waiting worker evidence is incomplete" >&2
    exit 1
}
[[ -f "$ENCRYPTED_CONFIG_SOURCE" && ! -L "$ENCRYPTED_CONFIG_SOURCE" && -s "$ENCRYPTED_CONFIG_SOURCE" ]] || {
    echo "ERROR: encrypted config payload is invalid" >&2
    exit 1
}
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: refusing to replace encrypted handoff" >&2; exit 1; }

python3 - "$REQUEST" "$STATUS_JSON" "$EXPORT_ID" <<'PY'
import json
import pathlib
import sys

request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
status = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
export_id = sys.argv[3]
expected_request = {
    "schema_version": "metadata_no_hash_v1",
    "kind": "rclone_config_encryption_request",
    "status": "waiting",
    "export_id": export_id,
    "envelope": "openssl-cms-enveloped-data-aes-256-cbc-der",
}
wrong = {key: (request.get(key), value) for key, value in expected_request.items() if request.get(key) != value}
if wrong:
    raise SystemExit(f"handoff request mismatch: {wrong}")
if not str(request.get("remote", "")).startswith("eailab-hdd2:eailab/chengjuntao/"):
    raise SystemExit("requested remote escapes personal prefix")
if status.get("kind") != "cwam_h_serving_export_status":
    raise SystemExit("unexpected worker status kind")
if status.get("export_id") != export_id or status.get("status") != "WAITING_FOR_CREDENTIAL":
    raise SystemExit("worker is not waiting for this export credential")
PY

openssl cms -cmsout -inform DER -in "$ENCRYPTED_CONFIG_SOURCE" -noout >/dev/null
python3 - "$ENCRYPTED_CONFIG_SOURCE" "$OUTPUT" <<'PY'
import os
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
with source.open("rb") as src, target.open("xb") as dst:
    shutil.copyfileobj(src, dst)
os.chmod(target, 0o444)
PY
printf 'CWAM_RCLONE_RELAY_COMPLETE export_id=%s output=%s bytes=%s\n' \
    "$EXPORT_ID" "$OUTPUT" "$(stat -c %s "$OUTPUT")"
