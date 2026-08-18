#!/usr/bin/env bash
# Encrypt the workspace rclone config for one waiting H export worker.

set -Eeuo pipefail
umask 077

[[ "$#" -eq 2 ]] || { echo "usage: $0 <handoff-root> <export-id>" >&2; exit 2; }
readonly HANDOFF_ROOT="$1"
readonly EXPORT_ID="$2"
readonly REQUEST="$HANDOFF_ROOT/request.json"
readonly CERTIFICATE="${RECIPIENT_CERTIFICATE_PATH:-$HANDOFF_ROOT/recipient.pem}"
readonly RCLONE_CONFIG=/root/.config/rclone/rclone.conf
readonly OUTPUT="${PROVISION_OUTPUT:-$HANDOFF_ROOT/rclone-config.cms.der}"
readonly PERSONAL_HANDOFF_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"

[[ "$HANDOFF_ROOT" == "$PERSONAL_HANDOFF_ROOT/"* ]] || {
    echo "ERROR: handoff root escapes personal GPFS" >&2
    exit 1
}
[[ -f "$CERTIFICATE" && ! -L "$CERTIFICATE" ]] || {
    echo "ERROR: recipient certificate is missing" >&2
    exit 1
}
[[ -f "$RCLONE_CONFIG" && ! -L "$RCLONE_CONFIG" ]] || {
    echo "ERROR: workspace rclone config is missing" >&2
    exit 1
}
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: refusing to replace encrypted handoff" >&2; exit 1; }

if [[ -z "${RECIPIENT_CERTIFICATE_PATH:-}" ]]; then
    [[ -f "$REQUEST" && ! -L "$REQUEST" ]] || { echo "ERROR: worker request is incomplete" >&2; exit 1; }
    python3 - "$REQUEST" "$CERTIFICATE" "$EXPORT_ID" <<'PY'
import json
import pathlib
import sys

request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": "metadata_no_hash_v1",
    "kind": "rclone_config_encryption_request",
    "status": "waiting",
    "export_id": sys.argv[3],
    "recipient_certificate": sys.argv[2],
    "envelope": "openssl-cms-enveloped-data-aes-256-cbc-der",
}
wrong = {key: (request.get(key), value) for key, value in expected.items() if request.get(key) != value}
if wrong:
    raise SystemExit(f"handoff request mismatch: {wrong}")
if not str(request.get("remote", "")).startswith("eailab-hdd2:eailab/chengjuntao/"):
    raise SystemExit("requested remote escapes personal prefix")
PY
else
    [[ "$(openssl x509 -in "$CERTIFICATE" -noout -subject -nameopt RFC2253)" == "subject=CN=$EXPORT_ID" ]] || {
        echo "ERROR: recipient certificate subject mismatch" >&2
        exit 1
    }
    openssl x509 -in "$CERTIFICATE" -noout -checkend 0 >/dev/null
fi

mkdir -p "$(dirname "$OUTPUT")"
temporary="$OUTPUT.tmp.$$"
trap 'rm -f "$temporary"' EXIT INT TERM
openssl cms -encrypt -binary -aes-256-cbc -outform DER \
    -in "$RCLONE_CONFIG" -out "$temporary" "$CERTIFICATE"
chmod 0444 "$temporary"
mv "$temporary" "$OUTPUT"
trap - EXIT INT TERM
printf 'CWAM_RCLONE_HANDOFF_COMPLETE export_id=%s output=%s bytes=%s\n' \
    "$EXPORT_ID" "$OUTPUT" "$(stat -c %s "$OUTPUT")"
