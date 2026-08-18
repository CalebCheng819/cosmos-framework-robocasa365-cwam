#!/usr/bin/env bash
# Export one CWAM training DCP from H object storage to an EMA-only BF16 artifact.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

required=(
    EXPORT_ID CONTROL_ROOT COSMOS_SOURCE COSMOS_ENV SOURCE_REMOTE SOURCE_CONFIG_REMOTE
    TARGET_REMOTE SOURCE_RUN_ID SOURCE_REVISION EXPECTED_EMA_KEY_COUNT
    EXPECTED_REGULAR_KEY_COUNT EXPECTED_SHARD_COUNT EXPECTED_SOURCE_EMA_LOGICAL_BYTES
    EXPECTED_TARGET_LOGICAL_BYTES SAMPLE_KEYS_CSV HANDOFF_ROOT
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done
export EXPORT_ID SOURCE_RUN_ID SOURCE_REVISION SOURCE_REMOTE TARGET_REMOTE

readonly EXPORTER="$CONTROL_ROOT/examples/h_cluster/export_cwam_serving_ema_bf16_dcp_metadata.py"
readonly WORK_ROOT="/nvme/cwam-serving-export/$EXPORT_ID"
readonly SOURCE_ROOT="$WORK_ROOT/source"
readonly ARTIFACT_ROOT="$WORK_ROOT/artifact"
readonly PRIVATE_ROOT="$WORK_ROOT/private"
readonly PRIVATE_KEY="$PRIVATE_ROOT/recipient-key.pem"
readonly CERTIFICATE="$HANDOFF_ROOT/recipient.pem"
readonly REQUEST="$HANDOFF_ROOT/request.json"
readonly ENCRYPTED_CONFIG="$HANDOFF_ROOT/rclone-config.cms.der"
readonly RCLONE_CONFIG="$PRIVATE_ROOT/rclone.conf"
readonly EVIDENCE_ROOT="$HANDOFF_ROOT/evidence"
readonly STATUS_JSON="$EVIDENCE_ROOT/status.json"
readonly INVENTORY="$EVIDENCE_ROOT/published-inventory.txt"
readonly PERSONAL_HANDOFF_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly PUBLIC_HANDOFF_ROOT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/cwam-chengjuntao/handoffs"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

write_status() {
    local status="$1"
    local detail="$2"
    local temporary="$STATUS_JSON.tmp.$$"
    mkdir -p "$EVIDENCE_ROOT"
    python3 - "$temporary" "$status" "$detail" <<'PY'
import datetime
import json
import os
import pathlib
import sys

payload = {
    "schema_version": "metadata_no_hash_v1",
    "kind": "cwam_h_serving_export_status",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "export_id": os.environ["EXPORT_ID"],
    "source_run_id": os.environ["SOURCE_RUN_ID"],
    "source_revision": os.environ["SOURCE_REVISION"],
    "source_remote": os.environ["SOURCE_REMOTE"],
    "target_remote": os.environ["TARGET_REMOTE"],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    chmod 0444 "$temporary"
    mv -f "$temporary" "$STATUS_JSON"
}

on_exit() {
    local status="$?"
    trap - EXIT INT TERM
    if ((status != 0)); then
        write_status FAILED "worker_exit_code=$status" || true
    fi
    rm -rf --one-file-system "$PRIVATE_ROOT" || true
    exit "$status"
}
[[ "$EXPORT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe EXPORT_ID"
[[ "$SOURCE_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || die "source remote escapes personal prefix"
[[ "$SOURCE_CONFIG_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || die "config remote escapes personal prefix"
[[ "$TARGET_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || die "target remote escapes personal prefix"
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "invalid source revision"
[[ -d "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || die "control root missing"
[[ -d "$COSMOS_SOURCE" && ! -L "$COSMOS_SOURCE" ]] || die "source snapshot missing"
[[ -x "$COSMOS_ENV/bin/python" ]] || die "Cosmos Python missing"
[[ -f "$EXPORTER" && ! -L "$EXPORTER" ]] || die "exporter missing"
[[ -f "$COSMOS_SOURCE/GIT_REVISION" && -f "$COSMOS_SOURCE/RUN_ID" ]] || die "source identity receipts missing"
[[ "$(tr -d '[:space:]' <"$COSMOS_SOURCE/GIT_REVISION")" == "$SOURCE_REVISION" ]] || die "source revision mismatch"
[[ "$(tr -d '[:space:]' <"$COSMOS_SOURCE/RUN_ID")" == "$SOURCE_RUN_ID" ]] || die "source run ID mismatch"
[[ "$HANDOFF_ROOT" == "$PERSONAL_HANDOFF_ROOT/"* || "$HANDOFF_ROOT" == "$PUBLIC_HANDOFF_ROOT/"* ]] || {
    die "handoff root escapes approved GPFS roots"
}
[[ ! -e "$WORK_ROOT" ]] || die "worker NVMe root already exists"
[[ ! -e "$HANDOFF_ROOT" ]] || die "handoff root already exists"

mkdir -p "$PRIVATE_ROOT" "$SOURCE_ROOT/model" "$(dirname "$ARTIFACT_ROOT")" "$HANDOFF_ROOT" "$EVIDENCE_ROOT"
chmod 0733 "$HANDOFF_ROOT"
chmod 0755 "$EVIDENCE_ROOT"
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
openssl req -x509 -newkey rsa:3072 -nodes -days 2 -subj "/CN=$EXPORT_ID" \
    -keyout "$PRIVATE_KEY" -out "$CERTIFICATE" >/dev/null 2>&1
chmod 0400 "$PRIVATE_KEY"
chmod 0444 "$CERTIFICATE"
python3 - "$REQUEST.tmp.$$" "$CERTIFICATE" <<'PY'
import json
import os
import pathlib
import sys

payload = {
    "schema_version": "metadata_no_hash_v1",
    "kind": "rclone_config_encryption_request",
    "status": "waiting",
    "export_id": os.environ["EXPORT_ID"],
    "recipient_certificate": sys.argv[2],
    "remote": os.environ["TARGET_REMOTE"],
    "envelope": "openssl-cms-enveloped-data-aes-256-cbc-der",
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
chmod 0444 "$REQUEST.tmp.$$"
mv "$REQUEST.tmp.$$" "$REQUEST"
write_status WAITING_FOR_CREDENTIAL "recipient request published"

deadline=$((SECONDS + 3600))
while [[ ! -s "$ENCRYPTED_CONFIG" && $SECONDS -lt $deadline ]]; do sleep 5; done
[[ -s "$ENCRYPTED_CONFIG" ]] || die "timed out waiting for encrypted rclone config"
chmod 0555 "$HANDOFF_ROOT"
openssl cms -decrypt -binary -inform DER -in "$ENCRYPTED_CONFIG" \
    -recip "$CERTIFICATE" -inkey "$PRIVATE_KEY" -out "$RCLONE_CONFIG"
chmod 0400 "$RCLONE_CONFIG"
rclone --config "$RCLONE_CONFIG" listremotes | grep -Fx 'eailab-hdd2:' >/dev/null

write_status RUNNING "downloading source checkpoint to worker NVMe"
rclone --config "$RCLONE_CONFIG" copy "$SOURCE_REMOTE/model" "$SOURCE_ROOT/model" \
    --size-only --immutable --transfers 8 --checkers 16 --retries 5 --low-level-retries 10
rclone --config "$RCLONE_CONFIG" copyto "$SOURCE_CONFIG_REMOTE" "$SOURCE_ROOT/config.yaml" \
    --size-only --immutable --retries 5 --low-level-retries 10

IFS=',' read -r -a sample_keys <<<"$SAMPLE_KEYS_CSV"
sample_args=()
for key in "${sample_keys[@]}"; do
    [[ -n "$key" ]] || die "empty sample key"
    sample_args+=(--sample-key "$key")
done

write_status RUNNING "exporting EMA BF16 and performing full semantic readback"
PYTHONPATH="$COSMOS_SOURCE:$CONTROL_ROOT" \
    "$COSMOS_ENV/bin/python" "$EXPORTER" \
    --source-model "$SOURCE_ROOT/model" \
    --source-config "$SOURCE_ROOT/config.yaml" \
    --final-artifact "$ARTIFACT_ROOT" \
    --expected-ema-key-count "$EXPECTED_EMA_KEY_COUNT" \
    --expected-regular-key-count "$EXPECTED_REGULAR_KEY_COUNT" \
    --expected-shard-count "$EXPECTED_SHARD_COUNT" \
    --expected-source-ema-logical-bytes "$EXPECTED_SOURCE_EMA_LOGICAL_BYTES" \
    --expected-target-logical-bytes "$EXPECTED_TARGET_LOGICAL_BYTES" \
    "${sample_args[@]}" \
    --source-run-id "$SOURCE_RUN_ID" \
    --source-git-commit "$SOURCE_REVISION" \
    >"$EVIDENCE_ROOT/export.log" 2>&1

write_status RUNNING "publishing exact artifact to H object storage"
rclone --config "$RCLONE_CONFIG" copy "$ARTIFACT_ROOT" "$TARGET_REMOTE" \
    --size-only --immutable --transfers 8 --checkers 16 --retries 5 --low-level-retries 10
(
    cd "$ARTIFACT_ROOT"
    find . -type f -printf '%s|%P\n' | LC_ALL=C sort
) >"$EVIDENCE_ROOT/local-inventory.txt"
rclone --config "$RCLONE_CONFIG" lsf --recursive --files-only --format sp --separator '|' \
    "$TARGET_REMOTE" | LC_ALL=C sort >"$INVENTORY"
cmp "$EVIDENCE_ROOT/local-inventory.txt" "$INVENTORY" || die "published path/size inventory mismatch"

write_status COMPLETE "EMA BF16 artifact exported, semantically validated, and published"
printf 'CWAM_SERVING_EXPORT_COMPLETE export_id=%s target=%s\n' "$EXPORT_ID" "$TARGET_REMOTE"
rm -rf --one-file-system "$SOURCE_ROOT" "$ARTIFACT_ROOT"
trap - EXIT INT TERM
rm -rf --one-file-system "$PRIVATE_ROOT"
