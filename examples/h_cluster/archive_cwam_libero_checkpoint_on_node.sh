#!/usr/bin/env bash
# Archive a checkpoint left on H254 host NVMe by an already-running LIBERO job.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${EXPERIMENT_ID:?Set the source experiment ID.}"
: "${SOURCE_COMMIT:?Set the source experiment commit.}"
: "${SOURCE_CHECKPOINTS_DIR:?Set the source checkpoint directory on host NVMe.}"
: "${TARGET_STEP:?Set the checkpoint step to archive.}"
: "${OSS_RUN_ROOT:?Set the source experiment OSS root.}"
: "${KUBEBRAIN_RJOB_NAME:=unknown}"
: "${PERSONAL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"
: "${RCLONE_BIN:=$PERSONAL_ROOT/tools/rclone}"

readonly ARCHIVER="$COSMOS_SOURCE/examples/h_cluster/archive_cwam_s1_cont100k.py"
readonly EVIDENCE_ROOT="/nvme/cwam-libero-s1-archive/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"
readonly HANDOFF_ROOT="$PERSONAL_ROOT/runtime-handoffs/cwam-libero-s1/$KUBEBRAIN_RJOB_NAME"
readonly SECRET_ROOT="/dev/shm/cwam-libero-s1-archive-rclone"
readonly RCLONE_CONFIG_PATH="$SECRET_ROOT/rclone.conf"

[[ "$TARGET_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: TARGET_STEP must be positive" >&2; exit 1; }
[[ "$OSS_RUN_ROOT" == eailab-hdd2:eailab/chengjuntao/* ]] || {
    echo "ERROR: invalid OSS_RUN_ROOT" >&2
    exit 1
}
for path in "$ARCHIVER" "$COSMOS_ENV/bin/python" "$RCLONE_BIN"; do
    [[ -e "$path" ]] || { echo "ERROR: required input missing: $path" >&2; exit 1; }
done

mkdir -p "$EVIDENCE_ROOT" "$HANDOFF_ROOT" "$SECRET_ROOT"
chmod 700 "$HANDOFF_ROOT" "$SECRET_ROOT"

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    if [[ -s "$RCLONE_CONFIG_PATH" ]]; then
        "$RCLONE_BIN" copy "$EVIDENCE_ROOT" "$OSS_RUN_ROOT/evidence/checkpoint-rescue" \
            --transfers=4 --checkers=8 || true
    fi
    rm -rf "$SECRET_ROOT"
    rm -f "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json" "$HANDOFF_ROOT/rclone-config.cms.der"
    rmdir "$HANDOFF_ROOT" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

openssl req -x509 -newkey rsa:3072 -nodes -days 1 -subj "/CN=$KUBEBRAIN_RJOB_NAME" \
    -keyout "$SECRET_ROOT/recipient.key" -out "$HANDOFF_ROOT/recipient.pem" >/dev/null 2>&1
export KUBEBRAIN_RJOB_NAME OSS_RUN_ROOT
"$COSMOS_ENV/bin/python" - "$HANDOFF_ROOT/request.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "job_id": os.environ["KUBEBRAIN_RJOB_NAME"],
    "oss_run_root": os.environ["OSS_RUN_ROOT"],
}, indent=2) + "\n")
PY
chmod 444 "$HANDOFF_ROOT/recipient.pem" "$HANDOFF_ROOT/request.json"
echo "RCLONE_HANDOFF_WAITING root=$HANDOFF_ROOT"
for _ in $(seq 1 240); do
    [[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] && break
    sleep 5
done
[[ -s "$HANDOFF_ROOT/rclone-config.cms.der" ]] || { echo "ERROR: timed out waiting for rclone handoff" >&2; exit 1; }
openssl cms -decrypt -binary -inform DER -in "$HANDOFF_ROOT/rclone-config.cms.der" \
    -recip "$HANDOFF_ROOT/recipient.pem" -inkey "$SECRET_ROOT/recipient.key" -out "$RCLONE_CONFIG_PATH"
chmod 400 "$RCLONE_CONFIG_PATH"
export PATH="$(dirname "$RCLONE_BIN"):$COSMOS_ENV/bin:$PATH" RCLONE_CONFIG="$RCLONE_CONFIG_PATH"
"$RCLONE_BIN" lsd eailab-hdd2:eailab/chengjuntao >/dev/null

echo "CHECKPOINT_RESCUE_WAIT source=$SOURCE_CHECKPOINTS_DIR step=$TARGET_STEP"
"$COSMOS_ENV/bin/python" "$ARCHIVER" \
    --checkpoints-dir "$SOURCE_CHECKPOINTS_DIR" --remote-root "$OSS_RUN_ROOT" \
    --receipts-dir "$EVIDENCE_ROOT/receipts" --experiment-id "$EXPERIMENT_ID" \
    --job-id="$KUBEBRAIN_RJOB_NAME" --source-commit "$SOURCE_COMMIT" \
    --start-step "$TARGET_STEP" --end-step "$TARGET_STEP" --interval "$TARGET_STEP" \
    --keep-local 2 --poll-seconds 15
echo "CHECKPOINT_RESCUE_COMPLETE step=$TARGET_STEP"
