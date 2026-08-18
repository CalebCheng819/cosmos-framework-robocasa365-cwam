#!/usr/bin/env bash
# Submit one isolated H worker that exports an EMA-only BF16 serving artifact.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_QUOTA_GROUP="eailabagent_gpu"
readonly PERSONAL_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PERSONAL_GPFS_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:$PERSONAL_ROOT"

required=(
    EXPORT_ID RJOB_NAME CONTROL_ROOT COSMOS_SOURCE COSMOS_ENV SOURCE_REMOTE
    SOURCE_CONFIG_REMOTE TARGET_REMOTE SOURCE_RUN_ID SOURCE_REVISION
    EXPECTED_EMA_KEY_COUNT EXPECTED_REGULAR_KEY_COUNT EXPECTED_SHARD_COUNT
    EXPECTED_SOURCE_EMA_LOGICAL_BYTES EXPECTED_TARGET_LOGICAL_BYTES SAMPLE_KEYS_CSV
    HANDOFF_ROOT SUBMISSION_ROOT
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done
: "${RJOB_PREDICT_ONLY:=1}"
export EXPORT_ID RJOB_NAME SOURCE_RUN_ID SOURCE_REVISION SOURCE_REMOTE TARGET_REMOTE

[[ "$EXPORT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: unsafe EXPORT_ID" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] || {
    echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2
    exit 1
}
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] || { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid source revision" >&2; exit 1; }
[[ "$CONTROL_ROOT" == "$PERSONAL_ROOT/chengjuntao/"* ]] || { echo "ERROR: invalid control root" >&2; exit 1; }
[[ "$COSMOS_SOURCE" == "$PERSONAL_ROOT/chengjuntao/"* ]] || { echo "ERROR: invalid source root" >&2; exit 1; }
[[ "$COSMOS_ENV" == "$PERSONAL_ROOT/chengjuntao/"* ]] || { echo "ERROR: invalid environment root" >&2; exit 1; }
[[ "$HANDOFF_ROOT" == "$PERSONAL_ROOT/chengjuntao/"* ]] || { echo "ERROR: invalid handoff root" >&2; exit 1; }
[[ "$SUBMISSION_ROOT" == "$PERSONAL_ROOT/chengjuntao/"* ]] || { echo "ERROR: invalid submission root" >&2; exit 1; }
[[ "$SOURCE_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || { echo "ERROR: invalid source remote" >&2; exit 1; }
[[ "$SOURCE_CONFIG_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || { echo "ERROR: invalid config remote" >&2; exit 1; }
[[ "$TARGET_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] || { echo "ERROR: invalid target remote" >&2; exit 1; }
[[ -d "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || { echo "ERROR: control root missing" >&2; exit 1; }
[[ -d "$COSMOS_SOURCE" && ! -L "$COSMOS_SOURCE" ]] || { echo "ERROR: source snapshot missing" >&2; exit 1; }
[[ -x "$COSMOS_ENV/bin/python" ]] || { echo "ERROR: Cosmos Python missing" >&2; exit 1; }
readonly WORKER="$CONTROL_ROOT/examples/h_cluster/export_cwam_serving_ema_bf16_from_hdd2.sh"
[[ -f "$WORKER" && ! -L "$WORKER" ]] || { echo "ERROR: worker script missing" >&2; exit 1; }
[[ ! -e "$HANDOFF_ROOT" ]] || { echo "ERROR: handoff root already exists" >&2; exit 1; }

worker_env=()
for name in "${required[@]}"; do
    [[ "$name" == RJOB_NAME || "$name" == SUBMISSION_ROOT || "$name" == CONTROL_ROOT ]] && continue
    worker_env+=("$name=${!name}")
done
worker_env+=("CONTROL_ROOT=.")
worker_command=(
    /usr/bin/env -u BASH_ENV -u ENV -u PYTHONPATH -u PYTHONHOME
    "${worker_env[@]}"
    /usr/bin/bash --noprofile --norc ./examples/h_cluster/export_cwam_serving_ema_bf16_from_hdd2.sh
)
submit_args=(
    submit
    --folder "$CONTROL_ROOT"
    --name "$RJOB_NAME"
    --group "$PINNED_QUOTA_GROUP"
    --charged-group "$PINNED_QUOTA_GROUP"
    --priority 9
    --task-type normal
    --replicas 1
    --restart-policy never
    --termination-grace-period-seconds 60
    --backoff_limit 1
    --enable-sshd
    --image "$PINNED_IMAGE"
    --cpu 32
    --gpu 1
    --memory 250000
    --store-host-nvme
    --mount "$PERSONAL_GPFS_MOUNT"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    submit_args+=(--predict-only true)
fi

printf 'EXPORT_ID=%s\nRJOB_NAME=%s\nRJOB_PREDICT_ONLY=%s\nSOURCE_RUN_ID=%s\nSOURCE_REVISION=%s\nTARGET_REMOTE=%s\n' \
    "$EXPORT_ID" "$RJOB_NAME" "$RJOB_PREDICT_ONLY" "$SOURCE_RUN_ID" "$SOURCE_REVISION" "$TARGET_REMOTE"
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
    rjob "${submit_args[@]}" -- "${worker_command[@]}"
    echo "RJOB_PREDICTION_COMPLETE reservation_created=false"
    exit 0
fi

mkdir -p "$SUBMISSION_ROOT"
readonly RESERVATION="$SUBMISSION_ROOT/$EXPORT_ID"
mkdir --mode=0700 "$RESERVATION"
python3 - "$RESERVATION/submission-intent.json" <<'PY'
import datetime
import json
import os
import pathlib

payload = {
    "schema_version": "metadata_no_hash_v1",
    "kind": "cwam_h_serving_export_submission_intent",
    "status": "submitting",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "export_id": os.environ["EXPORT_ID"],
    "rjob_name": os.environ["RJOB_NAME"],
    "source_run_id": os.environ["SOURCE_RUN_ID"],
    "source_revision": os.environ["SOURCE_REVISION"],
    "source_remote": os.environ["SOURCE_REMOTE"],
    "target_remote": os.environ["TARGET_REMOTE"],
    "resources": {"replicas": 1, "cpu": 32, "gpu": 1, "memory_mib": 250000},
}
path = pathlib.Path(os.sys.argv[1])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
path.chmod(0o444)
PY

set +e
rjob "${submit_args[@]}" -- "${worker_command[@]}" 2>&1 | tee "$RESERVATION/rjob-submit.log"
status=${PIPESTATUS[0]}
set -e
chmod 0444 "$RESERVATION/rjob-submit.log"
if ((status != 0)); then
    printf 'RJOB_SUBMISSION_FAILED exit_code=%s reservation=%s\n' "$status" "$RESERVATION" >&2
    exit "$status"
fi
printf 'RJOB_SUBMISSION_ACCEPTED export_id=%s rjob_name=%s reservation=%s\n' \
    "$EXPORT_ID" "$RJOB_NAME" "$RESERVATION"
