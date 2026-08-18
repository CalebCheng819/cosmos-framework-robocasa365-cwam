#!/usr/bin/env bash
# Submit one node-pinned eight-H200 job that diagnoses R13 before training V2.

set -Eeuo pipefail
umask 077

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable CWAM-v2 source snapshot.}"
: "${GIT_REVISION:?Set GIT_REVISION to the snapshot Git revision.}"
: "${FORMAL_RUN_NAME:=CWAM-COSMOS3-V2-20260818-R1}"
: "${DIAG_RUN_ID:=CWAM-COSMOS3-C1-DIAG-20260818-R1}"
: "${RJOB_NAME:=cwam-c1diag-v2-0818-r1}"
: "${EXPECTED_HOST:=gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"
: "${CONTROL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-v2-20260818/control}"
: "${RJOB_IMAGE:=registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7}"

readonly RJOB_QUOTA_GROUP=eailabagent_gpu
readonly RJOB_CPU=128
readonly RJOB_GPU=8
readonly RJOB_MEMORY_MIB=1500000

[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] || {
    echo "ERROR: invalid rjob name: $RJOB_NAME" >&2
    exit 1
}
for run_id in "$FORMAL_RUN_NAME" "$DIAG_RUN_ID"; do
    [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
        echo "ERROR: invalid run id: $run_id" >&2
        exit 1
    }
done
[[ -f "$COSMOS_SOURCE/GIT_REVISION" && "$(cat "$COSMOS_SOURCE/GIT_REVISION")" == "$GIT_REVISION" ]] || {
    echo "ERROR: source snapshot revision does not match GIT_REVISION" >&2
    exit 1
}
[[ -f "$COSMOS_SOURCE/RUN_ID" && "$(cat "$COSMOS_SOURCE/RUN_ID")" == "$FORMAL_RUN_NAME" ]] || {
    echo "ERROR: source snapshot RUN_ID does not match FORMAL_RUN_NAME" >&2
    exit 1
}
[[ -x "$COSMOS_SOURCE/examples/h_cluster/run_cwam_c1_diag_then_v2_h8.sh" ]] || {
    echo "ERROR: chained worker script is absent or not executable" >&2
    exit 1
}
command -v rjob >/dev/null

mkdir -p "$CONTROL_ROOT/results"
submission_log="$CONTROL_ROOT/results/${FORMAL_RUN_NAME}.submission.log"
receipt_path="$CONTROL_ROOT/results/${FORMAL_RUN_NAME}.submission.json"
[[ ! -e "$receipt_path" ]] || {
    echo "ERROR: refusing to reuse submission receipt: $receipt_path" >&2
    exit 1
}

printf -v q_source '%q' "$COSMOS_SOURCE"
printf -v q_revision '%q' "$GIT_REVISION"
printf -v q_formal '%q' "$FORMAL_RUN_NAME"
printf -v q_diag '%q' "$DIAG_RUN_ID"
printf -v q_host '%q' "$EXPECTED_HOST"
printf -v q_control '%q' "$CONTROL_ROOT"
worker_command="$(cat <<EOF
set -Eeuo pipefail
export COSMOS_SOURCE=$q_source
export GIT_REVISION=$q_revision
export FORMAL_RUN_NAME=$q_formal
export DIAG_RUN_ID=$q_diag
export EXPECTED_HOST=$q_host
export CONTROL_ROOT=$q_control
bash \"\$COSMOS_SOURCE/examples/h_cluster/run_cwam_c1_diag_then_v2_h8.sh\"
EOF
)"

set +e
rjob submit \
    --name "$RJOB_NAME" \
    --charged-group "$RJOB_QUOTA_GROUP" \
    --priority 9 \
    --task-type normal \
    --restart-policy never \
    --termination-grace-period-seconds 60 \
    --backoff_limit 1 \
    --enable-sshd \
    --store-host-nvme=True \
    --image "$RJOB_IMAGE" \
    --cpu "$RJOB_CPU" \
    --gpu "$RJOB_GPU" \
    --memory "$RJOB_MEMORY_MIB" \
    --positive-tags "node/$EXPECTED_HOST" \
    --mount \
        "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs" \
        "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public" \
    -- bash -lc "$worker_command" 2>&1 | tee "$submission_log"
submit_status="${PIPESTATUS[0]}"
set -e
[[ "$submit_status" -eq 0 ]] || exit "$submit_status"

scheduler_name="$(sed -n 's/.*created rjob_name:[[:space:]]*\([a-z0-9-]*\).*/\1/p' "$submission_log" | tail -1)"
[[ -n "$scheduler_name" ]] || {
    echo "ERROR: scheduler did not return a job name" >&2
    exit 1
}
get_output="$(rjob get "$scheduler_name")"
grep -Eq "rjob $scheduler_name \(showname=$RJOB_NAME\): (Inqueue|Running)$" <<<"$get_output" || {
    printf '%s\n' "$get_output" >&2
    exit 1
}

export RECEIPT_PATH="$receipt_path"
export SCHEDULER_NAME="$scheduler_name"
export SHOW_NAME="$RJOB_NAME"
export RECEIPT_FORMAL_RUN="$FORMAL_RUN_NAME"
export RECEIPT_DIAG_RUN="$DIAG_RUN_ID"
export RECEIPT_REVISION="$GIT_REVISION"
export RECEIPT_HOST="$EXPECTED_HOST"
python3 - <<'PY'
import datetime
import json
import os
from pathlib import Path

payload = {
    "status": "accepted",
    "accepted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scheduler_rjob_name": os.environ["SCHEDULER_NAME"],
    "show_name": os.environ["SHOW_NAME"],
    "formal_run_id": os.environ["RECEIPT_FORMAL_RUN"],
    "diagnostic_run_id": os.environ["RECEIPT_DIAG_RUN"],
    "git_revision": os.environ["RECEIPT_REVISION"],
    "expected_host": os.environ["RECEIPT_HOST"],
    "gpu_count": 8,
    "cpu": 128,
    "memory_mib": 1500000,
    "phase_order": ["fixed_r13_diagnostics", "cwam_v2_smoke", "cwam_v2_formal_training"],
    "diagnostic_oss_root": (
        "eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1-DIAG/"
        + os.environ["RECEIPT_DIAG_RUN"]
    ),
    "training_oss_root": (
        "eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/V2/"
        + os.environ["RECEIPT_FORMAL_RUN"]
    ),
}
path = Path(os.environ["RECEIPT_PATH"])
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

echo "CWAM_C1_DIAG_V2_SUBMISSION_ACCEPTED scheduler_rjob=$scheduler_name receipt=$receipt_path"

