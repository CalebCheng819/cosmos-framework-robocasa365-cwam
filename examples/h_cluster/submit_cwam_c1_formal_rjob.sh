#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Submit one 6/8-H200 C1 run. New runs execute a two-step production-batch
# smoke; explicit resume runs stage and validate a full DCP checkpoint before
# continuing the 20k-step schedule. Large outputs are uploaded to OSS.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable C1 source snapshot.}"
: "${FORMAL_RUN_NAME:=CWAM-COSMOS3-C1-20260817-R1}"
: "${GIT_REVISION:?Set GIT_REVISION to the snapshot Git revision.}"
: "${RJOB_NAME:=cwam-c1-20260817-r1}"
: "${EXPECTED_HOST:=gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn}"
: "${RJOB_IMAGE:=registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7}"
: "${RJOB_QUOTA_GROUP:=eailabagent_gpu}"
: "${CONTROL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-c1-20260817/control}"
: "${C1_GPU_COUNT:=6}"
: "${RESUME_FROM_OSS:=0}"
: "${RESUME_CHECKPOINT_REMOTE:=}"
: "${RESUME_CHECKPOINT_NAME:=}"
: "${RESUME_SOURCE_RUN_NAME:=}"

[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] || {
    echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2
    exit 1
}
[[ "$FORMAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: unsupported FORMAL_RUN_NAME: $FORMAL_RUN_NAME" >&2
    exit 1
}
[[ "$RESUME_FROM_OSS" =~ ^[01]$ ]] || {
    echo "ERROR: RESUME_FROM_OSS must be 0 or 1" >&2
    exit 1
}
[[ "$C1_GPU_COUNT" == 6 || "$C1_GPU_COUNT" == 8 ]] || {
    echo "ERROR: C1_GPU_COUNT must be exactly 6 or 8" >&2
    exit 1
}
if [[ "$C1_GPU_COUNT" == 8 ]]; then
    RJOB_CPU=128
    RJOB_MEMORY_MIB=1500000
else
    RJOB_CPU=96
    RJOB_MEMORY_MIB=1125000
fi
if [[ "$RESUME_FROM_OSS" == 1 ]]; then
    [[ "$RESUME_CHECKPOINT_NAME" =~ ^iter_[0-9]{9}$ ]] || {
        echo "ERROR: RESUME_CHECKPOINT_NAME must match iter_NNNNNNNNN" >&2
        exit 1
    }
    [[ "$RESUME_SOURCE_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
        echo "ERROR: invalid RESUME_SOURCE_RUN_NAME" >&2
        exit 1
    }
    [[ "$RESUME_CHECKPOINT_REMOTE" =~ ^[A-Za-z0-9._:/-]+$ ]] || {
        echo "ERROR: invalid RESUME_CHECKPOINT_REMOTE" >&2
        exit 1
    }
    [[ "${RESUME_CHECKPOINT_REMOTE##*/}" == "$RESUME_CHECKPOINT_NAME" ]] || {
        echo "ERROR: resume remote basename must match RESUME_CHECKPOINT_NAME" >&2
        exit 1
    }
fi
[[ -f "$COSMOS_SOURCE/GIT_REVISION" && "$(cat "$COSMOS_SOURCE/GIT_REVISION")" == "$GIT_REVISION" ]] || {
    echo "ERROR: source snapshot Git revision does not match GIT_REVISION" >&2
    exit 1
}
[[ -f "$COSMOS_SOURCE/RUN_ID" && "$(cat "$COSMOS_SOURCE/RUN_ID")" == "$FORMAL_RUN_NAME" ]] || {
    echo "ERROR: source snapshot RUN_ID does not match FORMAL_RUN_NAME" >&2
    exit 1
}
[[ -x "$COSMOS_SOURCE/examples/h_cluster/cwam_c1_prepare_assets.sh" ]] || {
    echo "ERROR: C1 worker scripts are absent from the snapshot" >&2
    exit 1
}
command -v rjob >/dev/null

mkdir -p "$CONTROL_ROOT/results"
submission_log="$CONTROL_ROOT/results/${FORMAL_RUN_NAME}.submission.log"
receipt_path="$CONTROL_ROOT/results/${FORMAL_RUN_NAME}.submission.json"
[[ ! -e "$receipt_path" ]] || {
    echo "ERROR: refusing to reuse an existing C1 submission receipt" >&2
    exit 1
}

printf -v q_source '%q' "$COSMOS_SOURCE"
printf -v q_run '%q' "$FORMAL_RUN_NAME"
printf -v q_revision '%q' "$GIT_REVISION"
printf -v q_host '%q' "$EXPECTED_HOST"
printf -v q_control '%q' "$CONTROL_ROOT"
printf -v q_world_size '%q' "$C1_GPU_COUNT"
printf -v q_resume '%q' "$RESUME_FROM_OSS"
printf -v q_resume_remote '%q' "$RESUME_CHECKPOINT_REMOTE"
printf -v q_resume_name '%q' "$RESUME_CHECKPOINT_NAME"
printf -v q_resume_source_run '%q' "$RESUME_SOURCE_RUN_NAME"
worker_command="$(cat <<EOF
set -Eeuo pipefail
export COSMOS_SOURCE=$q_source
export FORMAL_RUN_NAME=$q_run
export GIT_REVISION=$q_revision
export EXPECTED_HOST=$q_host
export CONTROL_ROOT=$q_control
export WORLD_SIZE=$q_world_size
export RESUME_FROM_OSS=$q_resume
export RESUME_CHECKPOINT_REMOTE=$q_resume_remote
export RESUME_CHECKPOINT_NAME=$q_resume_name
export RESUME_SOURCE_RUN_NAME=$q_resume_source_run
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1
bash "\$COSMOS_SOURCE/examples/h_cluster/cwam_c1_prepare_assets.sh"
bash "\$COSMOS_SOURCE/examples/h_cluster/cwam_c1_smoke_then_train.sh"
EOF
)"

set +e
rjob submit \
    --name "$RJOB_NAME" \
    --group "$RJOB_QUOTA_GROUP" \
    --charged-group "$RJOB_QUOTA_GROUP" \
    --private-machine group \
    --priority 9 \
    --task-type normal \
    --restart-policy never \
    --termination-grace-period-seconds 60 \
    --backoff_limit 1 \
    --enable-sshd \
    --store-host-nvme=True \
    --image "$RJOB_IMAGE" \
    --cpu "$RJOB_CPU" \
    --gpu "$C1_GPU_COUNT" \
    --memory "$RJOB_MEMORY_MIB" \
    --positive-tags "node/$EXPECTED_HOST" \
    --mount \
        "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs" \
        "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public" \
    -- bash -lc "$worker_command" 2>&1 | tee "$submission_log"
submit_status="${PIPESTATUS[0]}"
set -e
[[ "$submit_status" -eq 0 ]] || {
    echo "ERROR: rjob submission failed with rc=$submit_status" >&2
    exit "$submit_status"
}

scheduler_name="$(sed -n 's/.*created rjob_name:[[:space:]]*\([a-z0-9-]*\).*/\1/p' "$submission_log" | tail -1)"
[[ -n "$scheduler_name" ]] || {
    echo "ERROR: rjob accepted the command but no scheduler name was returned" >&2
    exit 1
}

get_output="$(rjob get "$scheduler_name")"
grep -Fq "rjob $scheduler_name (showname=$RJOB_NAME):" <<<"$get_output" || {
    echo "ERROR: scheduler readback did not match the submitted job" >&2
    exit 1
}
grep -Eq "rjob $scheduler_name \(showname=$RJOB_NAME\): (Inqueue|Running)$" <<<"$get_output" || {
    echo "ERROR: submitted job was not accepted into the queue" >&2
    printf '%s\n' "$get_output" >&2
    exit 1
}
export C1_RECEIPT_PATH="$receipt_path"
export C1_SCHEDULER_NAME="$scheduler_name"
export C1_SHOW_NAME="$RJOB_NAME"
export C1_RUN_NAME="$FORMAL_RUN_NAME"
export C1_GIT_REVISION="$GIT_REVISION"
export C1_EXPECTED_HOST="$EXPECTED_HOST"
export C1_GPU_COUNT
export C1_RJOB_CPU="$RJOB_CPU"
export C1_RJOB_MEMORY_MIB="$RJOB_MEMORY_MIB"
export C1_RESUME_FROM_OSS="$RESUME_FROM_OSS"
export C1_RESUME_CHECKPOINT_REMOTE="$RESUME_CHECKPOINT_REMOTE"
export C1_RESUME_CHECKPOINT_NAME="$RESUME_CHECKPOINT_NAME"
export C1_RESUME_SOURCE_RUN_NAME="$RESUME_SOURCE_RUN_NAME"
python3 - <<'PY'
import datetime
import json
import os
from pathlib import Path

payload = {
    "status": "accepted",
    "accepted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scheduler_rjob_name": os.environ["C1_SCHEDULER_NAME"],
    "show_name": os.environ["C1_SHOW_NAME"],
    "run_id": os.environ["C1_RUN_NAME"],
    "git_revision": os.environ["C1_GIT_REVISION"],
    "gpu_count": int(os.environ["C1_GPU_COUNT"]),
    "cpu": int(os.environ["C1_RJOB_CPU"]),
    "memory_mib": int(os.environ["C1_RJOB_MEMORY_MIB"]),
    "expected_host": os.environ["C1_EXPECTED_HOST"],
    "resume_from_oss": os.environ["C1_RESUME_FROM_OSS"] == "1",
    "resume_checkpoint_remote": os.environ["C1_RESUME_CHECKPOINT_REMOTE"] or None,
    "resume_checkpoint_name": os.environ["C1_RESUME_CHECKPOINT_NAME"] or None,
    "resume_source_run_id": os.environ["C1_RESUME_SOURCE_RUN_NAME"] or None,
    "data_stream_replay_acknowledged": os.environ["C1_RESUME_FROM_OSS"] == "1",
    "oss_root": f"eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1/{os.environ['C1_RUN_NAME']}",
}
path = Path(os.environ["C1_RECEIPT_PATH"])
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

echo "C1_FORMAL_SUBMISSION_ACCEPTED scheduler_rjob=$scheduler_name receipt=$receipt_path"
