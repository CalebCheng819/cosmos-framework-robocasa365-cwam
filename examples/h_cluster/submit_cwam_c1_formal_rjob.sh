#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Submit one six-H200 C1 run. The worker stages inputs on NVMe, runs a two-step
# production-batch smoke, then starts the 20k-step formal training. Large
# outputs are uploaded to OSS by the worker controller.

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

[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] || {
    echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2
    exit 1
}
[[ "$FORMAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: unsupported FORMAL_RUN_NAME: $FORMAL_RUN_NAME" >&2
    exit 1
}
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
worker_command="$(cat <<EOF
set -Eeuo pipefail
export COSMOS_SOURCE=$q_source
export FORMAL_RUN_NAME=$q_run
export GIT_REVISION=$q_revision
export EXPECTED_HOST=$q_host
export CONTROL_ROOT=$q_control
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
bash \"\$COSMOS_SOURCE/examples/h_cluster/cwam_c1_prepare_assets.sh\"
bash \"\$COSMOS_SOURCE/examples/h_cluster/cwam_c1_smoke_then_train.sh\"
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
    --store-host-nvme True \
    --image "$RJOB_IMAGE" \
    --cpu 96 \
    --gpu 6 \
    --memory 1125000 \
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
export C1_RECEIPT_PATH="$receipt_path"
export C1_SCHEDULER_NAME="$scheduler_name"
export C1_SHOW_NAME="$RJOB_NAME"
export C1_RUN_NAME="$FORMAL_RUN_NAME"
export C1_GIT_REVISION="$GIT_REVISION"
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
    "gpu_count": 6,
    "expected_host": "gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn",
    "oss_root": f"eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1/{os.environ['C1_RUN_NAME']}",
}
path = Path(os.environ["C1_RECEIPT_PATH"])
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

echo "C1_FORMAL_SUBMISSION_ACCEPTED scheduler_rjob=$scheduler_name receipt=$receipt_path"
