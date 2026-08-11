#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Long-lived command controller for the single-node B2 allocation. Commands
# are executed in lexical order from CONTROL_ROOT/queue and retain their logs
# and exit codes under CONTROL_ROOT/results.

set -Eeuo pipefail

: "${CONTROL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-b2-20260811/control}"
: "${EXPECTED_HOST:=gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"

QUEUE_DIR="$CONTROL_ROOT/queue"
RESULT_DIR="$CONTROL_ROOT/results"
mkdir -p "$QUEUE_DIR" "$RESULT_DIR"

exec > >(tee -a "$RESULT_DIR/controller.log") 2>&1

echo "CONTROLLER_STARTED=$(date -Is)"
echo "HOSTNAME=$(hostname -f)"
echo "EXPECTED_HOST=$EXPECTED_HOST"
[[ "$(hostname -f)" == "$EXPECTED_HOST" ]] || {
    echo "ERROR: B2 allocation landed on an unexpected node" >&2
    exit 1
}

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "GPU_COUNT=$GPU_COUNT"
[[ "$GPU_COUNT" -eq 8 ]] || {
    echo "ERROR: B2 requires exactly eight visible GPUs" >&2
    exit 1
}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
df -hT /nvme /dev/shm /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs 2>&1 || true
find /usr /opt -maxdepth 5 -type f -name 'python3.13' -print 2>/dev/null | head -20 || true
touch "$RESULT_DIR/controller.ready"

while true; do
    found_command=0
    shopt -s nullglob
    commands=("$QUEUE_DIR"/*.sh)
    shopt -u nullglob
    for command_path in "${commands[@]}"; do
        command_name="$(basename "$command_path" .sh)"
        done_path="$RESULT_DIR/$command_name.done"
        [[ -e "$done_path" ]] && continue
        found_command=1
        log_path="$RESULT_DIR/$command_name.log"
        rc_path="$RESULT_DIR/$command_name.rc"
        started_path="$RESULT_DIR/$command_name.started"
        date -Is > "$started_path"
        echo "COMMAND_STARTED=$command_name $(cat "$started_path")"
        set +e
        bash "$command_path" > >(tee -a "$log_path") 2>&1
        command_rc=$?
        set -e
        printf '%s\n' "$command_rc" > "$rc_path"
        date -Is > "$done_path"
        echo "COMMAND_FINISHED=$command_name rc=$command_rc $(cat "$done_path")"
    done
    date -Is > "$RESULT_DIR/controller.heartbeat"
    if [[ "$found_command" -eq 0 ]]; then
        sleep 5
    fi
done
