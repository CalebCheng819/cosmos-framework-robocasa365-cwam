#!/usr/bin/env bash
# Run fixed-R13 diagnostics first, then start a fresh CWAM-v2 training run.

set -Eeuo pipefail
umask 077

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable CWAM-v2 source snapshot.}"
: "${GIT_REVISION:?Set GIT_REVISION to the snapshot Git revision.}"
: "${FORMAL_RUN_NAME:=CWAM-COSMOS3-V2-20260818-R1}"
: "${DIAG_RUN_ID:=CWAM-COSMOS3-C1-DIAG-20260818-R1}"
: "${EXPECTED_HOST:=gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn}"
: "${CONTROL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-v2-20260818/control}"

readonly PERSONAL_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly SECRET_ROOT="/dev/shm/cwam-v2-rclone"
readonly RCLONE_CONFIG="$SECRET_ROOT/rclone.conf"

[[ "$(hostname -f)" == "$EXPECTED_HOST" ]] || {
    echo "ERROR: expected $EXPECTED_HOST, got $(hostname -f)" >&2
    exit 1
}
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8 ]] || {
    echo "ERROR: the chained diagnostic/training run requires exactly eight visible GPUs" >&2
    exit 1
}

export COSMOS_SOURCE GIT_REVISION FORMAL_RUN_NAME DIAG_RUN_ID EXPECTED_HOST CONTROL_ROOT
export WORLD_SIZE=8
export C1_INPUT_ROOT=/nvme/cwam-c1/inputs
export SECRET_ROOT
export RCLONE_CONFIG
export PATH="$PERSONAL_ROOT/tools:$PATH"
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1

bash "$COSMOS_SOURCE/examples/h_cluster/cwam_c1_prepare_assets.sh"

RUN_ID="$DIAG_RUN_ID" \
OSS_ROOT="eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1-DIAG/$DIAG_RUN_ID" \
    bash "$COSMOS_SOURCE/examples/h_cluster/run_cwam_c1_diagnostics_h8.sh"

bash "$COSMOS_SOURCE/examples/h_cluster/cwam_v2_smoke_then_train.sh"

