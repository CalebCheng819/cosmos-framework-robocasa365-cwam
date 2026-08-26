#!/usr/bin/env bash
# Submit a zero-GPU H254 helper that uploads the R5 step-500 DCP from host NVMe.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${SOURCE_COMMIT:=07b6b60}"
: "${EXPERIMENT_ID:=CWAM-LIBERO-S1-4SUITE-4H200-H254-S42-R5-20260826}"
: "${SOURCE_RJOB_NAME:=cosmos-s1-h254-r5}"
: "${RJOB_NAME:=cosmos-archive-h254-r5}"
: "${TARGET_STEP:=500}"
: "${OSS_RUN_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826/$EXPERIMENT_ID}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly H254_TAG="node/gpu-l-lg-cmc-h-h200-0254.host.h.pjlab.org.cn"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/archive_cwam_libero_checkpoint_on_node.sh"
readonly SOURCE_CHECKPOINTS_DIR="/nvme/cwam-libero-s1/$EXPERIMENT_ID/$SOURCE_RJOB_NAME/output/cwam_action_libero/action_sft/$EXPERIMENT_ID/checkpoints"

[[ -x "$RUNNER" ]]
[[ "$TARGET_STEP" -eq 500 ]] || { echo "ERROR: this rescue is scoped to R5 step 500" >&2; exit 1; }

printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" EXPERIMENT_ID="$EXPERIMENT_ID" SOURCE_COMMIT="$SOURCE_COMMIT" \
    SOURCE_CHECKPOINTS_DIR="$SOURCE_CHECKPOINTS_DIR" TARGET_STEP="$TARGET_STEP" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    KUBEBRAIN_RJOB_NAME="$RJOB_NAME" "$RUNNER"

rjob submit \
    --name="$RJOB_NAME" --priority=9 --group=eailabagent_gpu --charged-group=eailabagent_gpu \
    --private-machine=group --task-type=normal -P 1 --gpu=0 --cpu=8 --memory=32000 \
    --share-host-shm=true --store-host-nvme=true --positive-tags="$H254_TAG" \
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never \
    --termination-grace-period-seconds=300 --backoff_limit=1 --mount "$PERSONAL_GPFS" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
