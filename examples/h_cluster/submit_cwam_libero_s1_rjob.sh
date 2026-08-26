#!/usr/bin/env bash
# Submit a one-node CWAM LIBERO S1 train or runtime preflight to H cluster.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${EXPERIMENT_ID:=CWAM-LIBERO-S1-4SUITE-4H200-H254-S42-R1-20260826}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${RJOB_NAME:=cosmos-s1-h254-r1}"
: "${NUM_GPUS:=4}"
: "${ENABLE_OSS:=1}"
: "${OSS_RUN_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826/$EXPERIMENT_ID}"
: "${CPU_COUNT:=$((NUM_GPUS * 16))}"
: "${MEMORY_MB:=$((NUM_GPUS * 180000))}"
: "${EXTRA_TAIL_OVERRIDES:=}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly H254_TAG="node/gpu-l-lg-cmc-h-h200-0254.host.h.pjlab.org.cn"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_libero_s1.sh"

[[ -x "$RUNNER" ]]
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]]
printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" EXPERIMENT_ID="$EXPERIMENT_ID" RUN_NAME="$RUN_NAME" \
    SOURCE_COMMIT="$SOURCE_COMMIT" KUBEBRAIN_RJOB_NAME="$RJOB_NAME" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    NUM_GPUS="$NUM_GPUS" ENABLE_OSS="$ENABLE_OSS" EXTRA_TAIL_OVERRIDES="$EXTRA_TAIL_OVERRIDES" "$RUNNER"

rjob submit \
    --name="$RJOB_NAME" --priority=9 --group=eailabagent_gpu --charged-group=eailabagent_gpu \
    --private-machine=group --task-type=normal -P 1 --gpu="$NUM_GPUS" --cpu="$CPU_COUNT" \
    --memory="$MEMORY_MB" --share-host-shm=true --store-host-nvme=true \
    --positive-tags="$H254_TAG" \
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never \
    --termination-grace-period-seconds=300 --backoff_limit=1 \
    --mount "$PERSONAL_GPFS" "$PUBLIC_GPFS" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
