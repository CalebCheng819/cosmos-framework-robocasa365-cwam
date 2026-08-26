#!/usr/bin/env bash
# Submit the three-H200 LIBERO Wan VAE cache build to H299.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the committed Cosmos source.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${EXPERIMENT_ID:=CWAM-COSMOS3-LIBERO-WAN-CACHE-20260826-001}"
: "${RJOB_NAME:=cosmos-cache-h299-r1}"
: "${NUM_GPUS:=3}"
: "${CACHE_REMOTE:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826/wan-latent-cache-v1}"
: "${CPU_COUNT:=48}"
: "${MEMORY_MB:=500000}"
: "${VAE_BATCH_SIZE:=8}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly H299_TAG="node/gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_libero_wan_latent_cache.sh"

[[ -x "$RUNNER" ]]
[[ "$NUM_GPUS" == 3 ]]
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]]
printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" EXPERIMENT_ID="$EXPERIMENT_ID" SOURCE_COMMIT="$SOURCE_COMMIT" \
    KUBEBRAIN_RJOB_NAME="$RJOB_NAME" NUM_GPUS="$NUM_GPUS" CACHE_REMOTE="$CACHE_REMOTE" \
    VAE_BATCH_SIZE="$VAE_BATCH_SIZE" "$RUNNER"

rjob submit \
    --name="$RJOB_NAME" --priority=9 --group=eailabagent_gpu --charged-group=eailabagent_gpu \
    --private-machine=group --task-type=normal -P 1 --gpu="$NUM_GPUS" --cpu="$CPU_COUNT" \
    --memory="$MEMORY_MB" --share-host-shm=true --store-host-nvme=true \
    --positive-tags="$H299_TAG" \
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never \
    --termination-grace-period-seconds=300 --backoff_limit=1 \
    --mount "$PERSONAL_GPFS" "$PUBLIC_GPFS" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
