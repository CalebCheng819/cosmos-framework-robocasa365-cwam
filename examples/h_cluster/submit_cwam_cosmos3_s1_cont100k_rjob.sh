#!/usr/bin/env bash
# Submit the 8xH200 E35-S1 continuation specifically to private node 0252.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable GPFS code snapshot.}"
: "${EXPERIMENT_ID:=CWAM-E35-S1-CONT-10K-100K-8H200-S42-R1-20260820}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the Git commit in COSMOS_SOURCE.}"
: "${RJOB_NAME:=cwam-s1-cont100k-s42-r1-h252-0820}"
: "${OSS_RUN_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam-cosmos3/e35-s1-cont100k/$EXPERIMENT_ID}"
: "${SOURCE_CHECKPOINT_URI:=eailab-hdd2:eailab/chengjuntao/migrate-20260805_142643/outputs/cwam-cosmos3-formal/cosmos3_action_robocasa365/action_sft/CWAM-E35-S1-SCRATCH-ARR-8GPU-S42-R3-20260731/checkpoints/iter_000010000}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly NODE_TAG="node/gpu-l-lg-cmc-h-h200-0252.host.h.pjlab.org.cn"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_s1_cont100k.sh"

[[ -x "$RUNNER" ]] || { echo "ERROR: runner is absent or not executable: $RUNNER" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]] || {
    echo "ERROR: invalid rjob name: $RJOB_NAME" >&2
    exit 1
}

printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" EXPERIMENT_ID="$EXPERIMENT_ID" RUN_NAME="$RUN_NAME" \
    SOURCE_COMMIT="$SOURCE_COMMIT" KUBEBRAIN_RJOB_NAME="$RJOB_NAME" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    SOURCE_CHECKPOINT_URI="$SOURCE_CHECKPOINT_URI" "$RUNNER"
WORKER_COMMAND="set -Eeuo pipefail
printf 'WORKER_BOOTSTRAP_START %s\\n' \"\$(date -Is)\"
exec $COMMAND"

rjob submit \
    --name="$RJOB_NAME" \
    --priority=9 \
    --group=eailabagent_gpu \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --task-type=normal \
    -P 1 \
    --gpu=8 \
    --cpu=128 \
    --memory=1500000 \
    --share-host-shm=true \
    --store-host-nvme=true \
    --positive-tags="$NODE_TAG" \
    --image="$IMAGE" \
    --image-pull-policy=IfNotPresent \
    --restart-policy=never \
    --termination-grace-period-seconds=300 \
    --backoff_limit=1 \
    --mount \
    "$PERSONAL_GPFS" \
    "$PUBLIC_GPFS" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "$WORKER_COMMAND"
