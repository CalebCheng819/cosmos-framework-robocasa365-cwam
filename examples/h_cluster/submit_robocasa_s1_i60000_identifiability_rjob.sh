#!/usr/bin/env bash
# Submit the fixed RoboCasa S1 iter-60000 paired identifiability protocol.

set -Eeuo pipefail
umask 077

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${PROTOCOL_SOURCE:?Set PROTOCOL_SOURCE to the immutable protocol snapshot.}"
: "${DEMIAN_SOURCE:?Set DEMIAN_SOURCE.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${PROTOCOL_COMMIT:?Set PROTOCOL_COMMIT.}"
: "${EXPERIMENT_ID:=CWAM-ROB365-S1-I60000-IDENTIFIABILITY-R5-20260828}"
: "${RJOB_NAME:=rob365-id-r5-0828}"
: "${NUM_GPUS:=8}"
: "${TARGET_NODE_TAG:=}"
: "${PRIVATE_MACHINE:=0}"
: "${PRIORITY:=4}"
: "${SOURCE_CHECKPOINT_URI:=eailab-hdd2:eailab/chengjuntao/cwam/s1/serving/CWAM-E35-S1-I60000-EMA-BF16-20260826}"
: "${OSS_RUN_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam/identifiability-20260828/$EXPERIMENT_ID}"
: "${EPISODES_PER_TASK:=2}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_robocasa_s1_i60000_identifiability.sh"

[[ -x "$RUNNER" ]]
[[ "$NUM_GPUS" -eq 8 ]]
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]]
[[ "$SOURCE_CHECKPOINT_URI" == eailab-hdd2:eailab/chengjuntao/* ]]
[[ "$OSS_RUN_ROOT" == eailab-hdd2:eailab/chengjuntao/* ]]

printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" PROTOCOL_SOURCE="$PROTOCOL_SOURCE" DEMIAN_SOURCE="$DEMIAN_SOURCE" \
    SOURCE_COMMIT="$SOURCE_COMMIT" PROTOCOL_COMMIT="$PROTOCOL_COMMIT" \
    EXPERIMENT_ID="$EXPERIMENT_ID" KUBEBRAIN_RJOB_NAME="$RJOB_NAME" \
    SOURCE_CHECKPOINT_URI="$SOURCE_CHECKPOINT_URI" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    EXPECTED_NODE_TAG="${TARGET_NODE_TAG:-scheduler-selected}" NUM_GPUS="$NUM_GPUS" \
    EPISODES_PER_TASK="$EPISODES_PER_TASK" \
    "$RUNNER"

submit_args=(
    --name="$RJOB_NAME" --priority="$PRIORITY" --charged-group=eailabagent_gpu
    --task-type=normal -P 1 --gpu="$NUM_GPUS" --cpu=128 --memory=1280000
    --share-host-shm=true --store-host-nvme=true
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never
    --termination-grace-period-seconds=300 --backoff_limit=1
    --mount "$PERSONAL_GPFS" "$PUBLIC_GPFS"
)
[[ -z "$TARGET_NODE_TAG" ]] || submit_args+=(--positive-tags="$TARGET_NODE_TAG")
[[ "$PRIVATE_MACHINE" != 1 ]] || submit_args+=(--private-machine=group)

rjob submit "${submit_args[@]}" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
