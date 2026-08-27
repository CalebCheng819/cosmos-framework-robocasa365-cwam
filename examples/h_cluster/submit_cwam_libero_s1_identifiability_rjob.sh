#!/usr/bin/env bash
# Submit the paired Cosmos-LIBERO routed-C protocol to one eight-H200 worker.

set -Eeuo pipefail
umask 077

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${PROTOCOL_SOURCE:?Set PROTOCOL_SOURCE to the empirical protocol snapshot.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${PROTOCOL_COMMIT:?Set PROTOCOL_COMMIT.}"
: "${EXPERIMENT_ID:=CWAM-LIBERO-S1-I500-IDENTIFIABILITY-R5-20260828}"
: "${RJOB_NAME:=cosmos-id-r5-0828}"
: "${NUM_GPUS:=8}"
: "${TARGET_NODE_TAG:=}"
: "${PRIVATE_MACHINE:=0}"
: "${PRIORITY:=4}"
: "${SOURCE_CHECKPOINT_URI:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826/CWAM-LIBERO-S1-4SUITE-4H200-H254-S42-R5-20260826/checkpoints/iter_000000500}"
: "${OSS_RUN_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam/identifiability-20260828/$EXPERIMENT_ID}"
: "${CPU_COUNT:=$((NUM_GPUS * 16))}"
: "${MEMORY_MB:=$((NUM_GPUS * 160000))}"
: "${TRIALS_PER_TASK:=2}"
: "${ACTION_HORIZON:=16}"
: "${MAX_STEPS:=0}"
: "${POLICY_SEED:=42}"
: "${SPLIT_SEED:=37}"
: "${JACOBIAN_PROJECTIONS:=4}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_libero_s1_identifiability.sh"

[[ -x "$RUNNER" ]]
[[ "$NUM_GPUS" -eq 8 ]]
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]]
[[ "$SOURCE_CHECKPOINT_URI" == eailab-hdd2:eailab/chengjuntao/* ]]
[[ "$OSS_RUN_ROOT" == eailab-hdd2:eailab/chengjuntao/* ]]

printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" PROTOCOL_SOURCE="$PROTOCOL_SOURCE" \
    SOURCE_COMMIT="$SOURCE_COMMIT" PROTOCOL_COMMIT="$PROTOCOL_COMMIT" \
    EXPERIMENT_ID="$EXPERIMENT_ID" KUBEBRAIN_RJOB_NAME="$RJOB_NAME" \
    SOURCE_CHECKPOINT_URI="$SOURCE_CHECKPOINT_URI" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    NUM_GPUS="$NUM_GPUS" TRIALS_PER_TASK="$TRIALS_PER_TASK" \
    ACTION_HORIZON="$ACTION_HORIZON" MAX_STEPS="$MAX_STEPS" POLICY_SEED="$POLICY_SEED" \
    SPLIT_SEED="$SPLIT_SEED" JACOBIAN_PROJECTIONS="$JACOBIAN_PROJECTIONS" "$RUNNER"

submit_args=(
    --name="$RJOB_NAME" --priority="$PRIORITY" --charged-group=eailabagent_gpu
    --task-type=normal -P 1 --gpu="$NUM_GPUS" --cpu="$CPU_COUNT"
    --memory="$MEMORY_MB" --share-host-shm=true --store-host-nvme=true
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never
    --termination-grace-period-seconds=300 --backoff_limit=1
    --mount "$PERSONAL_GPFS" "$PUBLIC_GPFS"
)
[[ -z "$TARGET_NODE_TAG" ]] || submit_args+=(--positive-tags="$TARGET_NODE_TAG")
[[ "$PRIVATE_MACHINE" != 1 ]] || submit_args+=(--private-machine=group)

rjob submit "${submit_args[@]}" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
