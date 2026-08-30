#!/usr/bin/env bash
# Resume the durable LIBERO S1 13k checkpoint on all eight H252 GPUs.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${EXPERIMENT_ID:=CWAM-LIBERO-S1-CACHED-CONT100K-8H200-H252-S42-R3-20260831}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${RJOB_NAME:=cosmos-s1-h252-r13}"
: "${NUM_GPUS:=8}"
: "${RESUME_STEP:=13000}"
: "${OSS_BASE_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826}"
: "${SOURCE_CHECKPOINT_URI:=$OSS_BASE_ROOT/CWAM-LIBERO-S1-CACHED-CONT100K-4H200-H253-S42-R2-20260828/checkpoints/iter_000013000}"
: "${OSS_RUN_ROOT:=$OSS_BASE_ROOT/$EXPERIMENT_ID}"
: "${CPU_COUNT:=128}"
: "${MEMORY_MB:=1400000}"
: "${LIBERO_LATENT_CACHE_OSS_ROOT:=$OSS_BASE_ROOT/wan-latent-cache-v1}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly H252_TAG="node/gpu-l-lg-cmc-h-h200-0252.host.h.pjlab.org.cn"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_libero_s1.sh"
readonly TOML_FILE="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_libero_all_s1_cont100k_nano.toml"

[[ -x "$RUNNER" && -f "$TOML_FILE" ]]
[[ "$NUM_GPUS" -eq 8 ]] || { echo "ERROR: H252 continuation requires eight GPUs" >&2; exit 1; }
[[ "$RESUME_STEP" -eq 13000 ]] || { echo "ERROR: continuation must reload durable step 13000" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]]

printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" EXPERIMENT_ID="$EXPERIMENT_ID" RUN_NAME="$RUN_NAME" \
    SOURCE_COMMIT="$SOURCE_COMMIT" KUBEBRAIN_RJOB_NAME="$RJOB_NAME" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    NUM_GPUS="$NUM_GPUS" ENABLE_OSS=1 RUN_PREFLIGHT_FIRST=0 TOML_FILE="$TOML_FILE" \
    SOURCE_CHECKPOINT_URI="$SOURCE_CHECKPOINT_URI" RESUME_STEP="$RESUME_STEP" \
    ALLOW_DATA_STREAM_REPLAY=1 LIBERO_LATENT_CACHE_MODE=required \
    LIBERO_LATENT_CACHE_OSS_ROOT="$LIBERO_LATENT_CACHE_OSS_ROOT" \
    ARCHIVE_START_STEP=15000 ARCHIVE_END_STEP=100000 ARCHIVE_INTERVAL=5000 \
    ARCHIVE_COMPACT_INTERVAL=0 EXTRA_TAIL_OVERRIDES=checkpoint.save_iter=5000 "$RUNNER"

rjob submit \
    --name="$RJOB_NAME" --priority=9 --group=eailabagent_gpu --charged-group=eailabagent_gpu \
    --private-machine=group --task-type=normal -P 1 --gpu="$NUM_GPUS" --cpu="$CPU_COUNT" \
    --memory="$MEMORY_MB" --share-host-shm=true --store-host-nvme=true \
    --positive-tags="$H252_TAG" \
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never \
    --termination-grace-period-seconds=300 --backoff_limit=1 \
    --mount "$PERSONAL_GPFS" "$PUBLIC_GPFS" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
