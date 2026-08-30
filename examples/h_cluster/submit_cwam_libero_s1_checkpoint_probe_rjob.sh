#!/usr/bin/env bash
# Resume the recovered 13k state for one optimizer step and archive a new-format checkpoint.

set -Eeuo pipefail
umask 022

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable source snapshot.}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT.}"
: "${EXPERIMENT_ID:=CWAM-LIBERO-S1-CKPTSAFE-PROBE-4H200-H253-S42-R1-20260831}"
: "${RUN_NAME:=$EXPERIMENT_ID}"
: "${RJOB_NAME:=cosmos-s1-ckpt-probe-h253-r1}"
: "${NUM_GPUS:=4}"
: "${RESUME_STEP:=13000}"
: "${TARGET_STEP:=13001}"
: "${SOURCE_EXPERIMENT_ID:=CWAM-LIBERO-S1-CACHED-CONT100K-4H200-H253-S42-R2-20260828}"
: "${OSS_BASE_ROOT:=eailab-hdd2:eailab/chengjuntao/cwam/libero-s1-4suite-20260826}"
: "${SOURCE_CHECKPOINT_URI:=$OSS_BASE_ROOT/$SOURCE_EXPERIMENT_ID/checkpoints/iter_$(printf '%09d' "$RESUME_STEP")}"
: "${OSS_RUN_ROOT:=$OSS_BASE_ROOT/$EXPERIMENT_ID}"
: "${CPU_COUNT:=$((NUM_GPUS * 16))}"
: "${MEMORY_MB:=$((NUM_GPUS * 180000))}"
: "${LIBERO_LATENT_CACHE_OSS_ROOT:=$OSS_BASE_ROOT/wan-latent-cache-v1}"

readonly IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PERSONAL_GPFS="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_GPFS="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly H253_TAG="node/gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn"
readonly RUNNER="$COSMOS_SOURCE/examples/h_cluster/run_cwam_libero_s1.sh"
readonly TOML_FILE="$COSMOS_SOURCE/examples/toml/sft_config/action_policy_libero_all_s1_cont100k_nano.toml"

[[ -x "$RUNNER" && -f "$TOML_FILE" ]]
[[ "$NUM_GPUS" -eq 4 ]] || { echo "ERROR: checkpoint topology requires four GPUs" >&2; exit 1; }
[[ "$RESUME_STEP" -eq 13000 && "$TARGET_STEP" -eq 13001 ]] || {
    echo "ERROR: this probe is fixed to recovered step 13000 and target step 13001" >&2
    exit 1
}
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#RJOB_NAME} -le 50 ]]

printf -v COMMAND '%q ' env \
    COSMOS_SOURCE="$COSMOS_SOURCE" EXPERIMENT_ID="$EXPERIMENT_ID" RUN_NAME="$RUN_NAME" \
    SOURCE_COMMIT="$SOURCE_COMMIT" KUBEBRAIN_RJOB_NAME="$RJOB_NAME" OSS_RUN_ROOT="$OSS_RUN_ROOT" \
    NUM_GPUS="$NUM_GPUS" ENABLE_OSS=1 RUN_PREFLIGHT_FIRST=0 TOML_FILE="$TOML_FILE" \
    SOURCE_CHECKPOINT_URI="$SOURCE_CHECKPOINT_URI" RESUME_STEP="$RESUME_STEP" \
    ALLOW_DATA_STREAM_REPLAY=1 LIBERO_LATENT_CACHE_MODE=required \
    LIBERO_LATENT_CACHE_OSS_ROOT="$LIBERO_LATENT_CACHE_OSS_ROOT" \
    EXTRA_TAIL_OVERRIDES="trainer.max_iter=$TARGET_STEP trainer.logging_iter=1 checkpoint.save_iter=1 trainer.run_validation=false" \
    ARCHIVE_START_STEP="$TARGET_STEP" ARCHIVE_END_STEP="$TARGET_STEP" ARCHIVE_INTERVAL=1 \
    ARCHIVE_COMPACT_INTERVAL=0 ARCHIVER_STABILITY_SECONDS=15 "$RUNNER"

rjob submit \
    --name="$RJOB_NAME" --priority=9 --group=eailabagent_gpu --charged-group=eailabagent_gpu \
    --private-machine=group --task-type=normal -P 1 --gpu="$NUM_GPUS" --cpu="$CPU_COUNT" \
    --memory="$MEMORY_MB" --share-host-shm=true --store-host-nvme=true \
    --positive-tags="$H253_TAG" \
    --image="$IMAGE" --image-pull-policy=IfNotPresent --restart-policy=never \
    --termination-grace-period-seconds=300 --backoff_limit=1 \
    --mount "$PERSONAL_GPFS" "$PUBLIC_GPFS" \
    -- /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c "set -Eeuo pipefail; exec $COMMAND"
