#!/usr/bin/env bash
# Submit the strict R6-to-R7 four-GPU continuation to H200-0299.

set -Eeuo pipefail
umask 077

required=(
    EVAL_CONTROL_SOURCE COSMOS_SOURCE COSMOS_ENV ROBOCASA_ENV DEMIAN_REPO
    ROBOCASA_REPO ROBOSUITE_REPO SERVING_CHECKPOINT_PATH TRAINING_CONFIG
    ACTION_STATS WAN2PT2_VAE QWEN_TOKENIZER BASELINE_RECORD EVAL_OUTPUT_ROOT
    EVAL_EXPERIMENT_ID EVAL_LOGICAL_RUN_ID EVAL_RUN_NAME MODEL_LABEL
    MODEL_SOURCE_REVISION CONTROL_SOURCE_REVISION EVALUATOR_SOURCE_REVISION
    EXPECTED_EMA_KEY_COUNT EVAL_EGL_VENDOR_JSON EXPECTED_NODE_TAG EVAL_GPU_COUNT
    RJOB_NAME EVAL_R4_SOURCE_ROOT EVAL_R4_SOURCE_RUN_NAME
    EVAL_R6_SOURCE_ROOT EVAL_R6_SOURCE_RUN_NAME
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "$name" >&2; exit 2; }
done
readonly REQUIRED_NODE_TAG=node/gpu-l-lg-cmc-h-h200-0299.host.h.pjlab.org.cn
readonly RUNNER="$EVAL_CONTROL_SOURCE/examples/h_cluster/run_cwam_cosmos3_robocasa365_four_gpu_partial_resume_eval.sh"
readonly RJOB_PREDICT_ONLY="${RJOB_PREDICT_ONLY:-0}"
[[ "$EXPECTED_NODE_TAG" == "$REQUIRED_NODE_TAG" ]] || { echo 'ERROR: job must target H200-0299' >&2; exit 2; }
[[ "$EVAL_GPU_COUNT" == 4 ]] || { echo 'ERROR: job requires four GPUs' >&2; exit 2; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] || { echo 'ERROR: invalid RJOB_NAME' >&2; exit 2; }
[[ "$RJOB_PREDICT_ONLY" == 0 || "$RJOB_PREDICT_ONLY" == 1 ]] || { echo 'ERROR: invalid RJOB_PREDICT_ONLY' >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo 'ERROR: runner missing' >&2; exit 2; }
[[ "$(git -c safe.directory="$EVAL_CONTROL_SOURCE" -C "$EVAL_CONTROL_SOURCE" rev-parse HEAD)" == "$CONTROL_SOURCE_REVISION" ]] || { echo 'ERROR: control revision mismatch' >&2; exit 2; }
[[ "$(git -c safe.directory="$COSMOS_SOURCE" -C "$COSMOS_SOURCE" rev-parse HEAD)" == "$MODEL_SOURCE_REVISION" ]] || { echo 'ERROR: model revision mismatch' >&2; exit 2; }
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || { echo 'ERROR: output root exists' >&2; exit 2; }

worker_script=$'set -Eeuo pipefail\n'
worker_script+=$'unset BASH_ENV ENV PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD LD_AUDIT __EGL_VENDOR_LIBRARY_DIRS LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE LIBGL_DRIVERS_PATH\n'
worker_script+=$'export __GLX_VENDOR_LIBRARY_NAME=nvidia\nexport NCCL_IB_DISABLE=1\n'
for name in "${required[@]}"; do
    printf -v quoted '%q' "${!name}"
    worker_script+="export $name=$quoted"$'\n'
done
worker_script+="exec /usr/bin/bash $(printf '%q' "$RUNNER")"$'\n'

submit_args=(
    submit --name "$RJOB_NAME" --charged-group eailabagent_gpu
    --priority 9 --task-type normal --replicas 1 --restart-policy never
    --termination-grace-period-seconds 60 --backoff_limit 1 --enable-sshd
    --image registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7
    --cpu 48 --gpu 4 --memory 500000 --positive-tags "$EXPECTED_NODE_TAG"
    --store-host-nvme --mount
    gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs
    gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then submit_args+=(--predict-only true); fi
printf '%s rjob=%s node=%s gpu_count=4 output=%s\n' \
    "$([[ "$RJOB_PREDICT_ONLY" == 1 ]] && echo PREDICTING || echo SUBMITTING)" \
    "$RJOB_NAME" "$EXPECTED_NODE_TAG" "$EVAL_OUTPUT_ROOT"
/usr/local/bin/rjob "${submit_args[@]}" -- \
    /usr/bin/env -u BASH_ENV -u ENV /usr/bin/bash --noprofile --norc -c "$worker_script"
