#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run the exact 17-task x 20-episode RoboCasa365 formal protocol on one
# Alibaba PAI DSW with four local 4090 GPUs. The policy is one FSDP4 torchrun
# service on rank 0's endpoint; task partitioning remains eight logical shards.
# Workflow iteration: ITER-2026-07-31-CWAM-DSW974-FSDP4-EVAL-ADAPTER.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable Cosmos source snapshot}"
: "${COSMOS_ENV:?Set COSMOS_ENV to the pinned Cosmos environment}"
: "${ROBOCASA_ENV:?Set ROBOCASA_ENV to the pinned RoboCasa environment}"
: "${DEMIAN_REPO:?Set DEMIAN_REPO to the pinned DeMiAn checkout}"
: "${ROBOCASA_REPO:?Set ROBOCASA_REPO to the pinned RoboCasa checkout}"
: "${ROBOSUITE_REPO:?Set ROBOSUITE_REPO to the pinned robosuite checkout}"
: "${SERVING_CHECKPOINT_PATH:?Set SERVING_CHECKPOINT_PATH to the consolidated model directory}"
: "${H_SERVING_CHECKPOINT_MANIFEST:?Set the copied H serving checkpoint manifest}"
: "${EMA_EXPORT_RECEIPT:?Set the copied EMA BF16 export receipt}"
: "${EMA_KEY_MANIFEST:?Set the copied EMA BF16 key manifest}"
: "${EMA_VALIDATION:?Set the copied EMA BF16 validation receipt}"
: "${EMA_SOURCE_DCP_FULL_RECEIPT:?Set the copied independent source-DCP full verification receipt}"
: "${EMA_TRANSFER_MANIFEST:?Set the downloaded Ceph EMA transfer manifest}"
: "${CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT:?Set the DSW atomic checkpoint publication receipt}"
: "${TRAINING_CONFIG:?Set TRAINING_CONFIG to the exact training config}"
: "${ACTION_STATS:?Set ACTION_STATS to the exact RoboCasa365 action stats}"
: "${WAN2PT2_VAE:?Set WAN2PT2_VAE to the exact Wan2.2 VAE file}"
: "${QWEN_TOKENIZER:?Set QWEN_TOKENIZER to the exact local Qwen tokenizer directory}"
: "${QWEN_TOKENIZER_MANIFEST:?Set QWEN_TOKENIZER_MANIFEST to its external manifest}"
: "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to the immutable source manifest}"
: "${COSMOS_ENV_READY_MARKER:?Set the DSW-native Cosmos .cwam_env_ready.json}"
: "${ROBOCASA_ENV_READY_MARKER:?Set the DSW-native RoboCasa .cwam_env_ready.json}"
: "${ROBOCASA_TORCHCODEC_RELAY_RECEIPT:?Set the verified plain torchcodec relay receipt}"
: "${ASSET_ENV_CLOSURE_MANIFEST:?Set the copied asset/environment closure manifest}"
: "${ASSET_ENV_CLOSURE_H_VERIFICATION:?Set the copied H closure verification receipt}"
: "${ROBOCASA_ASSET_ROOT:?Set the installed RoboCasa root carrying .cwam_provenance}"
: "${EVAL_OUTPUT_ROOT:?Set EVAL_OUTPUT_ROOT to the exact new CPFS run directory}"
: "${DSW_CPFS_MOUNT:?Set DSW_CPFS_MOUNT to the exact CPFS mount target}"
: "${DSW_CPFS_OUTPUT_PARENT:?Set DSW_CPFS_OUTPUT_PARENT to the allowed CPFS runs parent}"
: "${DSW_LOCAL_SCRATCH_PARENT:?Set DSW_LOCAL_SCRATCH_PARENT to an existing local scratch directory}"
: "${EVAL_EXPERIMENT_ID:?Set EVAL_EXPERIMENT_ID}"
: "${EVAL_LOGICAL_RUN_ID:?Set EVAL_LOGICAL_RUN_ID}"
: "${EVAL_RUN_NAME:?Set EVAL_RUN_NAME}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?Set EXPECTED_SOURCE_MANIFEST_SHA256}"
: "${EXPECTED_H_SERVING_CHECKPOINT_MANIFEST_SHA256:?Set the H serving manifest SHA256}"
: "${EXPECTED_SERVING_CHECKPOINT_TREE_SHA256:?Set the byte-identical H/DSW model tree SHA256}"
: "${EXPECTED_EMA_EXPORT_RECEIPT_SHA256:?Set the EMA export receipt SHA256}"
: "${EXPECTED_EMA_KEY_MANIFEST_SHA256:?Set the EMA key manifest SHA256}"
: "${EXPECTED_EMA_VALIDATION_SHA256:?Set the EMA validation SHA256}"
: "${EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256:?Set the source-DCP full verification receipt SHA256}"
: "${EXPECTED_EMA_TRANSFER_MANIFEST_SHA256:?Set the Ceph EMA transfer manifest SHA256}"
: "${EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256:?Set the DSW atomic checkpoint publication receipt SHA256}"
: "${EXPECTED_TRAINING_CONFIG_SHA256:?Set EXPECTED_TRAINING_CONFIG_SHA256}"
: "${EXPECTED_ACTION_STATS_SHA256:?Set EXPECTED_ACTION_STATS_SHA256}"
: "${EXPECTED_WAN2PT2_VAE_SHA256:?Set EXPECTED_WAN2PT2_VAE_SHA256}"
: "${EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256:?Set EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256}"
: "${EXPECTED_COSMOS_ENV_READY_MARKER_SHA256:?Set the DSW-native Cosmos ready-marker SHA256}"
: "${EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256:?Set the DSW-native RoboCasa ready-marker SHA256}"
: "${EXPECTED_ROBOCASA_TORCHCODEC_RELAY_RECEIPT_SHA256:?Set the plain torchcodec relay receipt SHA256}"
: "${EGL_MAPPING_RECEIPT:?Set the verified four-GPU EGL mapping receipt}"
: "${EXPECTED_EGL_MAPPING_RECEIPT_SHA256:?Set the four-GPU EGL mapping receipt SHA256}"
: "${EXPECTED_EGL_VENDOR_JSON_SHA256:?Set the NVIDIA EGL vendor JSON SHA256}"
: "${EXPECTED_COSMOS_ENV_SOURCE_COMMIT:?Set the environment-spec Cosmos commit}"
: "${EXPECTED_ASSET_ENV_CLOSURE_MANIFEST_SHA256:?Set the closure manifest SHA256}"
: "${EXPECTED_ASSET_ENV_CLOSURE_H_VERIFICATION_SHA256:?Set the H closure verification SHA256}"
: "${EXPECTED_DEMIAN_COMMIT:?Set EXPECTED_DEMIAN_COMMIT}"
: "${EXPECTED_ROBOCASA_COMMIT:?Set EXPECTED_ROBOCASA_COMMIT}"
: "${EXPECTED_ROBOSUITE_COMMIT:?Set EXPECTED_ROBOSUITE_COMMIT}"
: "${EVAL_EGL_VENDOR_JSON:?Set EVAL_EGL_VENDOR_JSON to the NVIDIA EGL vendor JSON}"

readonly PHYSICAL_GPU_COUNT=4
readonly PHYSICAL_SERVER_COUNT=1
readonly LOGICAL_NUM_SHARDS=8
readonly TASK_COUNT=17
readonly EPISODES_PER_TASK=20
readonly EXPECTED_EPISODE_COUNT=340
readonly BASE_SEED=42
readonly MAX_STEPS=400
readonly OPEN_LOOP_HORIZON=12
readonly POLICY_SAMPLER=unipc
readonly POLICY_SEED=42
readonly POLICY_GUIDANCE=1.0
readonly POLICY_NUM_STEPS=30
readonly BASE_PORT=16000
readonly SERVER_READY_TIMEOUT_SECONDS=1800
readonly SERVER_SHUTDOWN_TIMEOUT_SECONDS=60
readonly SERVER_EXIT_TIMEOUT_SECONDS=180
# Shard 0 owns 3 tasks x 20 episodes.  At the 400-step limit with a
# 12-action open-loop horizon, it can require 60 x ceil(400 / 12) policy
# requests.  The observed 4x4090 request time is about 41 seconds, so the
# valid worst case is roughly 23 hours; retain substantial watchdog headroom.
readonly EVALUATOR_TIMEOUT_SECONDS=129600
readonly ONE_EPISODE_GATE_TIMEOUT_SECONDS=3600
readonly ONE_EPISODE_GATE_TASK=CloseBlenderLid
readonly EXPECTED_CPU_COUNT=128
readonly EXPECTED_MEMORY_GIB=400
readonly SERVER_MODULE="cosmos_framework.scripts.action_policy_server_robocasa365"
readonly SERVER_MODULE_REL="cosmos_framework/scripts/action_policy_server_robocasa365.py"
readonly EVALUATOR_REL="demian_wam/eval_utils/run_robocasa_eval.py"
readonly TASK_REGISTRY_REL="demian_wam/eval_utils/robocasa_task_list.py"
readonly CHECKPOINT_HELPER_REL="examples/h_cluster/cwam_serving_checkpoint_manifest.py"
readonly SOURCE_PROVENANCE_HELPER_REL="examples/aliyun_4090/cwam_dsw_input_provenance.py"
readonly DSW_EVIDENCE_HELPER_REL="examples/aliyun_4090/cwam_robocasa365_dsw_evidence.py"
readonly ASSET_VERIFY_HELPER_REL="examples/h_cluster/cwam_robocasa_asset_manifest.py"
readonly CLEANUP_GRACE_SECONDS=30
readonly SETSID=/usr/bin/setsid

readonly FORMAL_INPUT_ROOT="/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs"
readonly FORMAL_NVME_INPUT_ROOT="/tmp/cwam_cosmos3_robocasa365_eval_nvme/inputs"
readonly FORMAL_COSMOS_ENV="$FORMAL_NVME_INPUT_ROOT/envs/cosmos-framework-09f23119-cu128-train"
readonly FORMAL_ROBOCASA_REPO="$FORMAL_NVME_INPUT_ROOT/repos/robocasa-29f7ce8814c1"
readonly FORMAL_CHECKPOINT_ARTIFACT_ROOT="$FORMAL_INPUT_ROOT/checkpoints/R14-iter_000010000-serving-ema-bf16-v1"
readonly FORMAL_CHECKPOINT_PUBLICATION_RECEIPT="/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/receipts/ema-checkpoint-atomic-publication.json"
readonly PINNED_COSMOS_ENV_SOURCE_COMMIT="09f23119ea92c707207bba55565e7a09d16896a2"
readonly PINNED_DEMIAN_COMMIT="2600ece5bf0eeacf010a95f084dcda2de6d30d59"
readonly PINNED_ROBOCASA_COMMIT="29f7ce8814c1547f5af762a0997fbd4b64848dd7"
readonly PINNED_ROBOSUITE_COMMIT="232ce7d4a6ed89c949a9aba024a05c8c32fdd08b"
readonly PINNED_H_SERVING_MANIFEST_SHA256="7bcb9494b9f00b93dbad7d2b533032d3c8086e134986ae70f1c03abb1858f500"
readonly PINNED_SERVING_TREE_SHA256="77d2a84a35e81438c87a6e31bedc09299b294fcbe9841f4e943b985bbcb6f846"
readonly PINNED_EMA_EXPORT_SHA256="26bc5f658aa71a2ac8018dabb53574c9917171bb69bff9e4091d5621961c4200"
readonly PINNED_EMA_KEYS_SHA256="e2edc73c06b628bb9294b157af1f6d3be3ec11c626713a709c9f64667ca2d8fa"
readonly PINNED_EMA_VALIDATION_SHA256="887d7a962bda3f37f105d210945ec21bebed4becb0c42677b8c4ead256b673dd"
readonly PINNED_SOURCE_DCP_FULL_SHA256="1f9352ff4d97116d4861e2fec6dc5522a738a5ac58d5e15bf0025f3b1f45f646"
readonly PINNED_EMA_TRANSFER_SHA256="1f00b9d45f1d7ca2e6691fcd117d9da42c3444f9e517f5f3091362e1b3992f9b"
readonly PINNED_TRAINING_CONFIG_SHA256="c8269c4f12c99ac98e71e85334933d9749cd66a494960d6afe6e91368134caaa"
readonly PINNED_ACTION_STATS_SHA256="2b48eb8a42e0c9c7befc45034d152282dad328f36389c10cafdc5be2635b67df"
readonly PINNED_WAN2PT2_VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
readonly PINNED_QWEN_MANIFEST_SHA256="c71787b714a6876114b69d249e88bfa50de1262d33f46f8285dd00da729d4978"
readonly PINNED_CLOSURE_MANIFEST_SHA256="08b73f93a2ef191e2d82783e0a35689d1416a6d05e0384145e78551b3103f5bb"
readonly PINNED_CLOSURE_H_VERIFICATION_SHA256="60263e2de95efeeab89bc04b0b176331440a46ef8f027c8d708f8a9ecf739d45"
readonly PINNED_ROBOCASA_TORCHCODEC_RELAY_SHA256="1b16cc5419b7eb8558d5c0bd6dc4099493df629d50756ad3116b3c1da47f5ce4"
readonly PINNED_EGL_MAPPING_RECEIPT_SHA256="88c1b81ea58fae409a0578956ac9c8114effccba85d644bebe72fdc8aad1a066"
readonly PINNED_EGL_VENDOR_JSON_SHA256="9e6f14afaa7523370ad96e6675d3357441ee3985e60405cb3e0ced30b2ddaf76"

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

verify_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA256 is malformed"
    [[ -f "$path" && ! -L "$path" ]] || die "$label must be a regular non-symlink file: $path"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] ||
        die "$label SHA256 mismatch: expected=$expected actual=$actual path=$path"
}

canonical_existing_file() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || die "not a regular non-symlink file: $path"
    realpath -e -- "$path"
}

canonical_existing_dir() {
    local path="$1"
    [[ -d "$path" && ! -L "$path" ]] || die "not a directory or is a symlink: $path"
    realpath -e -- "$path"
}

verify_git_checkout() {
    local label="$1"
    local root="$2"
    local expected="$3"
    local actual
    [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die "$label expected commit is malformed"
    actual="$(git -c safe.directory="$root" -C "$root" rev-parse HEAD)"
    [[ "$actual" == "$expected" ]] ||
        die "$label commit mismatch: expected=$expected actual=$actual"
    [[ -z "$(git -c safe.directory="$root" -C "$root" status --porcelain --untracked-files=all)" ]] ||
        die "$label checkout is dirty: $root"
}

require_exact_value() {
    local label="$1"
    local actual="$2"
    local expected="$3"
    [[ "$actual" == "$expected" ]] ||
        die "$label differs from the audited formal value: expected=$expected actual=$actual"
}

for forbidden_loader_variable in \
    LD_PRELOAD \
    LD_AUDIT \
    __EGL_VENDOR_LIBRARY_DIRS \
    LIBGL_ALWAYS_SOFTWARE \
    MESA_LOADER_DRIVER_OVERRIDE \
    LIBGL_DRIVERS_PATH; do
    [[ -z "${!forbidden_loader_variable-}" ]] ||
        die "$forbidden_loader_variable must be unset before formal evaluation"
done

COSMOS_SOURCE="$(canonical_existing_dir "$COSMOS_SOURCE")"
COSMOS_ENV="$(canonical_existing_dir "$COSMOS_ENV")"
ROBOCASA_ENV="$(canonical_existing_dir "$ROBOCASA_ENV")"
DEMIAN_REPO="$(canonical_existing_dir "$DEMIAN_REPO")"
ROBOCASA_REPO="$(canonical_existing_dir "$ROBOCASA_REPO")"
ROBOSUITE_REPO="$(canonical_existing_dir "$ROBOSUITE_REPO")"
SERVING_CHECKPOINT_PATH="$(canonical_existing_dir "$SERVING_CHECKPOINT_PATH")"
H_SERVING_CHECKPOINT_MANIFEST="$(canonical_existing_file "$H_SERVING_CHECKPOINT_MANIFEST")"
EMA_EXPORT_RECEIPT="$(canonical_existing_file "$EMA_EXPORT_RECEIPT")"
EMA_KEY_MANIFEST="$(canonical_existing_file "$EMA_KEY_MANIFEST")"
EMA_VALIDATION="$(canonical_existing_file "$EMA_VALIDATION")"
EMA_SOURCE_DCP_FULL_RECEIPT="$(canonical_existing_file "$EMA_SOURCE_DCP_FULL_RECEIPT")"
EMA_TRANSFER_MANIFEST="$(canonical_existing_file "$EMA_TRANSFER_MANIFEST")"
CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT="$(
    canonical_existing_file "$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT"
)"
TRAINING_CONFIG="$(canonical_existing_file "$TRAINING_CONFIG")"
ACTION_STATS="$(canonical_existing_file "$ACTION_STATS")"
WAN2PT2_VAE="$(canonical_existing_file "$WAN2PT2_VAE")"
QWEN_TOKENIZER="$(canonical_existing_dir "$QWEN_TOKENIZER")"
QWEN_TOKENIZER_MANIFEST="$(canonical_existing_file "$QWEN_TOKENIZER_MANIFEST")"
SOURCE_MANIFEST="$(canonical_existing_file "$SOURCE_MANIFEST")"
COSMOS_ENV_READY_MARKER="$(canonical_existing_file "$COSMOS_ENV_READY_MARKER")"
ROBOCASA_ENV_READY_MARKER="$(canonical_existing_file "$ROBOCASA_ENV_READY_MARKER")"
ROBOCASA_TORCHCODEC_RELAY_RECEIPT="$(canonical_existing_file "$ROBOCASA_TORCHCODEC_RELAY_RECEIPT")"
ASSET_ENV_CLOSURE_MANIFEST="$(canonical_existing_file "$ASSET_ENV_CLOSURE_MANIFEST")"
ASSET_ENV_CLOSURE_H_VERIFICATION="$(canonical_existing_file "$ASSET_ENV_CLOSURE_H_VERIFICATION")"
ROBOCASA_ASSET_ROOT="$(canonical_existing_dir "$ROBOCASA_ASSET_ROOT")"
EVAL_EGL_VENDOR_JSON="$(canonical_existing_file "$EVAL_EGL_VENDOR_JSON")"
EGL_MAPPING_RECEIPT="$(canonical_existing_file "$EGL_MAPPING_RECEIPT")"
DSW_CPFS_MOUNT="$(canonical_existing_dir "$DSW_CPFS_MOUNT")"
DSW_CPFS_OUTPUT_PARENT="$(canonical_existing_dir "$DSW_CPFS_OUTPUT_PARENT")"
DSW_LOCAL_SCRATCH_PARENT="$(canonical_existing_dir "$DSW_LOCAL_SCRATCH_PARENT")"

require_exact_value \
    cosmos_environment_path \
    "$COSMOS_ENV" \
    "$FORMAL_COSMOS_ENV"
require_exact_value \
    robocasa_environment_path \
    "$ROBOCASA_ENV" \
    "$FORMAL_INPUT_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
require_exact_value \
    demian_repository_path \
    "$DEMIAN_REPO" \
    "$FORMAL_INPUT_ROOT/demian-2600ece5bf0e"
require_exact_value \
    robocasa_repository_path \
    "$ROBOCASA_REPO" \
    "$FORMAL_ROBOCASA_REPO"
require_exact_value \
    robocasa_asset_root \
    "$ROBOCASA_ASSET_ROOT" \
    "$FORMAL_ROBOCASA_REPO"
require_exact_value \
    robosuite_repository_path \
    "$ROBOSUITE_REPO" \
    "$FORMAL_INPUT_ROOT/robosuite-232ce7d4a6ed"
require_exact_value \
    serving_checkpoint_path \
    "$SERVING_CHECKPOINT_PATH" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/model"
require_exact_value \
    h_serving_checkpoint_manifest_path \
    "$H_SERVING_CHECKPOINT_MANIFEST" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/h-serving-checkpoint-manifest.json"
require_exact_value \
    ema_export_receipt_path \
    "$EMA_EXPORT_RECEIPT" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/ema-bf16-export-receipt.json"
require_exact_value \
    ema_key_manifest_path \
    "$EMA_KEY_MANIFEST" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/ema-bf16-key-manifest.json"
require_exact_value \
    ema_validation_path \
    "$EMA_VALIDATION" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/ema-bf16-validation.json"
require_exact_value \
    ema_source_dcp_full_receipt_path \
    "$EMA_SOURCE_DCP_FULL_RECEIPT" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/source-dcp-full-verification.json"
require_exact_value \
    ema_transfer_manifest_path \
    "$EMA_TRANSFER_MANIFEST" \
    "$FORMAL_CHECKPOINT_ARTIFACT_ROOT/ema-serving-dcp-transfer-manifest.json"
require_exact_value \
    checkpoint_atomic_publication_receipt_path \
    "$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT" \
    "$FORMAL_CHECKPOINT_PUBLICATION_RECEIPT"
require_exact_value \
    training_config_path \
    "$TRAINING_CONFIG" \
    "$FORMAL_INPUT_ROOT/config/R14-config.yaml"
require_exact_value \
    action_stats_path \
    "$ACTION_STATS" \
    "$FORMAL_INPUT_ROOT/config/robocasa365_native_frame_wise_relative_rot6d.json"
require_exact_value \
    wan2pt2_vae_path \
    "$WAN2PT2_VAE" \
    "$FORMAL_INPUT_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
require_exact_value \
    qwen_tokenizer_path \
    "$QWEN_TOKENIZER" \
    "$FORMAL_INPUT_ROOT/models/Qwen3-VL-8B-Instruct-tokenizer"
require_exact_value \
    qwen_tokenizer_manifest_path \
    "$QWEN_TOKENIZER_MANIFEST" \
    "$FORMAL_INPUT_ROOT/config/Qwen3-VL-8B-Instruct-tokenizer-manifest.json"
require_exact_value \
    egl_vendor_json_path \
    "$EVAL_EGL_VENDOR_JSON" \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json
require_exact_value \
    egl_mapping_receipt_path \
    "$EGL_MAPPING_RECEIPT" \
    "$FORMAL_INPUT_ROOT/config/robocasa-egl-device-mapping-dsw974.json"
require_exact_value \
    robocasa_torchcodec_relay_receipt_path \
    "$ROBOCASA_TORCHCODEC_RELAY_RECEIPT" \
    "$FORMAL_INPUT_ROOT/config/robocasa-torchcodec-plain-wheel-relay-receipt.json"
require_exact_value dsw_cpfs_mount "$DSW_CPFS_MOUNT" /mnt/workspace
require_exact_value \
    dsw_cpfs_output_parent \
    "$DSW_CPFS_OUTPUT_PARENT" \
    /mnt/workspace/cwam_cosmos3_robocasa365_eval/runs
require_exact_value \
    dsw_local_scratch_parent \
    "$DSW_LOCAL_SCRATCH_PARENT" \
    /tmp/cwam_cosmos3_robocasa365_eval

readonly COSMOS_PY="$COSMOS_ENV/bin/python"
readonly TORCHRUN="$COSMOS_ENV/bin/torchrun"
readonly ROBOCASA_PY="$ROBOCASA_ENV/bin/python"
[[ -x "$COSMOS_PY" ]] || die "Cosmos Python is not executable: $COSMOS_PY"
[[ -x "$TORCHRUN" ]] || die "torchrun is not executable: $TORCHRUN"
[[ -x "$ROBOCASA_PY" ]] || die "RoboCasa Python is not executable: $ROBOCASA_PY"
[[ -x "$SETSID" ]] || die "setsid is not executable: $SETSID"
[[ -f "$COSMOS_SOURCE/$SERVER_MODULE_REL" ]] || die "missing policy server module"
[[ -f "$COSMOS_SOURCE/$CHECKPOINT_HELPER_REL" ]] || die "missing checkpoint verifier"
[[ -f "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" ]] || die "missing DSW input provenance helper"
[[ -f "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" ]] || die "missing DSW evidence helper"
[[ -f "$COSMOS_SOURCE/$ASSET_VERIFY_HELPER_REL" ]] || die "missing RoboCasa asset verifier"
[[ -f "$DEMIAN_REPO/$EVALUATOR_REL" ]] || die "missing DeMiAn evaluator"
[[ -f "$DEMIAN_REPO/$TASK_REGISTRY_REL" ]] || die "missing DeMiAn task registry"

[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "EXPECTED_SOURCE_COMMIT is malformed"
require_exact_value \
    cosmos_environment_source_commit \
    "$EXPECTED_COSMOS_ENV_SOURCE_COMMIT" \
    "$PINNED_COSMOS_ENV_SOURCE_COMMIT"
require_exact_value demian_commit "$EXPECTED_DEMIAN_COMMIT" "$PINNED_DEMIAN_COMMIT"
require_exact_value robocasa_commit "$EXPECTED_ROBOCASA_COMMIT" "$PINNED_ROBOCASA_COMMIT"
require_exact_value robosuite_commit "$EXPECTED_ROBOSUITE_COMMIT" "$PINNED_ROBOSUITE_COMMIT"
require_exact_value \
    h_serving_manifest_sha256 \
    "$EXPECTED_H_SERVING_CHECKPOINT_MANIFEST_SHA256" \
    "$PINNED_H_SERVING_MANIFEST_SHA256"
require_exact_value \
    serving_checkpoint_tree_sha256 \
    "$EXPECTED_SERVING_CHECKPOINT_TREE_SHA256" \
    "$PINNED_SERVING_TREE_SHA256"
require_exact_value \
    ema_export_receipt_sha256 \
    "$EXPECTED_EMA_EXPORT_RECEIPT_SHA256" \
    "$PINNED_EMA_EXPORT_SHA256"
require_exact_value \
    ema_key_manifest_sha256 \
    "$EXPECTED_EMA_KEY_MANIFEST_SHA256" \
    "$PINNED_EMA_KEYS_SHA256"
require_exact_value \
    ema_validation_sha256 \
    "$EXPECTED_EMA_VALIDATION_SHA256" \
    "$PINNED_EMA_VALIDATION_SHA256"
require_exact_value \
    source_dcp_full_receipt_sha256 \
    "$EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256" \
    "$PINNED_SOURCE_DCP_FULL_SHA256"
require_exact_value \
    ema_transfer_manifest_sha256 \
    "$EXPECTED_EMA_TRANSFER_MANIFEST_SHA256" \
    "$PINNED_EMA_TRANSFER_SHA256"
require_exact_value \
    training_config_sha256 \
    "$EXPECTED_TRAINING_CONFIG_SHA256" \
    "$PINNED_TRAINING_CONFIG_SHA256"
require_exact_value \
    action_stats_sha256 \
    "$EXPECTED_ACTION_STATS_SHA256" \
    "$PINNED_ACTION_STATS_SHA256"
require_exact_value \
    wan2pt2_vae_sha256 \
    "$EXPECTED_WAN2PT2_VAE_SHA256" \
    "$PINNED_WAN2PT2_VAE_SHA256"
require_exact_value \
    qwen_tokenizer_manifest_sha256 \
    "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256" \
    "$PINNED_QWEN_MANIFEST_SHA256"
require_exact_value \
    asset_env_closure_manifest_sha256 \
    "$EXPECTED_ASSET_ENV_CLOSURE_MANIFEST_SHA256" \
    "$PINNED_CLOSURE_MANIFEST_SHA256"
require_exact_value \
    asset_env_closure_h_verification_sha256 \
    "$EXPECTED_ASSET_ENV_CLOSURE_H_VERIFICATION_SHA256" \
    "$PINNED_CLOSURE_H_VERIFICATION_SHA256"
require_exact_value \
    robocasa_torchcodec_relay_receipt_sha256 \
    "$EXPECTED_ROBOCASA_TORCHCODEC_RELAY_RECEIPT_SHA256" \
    "$PINNED_ROBOCASA_TORCHCODEC_RELAY_SHA256"
require_exact_value \
    egl_vendor_json_sha256 \
    "$EXPECTED_EGL_VENDOR_JSON_SHA256" \
    "$PINNED_EGL_VENDOR_JSON_SHA256"
require_exact_value \
    egl_mapping_receipt_sha256 \
    "$EXPECTED_EGL_MAPPING_RECEIPT_SHA256" \
    "$PINNED_EGL_MAPPING_RECEIPT_SHA256"
[[ "$SOURCE_MANIFEST" == "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json" ]] ||
    die "SOURCE_MANIFEST must be the snapshot-root SNAPSHOT_MANIFEST.json"
verify_sha256 source_manifest "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256"
[[ "$(stat -c %a -- "$SOURCE_MANIFEST")" == 444 ]] ||
    die "SOURCE_MANIFEST must be immutable mode 0444"
"$COSMOS_PY" - "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1:
    raise SystemExit("source snapshot manifest schema_version must be 1")
if manifest.get("source_head") != sys.argv[2]:
    raise SystemExit("source snapshot manifest does not bind EXPECTED_SOURCE_COMMIT")
if manifest.get("source_dirty") is not False:
    raise SystemExit("source snapshot manifest must assert source_dirty=false")
files = manifest.get("files")
if not isinstance(files, list) or manifest.get("file_count") != len(files):
    raise SystemExit("source snapshot manifest file inventory is malformed")
PY

mount_target="$(findmnt -n -T "$DSW_CPFS_MOUNT" -o TARGET)" ||
    die "could not resolve the CPFS mount target"
mount_source="$(findmnt -n -T "$DSW_CPFS_MOUNT" -o SOURCE)" ||
    die "could not resolve the CPFS mount source"
mount_fstype="$(findmnt -n -T "$DSW_CPFS_MOUNT" -o FSTYPE)" ||
    die "could not resolve the CPFS filesystem type"
[[ "$mount_target" == "$DSW_CPFS_MOUNT" ]] ||
    die "DSW_CPFS_MOUNT must name the exact mount target: observed=$mount_target"
mount_identity="${mount_source,,}:${mount_fstype,,}"
[[ "$mount_identity" == *cpfs* ]] ||
    die "declared CPFS mount does not identify CPFS: source=$mount_source fstype=$mount_fstype"
case "$DSW_CPFS_OUTPUT_PARENT/" in
    "$DSW_CPFS_MOUNT"/*/) ;;
    *) die "DSW_CPFS_OUTPUT_PARENT must be a strict child of DSW_CPFS_MOUNT" ;;
esac

EVAL_OUTPUT_ROOT="$(realpath -m -- "$EVAL_OUTPUT_ROOT")"
readonly EXPECTED_OUTPUT_ROOT="$DSW_CPFS_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
[[ "$EVAL_OUTPUT_ROOT" == "$EXPECTED_OUTPUT_ROOT" ]] ||
    die "EVAL_OUTPUT_ROOT must be exactly $EXPECTED_OUTPUT_ROOT"
case "$EVAL_OUTPUT_ROOT/" in
    "$DSW_CPFS_OUTPUT_PARENT"/*/) ;;
    *) die "EVAL_OUTPUT_ROOT must be a strict child of the allowed CPFS parent" ;;
esac
readonly EXPERIMENT_OUTPUT_ROOT="$DSW_CPFS_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID"
[[ ! -e "$EXPERIMENT_OUTPUT_ROOT" && ! -L "$EXPERIMENT_OUTPUT_ROOT" ]] ||
    die "refusing to reuse experiment output root: $EXPERIMENT_OUTPUT_ROOT"
mkdir -- "$EXPERIMENT_OUTPUT_ROOT"
mkdir -- "$EVAL_OUTPUT_ROOT"
readonly CONTROL_DIR="$EVAL_OUTPUT_ROOT/control"
readonly RESULTS_ROOT="$EVAL_OUTPUT_ROOT/results"
readonly RESULTS_DIR="$RESULTS_ROOT/$EVAL_RUN_NAME"
mkdir -- "$CONTROL_DIR" "$RESULTS_ROOT" "$RESULTS_DIR"
readonly SUMMARY_CSV="$RESULTS_DIR/summary.csv"
readonly TASK_SUCCESS_RATES_CSV="$RESULTS_DIR/task_success_rates.csv"
readonly EVIDENCE_JSON="$RESULTS_DIR/evaluation_evidence.json"
readonly RUNTIME_RECEIPT="$CONTROL_DIR/dsw-runtime-receipt.json"
readonly SHUTDOWN_RECEIPT="$CONTROL_DIR/dsw-controlled-shutdown-receipt.json"
readonly TERMINAL_RECEIPT="$CONTROL_DIR/dsw-server-terminal-receipt.json"
readonly SEALED_VERIFICATION_RECEIPT="$CONTROL_DIR/sealed-evidence-verification.json"
readonly QWEN_VERIFICATION_PRE="$CONTROL_DIR/qwen-tokenizer-verification-pre.json"
readonly QWEN_VERIFICATION_POST="$CONTROL_DIR/qwen-tokenizer-verification-post.json"
readonly SOURCE_VERIFICATION_PRE="$CONTROL_DIR/source-snapshot-verification-pre.json"
readonly SOURCE_VERIFICATION_POST="$CONTROL_DIR/source-snapshot-verification-post.json"
readonly CHECKPOINT_PUBLICATION_GATE_RECEIPT="$CONTROL_DIR/checkpoint-publication-pre-read-gate.json"
readonly CHECKPOINT_RELOCATION_RECEIPT="$CONTROL_DIR/checkpoint-relocation-receipt.json"
readonly DESTINATION_CHECKPOINT_MANIFEST="$CONTROL_DIR/dsw-serving-checkpoint-manifest.json"
readonly STATIC_INPUTS_PRE="$CONTROL_DIR/static-inputs-verification-pre.json"
readonly STATIC_INPUTS_POST="$CONTROL_DIR/static-inputs-verification-post.json"
readonly COSMOS_ENV_VERIFICATION_PRE="$CONTROL_DIR/cosmos-environment-verification-pre.json"
readonly COSMOS_ENV_VERIFICATION_POST="$CONTROL_DIR/cosmos-environment-verification-post.json"
readonly ROBOCASA_ENV_VERIFICATION_PRE="$CONTROL_DIR/robocasa-environment-verification-pre.json"
readonly ROBOCASA_ENV_VERIFICATION_POST="$CONTROL_DIR/robocasa-environment-verification-post.json"
readonly ROBOCASA_ASSET_VERIFICATION_PRE="$CONTROL_DIR/robocasa-asset-verification-pre.json"
readonly ROBOCASA_ASSET_VERIFICATION_POST="$CONTROL_DIR/robocasa-asset-verification-post.json"
readonly ASSET_ENV_CLOSURE_RECEIPT="$CONTROL_DIR/asset-env-closure-provenance.json"
readonly POLICY_SERVER_LOG="$CONTROL_DIR/policy-server-torchrun.log"
readonly ONE_EPISODE_GATE_RESULTS_ROOT="$CONTROL_DIR/one-episode-gate-results"
readonly ONE_EPISODE_GATE_RUN_NAME="$EVAL_RUN_NAME-one-episode-gate"
readonly ONE_EPISODE_GATE_RESULTS_DIR="$ONE_EPISODE_GATE_RESULTS_ROOT/$ONE_EPISODE_GATE_RUN_NAME"
readonly ONE_EPISODE_GATE_RECEIPT="$CONTROL_DIR/one-episode-gate-receipt.json"
mkdir -- "$ONE_EPISODE_GATE_RESULTS_ROOT"

readonly PROVENANCE_PYTHONPATH="$COSMOS_SOURCE/examples/h_cluster:$COSMOS_SOURCE/examples/aliyun_4090:$COSMOS_SOURCE"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-source-snapshot \
    --phase pre \
    --snapshot-root "$COSMOS_SOURCE" \
    --manifest-path "$SOURCE_MANIFEST" \
    --expected-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
    --output "$SOURCE_VERIFICATION_PRE" \
    >"$CONTROL_DIR/verify-source-snapshot-pre.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    verify-checkpoint-publication \
    --publication-receipt-path "$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT" \
    --expected-publication-receipt-sha256 \
    "$EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --expected-tree-sha256 "$EXPECTED_SERVING_CHECKPOINT_TREE_SHA256" \
    --output "$CHECKPOINT_PUBLICATION_GATE_RECEIPT" \
    >"$CONTROL_DIR/verify-checkpoint-publication-pre-read.log"
[[ "$(stat -c %a -- "$CHECKPOINT_PUBLICATION_GATE_RECEIPT")" == 444 ]] ||
    die "checkpoint publication pre-read gate receipt must have mode 0444"

verify_sha256 \
    h_serving_checkpoint_manifest \
    "$H_SERVING_CHECKPOINT_MANIFEST" \
    "$EXPECTED_H_SERVING_CHECKPOINT_MANIFEST_SHA256"
verify_sha256 ema_export_receipt "$EMA_EXPORT_RECEIPT" "$EXPECTED_EMA_EXPORT_RECEIPT_SHA256"
verify_sha256 ema_key_manifest "$EMA_KEY_MANIFEST" "$EXPECTED_EMA_KEY_MANIFEST_SHA256"
verify_sha256 ema_validation "$EMA_VALIDATION" "$EXPECTED_EMA_VALIDATION_SHA256"
verify_sha256 \
    ema_source_dcp_full_receipt \
    "$EMA_SOURCE_DCP_FULL_RECEIPT" \
    "$EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256"
verify_sha256 \
    ema_transfer_manifest \
    "$EMA_TRANSFER_MANIFEST" \
    "$EXPECTED_EMA_TRANSFER_MANIFEST_SHA256"
verify_sha256 \
    checkpoint_atomic_publication_receipt \
    "$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT" \
    "$EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256"
[[ "$(stat -c %a -- "$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT")" == 444 ]] ||
    die "checkpoint atomic publication receipt must have mode 0444"
"$COSMOS_PY" - \
    "$EMA_TRANSFER_MANIFEST" \
    "$EXPECTED_EMA_EXPORT_RECEIPT_SHA256" \
    "$EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256" \
    "$EXPECTED_SERVING_CHECKPOINT_TREE_SHA256" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_bindings = {
    "export_receipt_sha256": sys.argv[2],
    "source_dcp_full_verification_sha256": sys.argv[3],
    "target_tree_sha256": sys.argv[4],
}
if (
    manifest.get("kind") != "cwam_ema_serving_dcp_ceph_transfer_manifest"
    or manifest.get("status") != "uploaded_remote_size_verified"
    or manifest.get("item_count") != 17
    or not isinstance(manifest.get("items"), list)
    or len(manifest["items"]) != 17
    or manifest.get("bindings") != expected_bindings
):
    raise SystemExit("EMA Ceph transfer manifest provenance binding mismatch")
PY
verify_sha256 training_config "$TRAINING_CONFIG" "$EXPECTED_TRAINING_CONFIG_SHA256"
verify_sha256 action_stats "$ACTION_STATS" "$EXPECTED_ACTION_STATS_SHA256"
verify_sha256 wan2pt2_vae "$WAN2PT2_VAE" "$EXPECTED_WAN2PT2_VAE_SHA256"
verify_sha256 \
    qwen_tokenizer_manifest \
    "$QWEN_TOKENIZER_MANIFEST" \
    "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256"
verify_sha256 \
    cosmos_env_ready_marker \
    "$COSMOS_ENV_READY_MARKER" \
    "$EXPECTED_COSMOS_ENV_READY_MARKER_SHA256"
verify_sha256 \
    robocasa_env_ready_marker \
    "$ROBOCASA_ENV_READY_MARKER" \
    "$EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256"
verify_sha256 \
    robocasa_torchcodec_relay_receipt \
    "$ROBOCASA_TORCHCODEC_RELAY_RECEIPT" \
    "$EXPECTED_ROBOCASA_TORCHCODEC_RELAY_RECEIPT_SHA256"
verify_sha256 \
    egl_vendor_json \
    "$EVAL_EGL_VENDOR_JSON" \
    "$EXPECTED_EGL_VENDOR_JSON_SHA256"
verify_sha256 \
    egl_mapping_receipt \
    "$EGL_MAPPING_RECEIPT" \
    "$EXPECTED_EGL_MAPPING_RECEIPT_SHA256"
[[ "$(stat -c %a -- "$EVAL_EGL_VENDOR_JSON")" == 644 ]] ||
    die "NVIDIA EGL vendor JSON must have mode 0644"
[[ "$(stat -c %a -- "$EGL_MAPPING_RECEIPT")" == 444 ]] ||
    die "EGL mapping receipt must have mode 0444"
verify_sha256 \
    asset_env_closure_manifest \
    "$ASSET_ENV_CLOSURE_MANIFEST" \
    "$EXPECTED_ASSET_ENV_CLOSURE_MANIFEST_SHA256"
verify_sha256 \
    asset_env_closure_h_verification \
    "$ASSET_ENV_CLOSURE_H_VERIFICATION" \
    "$EXPECTED_ASSET_ENV_CLOSURE_H_VERIFICATION_SHA256"
[[ "$COSMOS_ENV_READY_MARKER" == "$COSMOS_ENV/.cwam_env_ready.json" ]] ||
    die "Cosmos formal environment must use its native .cwam_env_ready.json"
[[ "$ROBOCASA_ENV_READY_MARKER" == "$ROBOCASA_ENV/.cwam_env_ready.json" ]] ||
    die "RoboCasa formal environment must use its native .cwam_env_ready.json"
[[ "$ROBOCASA_ASSET_ROOT" == "$ROBOCASA_REPO" ]] ||
    die "ROBOCASA_ASSET_ROOT must be the exact RoboCasa checkout used by the evaluator"
if [[ -e "$COSMOS_SOURCE/.git" || -L "$COSMOS_SOURCE/.git" ]]; then
    verify_git_checkout Cosmos "$COSMOS_SOURCE" "$EXPECTED_SOURCE_COMMIT"
fi
verify_git_checkout DeMiAn "$DEMIAN_REPO" "$EXPECTED_DEMIAN_COMMIT"
if [[ -e "$ROBOCASA_REPO/.git" && ! -L "$ROBOCASA_REPO/.git" ]]; then
    verify_git_checkout RoboCasa "$ROBOCASA_REPO" "$EXPECTED_ROBOCASA_COMMIT"
fi
verify_git_checkout robosuite "$ROBOSUITE_REPO" "$EXPECTED_ROBOSUITE_COMMIT"

readonly SCRATCH_ROOT="$DSW_LOCAL_SCRATCH_PARENT/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID"
case "$SCRATCH_ROOT/" in
    "$DSW_LOCAL_SCRATCH_PARENT"/*/) ;;
    *) die "derived scratch path escaped DSW_LOCAL_SCRATCH_PARENT" ;;
esac
[[ ! -e "$SCRATCH_ROOT" && ! -L "$SCRATCH_ROOT" ]] ||
    die "refusing to reuse local scratch root: $SCRATCH_ROOT"
mkdir --parents --mode=0700 -- "$SCRATCH_ROOT"
readonly SERVER_OUTPUT="$SCRATCH_ROOT/server-output"
readonly COMPILE_CACHE="$SCRATCH_ROOT/torchinductor"
mkdir -- "$SERVER_OUTPUT" "$COMPILE_CACHE"

SERVER_LAUNCHER_PID=
SERVER_PROCESS_GROUP=
SCRATCH_CREATED=1
force_stop_server() {
    local deadline
    if [[ -n "$SERVER_PROCESS_GROUP" ]] &&
        kill -0 -- "-$SERVER_PROCESS_GROUP" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PROCESS_GROUP" 2>/dev/null || true
    elif [[ -n "$SERVER_LAUNCHER_PID" ]] &&
        kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_LAUNCHER_PID" 2>/dev/null || true
    fi
    deadline=$((SECONDS + CLEANUP_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        if { [[ -z "$SERVER_PROCESS_GROUP" ]] ||
            ! kill -0 -- "-$SERVER_PROCESS_GROUP" 2>/dev/null; } &&
            { [[ -z "$SERVER_LAUNCHER_PID" ]] ||
                ! kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; }; then
            break
        fi
        sleep 1
    done
    if [[ -n "$SERVER_PROCESS_GROUP" ]] &&
        kill -0 -- "-$SERVER_PROCESS_GROUP" 2>/dev/null; then
        kill -KILL -- "-$SERVER_PROCESS_GROUP" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_LAUNCHER_PID" ]] &&
        kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; then
        kill -KILL "$SERVER_LAUNCHER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_LAUNCHER_PID" ]]; then
        wait "$SERVER_LAUNCHER_PID" 2>/dev/null || true
    fi
    SERVER_LAUNCHER_PID=
    SERVER_PROCESS_GROUP=
}
cleanup_scratch() {
    [[ "$SCRATCH_CREATED" == 1 ]] || return 0
    [[ "$SCRATCH_ROOT" == "$DSW_LOCAL_SCRATCH_PARENT/.cwam-robocasa365-eval/$EVAL_LOGICAL_RUN_ID" ]] ||
        die "refusing to clean an unexpected scratch path: $SCRATCH_ROOT"
    [[ ! -L "$SCRATCH_ROOT" ]] || die "refusing to clean a symlinked scratch path"
    if [[ -d "$SCRATCH_ROOT" ]]; then
        /usr/bin/rm --one-file-system -rf -- "$SCRATCH_ROOT"
    fi
    [[ ! -e "$SCRATCH_ROOT" && ! -L "$SCRATCH_ROOT" ]] ||
        die "scoped scratch cleanup failed: $SCRATCH_ROOT"
    SCRATCH_CREATED=0
}
cleanup() {
    local status="$1"
    trap - EXIT INT TERM
    force_stop_server
    if ! cleanup_scratch; then
        status=1
    fi
    exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

readonly POLICY_PYTHONPATH="$COSMOS_SOURCE:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src"
readonly EVAL_PYTHONPATH="$DEMIAN_REPO/demian_wam:$DEMIAN_REPO/openpi/src:$DEMIAN_REPO/openpi/packages/openpi-client/src:$ROBOCASA_REPO:$ROBOSUITE_REPO"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    relocate-checkpoint \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --h-manifest-path "$H_SERVING_CHECKPOINT_MANIFEST" \
    --expected-h-manifest-sha256 "$EXPECTED_H_SERVING_CHECKPOINT_MANIFEST_SHA256" \
    --expected-target-tree-sha256 "$EXPECTED_SERVING_CHECKPOINT_TREE_SHA256" \
    --export-receipt-path "$EMA_EXPORT_RECEIPT" \
    --expected-export-receipt-sha256 "$EXPECTED_EMA_EXPORT_RECEIPT_SHA256" \
    --key-manifest-path "$EMA_KEY_MANIFEST" \
    --expected-key-manifest-sha256 "$EXPECTED_EMA_KEY_MANIFEST_SHA256" \
    --validation-path "$EMA_VALIDATION" \
    --expected-validation-sha256 "$EXPECTED_EMA_VALIDATION_SHA256" \
    --source-dcp-receipt-path "$EMA_SOURCE_DCP_FULL_RECEIPT" \
    --expected-source-dcp-receipt-sha256 "$EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256" \
    --training-config-path "$TRAINING_CONFIG" \
    --expected-training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
    --destination-manifest-output "$DESTINATION_CHECKPOINT_MANIFEST" \
    --relocation-receipt-output "$CHECKPOINT_RELOCATION_RECEIPT" \
    >"$CONTROL_DIR/relocate-checkpoint.log"
readonly DESTINATION_CHECKPOINT_MANIFEST_SHA256="$(sha256_file "$DESTINATION_CHECKPOINT_MANIFEST")"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-static-inputs \
    --phase pre \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --destination-manifest-path "$DESTINATION_CHECKPOINT_MANIFEST" \
    --expected-destination-manifest-sha256 "$DESTINATION_CHECKPOINT_MANIFEST_SHA256" \
    --training-config-path "$TRAINING_CONFIG" \
    --expected-training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
    --action-stats-path "$ACTION_STATS" \
    --expected-action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
    --wan2pt2-vae-path "$WAN2PT2_VAE" \
    --expected-wan2pt2-vae-sha256 "$EXPECTED_WAN2PT2_VAE_SHA256" \
    --output "$STATIC_INPUTS_PRE" \
    >"$CONTROL_DIR/verify-static-inputs-pre.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-environment \
    --phase pre \
    --environment "$COSMOS_ENV" \
    --environment-type cosmos \
    --installation-mode native_exact_rebuild \
    --marker-path "$COSMOS_ENV_READY_MARKER" \
    --expected-marker-sha256 "$EXPECTED_COSMOS_ENV_READY_MARKER_SHA256" \
    --expected-source-commit "$EXPECTED_COSMOS_ENV_SOURCE_COMMIT" \
    --import-probe torch \
    --import-probe pydantic \
    --import-probe yaml \
    --expect-cuda-gpus "$PHYSICAL_GPU_COUNT" \
    --output "$COSMOS_ENV_VERIFICATION_PRE" \
    >"$CONTROL_DIR/verify-cosmos-environment-pre.log"

# Verify the installed environment without checkout package metadata overlays.
PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$ROBOCASA_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-environment \
    --phase pre \
    --environment "$ROBOCASA_ENV" \
    --environment-type robocasa \
    --installation-mode native_exact_rebuild \
    --marker-path "$ROBOCASA_ENV_READY_MARKER" \
    --expected-marker-sha256 "$EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256" \
    --expected-robocasa-commit "$EXPECTED_ROBOCASA_COMMIT" \
    --expected-robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT" \
    --import-probe mujoco \
    --import-probe robocasa \
    --import-probe robosuite \
    --output "$ROBOCASA_ENV_VERIFICATION_PRE" \
    >"$CONTROL_DIR/verify-robocasa-environment-pre.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$ASSET_VERIFY_HELPER_REL" \
    verify \
    --asset-root "$ROBOCASA_ASSET_ROOT" \
    --workers 16 \
    --output "$ROBOCASA_ASSET_VERIFICATION_PRE" \
    >"$CONTROL_DIR/verify-robocasa-assets-pre.log"
"$COSMOS_PY" - "$ROBOCASA_ASSET_VERIFICATION_PRE" "$EXPECTED_ROBOCASA_COMMIT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "passed" or receipt.get("robocasa_git_commit") != sys.argv[2]:
    raise SystemExit("RoboCasa asset receipt does not bind EXPECTED_ROBOCASA_COMMIT")
PY

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-closure-provenance \
    --closure-manifest-path "$ASSET_ENV_CLOSURE_MANIFEST" \
    --expected-closure-manifest-sha256 "$EXPECTED_ASSET_ENV_CLOSURE_MANIFEST_SHA256" \
    --h-verification-receipt-path "$ASSET_ENV_CLOSURE_H_VERIFICATION" \
    --expected-h-verification-receipt-sha256 "$EXPECTED_ASSET_ENV_CLOSURE_H_VERIFICATION_SHA256" \
    --output "$ASSET_ENV_CLOSURE_RECEIPT" \
    >"$CONTROL_DIR/verify-asset-env-closure.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    verify-qwen-tokenizer \
    --tokenizer-path "$QWEN_TOKENIZER" \
    --manifest-path "$QWEN_TOKENIZER_MANIFEST" \
    --expected-manifest-sha256 "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256" \
    --output "$QWEN_VERIFICATION_PRE" \
    >"$CONTROL_DIR/verify-qwen-tokenizer-pre.log"

"$COSMOS_PY" - "$PHYSICAL_GPU_COUNT" "$EXPECTED_CPU_COUNT" <<'PY'
import os
import sys
import torch

expected_gpus = int(sys.argv[1])
expected_cpus = int(sys.argv[2])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
    raise SystemExit(
        f"expected exactly {expected_gpus} visible CUDA GPUs, got {torch.cuda.device_count()}"
    )
inventory = []
for index in range(expected_gpus):
    props = torch.cuda.get_device_properties(index)
    inventory.append((props.name, props.major, props.minor))
if any("4090" not in name or (major, minor) != (8, 9) for name, major, minor in inventory):
    raise SystemExit(f"expected four 4090 sm89 GPUs, got {inventory}")
if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != expected_cpus:
    raise SystemExit(
        f"expected effective CPU affinity count {expected_cpus}, "
        f"got {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}"
    )
print(f"DSW_CUDA_READY gpus={inventory} cpu_affinity={expected_cpus}")
PY

"$COSMOS_PY" - "$BASE_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", port))
print(f"DSW_POLICY_PORT_READY port={port}")
PY

launch_nonce="$("$COSMOS_PY" -c 'import secrets; print(secrets.token_hex(32))')"
# Bash exposes the background PID only after the asynchronous command has
# started.  During that one-command boundary, defer INT/TERM cleanup until the
# new launcher PID and its expected setsid process group are recorded.
PENDING_LAUNCH_SIGNAL_STATUS=
trap 'PENDING_LAUNCH_SIGNAL_STATUS=130' INT
trap 'PENDING_LAUNCH_SIGNAL_STATUS=143' TERM
(
    cd "$COSMOS_SOURCE"
    exec "$SETSID" env \
        -u LD_PRELOAD \
        -u LD_AUDIT \
        -u __EGL_VENDOR_LIBRARY_DIRS \
        -u LIBGL_ALWAYS_SOFTWARE \
        -u MESA_LOADER_DRIVER_OVERRIDE \
        -u LIBGL_DRIVERS_PATH \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PYTHONPATH="$POLICY_PYTHONPATH" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE" \
        "$TORCHRUN" \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="$PHYSICAL_GPU_COUNT" \
        --max-restarts=0 \
        -m "$SERVER_MODULE" \
        --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
        --config-file "$TRAINING_CONFIG" \
        --output-dir "$SERVER_OUTPUT" \
        --action-stats-path "$ACTION_STATS" \
        --wan2pt2-vae-path "$WAN2PT2_VAE" \
        --qwen-tokenizer-path "$QWEN_TOKENIZER" \
        --host 127.0.0.1 \
        --port "$BASE_PORT" \
        --sampler "$POLICY_SAMPLER" \
        --seed "$POLICY_SEED" \
        --guidance "$POLICY_GUIDANCE" \
        --num-steps "$POLICY_NUM_STEPS" \
        --launch-nonce "$launch_nonce"
) >"$POLICY_SERVER_LOG" 2>&1 &
SERVER_LAUNCHER_PID="$!"
# Reserve the expected process-group identity immediately.  Before setsid
# completes, cleanup falls back to signaling the still-child launcher PID; as
# soon as the group exists, the same value addresses the complete torchrun
# rank group.
SERVER_PROCESS_GROUP="$SERVER_LAUNCHER_PID"
trap 'cleanup 130' INT
trap 'cleanup 143' TERM
if [[ -n "$PENDING_LAUNCH_SIGNAL_STATUS" ]]; then
    cleanup "$PENDING_LAUNCH_SIGNAL_STATUS"
fi
server_process_group=
for ((process_group_attempt = 0; process_group_attempt < 100; process_group_attempt++)); do
    kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null ||
        die "distributed policy server exited before process-group verification"
    server_process_group="$(
        ps -o pgid= -p "$SERVER_LAUNCHER_PID" 2>/dev/null |
            tr -d '[:space:]'
    )"
    [[ "$server_process_group" == "$SERVER_LAUNCHER_PID" ]] && break
    sleep 0.05
done
[[ "$server_process_group" == "$SERVER_LAUNCHER_PID" ]] ||
    die "distributed policy server must lead an isolated process group"
SERVER_PROCESS_GROUP="$server_process_group"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    capture-runtime \
    --uri "ws://127.0.0.1:$BASE_PORT" \
    --output "$RUNTIME_RECEIPT" \
    --timeout-seconds "$SERVER_READY_TIMEOUT_SECONDS" \
    --launcher-pid "$SERVER_LAUNCHER_PID" \
    --expected-launch-nonce "$launch_nonce" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --checkpoint-manifest-path "$DESTINATION_CHECKPOINT_MANIFEST" \
    --checkpoint-manifest-sha256 "$DESTINATION_CHECKPOINT_MANIFEST_SHA256" \
    --checkpoint-tree-sha256 "$EXPECTED_SERVING_CHECKPOINT_TREE_SHA256" \
    --training-config-path "$TRAINING_CONFIG" \
    --training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
    --action-stats-path "$ACTION_STATS" \
    --action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
    --wan2pt2-vae-path "$WAN2PT2_VAE" \
    --wan2pt2-vae-sha256 "$EXPECTED_WAN2PT2_VAE_SHA256" \
    --qwen-tokenizer-path "$QWEN_TOKENIZER" \
    --qwen-tokenizer-manifest-path "$QWEN_TOKENIZER_MANIFEST" \
    --qwen-tokenizer-manifest-sha256 "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256" \
    --source-manifest-path "$SOURCE_MANIFEST" \
    --source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --source-commit "$EXPECTED_SOURCE_COMMIT" \
    --demian-commit "$EXPECTED_DEMIAN_COMMIT" \
    --demian-openpi-server-source-path \
    "$DEMIAN_REPO/openpi/src/openpi/serving/websocket_policy_server.py" \
    --robocasa-environment-path "$ROBOCASA_ENV" \
    --robocasa-ready-marker-path "$ROBOCASA_ENV_READY_MARKER" \
    --robocasa-ready-marker-sha256 "$EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256" \
    --egl-vendor-json-path "$EVAL_EGL_VENDOR_JSON" \
    --egl-vendor-json-sha256 "$EXPECTED_EGL_VENDOR_JSON_SHA256" \
    --egl-mapping-receipt-path "$EGL_MAPPING_RECEIPT" \
    --egl-mapping-receipt-sha256 "$EXPECTED_EGL_MAPPING_RECEIPT_SHA256" \
    --evaluator-timeout-seconds "$EVALUATOR_TIMEOUT_SECONDS" \
    --one-episode-gate-timeout-seconds "$ONE_EPISODE_GATE_TIMEOUT_SECONDS" \
    --server-exit-timeout-seconds "$SERVER_EXIT_TIMEOUT_SECONDS" \
    --provenance-receipt "asset_env_closure=$ASSET_ENV_CLOSURE_RECEIPT" \
    --provenance-receipt "checkpoint_atomic_publication=$CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT" \
    --provenance-receipt "checkpoint_publication_gate=$CHECKPOINT_PUBLICATION_GATE_RECEIPT" \
    --provenance-receipt "checkpoint_relocation=$CHECKPOINT_RELOCATION_RECEIPT" \
    --provenance-receipt "cosmos_environment_pre=$COSMOS_ENV_VERIFICATION_PRE" \
    --provenance-receipt "ema_export_receipt=$EMA_EXPORT_RECEIPT" \
    --provenance-receipt "ema_key_manifest=$EMA_KEY_MANIFEST" \
    --provenance-receipt "ema_source_dcp_full=$EMA_SOURCE_DCP_FULL_RECEIPT" \
    --provenance-receipt "ema_transfer_manifest=$EMA_TRANSFER_MANIFEST" \
    --provenance-receipt "ema_validation=$EMA_VALIDATION" \
    --provenance-receipt "h_checkpoint_manifest=$H_SERVING_CHECKPOINT_MANIFEST" \
    --provenance-receipt "qwen_tokenizer_pre=$QWEN_VERIFICATION_PRE" \
    --provenance-receipt "robocasa_assets_pre=$ROBOCASA_ASSET_VERIFICATION_PRE" \
    --provenance-receipt "robocasa_environment_pre=$ROBOCASA_ENV_VERIFICATION_PRE" \
    --provenance-receipt "robocasa_torchcodec_relay=$ROBOCASA_TORCHCODEC_RELAY_RECEIPT" \
    --provenance-receipt "source_snapshot_pre=$SOURCE_VERIFICATION_PRE" \
    --provenance-receipt "static_inputs_pre=$STATIC_INPUTS_PRE" \
    >"$CONTROL_DIR/capture-dsw-runtime.log"
echo "DSW_RUNTIME_VERIFIED" \
    "physical_servers=$PHYSICAL_SERVER_COUNT" \
    "physical_gpus=$PHYSICAL_GPU_COUNT" \
    "logical_shards=$LOGICAL_NUM_SHARDS" \
    "cpu=$EXPECTED_CPU_COUNT" \
    "memory_gib=$EXPECTED_MEMORY_GIB"

kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null ||
    die "distributed policy server exited before the one-episode preflight gate"
if ! timeout \
    --signal=TERM \
    --kill-after=30 \
    "$ONE_EPISODE_GATE_TIMEOUT_SECONDS" \
    env \
    -u LD_PRELOAD \
    -u LD_AUDIT \
    -u __EGL_VENDOR_LIBRARY_DIRS \
    -u LIBGL_ALWAYS_SOFTWARE \
    -u MESA_LOADER_DRIVER_OVERRIDE \
    -u LIBGL_DRIVERS_PATH \
    CUDA_VISIBLE_DEVICES=0 \
    MUJOCO_EGL_DEVICE_ID=0 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON" \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    PYTHONPATH="$EVAL_PYTHONPATH" \
    "$ROBOCASA_PY" "$DEMIAN_REPO/$EVALUATOR_REL" \
    --tasks "$ONE_EPISODE_GATE_TASK" \
    --task-shard 0 \
    --num-shards 1 \
    --episodes 1 \
    --episodes-start 0 \
    --max-steps "$MAX_STEPS" \
    --open-loop-horizon "$OPEN_LOOP_HORIZON" \
    --host 127.0.0.1 \
    --port "$BASE_PORT" \
    --checkpoint "$SERVING_CHECKPOINT_PATH" \
    --results-dir "$ONE_EPISODE_GATE_RESULTS_ROOT" \
    --run-name "$ONE_EPISODE_GATE_RUN_NAME" \
    --no-save-video \
    --headless \
    --camera-height 256 \
    --camera-width 256 \
    --seed "$BASE_SEED" \
    --split atomic >"$CONTROL_DIR/one-episode-gate-evaluator.log" 2>&1; then
    tail -200 "$CONTROL_DIR/one-episode-gate-evaluator.log" >&2 || true
    die "one-episode closed-loop preflight gate failed"
fi
kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null ||
    die "distributed policy server exited during the one-episode preflight gate"
[[ -s "$ONE_EPISODE_GATE_RESULTS_DIR/args.json" ]] ||
    die "one-episode gate args artifact is missing"
[[ -s "$ONE_EPISODE_GATE_RESULTS_DIR/results_shard0.json" ]] ||
    die "one-episode gate result artifact is missing"
chmod 0444 \
    "$ONE_EPISODE_GATE_RESULTS_DIR/args.json" \
    "$ONE_EPISODE_GATE_RESULTS_DIR/results_shard0.json"
PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    verify-one-episode-gate \
    --gate-results-dir "$ONE_EPISODE_GATE_RESULTS_DIR" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --run-name "$ONE_EPISODE_GATE_RUN_NAME" \
    --task "$ONE_EPISODE_GATE_TASK" \
    --runtime-receipt "$RUNTIME_RECEIPT" \
    --output "$ONE_EPISODE_GATE_RECEIPT" \
    >"$CONTROL_DIR/verify-one-episode-gate.log"
echo "ONE_EPISODE_GATE_PASSED task=$ONE_EPISODE_GATE_TASK"

for ((shard = 0; shard < LOGICAL_NUM_SHARDS; shard++)); do
    kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null ||
        die "distributed policy server exited before logical shard $shard"
    evaluator_log="$CONTROL_DIR/evaluator-shard${shard}.log"
    evaluator_gpu=$((shard % PHYSICAL_GPU_COUNT))
    echo "START_LOGICAL_SHARD shard=$shard num_shards=$LOGICAL_NUM_SHARDS evaluator_gpu=$evaluator_gpu"
    if ! timeout \
        --signal=TERM \
        --kill-after=30 \
        "$EVALUATOR_TIMEOUT_SECONDS" \
        env \
        -u LD_PRELOAD \
        -u LD_AUDIT \
        -u __EGL_VENDOR_LIBRARY_DIRS \
        -u LIBGL_ALWAYS_SOFTWARE \
        -u MESA_LOADER_DRIVER_OVERRIDE \
        -u LIBGL_DRIVERS_PATH \
        CUDA_VISIBLE_DEVICES="$evaluator_gpu" \
        MUJOCO_EGL_DEVICE_ID="$evaluator_gpu" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_EGL_VENDOR_JSON" \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        PYTHONPATH="$EVAL_PYTHONPATH" \
        "$ROBOCASA_PY" "$DEMIAN_REPO/$EVALUATOR_REL" \
        --task-shard "$shard" \
        --num-shards "$LOGICAL_NUM_SHARDS" \
        --episodes "$EPISODES_PER_TASK" \
        --episodes-start 0 \
        --max-steps "$MAX_STEPS" \
        --open-loop-horizon "$OPEN_LOOP_HORIZON" \
        --host 127.0.0.1 \
        --port "$BASE_PORT" \
        --checkpoint "$SERVING_CHECKPOINT_PATH" \
        --results-dir "$RESULTS_ROOT" \
        --run-name "$EVAL_RUN_NAME" \
        --no-save-video \
        --headless \
        --camera-height 256 \
        --camera-width 256 \
        --seed "$BASE_SEED" \
        --split atomic >"$evaluator_log" 2>&1; then
        tail -200 "$evaluator_log" >&2 || true
        die "RoboCasa evaluator logical shard $shard failed"
    fi
    [[ -s "$RESULTS_DIR/args_shard${shard}.json" ]] ||
        die "logical shard $shard args artifact is missing"
    [[ -s "$RESULTS_DIR/results_shard${shard}.json" ]] ||
        die "logical shard $shard result artifact is missing"
    echo "FINISH_LOGICAL_SHARD shard=$shard"
done

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    shutdown-server \
    --uri "ws://127.0.0.1:$BASE_PORT" \
    --output "$SHUTDOWN_RECEIPT" \
    --timeout-seconds "$SERVER_SHUTDOWN_TIMEOUT_SECONDS" \
    --expected-launch-nonce "$launch_nonce" \
    --runtime-receipt "$RUNTIME_RECEIPT" \
    >"$CONTROL_DIR/shutdown-policy-server.log"

readonly COMPLETED_SERVER_LAUNCHER_PID="$SERVER_LAUNCHER_PID"
server_exit_deadline=$((SECONDS + SERVER_EXIT_TIMEOUT_SECONDS))
while kill -0 "$COMPLETED_SERVER_LAUNCHER_PID" 2>/dev/null &&
    ((SECONDS < server_exit_deadline)); do
    sleep 1
done
if kill -0 "$COMPLETED_SERVER_LAUNCHER_PID" 2>/dev/null; then
    die "torchrun did not exit after the acknowledged controlled shutdown"
fi
set +e
wait "$COMPLETED_SERVER_LAUNCHER_PID"
torchrun_exit_code="$?"
set -e
SERVER_LAUNCHER_PID=
SERVER_PROCESS_GROUP=
[[ "$torchrun_exit_code" == 0 ]] ||
    die "torchrun exited nonzero after controlled shutdown: $torchrun_exit_code"
[[ -s "$POLICY_SERVER_LOG" ]] || die "policy server torchrun log is missing or empty"
chmod 0444 "$POLICY_SERVER_LOG"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    capture-terminal \
    --output "$TERMINAL_RECEIPT" \
    --runtime-receipt "$RUNTIME_RECEIPT" \
    --shutdown-receipt "$SHUTDOWN_RECEIPT" \
    --server-log "$POLICY_SERVER_LOG" \
    --torchrun-launcher-pid "$COMPLETED_SERVER_LAUNCHER_PID" \
    --torchrun-exit-code "$torchrun_exit_code" \
    >"$CONTROL_DIR/capture-server-terminal.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-static-inputs \
    --phase post \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --destination-manifest-path "$DESTINATION_CHECKPOINT_MANIFEST" \
    --expected-destination-manifest-sha256 "$DESTINATION_CHECKPOINT_MANIFEST_SHA256" \
    --training-config-path "$TRAINING_CONFIG" \
    --expected-training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
    --action-stats-path "$ACTION_STATS" \
    --expected-action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
    --wan2pt2-vae-path "$WAN2PT2_VAE" \
    --expected-wan2pt2-vae-sha256 "$EXPECTED_WAN2PT2_VAE_SHA256" \
    --output "$STATIC_INPUTS_POST" \
    >"$CONTROL_DIR/verify-static-inputs-post.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-source-snapshot \
    --phase post \
    --snapshot-root "$COSMOS_SOURCE" \
    --manifest-path "$SOURCE_MANIFEST" \
    --expected-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
    --output "$SOURCE_VERIFICATION_POST" \
    >"$CONTROL_DIR/verify-source-snapshot-post.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-environment \
    --phase post \
    --environment "$COSMOS_ENV" \
    --environment-type cosmos \
    --installation-mode native_exact_rebuild \
    --marker-path "$COSMOS_ENV_READY_MARKER" \
    --expected-marker-sha256 "$EXPECTED_COSMOS_ENV_READY_MARKER_SHA256" \
    --expected-source-commit "$EXPECTED_COSMOS_ENV_SOURCE_COMMIT" \
    --import-probe torch \
    --import-probe pydantic \
    --import-probe yaml \
    --expect-cuda-gpus "$PHYSICAL_GPU_COUNT" \
    --output "$COSMOS_ENV_VERIFICATION_POST" \
    >"$CONTROL_DIR/verify-cosmos-environment-post.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$ROBOCASA_PY" "$COSMOS_SOURCE/$SOURCE_PROVENANCE_HELPER_REL" \
    verify-environment \
    --phase post \
    --environment "$ROBOCASA_ENV" \
    --environment-type robocasa \
    --installation-mode native_exact_rebuild \
    --marker-path "$ROBOCASA_ENV_READY_MARKER" \
    --expected-marker-sha256 "$EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256" \
    --expected-robocasa-commit "$EXPECTED_ROBOCASA_COMMIT" \
    --expected-robosuite-commit "$EXPECTED_ROBOSUITE_COMMIT" \
    --import-probe mujoco \
    --import-probe robocasa \
    --import-probe robosuite \
    --output "$ROBOCASA_ENV_VERIFICATION_POST" \
    >"$CONTROL_DIR/verify-robocasa-environment-post.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$ASSET_VERIFY_HELPER_REL" \
    verify \
    --asset-root "$ROBOCASA_ASSET_ROOT" \
    --workers 16 \
    --output "$ROBOCASA_ASSET_VERIFICATION_POST" \
    >"$CONTROL_DIR/verify-robocasa-assets-post.log"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    verify-qwen-tokenizer \
    --tokenizer-path "$QWEN_TOKENIZER" \
    --manifest-path "$QWEN_TOKENIZER_MANIFEST" \
    --expected-manifest-sha256 "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256" \
    --output "$QWEN_VERIFICATION_POST" \
    >"$CONTROL_DIR/verify-qwen-tokenizer-post.log"

RAW_RESULT_FILES=()
for ((shard = 0; shard < LOGICAL_NUM_SHARDS; shard++)); do
    RAW_RESULT_FILES+=(
        "$RESULTS_DIR/args_shard${shard}.json"
        "$RESULTS_DIR/results_shard${shard}.json"
    )
done
chmod 0444 "${RAW_RESULT_FILES[@]}"

PYTHONPATH="$COSMOS_SOURCE/examples/h_cluster:$COSMOS_SOURCE" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    summarize \
    --experiment-id "$EVAL_EXPERIMENT_ID" \
    --logical-run-id "$EVAL_LOGICAL_RUN_ID" \
    --source-commit "$EXPECTED_SOURCE_COMMIT" \
    --run-name "$EVAL_RUN_NAME" \
    --results-dir "$RESULTS_DIR" \
    --checkpoint-path "$SERVING_CHECKPOINT_PATH" \
    --checkpoint-manifest-sha256 "$DESTINATION_CHECKPOINT_MANIFEST_SHA256" \
    --training-config-sha256 "$EXPECTED_TRAINING_CONFIG_SHA256" \
    --action-stats-sha256 "$EXPECTED_ACTION_STATS_SHA256" \
    --wan2pt2-vae-sha256 "$EXPECTED_WAN2PT2_VAE_SHA256" \
    --qwen-tokenizer-manifest-sha256 "$EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256" \
    --source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --demian-commit "$EXPECTED_DEMIAN_COMMIT" \
    --base-seed "$BASE_SEED" \
    --runtime-receipt "$RUNTIME_RECEIPT" \
    --terminal-receipt "$TERMINAL_RECEIPT" \
    --post-provenance-receipt "cosmos_environment_post=$COSMOS_ENV_VERIFICATION_POST" \
    --post-provenance-receipt "qwen_tokenizer_post=$QWEN_VERIFICATION_POST" \
    --post-provenance-receipt "robocasa_assets_post=$ROBOCASA_ASSET_VERIFICATION_POST" \
    --post-provenance-receipt "robocasa_environment_post=$ROBOCASA_ENV_VERIFICATION_POST" \
    --post-provenance-receipt "source_snapshot_post=$SOURCE_VERIFICATION_POST" \
    --post-provenance-receipt "static_inputs_post=$STATIC_INPUTS_POST" \
    --post-provenance-receipt "one_episode_gate=$ONE_EPISODE_GATE_RECEIPT" \
    --summary-csv "$SUMMARY_CSV" \
    --task-success-rates-csv "$TASK_SUCCESS_RATES_CSV" \
    --evidence-json "$EVIDENCE_JSON" \
    >"$CONTROL_DIR/summarize.log"

[[ -s "$SUMMARY_CSV" ]] || die "summary.csv is missing or empty"
[[ -s "$TASK_SUCCESS_RATES_CSV" ]] || die "task_success_rates.csv is missing or empty"
[[ -s "$EVIDENCE_JSON" ]] || die "evaluation_evidence.json is missing or empty"
chmod 0555 "$RESULTS_DIR"

PYTHONPATH="$PROVENANCE_PYTHONPATH" \
    "$COSMOS_PY" "$COSMOS_SOURCE/$DSW_EVIDENCE_HELPER_REL" \
    verify-sealed \
    --evidence-json "$EVIDENCE_JSON" \
    --runtime-receipt "$RUNTIME_RECEIPT" \
    --terminal-receipt "$TERMINAL_RECEIPT" \
    --results-dir "$RESULTS_DIR" \
    --summary-csv "$SUMMARY_CSV" \
    --task-success-rates-csv "$TASK_SUCCESS_RATES_CSV" \
    --output "$SEALED_VERIFICATION_RECEIPT" \
    >"$CONTROL_DIR/verify-sealed.log"

"$COSMOS_PY" - \
    "$SEALED_VERIFICATION_RECEIPT" \
    "$TASK_COUNT" \
    "$EXPECTED_EPISODE_COUNT" \
    "$LOGICAL_NUM_SHARDS" \
    "$PHYSICAL_GPU_COUNT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "tasks": int(sys.argv[2]),
    "episodes": int(sys.argv[3]),
    "logical_num_shards": int(sys.argv[4]),
    "physical_gpu_count": int(sys.argv[5]),
    "physical_server_count": 1,
}
if receipt.get("status") != "passed":
    raise SystemExit("sealed DSW evidence verification did not pass")
for key, value in expected.items():
    if receipt.get("counts", {}).get(key) != value:
        raise SystemExit(
            f"sealed DSW evidence count mismatch for {key}: "
            f"expected={value} actual={receipt.get('counts', {}).get(key)}"
        )
print("DSW_SEALED_EVIDENCE_COUNTS_MATCH")
PY

cleanup_scratch
trap - EXIT INT TERM
echo "ROBOCASA365_DSW_FORMAL_EVAL_SUCCESS" \
    "tasks=$TASK_COUNT" \
    "episodes_per_task=$EPISODES_PER_TASK" \
    "episodes=$EXPECTED_EPISODE_COUNT" \
    "logical_shards=$LOGICAL_NUM_SHARDS" \
    "physical_servers=$PHYSICAL_SERVER_COUNT" \
    "physical_gpus=$PHYSICAL_GPU_COUNT" \
    "summary_csv=$SUMMARY_CSV" \
    "task_success_rates_csv=$TASK_SUCCESS_RATES_CSV" \
    "sealed_receipt=$SEALED_VERIFICATION_RECEIPT"
