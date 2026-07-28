#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Same-allocation fail-closed production probe followed by the formal
# RoboCasa365 Cosmos3-Nano policy run.  The probe uses the final 128-sample
# microbatch, GA=2, EMA, compile, validation, full optimizer, and DCP save.
# The formal torchrun starts only after every probe artifact is verified.

set -Eeuo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_CONTAINER_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly PINNED_PERSONAL_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly PINNED_FORMAL_OUTPUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal"
readonly PINNED_FORMAL_LOG_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/logs/cwam-cosmos3-formal"
readonly PINNED_FORMAL_MIN_FREE_BYTES="5497558138880"
readonly PINNED_FORMAL_ESTIMATED_REQUIRED_BYTES="4398046511104"
readonly PINNED_RJOB_QUOTA_GROUP="eailabagent_gpu"
readonly PINNED_RJOB_EXCLUDED_NODE_TAG="node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the clean immutable formal snapshot.}"
: "${FORMAL_EXPERIMENT_ID:?Set the immutable formal experiment ID.}"
: "${FORMAL_RUN_NAME:=$FORMAL_EXPERIMENT_ID}"
: "${FORMAL_MANIFEST_PATH:?Set FORMAL_MANIFEST_PATH to the immutable experiment manifest.}"
: "${FORMAL_MANIFEST_SHA256:?Set the audited SHA-256 of FORMAL_MANIFEST_PATH.}"
: "${EXPECTED_SOURCE_COMMIT:?Set the pushed 40-character source commit.}"
: "${EXPECTED_WORKFLOW_SNAPSHOT_SHA256:?Set the clean workflow snapshot SHA-256.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set the audited SHA-256 of SNAPSHOT_MANIFEST.json.}"
: "${EXPECTED_CONTAINER_IMAGE:?Set the content-addressed repository@sha256 container image.}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE to 6 or 8.}"
: "${RJOB_NAME:?Set the unique H-cluster rjob show name for attempt ownership.}"
: "${SUBMISSION_RESERVATION_PATH:?Set the atomic submit-side reservation JSON path.}"
: "${RUN_FORMAL_AFTER_PROBE:=1}"
: "${COSMOS_ENV:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/cosmos-framework-09f23119-cu128-train}"
: "${ROBOCASA365_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/processed/cwam-cosmos3-v1-3b70fe1-d5eac857}"
: "${ROBOCASA365_SOURCE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_data/robocasa365-demian/raw/robocasa365-pretrain-atomic-d5eac857ee9b07ff089e89c594c64493b8df6e41}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/share_checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Wan-AI--Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${QWEN3_VL_TOKENIZER_PATH:=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-VL-8B-Instruct}"
: "${FORMAL_OUTPUT_ROOT:=$PINNED_FORMAL_OUTPUT_ROOT}"
: "${FORMAL_LOG_ROOT:=$PINNED_FORMAL_LOG_ROOT}"
: "${RJOB_QUOTA_GROUP:=$PINNED_RJOB_QUOTA_GROUP}"
: "${RJOB_EXCLUDED_NODE_TAG:=$PINNED_RJOB_EXCLUDED_NODE_TAG}"
: "${NVIDIA_SMI_INTERVAL_SECONDS:=2}"
: "${PROBE_MAX_MEMORY_FRACTION:=0.94}"
: "${FORMAL_MIN_FREE_BYTES:=$PINNED_FORMAL_MIN_FREE_BYTES}"
: "${FORMAL_ESTIMATED_REQUIRED_BYTES:=$PINNED_FORMAL_ESTIMATED_REQUIRED_BYTES}"
SNAPSHOT_ALLOWED_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots"
SECURE_PATH_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_secure_paths.py"
SECURE_TEE_HELPER="$COSMOS_SOURCE/examples/h_cluster/cwam_secure_tee.py"

[[ "$NPROC_PER_NODE" == 6 || "$NPROC_PER_NODE" == 8 ]] ||
    { echo "ERROR: NPROC_PER_NODE must be exactly 6 or 8" >&2; exit 1; }
if [[ "$NPROC_PER_NODE" == 8 ]]; then
    PROTOCOL_EXPERIMENT_FAMILY="CWAM-E35"
    PROTOCOL_LEARNING_RATE="5.0e-5"
    PROTOCOL_MAX_ITER=10000
    PROTOCOL_WARM_UP_STEPS=500
    PROTOCOL_CYCLE_LENGTH=16000
    PROTOCOL_VALIDATION_ITER=500
    PROTOCOL_MAX_VAL_ITER=8
    PROTOCOL_SAVE_ITER=500
    PROTOCOL_RJOB_CPU=128
    PROTOCOL_RJOB_MEMORY_MIB=1500000
else
    PROTOCOL_EXPERIMENT_FAMILY="CWAM-E35A"
    PROTOCOL_LEARNING_RATE="3.75e-5"
    PROTOCOL_MAX_ITER=13340
    PROTOCOL_WARM_UP_STEPS=667
    PROTOCOL_CYCLE_LENGTH=21344
    PROTOCOL_VALIDATION_ITER=667
    PROTOCOL_MAX_VAL_ITER=11
    PROTOCOL_SAVE_ITER=667
    PROTOCOL_RJOB_CPU=96
    PROTOCOL_RJOB_MEMORY_MIB=1125000
fi
[[ "$FORMAL_EXPERIMENT_ID" == "$PROTOCOL_EXPERIMENT_FAMILY"-* ]] ||
    {
        echo "ERROR: world_size=$NPROC_PER_NODE requires a $PROTOCOL_EXPERIMENT_FAMILY experiment ID" >&2
        exit 1
    }
[[ "$FORMAL_RUN_NAME" == "$FORMAL_EXPERIMENT_ID" ]] ||
    { echo "ERROR: FORMAL_RUN_NAME must equal FORMAL_EXPERIMENT_ID" >&2; exit 1; }
[[ "$RUN_FORMAL_AFTER_PROBE" =~ ^[01]$ ]] ||
    { echo "ERROR: RUN_FORMAL_AFTER_PROBE must be 0 or 1" >&2; exit 1; }
[[ "$NVIDIA_SMI_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: NVIDIA_SMI_INTERVAL_SECONDS must be positive" >&2; exit 1; }
[[ "$FORMAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: FORMAL_RUN_NAME contains unsupported characters" >&2; exit 1; }
[[ "$FORMAL_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "ERROR: FORMAL_EXPERIMENT_ID contains unsupported characters" >&2; exit 1; }
[[ "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
    { echo "ERROR: EXPECTED_SOURCE_COMMIT must be a lowercase 40-character Git SHA" >&2; exit 1; }
[[ "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_WORKFLOW_SNAPSHOT_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_SNAPSHOT_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$FORMAL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "ERROR: FORMAL_MANIFEST_SHA256 must be a lowercase SHA-256" >&2; exit 1; }
[[ "$EXPECTED_CONTAINER_IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] ||
    { echo "ERROR: EXPECTED_CONTAINER_IMAGE must be repository@sha256:<64 lowercase hex>" >&2; exit 1; }
[[ "$EXPECTED_CONTAINER_IMAGE" == "$PINNED_CONTAINER_IMAGE" ]] ||
    { echo "ERROR: EXPECTED_CONTAINER_IMAGE does not match the audited image digest" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
    { echo "ERROR: RJOB_NAME must be a lowercase Kubernetes name of at most 50 characters" >&2; exit 1; }
[[ "$FORMAL_MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: FORMAL_MIN_FREE_BYTES must be a positive integer" >&2; exit 1; }
[[ "$FORMAL_ESTIMATED_REQUIRED_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    { echo "ERROR: FORMAL_ESTIMATED_REQUIRED_BYTES must be a positive integer" >&2; exit 1; }
[[ "$FORMAL_OUTPUT_ROOT" == "$PINNED_FORMAL_OUTPUT_ROOT" ]] ||
    { echo "ERROR: FORMAL_OUTPUT_ROOT does not match the audited GPFS output root" >&2; exit 1; }
[[ "$FORMAL_LOG_ROOT" == "$PINNED_FORMAL_LOG_ROOT" ]] ||
    { echo "ERROR: FORMAL_LOG_ROOT does not match the audited GPFS log root" >&2; exit 1; }
[[ "$FORMAL_MIN_FREE_BYTES" == "$PINNED_FORMAL_MIN_FREE_BYTES" ]] ||
    { echo "ERROR: FORMAL_MIN_FREE_BYTES does not match the audited 5-TiB gate" >&2; exit 1; }
[[ "$FORMAL_ESTIMATED_REQUIRED_BYTES" == "$PINNED_FORMAL_ESTIMATED_REQUIRED_BYTES" ]] ||
    { echo "ERROR: FORMAL_ESTIMATED_REQUIRED_BYTES does not match the audited 4-TiB estimate" >&2; exit 1; }
[[ "$RJOB_QUOTA_GROUP" == "$PINNED_RJOB_QUOTA_GROUP" ]] ||
    { echo "ERROR: RJOB_QUOTA_GROUP does not match the audited quota group" >&2; exit 1; }
[[ "$RJOB_EXCLUDED_NODE_TAG" == "$PINNED_RJOB_EXCLUDED_NODE_TAG" ]] ||
    { echo "ERROR: RJOB_EXCLUDED_NODE_TAG does not match the audited exclusion" >&2; exit 1; }

test -d "$COSMOS_SOURCE"
test -s "$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
test -s "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"
test -s "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_training_evidence.py"
test -s "$SECURE_PATH_HELPER"
test -s "$SECURE_TEE_HELPER"
test -s "$FORMAL_MANIFEST_PATH"
test -d "$(dirname "$SUBMISSION_RESERVATION_PATH")"
test -x "$COSMOS_ENV/bin/python"
test -s "$COSMOS_ENV/.cwam_env_ready.json"
test -s "$ROBOCASA365_ROOT/_SUCCESS.json"
test -d "$ROBOCASA365_SOURCE_ROOT/videos"
test -s "$BASE_CHECKPOINT_PATH/model/.metadata"
test -s "$WAN_VAE_PATH"
test -s "$QWEN3_VL_TOKENIZER_PATH/tokenizer_config.json"
command -v nvidia-smi >/dev/null
command -v python3 >/dev/null
RUN_SCRIPT_REAL="$(realpath -e -- "${BASH_SOURCE[0]}")"
EXPECTED_RUN_SCRIPT="$COSMOS_SOURCE/examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh"
[[ "$RUN_SCRIPT_REAL" == "$(realpath -e -- "$EXPECTED_RUN_SCRIPT")" ]] ||
    {
        echo "ERROR: formal runtime wrapper must execute from the verified COSMOS_SOURCE snapshot" >&2
        exit 1
    }

python3 "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py" verify-snapshot \
    --snapshot-root "$COSMOS_SOURCE" \
    --allowed-root "$SNAPSHOT_ALLOWED_ROOT" \
    --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256"

"$COSMOS_ENV/bin/python" - \
    "$SUBMISSION_RESERVATION_PATH" \
    "$FORMAL_EXPERIMENT_ID" \
    "$FORMAL_RUN_NAME" \
    "$RJOB_NAME" \
    "$NPROC_PER_NODE" \
    "$EXPECTED_CONTAINER_IMAGE" \
    "$FORMAL_MANIFEST_SHA256" \
    "$EXPECTED_SOURCE_COMMIT" \
    "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$FORMAL_OUTPUT_ROOT" \
    "$FORMAL_MIN_FREE_BYTES" \
    "$FORMAL_ESTIMATED_REQUIRED_BYTES" \
    "$RJOB_QUOTA_GROUP" \
    "$PROTOCOL_RJOB_CPU" \
    "$PROTOCOL_RJOB_MEMORY_MIB" \
    "$RJOB_EXCLUDED_NODE_TAG" <<'PY'
import hashlib
import json
import os
import re
import socket
import stat
import sys
import time
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
output_root = Path(sys.argv[11])
literal_output_root = Path(os.path.abspath(output_root))
if literal_output_root != output_root or output_root.resolve(strict=True) != output_root:
    raise SystemExit(f"formal output root is not canonical: {output_root}")
expected_path = output_root / "formal_submission_reservations" / sys.argv[2] / "submission.json"
if Path(os.path.abspath(path)) != expected_path:
    raise SystemExit(
        f"submission reservation path is outside the canonical hierarchy: "
        f"expected={expected_path}, actual={path}"
    )

deadline = time.monotonic() + 600
unknown_path = expected_path.with_name("submission-unknown.json")
while True:
    if unknown_path.is_symlink():
        raise SystemExit(f"submission-unknown marker is unsafe: {unknown_path}")
    if unknown_path.is_file():
        raise SystemExit(
            f"submission entered an unknown state before acceptance: "
            f"{json.loads(unknown_path.read_text())!r}"
        )
    if path.is_symlink():
        raise SystemExit(f"submission reservation is unsafe: {path}")
    if path.is_file():
        if path.resolve(strict=True) != expected_path:
            raise SystemExit(f"submission reservation is non-canonical: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "accepted":
            raise SystemExit(f"submission reservation is not accepted: {payload!r}")
        if stat.S_IMODE(path.stat().st_mode) == 0o444:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("accepted submission reservation never became immutable")
        time.sleep(1)
        continue
    if time.monotonic() >= deadline:
        raise SystemExit("timed out waiting for an immutable accepted submission reservation")
    time.sleep(1)

expected_keys = {
    "schema_version",
    "status",
    "created_at",
    "accepted_at",
    "experiment_id",
    "run_name",
    "rjob_name",
    "scheduler_rjob_name",
    "world_size",
    "container_image",
    "formal_manifest_sha256",
    "source_commit",
    "workflow_snapshot_sha256",
    "source_manifest_sha256",
    "formal_output_root",
    "minimum_free_bytes",
    "estimated_required_bytes",
    "quota_group",
    "cpu",
    "memory_mib",
    "negative_tag",
    "task_type",
    "private_machine",
    "priority",
    "restart_policy",
    "backoff_limit",
    "submission_log",
    "submission_log_sha256",
    "scheduler_expected_contract",
    "scheduler_expected_contract_sha256",
    "scheduler_capture_log",
    "scheduler_capture_log_sha256",
    "scheduler_receipt",
    "scheduler_receipt_sha256",
    "scheduler_text_receipt",
    "scheduler_text_receipt_sha256",
}
if set(payload) != expected_keys:
    raise SystemExit(
        "submission reservation schema drift: "
        f"missing={sorted(expected_keys - set(payload))}, extra={sorted(set(payload) - expected_keys)}"
    )
expected = {
    "experiment_id": sys.argv[2],
    "run_name": sys.argv[3],
    "rjob_name": sys.argv[4],
    "world_size": int(sys.argv[5]),
    "container_image": sys.argv[6],
    "formal_manifest_sha256": sys.argv[7],
    "source_commit": sys.argv[8],
    "workflow_snapshot_sha256": sys.argv[9],
    "source_manifest_sha256": sys.argv[10],
    "formal_output_root": sys.argv[11],
    "minimum_free_bytes": int(sys.argv[12]),
    "estimated_required_bytes": int(sys.argv[13]),
    "quota_group": sys.argv[14],
    "cpu": int(sys.argv[15]),
    "memory_mib": int(sys.argv[16]),
    "negative_tag": sys.argv[17],
    "task_type": "normal",
    "private_machine": "group",
    "priority": 9,
    "restart_policy": "never",
    "backoff_limit": 1,
}
if payload.get("schema_version") != 2 or payload.get("status") != "accepted":
    raise SystemExit(f"submission reservation is not launchable: {payload!r}")
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"submission reservation {key} mismatch: expected={value!r}, actual={payload.get(key)!r}"
        )
created_at = datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
accepted_at = datetime.fromisoformat(payload["accepted_at"].replace("Z", "+00:00"))
if created_at.tzinfo is None or accepted_at.tzinfo is None or accepted_at < created_at:
    raise SystemExit("submission reservation timestamps are invalid or out of order")
scheduler_rjob_name = payload.get("scheduler_rjob_name")
if (
    not isinstance(scheduler_rjob_name, str)
    or re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", scheduler_rjob_name) is None
    or not scheduler_rjob_name.startswith(f"{sys.argv[4]}-")
):
    raise SystemExit(f"invalid scheduler rjob name: {scheduler_rjob_name!r}")

def sha256_file(candidate: Path) -> str:
    if candidate.is_symlink() or not candidate.is_file():
        raise SystemExit(f"submission receipt is missing or unsafe: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()

submission_log = expected_path.parent / "rjob-submit.log"
scheduler_expected_contract = expected_path.parent / "rjob-expected-contract.json"
scheduler_capture_log = expected_path.parent / "rjob-scheduler-capture.log"
scheduler_receipt = expected_path.parent / "rjob-scheduler-receipt.json"
scheduler_text_receipt = expected_path.parent / "rjob-get.log"
if payload.get("submission_log") != str(submission_log):
    raise SystemExit("submission log path is not canonical")
if payload.get("scheduler_expected_contract") != str(scheduler_expected_contract):
    raise SystemExit("scheduler expected-contract path is not canonical")
if payload.get("scheduler_capture_log") != str(scheduler_capture_log):
    raise SystemExit("scheduler capture-log path is not canonical")
if payload.get("scheduler_receipt") != str(scheduler_receipt):
    raise SystemExit("scheduler receipt path is not canonical")
if payload.get("scheduler_text_receipt") != str(scheduler_text_receipt):
    raise SystemExit("scheduler text-receipt path is not canonical")
if sha256_file(submission_log) != payload.get("submission_log_sha256"):
    raise SystemExit("submission log checksum mismatch")
if sha256_file(scheduler_expected_contract) != payload.get("scheduler_expected_contract_sha256"):
    raise SystemExit("scheduler expected-contract checksum mismatch")
if sha256_file(scheduler_capture_log) != payload.get("scheduler_capture_log_sha256"):
    raise SystemExit("scheduler capture-log checksum mismatch")
if sha256_file(scheduler_receipt) != payload.get("scheduler_receipt_sha256"):
    raise SystemExit("scheduler receipt checksum mismatch")
if sha256_file(scheduler_text_receipt) != payload.get("scheduler_text_receipt_sha256"):
    raise SystemExit("scheduler text-receipt checksum mismatch")
for immutable_path in (
    submission_log,
    scheduler_expected_contract,
    scheduler_capture_log,
    scheduler_receipt,
    scheduler_text_receipt,
):
    if stat.S_IMODE(immutable_path.stat().st_mode) != 0o444:
        raise SystemExit(f"scheduler evidence is not immutable: {immutable_path}")
if submission_log.read_text(errors="replace").count(
    f"created rjob_name: {scheduler_rjob_name}"
) != 1:
    raise SystemExit("submission log does not contain exactly one accepted scheduler rjob name")
if (
    f"rjob {scheduler_rjob_name} (showname={sys.argv[4]}):"
    not in scheduler_text_receipt.read_text(errors="replace")
):
    raise SystemExit("scheduler text receipt does not bind actual rjob name to requested showname")

scheduler = json.loads(scheduler_receipt.read_text())
scheduler_expected = json.loads(scheduler_expected_contract.read_text())
if (
    scheduler.get("schema_version") != 1
    or scheduler.get("kind") != "cwam_formal_rjob_scheduler_receipt"
    or scheduler.get("validation") != {
        "status": "passed",
        "expected_contract_schema_version": 1,
    }
):
    raise SystemExit("normalized scheduler receipt is not a passed schema-v1 receipt")
if scheduler.get("metadata", {}).get("name") != scheduler_rjob_name:
    raise SystemExit("normalized scheduler receipt has the wrong RJob identity")
if scheduler.get("metadata", {}).get("showname") != sys.argv[4]:
    raise SystemExit("normalized scheduler receipt has the wrong showname")
scheduler_rjob_uid = scheduler.get("metadata", {}).get("uid")
if not isinstance(scheduler_rjob_uid, str) or not scheduler_rjob_uid:
    raise SystemExit("normalized scheduler receipt has no RJob UID")
if os.environ.get("KUBEBRAIN_RJOB_UID") != scheduler_rjob_uid:
    raise SystemExit("KUBEBRAIN_RJOB_UID does not match the accepted scheduler RJob")
scheduler_spec = scheduler.get("spec", {})
scheduler_task = scheduler_spec.get("task", {})
scheduler_container = scheduler_task.get("container", {})
expected_scheduler_spec = {
    "quota_group": sys.argv[14],
    "task_type": "normal",
    "priority": 9,
    "backoff_limit": 1,
}
for key, value in expected_scheduler_spec.items():
    if scheduler_spec.get(key) != value:
        raise SystemExit(f"normalized scheduler receipt {key} mismatch")
if scheduler_task.get("replicas") != 1:
    raise SystemExit("normalized scheduler receipt replica count mismatch")
if scheduler_task.get("negative_tags") != [sys.argv[17]]:
    raise SystemExit("normalized scheduler receipt negative-tag mismatch")
if scheduler_task.get("private_machine") != "group":
    raise SystemExit("normalized scheduler receipt private-machine mismatch")
if scheduler_task.get("restart_policy") != "Never":
    raise SystemExit("normalized scheduler receipt restart-policy mismatch")
if scheduler_task.get("termination_grace_period_seconds") != 60:
    raise SystemExit("normalized scheduler receipt termination-grace mismatch")
if scheduler_container.get("image") != sys.argv[6]:
    raise SystemExit("normalized scheduler receipt image mismatch")
if scheduler_container.get("resources") != {
    "requests_equal_core_limits": True,
    "cpu": int(sys.argv[15]),
    "memory_mib": int(sys.argv[16]),
    "gpu": int(sys.argv[5]),
    "platform_injected_limit_keys": [],
}:
    raise SystemExit("normalized scheduler receipt resource mismatch")
if scheduler_container.get("snapshot_root") != os.environ["COSMOS_SOURCE"]:
    raise SystemExit("normalized scheduler receipt snapshot mismatch")
if scheduler_container.get("experiment_id") != sys.argv[2]:
    raise SystemExit("normalized scheduler receipt experiment mismatch")
pod_names = scheduler.get("pod_names")
replica_names = scheduler.get("replica_names")
if not isinstance(pod_names, list) or len(pod_names) != 1:
    raise SystemExit("normalized scheduler receipt must bind exactly one pod hostname")
if not isinstance(replica_names, list) or len(replica_names) != 1:
    raise SystemExit("normalized scheduler receipt must bind exactly one Replica CR")
replicas = scheduler.get("replicas")
if not isinstance(replicas, list) or len(replicas) != 1 or not isinstance(replicas[0], dict):
    raise SystemExit("normalized scheduler receipt must contain exactly one expanded Replica")
replica = replicas[0]
if replica.get("replica_name") != replica_names[0]:
    raise SystemExit("normalized expanded Replica identity mismatch")
if replica.get("pod_name") != pod_names[0] or replica.get("pod_hostname") != pod_names[0]:
    raise SystemExit("normalized expanded Replica pod identity mismatch")
worker_spec = replica.get("worker_spec")
if not isinstance(worker_spec, dict) or set(worker_spec) != {
    "image",
    "resources",
    "restart_policy",
    "termination_grace_period_seconds",
    "mounts",
    "command",
    "worker_init_prefix",
}:
    raise SystemExit("normalized expanded worker spec shape mismatch")
if worker_spec.get("image") != sys.argv[6]:
    raise SystemExit("normalized expanded worker image mismatch")
if worker_spec.get("restart_policy") != "Never":
    raise SystemExit("normalized expanded worker restart-policy mismatch")
if worker_spec.get("termination_grace_period_seconds") != 60:
    raise SystemExit("normalized expanded worker termination-grace mismatch")
if worker_spec.get("mounts") != scheduler_task.get("mounts"):
    raise SystemExit("normalized expanded worker mounts mismatch")
worker_resources = worker_spec.get("resources")
if not isinstance(worker_resources, dict):
    raise SystemExit("normalized expanded worker resources must be an object")
injected_limit_keys = worker_resources.get("platform_injected_limit_keys")
allowed_injected_limit_keys = {
    "mellanox.com/mlnx_rdma",
    "rdma/mlnx_shared",
}
if (
    not isinstance(injected_limit_keys, list)
    or injected_limit_keys != sorted(set(injected_limit_keys))
    or not set(injected_limit_keys).issubset(allowed_injected_limit_keys)
):
    raise SystemExit("normalized expanded worker has unsupported injected resource limits")
if worker_resources != {
    "requests_equal_core_limits": True,
    "cpu": int(sys.argv[15]),
    "memory_mib": int(sys.argv[16]),
    "gpu": int(sys.argv[5]),
    "platform_injected_limit_keys": injected_limit_keys,
}:
    raise SystemExit("normalized expanded worker resource mismatch")
template_command = scheduler_container.get("command")
worker_command = worker_spec.get("command")
worker_init_prefix = worker_spec.get("worker_init_prefix")
audited_worker_init_prefix = [
    "/kubebrain/worker-init",
    "run",
    "--enable-pid-monitor",
    "--",
]
if not isinstance(template_command, list) or not all(isinstance(item, str) for item in template_command):
    raise SystemExit("normalized scheduler template command is invalid")
if len(template_command) != 3 or template_command[:2] != ["bash", "-lc"]:
    raise SystemExit("normalized scheduler template command shape mismatch")
expected_command_sha256 = scheduler_expected.get("command_sha256")
if (
    not isinstance(expected_command_sha256, str)
    or re.fullmatch(r"[0-9a-f]{64}", expected_command_sha256) is None
    or hashlib.sha256(template_command[2].encode("utf-8")).hexdigest()
    != expected_command_sha256
):
    raise SystemExit("normalized scheduler formal-command SHA-256 mismatch")
if worker_init_prefix not in ([], audited_worker_init_prefix):
    raise SystemExit("normalized expanded worker-init prefix mismatch")
if worker_command != worker_init_prefix + template_command:
    raise SystemExit("normalized expanded worker command mismatch")
actual_hostname = socket.gethostname()
if pod_names[0] != actual_hostname:
    raise SystemExit(
        f"current worker hostname is not owned by scheduler receipt: "
        f"expected={pod_names[0]!r}, actual={actual_hostname!r}"
    )
if os.environ.get("KUBEBRAIN_RJOB_NAME") != scheduler_rjob_name:
    raise SystemExit("KUBEBRAIN_RJOB_NAME does not match the accepted scheduler RJob")
if os.environ.get("KUBEBRAIN_REPLICA_NAME") != replica_names[0]:
    raise SystemExit("KUBEBRAIN_REPLICA_NAME does not match the accepted Replica CR")
if os.environ.get("KUBEBRAIN_QUOTA_GROUP") != sys.argv[14]:
    raise SystemExit("KUBEBRAIN_QUOTA_GROUP does not match the formal quota group")
PY

PROBE_RUN_NAME="${FORMAL_RUN_NAME}-prod128-probe"
CANONICAL_RUN_ROOT="$FORMAL_OUTPUT_ROOT/cosmos3_action_robocasa365/action_sft"
PROBE_RUN_DIR="$CANONICAL_RUN_ROOT/$PROBE_RUN_NAME"
FORMAL_RUN_DIR="$CANONICAL_RUN_ROOT/$FORMAL_RUN_NAME"
EVIDENCE_DIR="$FORMAL_OUTPUT_ROOT/formal_evidence/$FORMAL_RUN_NAME"
PROBE_LOG="$FORMAL_LOG_ROOT/${FORMAL_RUN_NAME}.production-probe.log"
PROBE_RESUME_LOG="$FORMAL_LOG_ROOT/${FORMAL_RUN_NAME}.production-resume.log"
FORMAL_OUTER_LOG="$FORMAL_LOG_ROOT/${FORMAL_RUN_NAME}.formal.log"
SUMMARY_PATH="$EVIDENCE_DIR/summary.json"
INDUCTOR_CACHE_DIR="$EVIDENCE_DIR/inductor-cache"
OWNER_PATH="$EVIDENCE_DIR/owner.json"
PROVENANCE_PATH="$EVIDENCE_DIR/provenance.json"

"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" validate-root \
    --root "$PINNED_PERSONAL_GPFS_ROOT" >/dev/null
"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" validate-existing \
    --root "$PINNED_PERSONAL_GPFS_ROOT" \
    --path "$FORMAL_OUTPUT_ROOT" \
    --kind directory >/dev/null
"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" ensure-dir \
    --root "$PINNED_PERSONAL_GPFS_ROOT" \
    --path "$FORMAL_LOG_ROOT" \
    --create-parents \
    --directory-mode 0755 >/dev/null
"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" ensure-dir \
    --root "$FORMAL_OUTPUT_ROOT" \
    --path "$CANONICAL_RUN_ROOT" \
    --create-parents \
    --directory-mode 0755 >/dev/null
"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" ensure-dir \
    --root "$FORMAL_OUTPUT_ROOT" \
    --path "$(dirname "$EVIDENCE_DIR")" \
    --create-parents \
    --directory-mode 0755 >/dev/null
for fresh_path in "$PROBE_RUN_DIR" "$FORMAL_RUN_DIR"; do
    if [[ -e "$fresh_path" || -L "$fresh_path" ]]; then
        echo "ERROR: refusing to reuse formal/probe path: $fresh_path" >&2
        exit 1
    fi
done
if ! EVIDENCE_PATH_RECEIPT="$(
    "$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" create-dir-exclusive \
        --root "$FORMAL_OUTPUT_ROOT" \
        --path "$EVIDENCE_DIR" \
        --directory-mode 0755
)"; then
    echo "ERROR: another formal attempt already owns the evidence lock: $EVIDENCE_DIR" >&2
    exit 1
fi
"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" write-json-exclusive \
    --root "$FORMAL_OUTPUT_ROOT" \
    --path "$EVIDENCE_DIR/path-security.json" \
    --json "$EVIDENCE_PATH_RECEIPT" \
    --file-mode 0444 >/dev/null
"$COSMOS_ENV/bin/python" "$SECURE_PATH_HELPER" create-dir-exclusive \
    --root "$FORMAL_OUTPUT_ROOT" \
    --path "$INDUCTOR_CACHE_DIR" \
    --directory-mode 0755 >/dev/null

CURRENT_PHASE="preflight"
FINAL_STATUS="failed"
SAMPLER_PID=""

write_failure_summary() {
    local exit_code="$1"
    set +e
    if [[ "$FINAL_STATUS" != "passed" && -d "$EVIDENCE_DIR" ]]; then
        "$COSMOS_ENV/bin/python" - "$SUMMARY_PATH" "$FORMAL_EXPERIMENT_ID" \
            "$FORMAL_RUN_NAME" "$CURRENT_PHASE" "$exit_code" "$RJOB_NAME" \
            "$EXPECTED_CONTAINER_IMAGE" "$OWNER_PATH" "$PROVENANCE_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "failed",
    "experiment_id": sys.argv[2],
    "run_name": sys.argv[3],
    "failed_phase": sys.argv[4],
    "exit_code": int(sys.argv[5]),
    "rjob_name": sys.argv[6],
    "container_image": sys.argv[7],
    "attempt_owner": sys.argv[8] if Path(sys.argv[8]).is_file() else None,
    "provenance": sys.argv[9] if Path(sys.argv[9]).is_file() else None,
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY
    fi
}

stop_sampler() {
    local require_alive="${1:-0}"
    local pid="${SAMPLER_PID:-}"
    if [[ -z "$pid" ]]; then
        [[ "$require_alive" == 0 ]] && return 0
        echo "ERROR: nvidia-smi sampler PID is absent" >&2
        return 1
    fi
    local was_alive=0
    if kill -0 "$pid" 2>/dev/null; then
        was_alive=1
        kill -TERM "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    SAMPLER_PID=""
    if [[ "$require_alive" == 1 && "$was_alive" != 1 ]]; then
        echo "ERROR: nvidia-smi sampler exited before the production probe completed" >&2
        return 1
    fi
}

on_exit() {
    local exit_code="$1"
    set +e
    stop_sampler
    write_failure_summary "$exit_code"
}
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

"$COSMOS_ENV/bin/python" - \
    "$OWNER_PATH" \
    "$FORMAL_EXPERIMENT_ID" \
    "$FORMAL_RUN_NAME" \
    "$RJOB_NAME" \
    "$NPROC_PER_NODE" \
    "$EXPECTED_CONTAINER_IMAGE" \
    "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    "$FORMAL_MANIFEST_SHA256" \
    "$SUBMISSION_RESERVATION_PATH" \
    "$$" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_name,
    experiment_id,
    run_name,
    rjob_name,
    world_size,
    container_image,
    source_manifest_sha256,
    formal_manifest_sha256,
    submission_reservation_path,
    shell_pid,
) = sys.argv[1:]
reservation = json.loads(Path(submission_reservation_path).read_text())
scheduler_receipt_path = Path(reservation["scheduler_receipt"])
scheduler_receipt = json.loads(scheduler_receipt_path.read_text())
pod_names = scheduler_receipt["pod_names"]
replica_names = scheduler_receipt["replica_names"]
payload = {
    "schema_version": 1,
    "status": "owned",
    "acquired_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": experiment_id,
    "run_name": run_name,
    "rjob_name": rjob_name,
    "scheduler_rjob_name": reservation["scheduler_rjob_name"],
    "scheduler_rjob_uid": scheduler_receipt["metadata"]["uid"],
    "scheduler_replica_name": replica_names[0],
    "scheduler_pod_name": pod_names[0],
    "scheduler_receipt": str(scheduler_receipt_path),
    "scheduler_receipt_sha256": reservation["scheduler_receipt_sha256"],
    "hostname": socket.gethostname(),
    "node_name": os.environ["KUBEBRAIN_NODE_NAME"],
    "shell_pid": int(shell_pid),
    "world_size": int(world_size),
    "container_image": container_image,
    "source_manifest_sha256": source_manifest_sha256,
    "formal_manifest_sha256": formal_manifest_sha256,
}
target = Path(output_name)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY

export PATH="$COSMOS_ENV/bin:$PATH"
export PYTHONPATH="$COSMOS_SOURCE"
unset PYTHONOPTIMIZE
export COSMOS_DEVICE=cuda
export COSMOS_INTERNAL=0
export COSMOS_TRAINING=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OUTPUT_ROOT="$FORMAL_OUTPUT_ROOT"
export IMAGINAIRE_OUTPUT_ROOT="$FORMAL_OUTPUT_ROOT"
export COSMOS_ENV ROBOCASA365_ROOT ROBOCASA365_SOURCE_ROOT
export BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN3_VL_TOKENIZER_PATH
export NPROC_PER_NODE
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR"
export TORCHDYNAMO_DISABLE=0

source "$COSMOS_SOURCE/examples/h_cluster/configure_torchcodec_runtime.sh"
"$COSMOS_ENV/bin/python" "$COSMOS_SOURCE/examples/h_cluster/validate_cu128_train_env_marker.py" \
    --environment "$COSMOS_ENV"

cd "$COSMOS_SOURCE"

CURRENT_PHASE="storage_preflight"
"$COSMOS_ENV/bin/python" examples/h_cluster/cwam_formal_training_evidence.py verify-storage \
    --path "$FORMAL_OUTPUT_ROOT" \
    --minimum-free-bytes "$FORMAL_MIN_FREE_BYTES" \
    --estimated-required-bytes "$FORMAL_ESTIMATED_REQUIRED_BYTES" \
    --output-json "$EVIDENCE_DIR/storage-preflight.json"

CURRENT_PHASE="provenance_preflight"
"$COSMOS_ENV/bin/python" examples/h_cluster/cwam_formal_provenance.py verify-formal-contract \
    --snapshot-root "$COSMOS_SOURCE" \
    --snapshot-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
    --formal-manifest "$FORMAL_MANIFEST_PATH" \
    --formal-manifest-sha256 "$FORMAL_MANIFEST_SHA256" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --expected-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-workflow-snapshot-sha256 "$EXPECTED_WORKFLOW_SNAPSHOT_SHA256" \
    --nproc-per-node "$NPROC_PER_NODE" \
    --expected-container-image "$EXPECTED_CONTAINER_IMAGE" \
    --dataset-root "$ROBOCASA365_ROOT" \
    --source-root "$ROBOCASA365_SOURCE_ROOT" \
    --base-checkpoint "$BASE_CHECKPOINT_PATH" \
    --wan-vae "$WAN_VAE_PATH" \
    --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
    --environment "$COSMOS_ENV" \
    --output-json "$PROVENANCE_PATH"

"$COSMOS_ENV/bin/python" - \
    "$PROVENANCE_PATH" \
    "$OWNER_PATH" \
    "$FORMAL_RUN_NAME" \
    "$NPROC_PER_NODE" \
    "$EVIDENCE_DIR/storage-preflight.json" \
    "$EVIDENCE_DIR/preflight.json" <<'PY'
import json
import sys
from pathlib import Path

import torch

provenance_path, owner_path, run_name, nproc_text, storage_path, output_name = sys.argv[1:]
nproc = int(nproc_text)
provenance = json.loads(Path(provenance_path).read_text())
owner = json.loads(Path(owner_path).read_text())
storage = json.loads(Path(storage_path).read_text())
if provenance.get("status") != "passed":
    raise SystemExit(f"formal provenance gate did not pass: {provenance!r}")
if provenance.get("nproc_per_node") != nproc:
    raise SystemExit(f"formal provenance world-size mismatch: {provenance.get('nproc_per_node')!r}")
if storage.get("status") != "passed":
    raise SystemExit(f"formal storage gate did not pass: {storage!r}")
if torch.cuda.device_count() != nproc:
    raise SystemExit(f"expected {nproc} visible GPUs, got {torch.cuda.device_count()}")
devices = []
for index in range(nproc):
    props = torch.cuda.get_device_properties(index)
    if "H200" not in props.name:
        raise SystemExit(f"device {index} is not H200: {props.name}")
    devices.append({"index": index, "name": props.name, "total_memory_bytes": int(props.total_memory)})
payload = {
    "schema_version": 1,
    "status": "passed",
    "experiment_id": provenance["experiment_id"],
    "run_name": run_name,
    "attempt_owner": owner,
    "provenance": provenance,
    "storage": storage,
    "nproc_per_node": nproc,
    "devices": devices,
}
target = Path(output_name)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY

CURRENT_PHASE="cpu_production_batch"
"$COSMOS_ENV/bin/python" examples/validate_action_policy_robocasa365_nano_batch.py \
    --dataset-root "$ROBOCASA365_ROOT" \
    --source-root "$ROBOCASA365_SOURCE_ROOT" \
    --checkpoint "$BASE_CHECKPOINT_PATH" \
    --wan-vae "$WAN_VAE_PATH" \
    --qwen-tokenizer "$QWEN3_VL_TOKENIZER_PATH" \
    --output-json "$EVIDENCE_DIR/cpu-production-batch.json"

start_sampler() {
    local csv_path="$1"
    local stderr_path="$2"
    nvidia-smi \
        --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits \
        --loop="$NVIDIA_SMI_INTERVAL_SECONDS" \
        >"$csv_path" \
        2>"$stderr_path" &
    SAMPLER_PID=$!
    for _ in {1..50}; do
        [[ -s "$csv_path" ]] && return 0
        kill -0 "$SAMPLER_PID" 2>/dev/null || return 1
        sleep 0.2
    done
    echo "ERROR: nvidia-smi sampler did not produce probe samples" >&2
    return 1
}

validate_sampler() {
    local csv_path="$1"
    local output_path="$2"
    "$COSMOS_ENV/bin/python" - \
        "$csv_path" \
        "$output_path" \
        "$NPROC_PER_NODE" \
        "$PROBE_MAX_MEMORY_FRACTION" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected_gpus = int(sys.argv[3])
maximum_fraction = float(sys.argv[4])
if not 0.0 < maximum_fraction < 1.0:
    raise SystemExit(f"invalid maximum memory fraction: {maximum_fraction}")
records = []
with csv_path.open(newline="") as handle:
    for number, row in enumerate(csv.reader(handle), start=1):
        row = [value.strip() for value in row]
        if not row or not any(row):
            continue
        if len(row) != 9:
            raise SystemExit(f"malformed nvidia-smi row {number}: {row!r}")
        timestamp, index, uuid, name, total, used, free, utilization, power = row
        parsed = {
            "timestamp": timestamp,
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": float(total),
            "memory_used_mib": float(used),
            "memory_free_mib": float(free),
            "utilization_gpu_percent": float(utilization),
            "power_draw_w": None if power.casefold() in {"n/a", "[n/a]", "not supported"} else float(power),
        }
        for key in ("memory_total_mib", "memory_used_mib", "memory_free_mib"):
            if not math.isfinite(parsed[key]):
                raise SystemExit(f"non-finite {key} in row {number}")
        if "H200" not in name or not 140000 <= parsed["memory_total_mib"] <= 150000:
            raise SystemExit(f"unexpected GPU identity/capacity: {parsed}")
        parsed["memory_used_fraction"] = parsed["memory_used_mib"] / parsed["memory_total_mib"]
        records.append(parsed)
by_uuid = defaultdict(list)
for record in records:
    by_uuid[record["uuid"]].append(record)
if len(by_uuid) != expected_gpus:
    raise SystemExit(f"expected {expected_gpus} GPU UUIDs, got {len(by_uuid)}")
per_gpu = {}
maximum_observed = 0.0
for uuid, samples in sorted(by_uuid.items()):
    peak = max(item["memory_used_fraction"] for item in samples)
    maximum_observed = max(maximum_observed, peak)
    per_gpu[uuid] = {
        "index": samples[0]["index"],
        "name": samples[0]["name"],
        "sample_count": len(samples),
        "max_memory_used_fraction": peak,
        "max_memory_used_mib": max(item["memory_used_mib"] for item in samples),
        "min_memory_free_mib": min(item["memory_free_mib"] for item in samples),
        "max_utilization_gpu_percent": max(item["utilization_gpu_percent"] for item in samples),
    }
if maximum_observed >= maximum_fraction:
    raise SystemExit(
        f"probe sampled memory fraction {maximum_observed:.6f} >= limit {maximum_fraction:.6f}"
    )
payload = {
    "schema_version": 1,
    "status": "passed",
    "gpu_count": len(by_uuid),
    "sample_count": len(records),
    "max_memory_used_fraction": maximum_observed,
    "maximum_allowed_fraction": maximum_fraction,
    "per_gpu": per_gpu,
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
}

PROTOCOL_COMMON_TAIL_OVERRIDES="
optimizer.lr=$PROTOCOL_LEARNING_RATE
scheduler.warm_up_steps=[$PROTOCOL_WARM_UP_STEPS]
scheduler.cycle_lengths=[$PROTOCOL_CYCLE_LENGTH]
trainer.validation_iter=$PROTOCOL_VALIDATION_ITER
"

run_probe() {
    export RUN_NAME="$PROBE_RUN_NAME"
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export LOG_FILENAME="${FORMAL_RUN_NAME}.production-probe.torchrun.log"
    export EXTRA_TAIL_OVERRIDES="
$PROTOCOL_COMMON_TAIL_OVERRIDES
checkpoint.strict_resume=true
checkpoint.save_iter=1
trainer.max_iter=1
trainer.max_val_iter=1
trainer.logging_iter=1
trainer.callbacks.action_gradient_evidence.every_n=1
trainer.callbacks.device_monitor.every_n=1
trainer.callbacks.production_batch_memory_gate.required=true
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

CURRENT_PHASE="gpu_production_probe"
start_sampler \
    "$EVIDENCE_DIR/probe-gpu-samples.csv" \
    "$EVIDENCE_DIR/probe-gpu-samples.stderr"
set +e
run_probe 2>&1 |
    "$COSMOS_ENV/bin/python" "$SECURE_TEE_HELPER" \
        --root "$FORMAL_LOG_ROOT" \
        --path "$PROBE_LOG" \
        --file-mode 0444
PROBE_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
SAMPLER_STATUS=0
stop_sampler 1 || SAMPLER_STATUS=$?
if [[ "${PROBE_PIPE_STATUS[0]}" -ne 0 ]]; then
    echo "ERROR: production probe failed with exit ${PROBE_PIPE_STATUS[0]}" >&2
    exit "${PROBE_PIPE_STATUS[0]}"
fi
if [[ "${PROBE_PIPE_STATUS[1]}" -ne 0 ]]; then
    echo "ERROR: production probe tee failed with exit ${PROBE_PIPE_STATUS[1]}" >&2
    exit "${PROBE_PIPE_STATUS[1]}"
fi
if [[ "$SAMPLER_STATUS" -ne 0 ]]; then
    exit "$SAMPLER_STATUS"
fi
validate_sampler \
    "$EVIDENCE_DIR/probe-gpu-samples.csv" \
    "$EVIDENCE_DIR/probe-gpu-memory.json"

CURRENT_PHASE="probe_artifact_assertions"
"$COSMOS_ENV/bin/python" examples/h_cluster/cwam_formal_training_evidence.py verify-probe \
    --run-dir "$PROBE_RUN_DIR" \
    --log "$PROBE_LOG" \
    --cpu-batch "$EVIDENCE_DIR/cpu-production-batch.json" \
    --sampled-memory "$EVIDENCE_DIR/probe-gpu-memory.json" \
    --world-size "$NPROC_PER_NODE" \
    --inductor-cache "$INDUCTOR_CACHE_DIR" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --formal-run-name "$FORMAL_RUN_NAME" \
    --output-json "$EVIDENCE_DIR/probe-assertions.json"

run_probe_resume() {
    export RUN_NAME="$PROBE_RUN_NAME"
    export RESUME_EXISTING_RUN=1
    export ALLOW_DATA_STREAM_REPLAY=1
    export LOG_FILENAME="${FORMAL_RUN_NAME}.production-resume.torchrun.log"
    export EXTRA_TAIL_OVERRIDES="
$PROTOCOL_COMMON_TAIL_OVERRIDES
checkpoint.strict_resume=true
checkpoint.save_iter=1
trainer.max_iter=2
trainer.max_val_iter=1
trainer.logging_iter=1
trainer.callbacks.action_gradient_evidence.every_n=1
trainer.callbacks.device_monitor.every_n=1
trainer.callbacks.production_batch_memory_gate.required=true
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

CURRENT_PHASE="gpu_production_resume"
start_sampler \
    "$EVIDENCE_DIR/resume-gpu-samples.csv" \
    "$EVIDENCE_DIR/resume-gpu-samples.stderr"
set +e
run_probe_resume 2>&1 |
    "$COSMOS_ENV/bin/python" "$SECURE_TEE_HELPER" \
        --root "$FORMAL_LOG_ROOT" \
        --path "$PROBE_RESUME_LOG" \
        --file-mode 0444
RESUME_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
SAMPLER_STATUS=0
stop_sampler 1 || SAMPLER_STATUS=$?
if [[ "${RESUME_PIPE_STATUS[0]}" -ne 0 ]]; then
    echo "ERROR: production resume probe failed with exit ${RESUME_PIPE_STATUS[0]}" >&2
    exit "${RESUME_PIPE_STATUS[0]}"
fi
if [[ "${RESUME_PIPE_STATUS[1]}" -ne 0 ]]; then
    echo "ERROR: production resume tee failed with exit ${RESUME_PIPE_STATUS[1]}" >&2
    exit "${RESUME_PIPE_STATUS[1]}"
fi
if [[ "$SAMPLER_STATUS" -ne 0 ]]; then
    exit "$SAMPLER_STATUS"
fi
validate_sampler \
    "$EVIDENCE_DIR/resume-gpu-samples.csv" \
    "$EVIDENCE_DIR/resume-gpu-memory.json"

CURRENT_PHASE="resume_artifact_assertions"
"$COSMOS_ENV/bin/python" examples/h_cluster/cwam_formal_training_evidence.py verify-resume \
    --run-dir "$PROBE_RUN_DIR" \
    --log "$PROBE_RESUME_LOG" \
    --sampled-memory "$EVIDENCE_DIR/resume-gpu-memory.json" \
    --world-size "$NPROC_PER_NODE" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --formal-run-name "$FORMAL_RUN_NAME" \
    --output-json "$EVIDENCE_DIR/resume-assertions.json"

if [[ "$RUN_FORMAL_AFTER_PROBE" == 0 ]]; then
    FINAL_STATUS="passed"
    CURRENT_PHASE="probe_and_resume_complete"
    "$COSMOS_ENV/bin/python" - "$SUMMARY_PATH" "$FORMAL_EXPERIMENT_ID" "$FORMAL_RUN_NAME" \
        "$EVIDENCE_DIR/probe-assertions.json" "$EVIDENCE_DIR/resume-assertions.json" \
        "$EVIDENCE_DIR/storage-preflight.json" "$OWNER_PATH" "$PROVENANCE_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "probe_and_resume_passed_formal_not_requested",
    "experiment_id": sys.argv[2],
    "run_name": sys.argv[3],
    "probe_assertions": sys.argv[4],
    "resume_assertions": sys.argv[5],
    "storage_preflight": sys.argv[6],
    "attempt_owner": sys.argv[7],
    "provenance": sys.argv[8],
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
PY
    echo "CWAM_COSMOS3_PRODUCTION_PROBE_AND_RESUME_PASSED formal_started=0 summary=$SUMMARY_PATH"
    exit 0
fi

CURRENT_PHASE="formal_storage_preflight"
"$COSMOS_ENV/bin/python" examples/h_cluster/cwam_formal_training_evidence.py verify-storage \
    --path "$FORMAL_OUTPUT_ROOT" \
    --minimum-free-bytes "$FORMAL_MIN_FREE_BYTES" \
    --estimated-required-bytes "$FORMAL_ESTIMATED_REQUIRED_BYTES" \
    --output-json "$EVIDENCE_DIR/formal-storage-preflight.json"

run_formal() {
    export RUN_NAME="$FORMAL_RUN_NAME"
    export RESUME_EXISTING_RUN=0
    export ALLOW_DATA_STREAM_REPLAY=0
    export LOG_FILENAME="${FORMAL_RUN_NAME}.formal.torchrun.log"
    export EXTRA_TAIL_OVERRIDES="
$PROTOCOL_COMMON_TAIL_OVERRIDES
checkpoint.strict_resume=true
checkpoint.save_iter=$PROTOCOL_SAVE_ITER
trainer.max_iter=$PROTOCOL_MAX_ITER
trainer.max_val_iter=$PROTOCOL_MAX_VAL_ITER
trainer.callbacks.action_gradient_evidence.every_n=50
trainer.callbacks.production_batch_memory_gate.required=true
"
    bash examples/launch_sft_action_policy_robocasa365_nano.sh
}

CURRENT_PHASE="formal_training"
start_sampler \
    "$EVIDENCE_DIR/formal-gpu-samples.csv" \
    "$EVIDENCE_DIR/formal-gpu-samples.stderr"
set +e
run_formal 2>&1 |
    "$COSMOS_ENV/bin/python" "$SECURE_TEE_HELPER" \
        --root "$FORMAL_LOG_ROOT" \
        --path "$FORMAL_OUTER_LOG" \
        --file-mode 0444
FORMAL_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
SAMPLER_STATUS=0
stop_sampler 1 || SAMPLER_STATUS=$?
if [[ "${FORMAL_PIPE_STATUS[0]}" -ne 0 ]]; then
    echo "ERROR: formal training failed with exit ${FORMAL_PIPE_STATUS[0]}" >&2
    exit "${FORMAL_PIPE_STATUS[0]}"
fi
if [[ "${FORMAL_PIPE_STATUS[1]}" -ne 0 ]]; then
    echo "ERROR: formal training tee failed with exit ${FORMAL_PIPE_STATUS[1]}" >&2
    exit "${FORMAL_PIPE_STATUS[1]}"
fi
if [[ "$SAMPLER_STATUS" -ne 0 ]]; then
    exit "$SAMPLER_STATUS"
fi
validate_sampler \
    "$EVIDENCE_DIR/formal-gpu-samples.csv" \
    "$EVIDENCE_DIR/formal-gpu-memory.json"

CURRENT_PHASE="formal_final_assertions"
"$COSMOS_ENV/bin/python" examples/h_cluster/cwam_formal_training_evidence.py verify-formal \
    --run-dir "$FORMAL_RUN_DIR" \
    --log "$FORMAL_OUTER_LOG" \
    --world-size "$NPROC_PER_NODE" \
    --formal-run-name "$FORMAL_RUN_NAME" \
    --probe-assertions "$EVIDENCE_DIR/probe-assertions.json" \
    --resume-assertions "$EVIDENCE_DIR/resume-assertions.json" \
    --experiment-id "$FORMAL_EXPERIMENT_ID" \
    --formal-manifest "$FORMAL_MANIFEST_PATH" \
    --formal-manifest-sha256 "$FORMAL_MANIFEST_SHA256" \
    --owner "$OWNER_PATH" \
    --provenance "$PROVENANCE_PATH" \
    --storage-preflight "$EVIDENCE_DIR/formal-storage-preflight.json" \
    --submission-reservation "$SUBMISSION_RESERVATION_PATH" \
    --sampled-memory "$EVIDENCE_DIR/formal-gpu-memory.json" \
    --output-json "$SUMMARY_PATH"

FINAL_STATUS="passed"
CURRENT_PHASE="complete"
echo "CWAM_COSMOS3_FORMAL_TRAINING_COMPLETED summary=$SUMMARY_PATH control_success_claim=false"
date -Is
